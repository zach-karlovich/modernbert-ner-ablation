"""
Fine-tune ModernBERT-base + CRF on CoNLL-2003 NER with document context.

Single config: doc_5e5_bs2 (lr=5e-5, crf_lr=2.5e-4, 5 epochs).

Optimizations vs the original combined script:
- bf16 autocast (CRF emissions kept in fp32 via model)
- gradient checkpointing
- batch_size 4 (up from 2), grad_accum 4 (effective batch 16)
- pin_memory on DataLoaders
"""

import copy
import json
import random
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from seqeval.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from conll2003_labels import id2label, label2id
from dense_bio_labels import (
    assign_dense_bio_labels,
    collapse_to_word_labels,
    word_ids_list_to_tensor_ids,
)
from modernbert_crf_model import ModernBertTokenCRF, build_crf_optimizer

MODEL_ID = "answerdotai/ModernBERT-base"
MAX_SEQ_LENGTH = 8192
GRAD_ACCUM_STEPS = 4
WORD_PAD_ID = -99


def parse_conll_documents(filepath):
    documents = []
    current_doc = []
    current_sent = []

    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if line.startswith("-DOCSTART-"):
                if current_sent:
                    current_doc.append(current_sent)
                    current_sent = []
                if current_doc:
                    documents.append(current_doc)
                    current_doc = []
                continue

            if line == "":
                if current_sent:
                    current_doc.append(current_sent)
                    current_sent = []
                continue

            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Malformed line {line_num} in {filepath}: {line!r}"
                )
            current_sent.append((parts[0], parts[-1]))

    if current_sent:
        current_doc.append(current_sent)
    if current_doc:
        documents.append(current_doc)

    return documents


def set_seeds_to(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_info):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ConllDocContextDatasetCRF(Dataset):
    def __init__(self, documents, tokenizer, max_length=MAX_SEQ_LENGTH):
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.special_tokens = tokenizer.num_special_tokens_to_add(pair=False)

        self.targets = []
        self.sent_token_lens = []
        for d_idx, doc in enumerate(documents):
            doc_lens = []
            for s_idx, sentence in enumerate(doc):
                words = [w for w, _ in sentence]
                token_len = len(
                    tokenizer(
                        words,
                        is_split_into_words=True,
                        add_special_tokens=False,
                    )["input_ids"]
                )
                doc_lens.append(token_len)
                self.targets.append((d_idx, s_idx))
            self.sent_token_lens.append(doc_lens)

    def __len__(self):
        return len(self.targets)

    def _select_sentence_indices(self, d_idx, s_idx):
        budget = self.max_length - self.special_tokens
        selected = {s_idx}
        used = self.sent_token_lens[d_idx][s_idx]

        left = s_idx - 1
        right = s_idx + 1
        add_right_first = True

        while True:
            added = False
            order = ("right", "left") if add_right_first else ("left", "right")
            for side in order:
                if side == "left" and left >= 0:
                    candidate_len = self.sent_token_lens[d_idx][left]
                    if used + candidate_len <= budget:
                        selected.add(left)
                        used += candidate_len
                        left -= 1
                        added = True
                    else:
                        left = -1
                if side == "right" and right < len(self.documents[d_idx]):
                    candidate_len = self.sent_token_lens[d_idx][right]
                    if used + candidate_len <= budget:
                        selected.add(right)
                        used += candidate_len
                        right += 1
                        added = True
                    else:
                        right = len(self.documents[d_idx])

            if not added:
                break
            add_right_first = not add_right_first

        return sorted(selected)

    def __getitem__(self, idx):
        d_idx, s_idx = self.targets[idx]
        sentence_ids = self._select_sentence_indices(d_idx, s_idx)

        words = []
        tags = []
        is_target_word = []
        for cur_s_idx in sentence_ids:
            sent = self.documents[d_idx][cur_s_idx]
            for word, tag in sent:
                words.append(word)
                tags.append(tag)
                is_target_word.append(cur_s_idx == s_idx)

        def _current_len():
            return (
                len(
                    self.tokenizer(
                        words,
                        is_split_into_words=True,
                        add_special_tokens=False,
                    )["input_ids"]
                )
                + self.special_tokens
            )

        while len(words) > 0 and _current_len() > self.max_length:
            if not is_target_word[0]:
                words.pop(0)
                tags.pop(0)
                is_target_word.pop(0)
            elif not is_target_word[-1]:
                words.pop()
                tags.pop()
                is_target_word.pop()
            else:
                break

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )

        word_ids = encoding.word_ids()
        label_ids = assign_dense_bio_labels(word_ids, tags)
        w_tensor = torch.tensor(
            word_ids_list_to_tensor_ids(word_ids), dtype=torch.long
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "word_ids": w_tensor,
            "word_include_mask": is_target_word,
        }


class ConllDocCollatorCRF:
    def __init__(self, pad_token_id: int, label_pad_id: int):
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    def __call__(self, batch):
        max_len = max(item["input_ids"].size(0) for item in batch)
        pad_id = self.pad_token_id
        lp = self.label_pad_id
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        word_ids_list = []
        includes = []
        for item in batch:
            pad_len = max_len - item["input_ids"].size(0)
            input_ids_list.append(
                F.pad(item["input_ids"], (0, pad_len), value=pad_id)
            )
            attention_mask_list.append(
                F.pad(item["attention_mask"], (0, pad_len), value=0)
            )
            labels_list.append(
                F.pad(item["labels"], (0, pad_len), value=lp)
            )
            word_ids_list.append(
                F.pad(item["word_ids"], (0, pad_len), value=WORD_PAD_ID)
            )
            includes.append(item["word_include_mask"])
        return (
            torch.stack(input_ids_list),
            torch.stack(attention_mask_list),
            torch.stack(labels_list),
            torch.stack(word_ids_list),
            includes,
        )


def train_epoch(model, dataloader, optimizer, scheduler, device, clip,
                grad_accum_steps=1):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    accum_count = 0
    for step, batch in enumerate(tqdm(dataloader, desc="Training"), 1):
        input_ids, attention_mask, labels, _, _ = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(input_ids, attention_mask, labels)
        loss = loss / grad_accum_steps
        loss.backward()
        running_loss += (loss.item() * grad_accum_steps)
        accum_count += 1

        if accum_count == grad_accum_steps or step == len(dataloader):
            if accum_count < grad_accum_steps:
                scale = grad_accum_steps / accum_count
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accum_count = 0
    return running_loss / len(dataloader)


def evaluate(model, dataloader, device, id2label_map):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids, attention_mask, labels, word_ids, includes = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model(input_ids, attention_mask, labels)
            running_loss += loss.item()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                paths = model.decode(input_ids, attention_mask)
            bsz = input_ids.size(0)
            for i in range(bsz):
                L = int(attention_mask[i].sum().item())
                row_w = word_ids[i, :L].tolist()
                row_m = attention_mask[i, :L].tolist()
                pred_ids = paths[i][:L]
                gold_ids = labels[i, :L].tolist()
                pred_seq = collapse_to_word_labels(
                    row_w, pred_ids, row_m, id2label_map, includes[i]
                )
                label_seq = collapse_to_word_labels(
                    row_w, gold_ids, row_m, id2label_map, includes[i]
                )
                all_preds.append(pred_seq)
                all_true.append(label_seq)
    epoch_loss = running_loss / len(dataloader)
    f1 = cast(float, f1_score(all_true, all_preds))
    return epoch_loss, f1


def get_predictions(model, dataloader, device, id2label_map):
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids, attention_mask, labels, word_ids, includes = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                paths = model.decode(input_ids, attention_mask)
            bsz = input_ids.size(0)
            for i in range(bsz):
                L = int(attention_mask[i].sum().item())
                row_w = word_ids[i, :L].tolist()
                row_m = attention_mask[i, :L].tolist()
                pred_ids = paths[i][:L]
                gold_ids = labels[i, :L].tolist()
                pred_seq = collapse_to_word_labels(
                    row_w, pred_ids, row_m, id2label_map, includes[i]
                )
                label_seq = collapse_to_word_labels(
                    row_w, gold_ids, row_m, id2label_map, includes[i]
                )
                all_preds.append(pred_seq)
                all_true.append(label_seq)
    return all_preds, all_true


def aggregate_reports(reports):
    metric_keys = ["precision", "recall", "f1-score"]
    all_keys = []
    seen = set()
    for r in reports:
        for key in r:
            if key != "accuracy" and key not in seen:
                all_keys.append(key)
                seen.add(key)

    rows = []
    for key in all_keys:
        row = {"": key}
        for r in reports:
            if key in r and "support" in r[key]:
                support = r[key].get("support", np.nan)
                if not np.isnan(support):
                    row["support"] = int(support)
                break
        for m in metric_keys:
            vals = [r[key][m] if key in r and m in r[key] else 0.0 for r in reports]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            row[m] = f"{mean_val:.4f} ± {std_val:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def save_run_config(
    path: Path,
    cfg_name: str,
    cfg: dict[str, Any],
    seeds: list[int],
    extra: dict[str, Any],
) -> None:
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path.parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = "unknown"
    import importlib.metadata as im

    payload = {
        "config_name": cfg_name,
        "seeds": seeds,
        "git_commit": git_hash,
        "hyperparameters": cfg,
        "torch_version": torch.__version__,
        "transformers_version": im.version("transformers"),
        "pytorch_crf_version": im.version("pytorch-crf"),
        **extra,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


NUM_WORKERS = 2

SCRIPT_NAME = "train_modernbert_doc_crf_ner_doc5e5.py"

HP_CONFIG = {
    "name": "doc_5e5_bs2",
    "lr": 5e-5,
    "crf_lr": 2.5e-4,
    "epochs": 5,
    "warmup_ratio": 0.10,
    "weight_decay": 0.01,
    "batch_size": 4,
}

if __name__ == "__main__":
    SEEDS = [21, 42, 63]
    CLIP = 1
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(
        f"Model: {MODEL_ID}, max_seq_length={MAX_SEQ_LENGTH}, "
        f"grad_accum_steps={GRAD_ACCUM_STEPS}"
    )

    data_dir = Path(__file__).resolve().parent.parent / "data" / "conll2003"
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    train_docs = parse_conll_documents(data_dir / "eng.train")
    dev_docs = parse_conll_documents(data_dir / "eng.testa")
    test_docs = parse_conll_documents(data_dir / "eng.testb")
    print(f"Train: {sum(len(doc) for doc in train_docs)} sentences")
    print(f"Dev: {sum(len(doc) for doc in dev_docs)} sentences")
    print(f"Test: {sum(len(doc) for doc in test_docs)} sentences")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = ConllDocContextDatasetCRF(
        train_docs, tokenizer, max_length=MAX_SEQ_LENGTH
    )
    dev_dataset = ConllDocContextDatasetCRF(
        dev_docs, tokenizer, max_length=MAX_SEQ_LENGTH
    )
    test_dataset = ConllDocContextDatasetCRF(
        test_docs, tokenizer, max_length=MAX_SEQ_LENGTH
    )

    collate_fn = ConllDocCollatorCRF(
        tokenizer.pad_token_id, label_pad_id=label2id["O"]
    )

    loader_generator = torch.Generator()

    cfg = HP_CONFIG
    cfg_name = cfg["name"]
    lr = cfg["lr"]
    crf_lr = cfg["crf_lr"]
    n_epochs = cfg["epochs"]
    warmup_ratio = cfg["warmup_ratio"]
    weight_decay = cfg["weight_decay"]
    batch_size = cfg["batch_size"]

    save_run_config(
        results_dir / f"modernbert_doc_crf_ner_config_{cfg_name}.json",
        cfg_name,
        cfg,
        SEEDS,
        {
            "model_id": MODEL_ID,
            "max_seq_length": MAX_SEQ_LENGTH,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "script": SCRIPT_NAME,
        },
    )

    print(f"\n{'#' * 60}")
    print(
        f"CONFIG {cfg_name}: lr={lr}, crf_lr={crf_lr}, epochs={n_epochs}, "
        f"warmup={warmup_ratio}, wd={weight_decay}, bs={batch_size}"
    )
    print(f"{'#' * 60}")

    reports = []
    best_val_f1s = []
    best_epochs = []

    for run, seed in enumerate(SEEDS, 1):
        print(f"\n{'=' * 50}")
        print(f"[Config {cfg_name}] Run {run}/{len(SEEDS)} — Seed {seed}")
        print("=" * 50)
        set_seeds_to(seed)
        loader_generator.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            persistent_workers=NUM_WORKERS > 0,
            pin_memory=True,
            worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
            generator=loader_generator,
        )
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            persistent_workers=NUM_WORKERS > 0,
            pin_memory=True,
            worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            persistent_workers=NUM_WORKERS > 0,
            pin_memory=True,
            worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
        )

        model = ModernBertTokenCRF(
            MODEL_ID, trust_remote_code=True
        ).to(device)
        model.base.gradient_checkpointing_enable()

        opt_steps_per_epoch = (
            (len(train_loader) + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS
        )
        total_steps = opt_steps_per_epoch * n_epochs
        warmup_steps = int(warmup_ratio * total_steps)
        optimizer = build_crf_optimizer(
            model, lr=lr, crf_lr=crf_lr, weight_decay=weight_decay
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_val_f1 = 0.0
        best_epoch = -1
        best_state_dict = None
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            print("-" * 30)
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, device, CLIP,
                grad_accum_steps=GRAD_ACCUM_STEPS,
            )
            val_loss, val_f1 = evaluate(model, dev_loader, device, id2label)
            print(f"Train Loss: {train_loss:.3f}")
            print(f"Val Loss: {val_loss:.3f}, Val F1: {val_f1:.4f}")
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                best_state_dict = copy.deepcopy(model.state_dict())

        best_val_f1s.append(best_val_f1)
        best_epochs.append(best_epoch)
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        print(
            f"[Config {cfg_name}] Seed {seed} best dev F1 "
            f"{best_val_f1:.4f} at epoch {best_epoch}; "
            "restored best checkpoint for test evaluation."
        )

        all_preds, all_true = get_predictions(model, test_loader, device, id2label)
        report = classification_report(all_true, all_preds, output_dict=True)
        reports.append(report)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = aggregate_reports(reports).set_index("")
    print(f"\n=== Config {cfg_name} Test Results (seeds {SEEDS}) — mean ± std ===")
    print(df.to_string())

    csv_name = f"modernbert_doc_crf_ner_config_{cfg_name}.csv"
    df.to_csv(results_dir / csv_name)
    print(f"Saved to {csv_name}")
