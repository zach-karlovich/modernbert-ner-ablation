"""Fine-tune ModernBERT-base on CoNLL-2003 NER with in-document context (8192 tokens).

Run identity: `-DOCSTART-` respected; sliding windows; softmax head; seqeval. Doc side of
the 2×2 ablation vs sentence-level + CRF variants. Single HP config **doc_4e5_bs2**
(lr 4e-5). Outputs `ner_mbert_doc_4e5_bs2.{csv,json}`."""

import copy
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from seqeval.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from conll2003_expectations import assert_conll2003_dataset
from sliding_window_conll import (
    DEFAULT_TOKEN_OVERLAP,
    build_windows_word_ranges,
    filter_windows_intersecting_target,
    pick_best_centered_window,
    prefix_subwords_per_word,
    word_window_from_start,
)

MODEL_ID = "answerdotai/ModernBERT-base"
MAX_SEQ_LENGTH = 8192
GRAD_ACCUM_STEPS = 8
OUTPUT_STEM = "ner_mbert_doc_4e5_bs2"

RUN_DESCRIPTION = (
    "Document-context ModernBERT-base on CoNLL-2003; max 8192 subwords; sliding-window "
    "packing. Config doc_4e5_bs2: lr 4e-5, wd 0.01, 5 epochs, batch 2, grad_accum 8, "
    "warmup 0.1; default classifier dropout; no early stopping. "
    "Writes ner_mbert_doc_4e5_bs2.{csv,json}."
)


def parse_conll_documents(filepath):
    """Parse CoNLL-2003 into documents -> sentences -> (word, tag)."""
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


label_list = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}


def set_seeds_to(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_info):
    """Ensure each DataLoader worker has a unique but reproducible seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ConllDocContextDataset(Dataset):
    """Per-sentence supervision with same-document neighbor context (ModernBERT length)."""

    def __init__(
        self,
        documents,
        tokenizer,
        label2id,
        max_length=MAX_SEQ_LENGTH,
        window_mode: Literal["train", "eval"] = "train",
        token_overlap: int = DEFAULT_TOKEN_OVERLAP,
    ):
        self.documents = documents
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
        self.window_mode = window_mode
        self.token_overlap = token_overlap
        self._trunc_logged: set[tuple[int, int, int, int]] = set()

        self.sent_token_lens = []
        for d_idx, doc in enumerate(documents):
            doc_lens = []
            for sentence in doc:
                words = [w for w, _ in sentence]
                token_len = len(
                    tokenizer(
                        words,
                        is_split_into_words=True,
                        add_special_tokens=False,
                    )["input_ids"]
                )
                doc_lens.append(token_len)
            self.sent_token_lens.append(doc_lens)

        self._pack_cache: dict[
            tuple[int, int], tuple[list[str], list[str], list[bool]]
        ] = {}
        self.rows: list[tuple[int, int, int, int]] = []
        self.n_multi_window_overflow = 0

        for d_idx, doc in enumerate(documents):
            for s_idx in range(len(doc)):
                words, tags, is_tw = self._build_packed_lists(d_idx, s_idx)
                self._pack_cache[(d_idx, s_idx)] = (words, tags, is_tw)
                prefix = prefix_subwords_per_word(tokenizer, words)
                budget = max_length - self.special_tokens
                tw_idx = [i for i, v in enumerate(is_tw) if v]
                target_lo, target_hi_excl = tw_idx[0], tw_idx[-1] + 1

                if prefix[-1] <= budget:
                    self.rows.append((d_idx, s_idx, 0, len(words)))
                else:
                    ranges = build_windows_word_ranges(
                        prefix, budget, token_overlap
                    )
                    cand = filter_windows_intersecting_target(
                        ranges, target_lo, target_hi_excl
                    )
                    if not cand:
                        cand = [
                            word_window_from_start(
                                prefix, target_lo, budget
                            )
                        ]
                    if len(cand) > 1:
                        self.n_multi_window_overflow += 1
                    if window_mode == "train":
                        for w0, w1 in cand:
                            self.rows.append((d_idx, s_idx, w0, w1))
                    else:
                        w0, w1 = pick_best_centered_window(
                            prefix, cand, target_lo, target_hi_excl
                        )
                        self.rows.append((d_idx, s_idx, w0, w1))

        self.targets = [(d, s) for d, s, _, _ in self.rows]
        self.n_base_sentences = sum(len(d) for d in documents)

    def __len__(self):
        return len(self.rows)

    def _build_packed_lists(
        self, d_idx: int, s_idx: int
    ) -> tuple[list[str], list[str], list[bool]]:
        sentence_ids = self._select_sentence_indices(d_idx, s_idx)
        words: list[str] = []
        tags: list[str] = []
        is_target_word: list[bool] = []
        for cur_s_idx in sentence_ids:
            sent = self.documents[d_idx][cur_s_idx]
            for word, tag in sent:
                words.append(word)
                tags.append(tag)
                is_target_word.append(cur_s_idx == s_idx)
        return words, tags, is_target_word

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
        d_idx, s_idx, w0, w1 = self.rows[idx]
        words, tags, is_tw = self._pack_cache[(d_idx, s_idx)]
        words = words[w0:w1]
        tags = tags[w0:w1]
        is_target_word = is_tw[w0:w1]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=False,
            truncation=False,
            max_length=self.max_length,
        )
        n_tok = encoding["input_ids"].shape[-1]
        if n_tok > self.max_length:
            encoding = self.tokenizer(
                words,
                is_split_into_words=True,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            key = (d_idx, s_idx, w0, w1)
            if key not in self._trunc_logged:
                self._trunc_logged.add(key)
                print(
                    "WARNING: truncation fallback for row "
                    f"{key} (len was {n_tok})"
                )
            n_tok = encoding["input_ids"].shape[-1]

        assert n_tok <= self.max_length, (
            f"sequence length {n_tok} > max_length {self.max_length}"
        )

        word_ids = encoding.word_ids()
        labels = []
        for i in range(len(word_ids)):
            if word_ids[i] is None:
                labels.append(-100)
            elif i > 0 and word_ids[i] == word_ids[i - 1]:
                labels.append(-100)
            elif is_target_word[word_ids[i]]:
                labels.append(self.label2id[tags[word_ids[i]]])
            else:
                labels.append(-100)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class ConllDocCollator:
    """Top-level collator so DataLoader workers can pickle it (Python 3.14+ forkserver)."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(item["input_ids"].size(0) for item in batch)
        pad_id = self.pad_token_id
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        for item in batch:
            pad_len = max_len - item["input_ids"].size(0)
            input_ids_list.append(
                torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=pad_id)
            )
            attention_mask_list.append(
                torch.nn.functional.pad(item["attention_mask"], (0, pad_len), value=0)
            )
            labels_list.append(
                torch.nn.functional.pad(item["labels"], (0, pad_len), value=-100)
            )
        return (
            torch.stack(input_ids_list),
            torch.stack(attention_mask_list),
            torch.stack(labels_list),
        )


def train_epoch(model, dataloader, optimizer, scheduler, device, clip,
                grad_accum_steps=1):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    accum_count = 0
    for step, batch in enumerate(tqdm(dataloader, desc="Training"), 1):
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss / grad_accum_steps
        loss.backward()
        running_loss += outputs.loss.item()
        accum_count += 1

        if accum_count == grad_accum_steps or step == len(dataloader):
            # Scale gradients for a partial final accumulation window
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


def evaluate(model, dataloader, device, id2label):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            running_loss += outputs.loss.item()
            preds = outputs.logits.argmax(-1)
            for i in range(labels.size(0)):
                pred_seq = []
                label_seq = []
                for j in range(labels.size(1)):
                    if labels[i, j].item() != -100:
                        pred_seq.append(id2label[preds[i, j].item()])
                        label_seq.append(id2label[labels[i, j].item()])
                all_preds.append(pred_seq)
                all_true.append(label_seq)
    epoch_loss = running_loss / len(dataloader)
    f1 = cast(float, f1_score(all_true, all_preds))
    return epoch_loss, f1


def get_predictions(model, dataloader, device, id2label):
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(-1)
            for i in range(labels.size(0)):
                pred_seq = []
                label_seq = []
                for j in range(labels.size(1)):
                    if labels[i, j].item() != -100:
                        pred_seq.append(id2label[preds[i, j].item()])
                        label_seq.append(id2label[labels[i, j].item()])
                all_preds.append(pred_seq)
                all_true.append(label_seq)
    return all_preds, all_true


def aggregate_reports(reports):
    """Aggregate seqeval classification reports across seeds (mean ± std).

    Handles the case where a label appears in some seed runs but not others
    by treating missing labels as having 0.0 for precision/recall/f1.
    """
    metric_keys = ["precision", "recall", "f1-score"]

    # Collect the union of all keys across runs
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
        # Use support from the first report that has this key
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


def build_optimizer(model, lr, weight_decay=0.01):
    no_decay = ["bias", "norm.weight"]
    grouped_params = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optim.AdamW(grouped_params, lr=lr)


def save_run_manifest(
    path: Path,
    cfg_name: str,
    cfg: dict[str, Any],
    seeds: list[int],
    *,
    model_id: str,
    max_seq_length: int,
    grad_accum_steps: int,
    script_name: str,
    run_description: str,
    extra: dict[str, Any] | None = None,
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
        "model_id": model_id,
        "max_seq_length": max_seq_length,
        "grad_accum_steps": grad_accum_steps,
        "script": script_name,
        "run_description": run_description,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


NUM_WORKERS = 2  # Tune for Rivanna; 0 to disable multiprocess loading


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
    assert_conll2003_dataset(data_dir)
    print(f"CoNLL-2003 dataset checksums OK: {data_dir}")
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

    train_dataset = ConllDocContextDataset(
        train_docs,
        tokenizer,
        label2id,
        max_length=MAX_SEQ_LENGTH,
        window_mode="train",
        token_overlap=DEFAULT_TOKEN_OVERLAP,
    )
    dev_dataset = ConllDocContextDataset(
        dev_docs,
        tokenizer,
        label2id,
        max_length=MAX_SEQ_LENGTH,
        window_mode="eval",
        token_overlap=DEFAULT_TOKEN_OVERLAP,
    )
    test_dataset = ConllDocContextDataset(
        test_docs,
        tokenizer,
        label2id,
        max_length=MAX_SEQ_LENGTH,
        window_mode="eval",
        token_overlap=DEFAULT_TOKEN_OVERLAP,
    )
    for name, ds in (
        ("Train", train_dataset),
        ("Dev", dev_dataset),
        ("Test", test_dataset),
    ):
        print(
            f"{name} dataset: base_sentences={ds.n_base_sentences} "
            f"rows={len(ds)} "
            f"multi_window_overflow_sources={ds.n_multi_window_overflow} "
            f"window_mode={ds.window_mode} overlap={ds.token_overlap}"
        )

    collate_fn = ConllDocCollator(tokenizer.pad_token_id)

    # Generator for reproducible DataLoader shuffling per seed
    loader_generator = torch.Generator()

    HP_CONFIGS = [
        {
            "name": "doc_4e5_bs2",
            "lr": 4e-5,
            "epochs": 5,
            "warmup_ratio": 0.10,
            "weight_decay": 0.01,
            "batch_size": 2,
        },
    ]

    summary_rows = []
    prev_batch_size = None

    for cfg in HP_CONFIGS:
        cfg_name = cfg["name"]
        lr = cfg["lr"]
        n_epochs = cfg["epochs"]
        warmup_ratio = cfg["warmup_ratio"]
        weight_decay = cfg["weight_decay"]
        batch_size = cfg["batch_size"]
        early_stopping_patience = cfg.get("early_stopping_patience")
        classifier_dropout = cfg.get("classifier_dropout")

        print(f"\n{'#' * 60}")
        manifest_hp = {
            **cfg,
            "gradient_clip": CLIP,
            "context": "document",
            "max_seq_length": MAX_SEQ_LENGTH,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
        }
        es_note = (
            f", early_stopping_patience={early_stopping_patience}"
            if early_stopping_patience is not None
            else ""
        )
        cd_note = (
            f", classifier_dropout={classifier_dropout}"
            if classifier_dropout is not None
            else ""
        )
        print(
            f"CONFIG {cfg_name}: lr={lr}, epochs={n_epochs}{es_note}, "
            f"warmup={warmup_ratio}, wd={weight_decay}, bs={batch_size}{cd_note}, "
            f"max_seq_length={MAX_SEQ_LENGTH}, clip={CLIP}, "
            f"grad_accum={GRAD_ACCUM_STEPS}"
        )
        print(f"{'#' * 60}")

        save_run_manifest(
            results_dir / f"{OUTPUT_STEM}.json",
            cfg_name,
            manifest_hp,
            SEEDS,
            model_id=MODEL_ID,
            max_seq_length=MAX_SEQ_LENGTH,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            script_name="train_modernbert_doc_ner.py",
            run_description=RUN_DESCRIPTION,
            extra={
                "sliding_window_token_overlap": DEFAULT_TOKEN_OVERLAP,
                "dataset_train_rows": len(train_dataset),
                "dataset_dev_rows": len(dev_dataset),
                "dataset_test_rows": len(test_dataset),
                "dataset_train_multi_window_overflow": (
                    train_dataset.n_multi_window_overflow
                ),
                "dataset_dev_multi_window_overflow": (
                    dev_dataset.n_multi_window_overflow
                ),
                "dataset_test_multi_window_overflow": (
                    test_dataset.n_multi_window_overflow
                ),
            },
        )

        reports = []
        best_val_f1s = []
        best_epochs = []

        for run, seed in enumerate(SEEDS, 1):
            print(f"\n{'=' * 50}")
            print(f"[Config {cfg_name}] Run {run}/{len(SEEDS)} — Seed {seed}")
            print("=" * 50)
            set_seeds_to(seed)
            loader_generator.manual_seed(seed)

            # Recreate loaders per seed so shuffle order is deterministic
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=NUM_WORKERS,
                persistent_workers=NUM_WORKERS > 0,
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
                worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=NUM_WORKERS,
                persistent_workers=NUM_WORKERS > 0,
                worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
            )

            model_kwargs: dict[str, object] = {}
            if classifier_dropout is not None:
                model_kwargs["classifier_dropout"] = classifier_dropout
            model = AutoModelForTokenClassification.from_pretrained(
                MODEL_ID,
                num_labels=len(label_list),
                id2label=id2label,
                label2id=label2id,
                trust_remote_code=True,
                **model_kwargs,
            ).to(device)

            opt_steps_per_epoch = (
                (len(train_loader) + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS
            )
            total_steps = opt_steps_per_epoch * n_epochs
            warmup_steps = int(warmup_ratio * total_steps)
            optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

            best_val_f1 = 0.0
            best_epoch = -1
            best_state_dict = None
            epochs_no_improve = 0
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
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if (
                        early_stopping_patience is not None
                        and epochs_no_improve >= early_stopping_patience
                    ):
                        print(
                            f"Early stopping: no dev F1 improvement for "
                            f"{early_stopping_patience} epoch(s)."
                        )
                        break

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

        csv_name = f"{OUTPUT_STEM}.csv"
        df.to_csv(results_dir / csv_name)
        print(f"Saved to {csv_name}")

        test_micro_f1 = df.loc["micro avg", "f1-score"]
        dev_f1_mean = float(np.mean(best_val_f1s))
        dev_f1_std = float(np.std(best_val_f1s))
        summary_rows.append(
            {
                "config": cfg_name,
                "lr": lr,
                "epochs": n_epochs,
                "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "max_seq_length": MAX_SEQ_LENGTH,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "gradient_clip": CLIP,
                "test_micro_f1": test_micro_f1,
                "best_dev_f1": f"{dev_f1_mean:.4f} ± {dev_f1_std:.4f}",
                "best_epoch_mean": f"{np.mean(best_epochs):.2f}",
            }
        )

    print("\n" + "=" * 70)
    print("RUN SUMMARY")
    print("=" * 70)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
