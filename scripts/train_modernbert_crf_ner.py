"""Fine-tune ModernBERT-base + constrained CRF on CoNLL-2003 NER (sentence-level)."""

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
WORD_PAD_ID = -99


def parse_conll(filepath):
    sentences = []
    current = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                continue
            if line == "":
                if current:
                    sentences.append(current)
                    current = []
            else:
                parts = line.split()
                word = parts[0]
                ner = parts[-1]
                current.append((word, ner))
        if current:
            sentences.append(current)
    return sentences


def set_seeds_to(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ConllDatasetCRF(Dataset):
    def __init__(self, sentences, tokenizer, max_length=512):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        words = [w for w, _ in sentence]
        tags = [t for _, t in sentence]
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
        n_words = len(words)
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "word_ids": w_tensor,
            "word_include_mask": [True] * n_words,
        }


def make_collate_fn_crf(pad_token_id: int, label_pad_id: int):
    def collate_fn(batch):
        max_len = max(item["input_ids"].size(0) for item in batch)
        pad_id = pad_token_id
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
                F.pad(item["labels"], (0, pad_len), value=label_pad_id)
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

    return collate_fn


def train_epoch(model, dataloader, optimizer, scheduler, device, clip):
    model.train()
    running_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids, attention_mask, labels, _, _ = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss, _ = model(input_ids, attention_mask, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def evaluate(model, dataloader, device, id2label_map):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids, attention_mask, labels, word_ids, includes = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            loss, _ = model(input_ids, attention_mask, labels)
            running_loss += loss.item()
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
                all_predictions.append(pred_seq)
                all_labels.append(label_seq)
    epoch_loss = running_loss / len(dataloader)
    f1 = cast(float, f1_score(all_labels, all_predictions))
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
    rows = []
    for key in reports[0]:
        if key == "accuracy":
            continue
        row = {"": key}
        support = reports[0][key].get("support", np.nan)
        if not np.isnan(support):
            row["support"] = int(support)
        for m in metric_keys:
            if m in reports[0][key]:
                vals = [r[key][m] for r in reports]
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


if __name__ == "__main__":
    SEEDS = [21, 42, 63]
    CLIP = 1
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path(__file__).resolve().parent.parent / "data" / "conll2003"
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    train_sentences = parse_conll(data_dir / "eng.train")
    dev_sentences = parse_conll(data_dir / "eng.testa")
    test_sentences = parse_conll(data_dir / "eng.testb")
    print(f"Train: {len(train_sentences)} sentences")
    print(f"Dev: {len(dev_sentences)} sentences")
    print(f"Test: {len(test_sentences)} sentences")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    train_dataset = ConllDatasetCRF(train_sentences, tokenizer)
    dev_dataset = ConllDatasetCRF(dev_sentences, tokenizer)
    test_dataset = ConllDatasetCRF(test_sentences, tokenizer)

    collate_fn = make_collate_fn_crf(tokenizer.pad_token_id, label_pad_id=label2id["O"])

    HP_CONFIGS = [
        {
            "name": "0",
            "lr": 2e-5,
            "crf_lr": 2e-5,
            "epochs": 5,
            "warmup_ratio": 0.10,
            "weight_decay": 0.01,
            "batch_size": 16,
        },
        {
            "name": "G",
            "lr": 6e-5,
            "crf_lr": 3e-4,
            "epochs": 10,
            "warmup_ratio": 0.10,
            "weight_decay": 0.01,
            "batch_size": 32,
        },
    ]

    summary_rows = []
    prev_batch_size = None

    for cfg in HP_CONFIGS:
        cfg_name = cfg["name"]
        lr = cfg["lr"]
        crf_lr = cfg["crf_lr"]
        n_epochs = cfg["epochs"]
        warmup_ratio = cfg["warmup_ratio"]
        weight_decay = cfg["weight_decay"]
        batch_size = cfg["batch_size"]

        save_run_config(
            results_dir / f"modernbert_crf_ner_config_{cfg_name}.json",
            cfg_name,
            cfg,
            SEEDS,
            {
                "model_id": MODEL_ID,
                "max_seq_length": 512,
                "script": "train_modernbert_crf_ner.py",
            },
        )

        if batch_size != prev_batch_size:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
            )
            dev_loader = DataLoader(
                dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
            prev_batch_size = batch_size

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

            model = ModernBertTokenCRF(MODEL_ID, trust_remote_code=True).to(device)

            total_steps = len(train_loader) * n_epochs
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
                    model, train_loader, optimizer, scheduler, device, CLIP
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
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        df = aggregate_reports(reports).set_index("")
        print(f"\n=== Config {cfg_name} Test Results (seeds {SEEDS}) — mean ± std ===")
        print(df.to_string())

        csv_name = f"modernbert_crf_ner_config_{cfg_name}.csv"
        df.to_csv(results_dir / csv_name)
        print(f"Saved to {csv_name}")

        test_micro_f1 = df.loc["micro avg", "f1-score"]
        dev_f1_mean = float(np.mean(best_val_f1s))
        dev_f1_std = float(np.std(best_val_f1s))
        summary_rows.append(
            {
                "config": cfg_name,
                "lr": lr,
                "crf_lr": crf_lr,
                "epochs": n_epochs,
                "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "test_micro_f1": test_micro_f1,
                "best_dev_f1": f"{dev_f1_mean:.4f} ± {dev_f1_std:.4f}",
                "best_epoch_mean": f"{np.mean(best_epochs):.2f}",
            }
        )

    print("\n" + "=" * 70)
    print("HYPERPARAMETER SWEEP SUMMARY")
    print("=" * 70)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
