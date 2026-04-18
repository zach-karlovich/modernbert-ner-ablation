"""Fine-tune bert-base-cased on CoNLL-2003 NER.

Patterns from deep learning coursework assignment 2 and 3; classification report
logic from notebooks/00bert_baseline.ipynb.

Run identity: sentence-level examples only (`parse_conll` skips `-DOCSTART-`);
softmax token head; seqeval F1. Reference encoder for the ModernBERT ablation.
Outputs `ner_bert.{csv,json}`. HPs live in `HP_CONFIG` below."""

import copy
import json
import random
import subprocess
from pathlib import Path
from typing import Any, cast

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

from conll2003_expectations import (
    assert_conll2003_dataset,
    assert_parsed_sentence_counts_match_expected,
)
from conll2003_parse import parse_conll

OUTPUT_STEM = "ner_bert"

RUN_DESCRIPTION = (
    "Sentence-level bert-base-cased on CoNLL-2003; softmax head; -DOCSTART- "
    "ignored so no cross-sentence context. Config 0: lr 2e-5, wd 0.01, "
    "5 epochs, batch size 16, max_seq_length 512. Writes ner_bert.{csv,json}."
)


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
    """assignment 2 runner."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class ConllDataset(Dataset):
    """Tokenize sentences, align labels to subwords. First subword = label, rest = -100 (ignored)."""

    def __init__(self, sentences, tokenizer, label2id, max_length=512):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.label2id = label2id
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
        labels = []
        for i in range(len(word_ids)):
            if word_ids[i] is None:
                labels.append(-100)
            elif i == 0 or word_ids[i] != word_ids[i - 1]:
                labels.append(self.label2id[tags[word_ids[i]]])
            else:
                labels.append(-100)
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch):
    """Pad to max len in batch. assignment 3 style."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    pad_id = 0
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


def train_epoch(model, dataloader, optimizer, scheduler, device, clip):
    """Modeled after assignment 2 runner.train_epoch. BERT returns loss directly."""
    model.train()
    running_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def evaluate(model, dataloader, device, id2label):
    """assignment 2 runner.evaluate. Uses seqeval for entity-level F1."""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
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
                all_predictions.append(pred_seq)
                all_labels.append(label_seq)
    epoch_loss = running_loss / len(dataloader)
    f1 = cast(float, f1_score(all_labels, all_predictions))
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


def build_optimizer(model, lr, weight_decay=0.01):
    no_decay = ["bias", "LayerNorm.weight"]
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
    script_name: str,
    run_description: str,
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
        "script": script_name,
        "run_description": run_description,
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
    assert_conll2003_dataset(data_dir)
    print(f"CoNLL-2003 dataset checksums OK: {data_dir}")
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    train_sentences = parse_conll(data_dir / "eng.train")
    dev_sentences = parse_conll(data_dir / "eng.testa")
    test_sentences = parse_conll(data_dir / "eng.testb")
    assert_parsed_sentence_counts_match_expected(
        len(train_sentences), len(dev_sentences), len(test_sentences)
    )
    print(f"Train: {len(train_sentences)} sentences")
    print(f"Dev: {len(dev_sentences)} sentences")
    print(f"Test: {len(test_sentences)} sentences")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    HP_CONFIG = {
        "name": "0",
        "lr": 2e-5,
        "epochs": 5,
        "warmup_ratio": 0.10,
        "weight_decay": 0.01,
        "batch_size": 16,
        "max_seq_length": 512,
    }

    summary_rows = []
    prev_max_seq: int | None = None
    loader_generator = torch.Generator()

    for cfg in [HP_CONFIG]:
        cfg_name = cfg["name"]
        lr = cfg["lr"]
        n_epochs = cfg["epochs"]
        warmup_ratio = cfg["warmup_ratio"]
        weight_decay = cfg["weight_decay"]
        batch_size = cfg["batch_size"]
        max_seq_length = cfg["max_seq_length"]
        early_stopping_patience = cfg.get("early_stopping_patience")

        if max_seq_length != prev_max_seq:
            train_dataset = ConllDataset(
                train_sentences, tokenizer, label2id, max_length=max_seq_length
            )
            dev_dataset = ConllDataset(
                dev_sentences, tokenizer, label2id, max_length=max_seq_length
            )
            test_dataset = ConllDataset(
                test_sentences, tokenizer, label2id, max_length=max_seq_length
            )
            prev_max_seq = max_seq_length

        manifest_hp = {**cfg, "gradient_clip": CLIP, "context": "sentence"}

        print(f"\n{'#' * 60}")
        es_note = (
            f", early_stopping_patience={early_stopping_patience}"
            if early_stopping_patience is not None
            else ""
        )
        print(
            f"CONFIG {cfg_name}: lr={lr}, epochs={n_epochs}{es_note}, "
            f"warmup={warmup_ratio}, wd={weight_decay}, bs={batch_size}, "
            f"max_seq_length={max_seq_length}, clip={CLIP}"
        )
        print(f"{'#' * 60}")

        save_run_manifest(
            results_dir / f"{OUTPUT_STEM}.json",
            cfg_name,
            manifest_hp,
            SEEDS,
            model_id="bert-base-cased",
            max_seq_length=max_seq_length,
            script_name="train_bert_ner.py",
            run_description=RUN_DESCRIPTION,
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
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                generator=loader_generator,
            )
            dev_loader = DataLoader(
                dev_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )

            model = AutoModelForTokenClassification.from_pretrained(
                "bert-base-cased",
                num_labels=len(label_list),
                id2label=id2label,
                label2id=label2id,
            ).to(device)

            total_steps = len(train_loader) * n_epochs
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
                    model, train_loader, optimizer, scheduler, device, CLIP
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
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

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
                "max_seq_length": max_seq_length,
                "gradient_clip": CLIP,
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
