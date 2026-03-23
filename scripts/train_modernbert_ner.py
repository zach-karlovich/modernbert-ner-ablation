"""Fine-tune ModernBERT-base on CoNLL-2003 NER. Adapted from train_bert_ner.py."""

import copy
import random
from pathlib import Path

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


def parse_conll(filepath):
    """00bert_baseline. Sentence-level, skips -DOCSTART-."""
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
        # Subword alignment (first-subtoken strategy): ModernBERT uses BPE, not WordPiece.
        # word_ids() maps each token to its source word; the first token of each word
        # gets the label, continuation subwords get -100. Tokenizer-agnostic.
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


def make_collate_fn(pad_token_id):
    """Pad to max len in batch. assignment 3 style."""

    def collate_fn(batch):
        max_len = max(item["input_ids"].size(0) for item in batch)
        # ModernBERT pad_token_id != 0; use the tokenizer's actual pad token
        pad_id = pad_token_id
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

    return collate_fn


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
    f1 = f1_score(all_labels, all_predictions)
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


if __name__ == "__main__":
    SEEDS = [21, 42, 63]
    CLIP = 1
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path(__file__).resolve().parent.parent / "data" / "conll2003"
    results_dir = Path(__file__).resolve().parent.parent / "results"
    train_sentences = parse_conll(data_dir / "eng.train")
    dev_sentences = parse_conll(data_dir / "eng.testa")
    test_sentences = parse_conll(data_dir / "eng.testb")
    print(f"Train: {len(train_sentences)} sentences")
    print(f"Dev: {len(dev_sentences)} sentences")
    print(f"Test: {len(test_sentences)} sentences")

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    # UNCOMMENT TO RUN ALIGNMENT VERIFICATION
    # VERIFY_ALIGNMENT = True
    # if VERIFY_ALIGNMENT:
    #     test_words = ["Enter", "Sandman", "at", "Lane", "Stadium", "is", "incredible"]
    #     encoding = tokenizer(test_words, is_split_into_words=True, return_tensors="pt")
    #     word_ids = encoding.word_ids()
    #     tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    #     for i, (tok, wid) in enumerate(zip(tokens, word_ids)):
    #         print(f"{i}: {tok!r} -> word_id={wid}")

    train_dataset = ConllDataset(train_sentences, tokenizer, label2id)
    dev_dataset = ConllDataset(dev_sentences, tokenizer, label2id)
    test_dataset = ConllDataset(test_sentences, tokenizer, label2id)

    collate_fn = make_collate_fn(tokenizer.pad_token_id)

    # Hyperparameter configurations - Claude generated
    HP_CONFIGS = [
        {
            "name": "0",
            "lr": 2e-5,
            "epochs": 5,
            "warmup_ratio": 0.10,
            "weight_decay": 0.01,
            "batch_size": 16,
        },
        {
            "name": "0a",
            "lr": 1.5e-5,
            "epochs": 5,
            "warmup_ratio": 0.10,
            "weight_decay": 0.01,
            "batch_size": 16,
        },
        {
            "name": "0b",
            "lr": 2.5e-5,
            "epochs": 5,
            "warmup_ratio": 0.10,
            "weight_decay": 0.01,
            "batch_size": 16,
        },
        {
            "name": "0c",
            "lr": 3e-5,
            "epochs": 5,
            "warmup_ratio": 0.10,
            "weight_decay": 0.01,
            "batch_size": 16,
        },
        {
            "name": "L",
            "lr": 2e-5,
            "epochs": 7,
            "warmup_ratio": 0.10,
            "weight_decay": 0.01,
            "batch_size": 16,
        },
        {
            "name": "M",
            "lr": 2e-5,
            "epochs": 5,
            "warmup_ratio": 0.10,
            "weight_decay": 0.001,
            "batch_size": 16,
        },
        # Completed sweep configs 
        # {
        #     "name": "A",
        #     "lr": 3e-5,
        #     "epochs": 5,
        #     "warmup_ratio": 0.10,
        #     "weight_decay": 0.01,
        #     "batch_size": 16,
        # },
        # {
        #     "name": "B",
        #     "lr": 5e-5,
        #     "epochs": 5,
        #     "warmup_ratio": 0.10,
        #     "weight_decay": 0.01,
        #     "batch_size": 16,
        # },
        # {
        #     "name": "C",
        #     "lr": 3e-5,
        #     "epochs": 10,
        #     "warmup_ratio": 0.10,
        #     "weight_decay": 0.01,
        #     "batch_size": 16,
        # },
        # {
        #     "name": "D",
        #     "lr": 5e-5,
        #     "epochs": 10,
        #     "warmup_ratio": 0.15,
        #     "weight_decay": 0.01,
        #     "batch_size": 16,
        # },
        # {
        #     "name": "E",
        #     "lr": 3e-5,
        #     "epochs": 10,
        #     "warmup_ratio": 0.10,
        #     "weight_decay": 0.001,
        #     "batch_size": 16,
        # },
        # {
        #     "name": "F",
        #     "lr": 3e-5,
        #     "epochs": 10,
        #     "warmup_ratio": 0.20,
        #     "weight_decay": 0.01,
        #     "batch_size": 16,
        # },
        # {
        #     "name": "G",
        #     "lr": 6e-5,
        #     "epochs": 10,
        #     "warmup_ratio": 0.10,
        #     "weight_decay": 0.01,
        #     "batch_size": 32,
        # },
        # {
        #     "name": "H",
        #     "lr": 7e-5,
        #     "epochs": 10,
        #     "warmup_ratio": 0.10,
        #     "weight_decay": 0.01,
        #     "batch_size": 32,
        # },
        # {
        #     "name": "I",
        #     "lr": 6e-5,
        #     "epochs": 15,
        #     "warmup_ratio": 0.15,
        #     "weight_decay": 0.01,
        #     "batch_size": 32,
        # },
        # {
        #     "name": "J",
        #     "lr": 8e-5,
        #     "epochs": 10,
        #     "warmup_ratio": 0.15,
        #     "weight_decay": 0.01,
        #     "batch_size": 48,
        # },
        # {
        #     "name": "K",
        #     "lr": 5e-5,
        #     "epochs": 15,
        #     "warmup_ratio": 0.10,
        #     "weight_decay": 0.02,
        #     "batch_size": 32,
        # },
    ]

    summary_csv_path = results_dir / "modernbert_hp_sweep_summary.csv"
    if summary_csv_path.exists():
        existing_summary = pd.read_csv(summary_csv_path)
        summary_rows = existing_summary.to_dict("records")
    else:
        summary_rows = []
    prev_batch_size = None

    for cfg in HP_CONFIGS:
        cfg_name = cfg["name"]
        lr = cfg["lr"]
        n_epochs = cfg["epochs"]
        warmup_ratio = cfg["warmup_ratio"]
        weight_decay = cfg["weight_decay"]
        batch_size = cfg["batch_size"]

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
            f"CONFIG {cfg_name}: lr={lr}, epochs={n_epochs}, "
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

            model = AutoModelForTokenClassification.from_pretrained(
                "answerdotai/ModernBERT-base",
                num_labels=len(label_list),
                id2label=id2label,
                label2id=label2id,
            ).to(device)

            total_steps = len(train_loader) * n_epochs
            warmup_steps = int(warmup_ratio * total_steps)
            optimizer = optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
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
            all_preds, all_true = get_predictions(
                model, test_loader, device, id2label
            )
            report = classification_report(all_true, all_preds, output_dict=True)
            reports.append(report)
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        df = aggregate_reports(reports).set_index("")
        print(f"\n=== Config {cfg_name} Test Results (seeds {SEEDS}) — mean ± std ===")
        print(df.to_string())

        csv_name = f"modernbert_ner_config_{cfg_name}.csv"
        df.to_csv(results_dir / csv_name)
        print(f"Saved to {csv_name}")

        micro_f1 = df.loc["micro avg", "f1-score"]
        summary_rows.append(
            {
                "config": cfg_name,
                "lr": lr,
                "epochs": n_epochs,
                "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "micro_f1": micro_f1,
                "best_val_f1_mean": f"{np.mean(best_val_f1s):.4f}",
                "best_epoch_mean": f"{np.mean(best_epochs):.2f}",
            }
        )

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_csv_path, index=False)

    print("\n" + "=" * 70)
    print("HYPERPARAMETER SWEEP SUMMARY")
    print("=" * 70)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(summary_csv_path, index=False)
