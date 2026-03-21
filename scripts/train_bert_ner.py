"""Fine-tune bert-base-cased on CoNLL-2003 NER. Patterns from deep learning coursework assignment 2 and 3.
Classification report logic copied from notebooks/00bert_baseline.ipynb."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from seqeval.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer


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


def train_epoch(model, dataloader, optimizer, device, clip):
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


if __name__ == "__main__":
    SEED = 21
    set_seeds_to(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path(__file__).resolve().parent.parent / "data" / "conll2003"
    train_sentences = parse_conll(data_dir / "eng.train")
    dev_sentences = parse_conll(data_dir / "eng.testa")
    test_sentences = parse_conll(data_dir / "eng.testb")
    print(f"Train: {len(train_sentences)} sentences")
    print(f"Dev: {len(dev_sentences)} sentences")
    print(f"Test: {len(test_sentences)} sentences")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-cased", num_labels=len(label_list)
    ).to(device)

    train_dataset = ConllDataset(train_sentences, tokenizer, label2id)
    dev_dataset = ConllDataset(dev_sentences, tokenizer, label2id)
    test_dataset = ConllDataset(test_sentences, tokenizer, label2id)

    # TUNEABLE PARAMETERS
    BATCH_SIZE = 16
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    lr = 2e-5
    N_EPOCHS = 5
    CLIP = 1  # assignment 3 gradient clipping
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)  # a2

    for epoch in range(N_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{N_EPOCHS}")
        print("-" * 30)

        train_loss = train_epoch(model, train_loader, optimizer, device, CLIP)
        val_loss, val_f1 = evaluate(model, dev_loader, device, id2label)
        print(f"Train Loss: {train_loss:.3f}")
        print(f"Val Loss: {val_loss:.3f}, Val F1: {val_f1:.4f}")
        scheduler.step()

    all_preds, all_true = get_predictions(model, test_loader, device, id2label)
    report = classification_report(all_true, all_preds, output_dict=True)
    df = pd.DataFrame(report).T.round(4)
    print("\n=== Test Set Evaluation (detailed) ===")
    print(df.to_string())
