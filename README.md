# ModernBERT NER Ablation

BERT and ModernBERT results (linear and CRF heads, sentence and document context) live under [`results/`](results/) (see **Results**).

Evaluating document-level context and CRF decoding in ModernBERT for CoNLL-2003 named entity recognition (NER).

Overleaf document: [moderbert-ner-ablation](https://www.overleaf.com/project/6996373c44b841199bc3c599)

## Abstract

We evaluate whether document-level context and CRF decoding provide additive or synergistic gains over sentence-level ModernBERT for named entity recognition on CoNLL-2003. Using a 2x2 factorial ablation (context on/off, CRF on/off), we compare entity-level F1 across all configurations and analyze which entity types benefit most from each modification.

### Ablations

We use two factors:

- Document context: off/on
- CRF decoding head: off/on

![2×2 factorial: rows = sentence vs document context; columns = softmax vs CRF. Config A baseline (softmax, sentence); B +CRF only; C +document context only; D +document context +CRF.](images/ner_ablation_2x2.png)

_Figure: Same design as the list below—**A** = baseline (sentence, softmax), **B** = +CRF only, **C** = +document context only, **D** = +document context and CRF._

This yields four configurations (same labels as the figure, row-major: A and B on the sentence row, C and D on the document row):

1. **Config A —** Baseline ModernBERT (sentence-level, softmax; no CRF)
2. **Config B —** ModernBERT + CRF (sentence-level)
3. **Config C —** ModernBERT + document context (softmax)
4. **Config D —** ModernBERT + document context + CRF

Primary metric: entity-level F1 (seqeval), with per-entity-type F1 for PER/ORG/LOC/MISC.

## Results

**Last updated:** 2026-04-12. Headline metrics use the **best test micro F1** per model family among checked-in CSVs. Sources: [`results/ner_bert_ref.csv`](results/ner_bert_ref.csv) (same numbers as [`results/old/ner_bert_ref.csv`](results/old/ner_bert_ref.csv)), [`results/ner_bert_doc_ref.csv`](results/ner_bert_doc_ref.csv), [`results/ner_mbert_sent_v2_B_dropout_ls.csv`](results/ner_mbert_sent_v2_B_dropout_ls.csv), [`results/modernbert_doc_ner_config_doc_4e5_bs2.csv`](results/modernbert_doc_ner_config_doc_4e5_bs2.csv), [`results/ner_mbert_sent_crf_best.csv`](results/ner_mbert_sent_crf_best.csv), [`results/ner_mbert_doc_crf_tuned.csv`](results/ner_mbert_doc_crf_tuned.csv). Each has a paired `.json` manifest under [`results/`](results/).

Numbers are mean ± std over seeds 21, 42, 63. Per seed, evaluation uses the checkpoint with **best dev F1** on `eng.testa`. Configurations are **not** matched for a controlled comparison (context length, batch, head type vary); see each JSON for exact hyperparameters.

### Overall F1

| Model                                                      | Micro F1            | Macro F1            |
| ---------------------------------------------------------- | ------------------- | ------------------- |
| BERT-base-cased (sentence-level, no CRF)                   | 0.9137 ± 0.0017     | 0.8993 ± 0.0018     |
| BERT-base-cased (document windows, no CRF)                 | 0.8915 ± 0.0031     | 0.8777 ± 0.0022     |
| ModernBERT-base (sentence-level, `B_dropout_ls`)           | 0.9012 ± 0.0031     | 0.8875 ± 0.0027     |
| ModernBERT-base + document context (`doc_4e5_bs2`)         | **0.9161 ± 0.0023** | **0.9000 ± 0.0023** |
| ModernBERT-base + CRF (sentence-level, config G)           | 0.9015 ± 0.0021     | 0.8887 ± 0.0026     |
| ModernBERT-base + document context + CRF (`doc_crf_tuned`) | 0.9012 ± 0.0013     | 0.8843 ± 0.0022     |

### Per-entity F1

Entity order: PER, ORG, LOC, MISC.

| Entity | BERT (sentence)     | BERT (document) | ModernBERT (sentence) | ModernBERT (document) | ModernBERT (sent. + CRF) | ModernBERT (doc + CRF) |
| ------ | ------------------- | --------------- | --------------------- | --------------------- | ------------------------ | ---------------------- |
| PER    | 0.9612 ± 0.0014     | 0.9424 ± 0.0072 | 0.9571 ± 0.0005       | **0.9788** ± 0.0001   | 0.9559 ± 0.0016          | 0.9707 ± 0.0043        |
| ORG    | **0.8960** ± 0.0036 | 0.8696 ± 0.0028 | 0.8712 ± 0.0066       | 0.8946 ± 0.0050       | 0.8682 ± 0.0043          | 0.8734 ± 0.0004        |
| LOC    | **0.9309** ± 0.0003 | 0.9077 ± 0.0068 | 0.9222 ± 0.0034       | 0.9283 ± 0.0023       | 0.9233 ± 0.0022          | 0.9170 ± 0.0008        |
| MISC   | **0.8093** ± 0.0030 | 0.7910 ± 0.0047 | 0.7993 ± 0.0019       | 0.7981 ± 0.0050       | 0.8072 ± 0.0059          | 0.7759 ± 0.0099        |

## Planned Final Model

Reported above: BERT and ModernBERT variants across document context and CRF (plus BERT/ModernBERT sentence baselines). **ModernBERT + document context** (`doc_4e5_bs2`) achieves the highest test micro F1 in this table (**0.9161**). Document **CRF** is slightly below document linear and ties sentence CRF on micro F1 (0.9012 vs 0.9015).

## Environment Setup

### Install (uv, Python 3.14)

```bash
uv python install 3.14
uv sync
```

### Dataset download

Training expects **CoNLL-2003** files (`eng.train`, `eng.testa`, `eng.testb`) under [`data/conll2003/`](data/conll2003/). From the project root (where `pyproject.toml` lives), fetch them with [`scripts/download_data.py`](scripts/download_data.py):

```bash
uv run python scripts/download_data.py
```

The script uses [kagglehub](https://github.com/Kaggle/kagglehub) to download [juliangarratt/conll2003-dataset](https://www.kaggle.com/datasets/juliangarratt/conll2003-dataset) and copies the three splits into `data/conll2003/`. Configure [Kaggle API credentials](https://www.kaggle.com/docs/api) locally if prompted; do not commit tokens.

### Project layout

- [`data/conll2003/`](data/conll2003/): CoNLL-2003 splits (from **Dataset download** above)
- [`scripts/`](scripts/): training, verification, and data helpers
- [`results/`](results/): headline `*.csv` / `*.json` pairs (see **Results**)
- [`images/`](images/): figures for docs
- `notebooks/`: experiment notebooks and ablations
- `references/`: bibliography sources
- `documents/`: milestone and supporting course documents

## Training

From the project root (after `uv sync`), run a trainer with:

```bash
uv run python scripts/<training_file>.py
```

Each script writes metrics under [`results/`](results/) per its `OUTPUT_STEM` (paired `.csv` + `.json`). Hyperparameters live in that script’s `HP_CONFIG` (see [Hyperparameters (`HP_CONFIG`)](#hyperparameters-hp_config)). Headline aggregates in **Results** may match a sweep file copied into `results/`; ad-hoc runs are archived under [`results/old/`](results/old/) or [`results/old/archive_from_root_2026-04-12/`](results/old/archive_from_root_2026-04-12/).

| Script                                                                       | Default output stem (see script) |
| ---------------------------------------------------------------------------- | -------------------------------- |
| [`train_bert_ner.py`](scripts/train_bert_ner.py)                             | `ner_bert`                       |
| [`train_bert_doc_ner.py`](scripts/train_bert_doc_ner.py)                     | `ner_bert_doc`                   |
| [`train_modernbert_ner.py`](scripts/train_modernbert_ner.py)                 | `ner_mbert`                      |
| [`train_modernbert_doc_ner.py`](scripts/train_modernbert_doc_ner.py)         | `ner_mbert_doc`                  |
| [`train_modernbert_crf_ner.py`](scripts/train_modernbert_crf_ner.py)         | `ner_mbert_crf`                  |
| [`train_modernbert_doc_crf_ner.py`](scripts/train_modernbert_doc_crf_ner.py) | `ner_mbert_doc_crf`              |

### Hyperparameters (`HP_CONFIG`)

Each training script defines a single `HP_CONFIG` dict with the tunable values for that run. The paired [`results/<stem>.json`](results/) manifest records what was used.

Look for `HP_CONFIG` in each [`scripts/train_*.py`](scripts/) (a few files still wrap the same dict in a one-element `HP_CONFIGS` list until renamed). CRF trainers add decoder-specific keys (e.g. `crf_lr`) alongside the usual optimization and schedule fields.

Document ModernBERT softmax ([`train_modernbert_doc_ner.py`](scripts/train_modernbert_doc_ner.py)):

```python
HP_CONFIG = {
    "name": "test_config",
    "lr": 4e-5,
    "epochs": 5,
    "warmup_ratio": 0.10,
    "weight_decay": 0.01,
    "batch_size": 2,
}
```

### Data

Use the same `data/conll2003/` tree as in [Dataset download](#dataset-download). Training scripts resolve `data_dir` relative to the repo (see each file).

### Sanity checks (optional)

| Script                                                                           | What it checks                     |
| -------------------------------------------------------------------------------- | ---------------------------------- |
| [`conll2003_dataset_verification.py`](scripts/conll2003_dataset_verification.py) | Dataset layout / expectations      |
| [`conll2003_concat_verification.py`](scripts/conll2003_concat_verification.py)   | Sentence concatenation             |
| [`conll2003_tokenization_compare.py`](scripts/conll2003_tokenization_compare.py) | Tokenization alignment             |
| [`conll2003_crf_verification.py`](scripts/conll2003_crf_verification.py)         | CRF mask + illegal BIO transitions |

Run any of them from the project root, for example:

```bash
uv run python scripts/conll2003_dataset_verification.py
```
