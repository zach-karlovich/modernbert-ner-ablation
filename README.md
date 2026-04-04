# ModernBERT NER Ablation

**_IN PROGRESS_**

Evaluating document-level context and CRF decoding in ModernBERT for CoNLL-2003 named entity recognition (NER).

Overleaf document: [moderbert-ner-ablation](https://www.overleaf.com/project/6996373c44b841199bc3c599)

## Abstract

We evaluate whether document-level context and CRF decoding provide additive or synergistic gains over sentence-level ModernBERT for named entity recognition on CoNLL-2003. Using a 2x2 factorial ablation (context on/off, CRF on/off), we compare entity-level F1 across all configurations and analyze which entity types benefit most from each modification.

### Ablations

We use two factors:

- Document context: off/on
- CRF decoding head: off/on

This yields four configurations:

1. Baseline ModernBERT (sentence-level, no CRF)
2. ModernBERT + document context
3. ModernBERT + CRF
4. ModernBERT + document context + CRF

Primary metric: entity-level F1 (seqeval), with per-entity-type F1 for PER/ORG/LOC/MISC.

## Results

**Last updated:** 2026-04-04 (aligned with [results/results_summary.md](results/results_summary.md)).

Entity-level F1 on the CoNLL-2003 **test** set (`eng.testb`). Mean ± std over 3 seeds (21, 42, 63). For each seed, **test evaluation uses the checkpoint with highest dev F1** on `eng.testa`.

**BERT** and sentence-level **ModernBERT (config 0)** use the **matched-HP** recipe: LR **2e-5**, **5** epochs, batch **16**. **ModernBERT + document** uses LR **5e-5**, batch **2**, **8192** max length, grad accumulation **8** ([`results/modernbert_doc_ner_config_doc_5e5_bs2.csv`](results/modernbert_doc_ner_config_doc_5e5_bs2.csv))—not matched to the sentence baselines. HP sweep (e.g. sentence **config B** at **0.8984** micro F1), tuned BERT, and CRF ablations are in [results/results_summary.md](results/results_summary.md) and [results/results_summary.csv](results/results_summary.csv).

### Overall F1

| Model                                                  | Micro F1        | Macro F1        |
| ------------------------------------------------------ | --------------- | --------------- |
| BERT-base-cased (sentence-level, no CRF)               | 0.9128 ± 0.0025 | 0.8969 ± 0.0025 |
| ModernBERT-base (sentence-level, matched HP, config 0) | 0.8862 ± 0.0023 | 0.8720 ± 0.0024 |
| ModernBERT-base + document context                     | 0.9162 ± 0.0017 | 0.9004 ± 0.0015 |
| ModernBERT-base + CRF (sentence-level)                 | —               | —               |
| ModernBERT-base + document context + CRF               | —               | —               |

### Per-entity F1

Entity order: PER, ORG, LOC, MISC. BERT and sentence ModernBERT cells match [`results/bert_ner_config_0.csv`](results/bert_ner_config_0.csv) and [`results/modernbert_ner_config_0.csv`](results/modernbert_ner_config_0.csv). Document column: [`results/modernbert_doc_ner_config_doc_5e5_bs2.csv`](results/modernbert_doc_ner_config_doc_5e5_bs2.csv).

| Entity | BERT                | ModernBERT (sentence, config 0) | ModernBERT (document) | ModernBERT (sentence + CRF) | ModernBERT (document + CRF) |
| ------ | ------------------- | ------------------------------- | --------------------- | --------------------------- | --------------------------- |
| PER    | 0.9622 ± 0.0019     | 0.9528 ± 0.0035                 | **0.9808** ± 0.0022   | —                           | —                           |
| ORG    | **0.8975** ± 0.0037 | 0.8458 ± 0.0030                 | 0.8942 ± 0.0041       | —                           | —                           |
| LOC    | **0.9305** ± 0.0025 | 0.9100 ± 0.0021                 | 0.9268 ± 0.0016       | —                           | —                           |
| MISC   | 0.7973 ± 0.0021     | 0.7796 ± 0.0045                 | **0.7999** ± 0.0008   | —                           | —                           |

## Planned Final Model

ModernBERT with document-level context and a CRF decoding head, evaluated against the sentence-level ModernBERT baseline and single-factor variants.

### How to Load

## Environment Setup

### Install (uv, Python 3.14)

```bash
uv python install 3.14
uv sync
```

### UVA HPC Rivanna Setup

```bash
module load uv/0.9.9
uv sync
```

### Run notebooks

```bash
uv run jupyter lab
```

If you use Kaggle data access, configure Kaggle API credentials locally and do not commit tokens.

### Directory Structure

- `notebooks/`: experiment notebooks and ablations
- `references/`: bibliography sources
- `documents/`: milestone and supporting course documents
- `results/`: training run outputs

## Training

Place CoNLL-2003 files under `data/conll2003/` (`eng.train`, `eng.testa`, `eng.testb`). Example:

```bash
uv run python scripts/download_data.py
```

### 2×2 ModernBERT ablation (exact scripts)

Each cell is produced by the corresponding script; metrics are written under `results/` as CSV plus a `config_<name>.json` for CRF runs (hyperparameters, seeds, git hash, library versions).

| Configuration | Command | Primary result artifacts |
| ------------- | ------- | ------------------------ |
| 1. Sentence, linear head | `uv run python scripts/train_modernbert_ner.py` | `results/modernbert_ner_config_0.csv` (matched HP; script may also run other `HP_CONFIGS` entries) |
| 2. Document context, linear head | `uv run python scripts/train_modernbert_doc_ner.py` | `results/modernbert_doc_ner_config_doc_5e5_bs2.csv` (among sweep files in that script) |
| 3. Sentence, CRF head | `uv run python scripts/train_modernbert_crf_ner.py` | `results/modernbert_crf_ner_config_0.csv`, `results/modernbert_crf_ner_config_0.json` |
| 4. Document context, CRF head | `uv run python scripts/train_modernbert_doc_crf_ner.py` | `results/modernbert_doc_crf_ner_config_doc_5e5_bs2.csv`, `results/modernbert_doc_crf_ner_config_doc_5e5_bs2.json` |

Copy-paste:

```bash
uv run python scripts/train_modernbert_ner.py
uv run python scripts/train_modernbert_doc_ner.py
uv run python scripts/train_modernbert_crf_ner.py
uv run python scripts/train_modernbert_doc_crf_ner.py
```

[`train_bert_ner.py`](scripts/train_bert_ner.py) writes e.g. [`results/bert_ner_config_0.csv`](results/bert_ner_config_0.csv). [`train_runner.py`](scripts/train_runner.py) can launch trainers by name (`bert`, `modernbert`, `modernbert_doc`, `modernbert_crf`, `modernbert_doc_crf`, `all`). Note: `bert_doc` in the runner points to a script that is not in this repo.

The full list of hyperparameter entries is defined in each training script’s `HP_CONFIGS`; a full sweep is heavier than a single-config run.

Verification helpers (toy checks / data sanity): `uv run python scripts/conll2003_dataset_verification.py`, `conll2003_concat_verification.py`, `conll2003_tokenization_compare.py`, and `conll2003_crf_verification.py` (pytorch-crf padding mask, BIO constraints, dense-label roundtrip vs. datasets).
