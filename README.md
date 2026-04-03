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

**Last updated:** 2026-04-03 (aligned with [results/results_summary.md](results/results_summary.md)).

Entity-level F1 on the CoNLL-2003 **test** set (`eng.testb`). Mean ± std over 3 seeds (21, 42, 63). For each seed, **test evaluation uses the checkpoint with highest dev F1** on `eng.testa`.

This section highlights the **matched-HP baseline** only: **BERT** and **ModernBERT** both use LR **2e-5**, **5** epochs, batch **16** (ModernBERT **config 0**). That isolates encoder/tokenizer differences under the same training recipe. HP sweep (e.g. best sentence-level **config B** at **0.8984** micro F1), tuned BERT, and ablation columns are in [results/results_summary.md](results/results_summary.md) and [results/results_summary.csv](results/results_summary.csv).

### Overall F1

| Model                                                  | Micro F1        | Macro F1        |
| ------------------------------------------------------ | --------------- | --------------- |
| BERT-base-cased (sentence-level, no CRF)               | 0.9128 ± 0.0025 | 0.8969 ± 0.0025 |
| ModernBERT-base (sentence-level, matched HP, config 0) | 0.8862 ± 0.0023 | 0.8720 ± 0.0024 |
| ModernBERT-base + document context                     | —               | —               |
| ModernBERT-base + CRF (sentence-level)                 | —               | —               |
| ModernBERT-base + document context + CRF               | —               | —               |

### Per-entity F1

Entity order: PER, ORG, LOC, MISC. Baseline cells match [`results/bert_ner_config_0.csv`](results/bert_ner_config_0.csv) and [`results/modernbert_ner_config_0.csv`](results/modernbert_ner_config_0.csv).

| Entity | BERT                | ModernBERT (sentence, config 0) | ModernBERT (document) | ModernBERT (sentence + CRF) | ModernBERT (document + CRF) |
| ------ | ------------------- | ------------------------------- | --------------------- | --------------------------- | --------------------------- |
| PER    | **0.9622** ± 0.0019 | 0.9528 ± 0.0035                 | —                     | —                           | —                           |
| ORG    | **0.8975** ± 0.0037 | 0.8458 ± 0.0030                 | —                     | —                           | —                           |
| LOC    | **0.9305** ± 0.0025 | 0.9100 ± 0.0021                 | —                     | —                           | —                           |
| MISC   | **0.7973** ± 0.0021 | 0.7796 ± 0.0045                 | —                     | —                           | —                           |

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

Training scripts live under `scripts/`: [`train_bert_ner.py`](scripts/train_bert_ner.py) writes aggregated metrics to e.g. [`results/bert_ner_config_0.csv`](results/bert_ner_config_0.csv); [`train_modernbert_ner.py`](scripts/train_modernbert_ner.py) writes one `results/modernbert_ner_config_<name>.csv` per sweep entry (e.g. [`results/modernbert_ner_config_G.csv`](results/modernbert_ner_config_G.csv)).

[`train_runner.py`](scripts/train_runner.py) runs them in subprocess order:

```bash
uv run python scripts/train_runner.py bert
uv run python scripts/train_runner.py modernbert
uv run python scripts/train_runner.py all
```

The full list of ModernBERT configurations is commented out in `train_modernbert_ner.py`; to review or run specific configs, refer directly to that script. By default, running `modernbert` will execute all configs defined there, so a complete sweep is much heavier than a single BERT baseline run.

Verification helpers (toy checks / data sanity): `uv run python scripts/conll2003_dataset_verification.py`, `conll2003_concat_verification.py`, `conll2003_tokenization_compare.py`, and `conll2003_crf_verification.py` (pytorch-crf padding mask + BIO illegal transitions).
