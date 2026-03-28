# ModernBERT NER Ablation

***IN PROGRESS***

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

**Last updated:** 2026-03-28 (aligned with [results/results_summary.md](results/results_summary.md)).

Entity-level F1 on the CoNLL-2003 **test** set (`eng.testb`). Mean ± std over 3 seeds (21, 42, 63). For each seed, **test evaluation uses the checkpoint with highest dev F1** on `eng.testa`.

The ModernBERT row is sweep **config G** (learning rate 6e-5, 10 epochs, batch size 32).

| Model | Micro F1 | Macro F1 |
|-------|----------|----------|
| BERT-base-cased (sentence-level, no CRF) | 0.9131 ± 0.0014 | 0.8983 ± 0.0013 |
| ModernBERT-base (config G) | 0.9000 ± 0.0002 | 0.8854 ± 0.0016 |

Full hyperparameter grid, per-entity breakdown, and provenance: [results/results_summary.md](results/results_summary.md), [results/results_summary.csv](results/results_summary.csv).

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

