# moderbert-ner-ablation

Evaluating document-level context and CRF decoding in ModernBERT for CoNLL-2003 named entity recognition (NER).

Overleaf document: [ds6050-project](https://www.overleaf.com/project/6996373c44b841199bc3c599)

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

## Planned Final Model

ModernBERT with document-level context and a CRF decoding head, evaluated against the sentence-level ModernBERT baseline and single-factor variants.

### How to Load

## Environment Setup

### Install (uv, Python 3.14)

```bash
uv python install 3.14
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

## Training

