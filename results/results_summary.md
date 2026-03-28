# NER Results Summary — CoNLL-2003

All runs use 3 seeds (21, 42, 63). Metrics are mean ± std on the **test set**.

## Micro F1 Overview

| Model | Config | LR | Epochs | BS | Test Micro F1 | Delta vs BERT |
|---|---|---|---|---|---|---|
| **BERT** (baseline) | -- | 2e-05 | 5 | 16 | **0.9130 ± 0.0003** | -- |
| ModernBERT | 0 (matched HP) | 2e-05 | 5 | 16 | 0.8895 ± 0.0018 | -2.35 pp |
| ModernBERT | B (best 5-ep) | 5e-05 | 5 | 16 | 0.8984 ± 0.0021 | -1.46 pp |
| ModernBERT | **G (best overall)** | 6e-05 | 10 | 32 | **0.8999 ± 0.0002** | -1.31 pp |

## Per-Entity F1 Breakdown

| Entity | BERT | ModernBERT 0 | ModernBERT B | ModernBERT G |
|---|---|---|---|---|
| LOC | **0.9306** ± 0.0024 | 0.9148 ± 0.0011 | 0.9178 ± 0.0020 | 0.9215 ± 0.0022 |
| MISC | **0.8024** ± 0.0011 | 0.7776 ± 0.0064 | 0.7969 ± 0.0049 | 0.7922 ± 0.0111 |
| ORG | **0.8975** ± 0.0030 | 0.8519 ± 0.0028 | 0.8669 ± 0.0039 | 0.8688 ± 0.0059 |
| PER | **0.9605** ± 0.0018 | 0.9541 ± 0.0016 | 0.9573 ± 0.0016 | 0.9587 ± 0.0010 |

BERT leads on every entity type. The largest gap is on ORG (~2.9 pp for the best ModernBERT config).

## Notes

- ModernBERT used **best-checkpoint selection** (restoring the best val-F1 epoch for test evaluation); the original BERT run evaluated the **last epoch** only, giving ModernBERT a methodological advantage it still could not overcome.
- Full hyperparameter sweep details (17 configs) are in `modernbert_hp_sweep_summary.csv`. Individual config results are archived in `old/`.
