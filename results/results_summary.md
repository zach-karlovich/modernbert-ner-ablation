# NER Results Summary — CoNLL-2003

**Last updated:** 2026-03-28. Test metrics for BERT and ModernBERT G match [`bert_ner_config_0.csv`](bert_ner_config_0.csv) and [`modernbert_ner_config_G.csv`](modernbert_ner_config_G.csv). BERT checkpoint statistics (mean best dev F1, mean best epoch) match the training log [`old/train_ner_all.txt`](old/train_ner_all.txt) from the same run.

All runs use 3 seeds (21, 42, 63). Metrics are mean ± std on the **test set**.

## Micro F1 Overview


| Model               | Config               | LR    | Epochs | BS  | Test Micro F1       | Delta vs BERT |
| ------------------- | -------------------- | ----- | ------ | --- | ------------------- | ------------- |
| **BERT** (baseline) | --                   | 2e-05 | 5      | 16  | **0.9131 ± 0.0014** | --            |
| ModernBERT          | 0 (matched HP)       | 2e-05 | 5      | 16  | 0.8895 ± 0.0018     | -2.36 pp      |
| ModernBERT          | B (best 5-ep)        | 5e-05 | 5      | 16  | 0.8984 ± 0.0021     | -1.47 pp      |
| ModernBERT          | **G (best overall)** | 6e-05 | 10     | 32  | **0.9000 ± 0.0002** | -1.31 pp      |


## Per-Entity F1 Breakdown


| Entity | BERT                | ModernBERT 0    | ModernBERT B    | ModernBERT G    |
| ------ | ------------------- | --------------- | --------------- | --------------- |
| LOC    | **0.9308** ± 0.0005 | 0.9148 ± 0.0011 | 0.9178 ± 0.0020 | 0.9215 ± 0.0021 |
| MISC   | **0.8054** ± 0.0047 | 0.7776 ± 0.0064 | 0.7969 ± 0.0049 | 0.7925 ± 0.0116 |
| ORG    | **0.8961** ± 0.0043 | 0.8519 ± 0.0028 | 0.8669 ± 0.0039 | 0.8689 ± 0.0058 |
| PER    | **0.9607** ± 0.0016 | 0.9541 ± 0.0016 | 0.9573 ± 0.0016 | 0.9588 ± 0.0009 |


BERT leads on every entity type. The largest gap is on ORG (~2.7 pp for the best ModernBERT config).

## Notes

- **BERT** and **ModernBERT** both evaluate on the test set using the checkpoint with **highest dev F1** for each seed (best-checkpoint selection), then aggregate mean ± std across seeds.
- **BERT** (this run): mean best dev F1 **0.9515**; mean best epoch **4.67** (epochs 4, 5, 5 for seeds 21, 42, 63). Source: [`old/train_ner_all.txt`](old/train_ner_all.txt).
- Full hyperparameter grid (17 ModernBERT configs) is in [`results_summary.csv`](results_summary.csv). An archived sweep table is in [`old/modernbert_hp_sweep_summary.csv`](old/modernbert_hp_sweep_summary.csv). Individual per-config CSVs are in [`old/`](old/).
