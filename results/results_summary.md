# NER Results Summary — CoNLL-2003

**Last updated:** 2026-04-03. **Sentence-level ModernBERT** in the main tables uses **config B** (best test micro F1 in the sweep: [`old/modernbert_ner_config_B.csv`](old/modernbert_ner_config_B.csv)). **BERT** in those tables stays **config 0** (2e-5) from [`bert_ner_config_0.csv`](bert_ner_config_0.csv). Matched-HP ModernBERT (config 0): [`modernbert_ner_config_0.csv`](modernbert_ner_config_0.csv); logs [`train_bert.txt`](train_bert.txt), [`train_mbert.txt`](train_mbert.txt). **Document / CRF / Doc+CRF** columns are `—` until runs finish ([`train_modernbert_doc_ner.py`](../scripts/train_modernbert_doc_ner.py), [`train_runner.py`](../scripts/train_runner.py)). Full sweep: [`results_summary.csv`](results_summary.csv), [`old/`](old/).

All runs use 3 seeds (21, 42, 63). Metrics are mean ± std on the **test set**.

## Micro F1 Overview


| Model                          | Config / variant                | LR    | Epochs | BS  | Test Micro F1       | Δ vs BERT |
| ------------------------------ | ------------------------------- | ----- | ------ | --- | ------------------- | --------- |
| **BERT** (baseline, config 0)  | —                               | 2e-05 | 5      | 16  | **0.9128 ± 0.0025** | —         |
| ModernBERT                     | Sentence (**best tuned B**)     | 5e-05 | 5      | 16  | **0.8984 ± 0.0021** | -1.44 pp  |
| ModernBERT                     | Sentence (matched HP, config 0) | 2e-05 | 5      | 16  | 0.8862 ± 0.0023     | -2.66 pp  |
| ModernBERT                     | + document context              | 5e-05 | 5      | 2   | —                   | —         |
| ModernBERT                     | + CRF (sentence)                | —     | —      | —   | —                   | —         |
| ModernBERT                     | + document + CRF                | —     | —      | —   | —                   | —         |


## Per-Entity F1 Breakdown

**BERT** (config 0, 2e-5) vs **ModernBERT sentence config B** (best test micro F1 in the sweep; [`old/modernbert_ner_config_B.csv`](old/modernbert_ner_config_B.csv)), plus **ablation** columns. Entity order matches `tab:entity-dist` (PER, ORG, LOC, MISC). Test set, mean ± std.

| Entity | BERT (config 0)     | ModernBERT (sentence, **B**) | ModernBERT (document) | ModernBERT (sentence + CRF) | ModernBERT (document + CRF) |
| ------ | ------------------- | ---------------------------- | --------------------- | --------------------------- | ----------------------------- |
| PER    | **0.9622** ± 0.0019 | 0.9573 ± 0.0016              | —                     | —                           | —                             |
| ORG    | **0.8975** ± 0.0037 | 0.8669 ± 0.0039              | —                     | —                           | —                             |
| LOC    | **0.9305** ± 0.0025 | 0.9178 ± 0.0020              | —                     | —                           | —                             |
| MISC   | **0.7973** ± 0.0021 | 0.7969 ± 0.0049              | —                     | —                           | —                             |


**BERT** leads on every entity type vs tuned sentence-level ModernBERT. Largest gap: **ORG** (~3.1 pp). **MISC** is nearly tied (~0.04 pp). Matched-HP ModernBERT (config 0) per-entity values: [`modernbert_ner_config_0.csv`](modernbert_ner_config_0.csv).

## Per-Entity F1 Breakdown — Tuned Results

**Both encoders** at **best test micro F1** HP from archived sweeps ([`results_summary.csv`](results_summary.csv), [`old/bert_ner_config_*.csv`](old/bert_ner_config_3e5.csv)). **BERT**: LR **3e-5** ([`old/bert_ner_config_3e5.csv`](old/bert_ner_config_3e5.csv); ties **2e-5** at 0.9137 micro; 3e-5 has slightly lower std). **ModernBERT** sentence: config **B** (same as the main Per-Entity table above). Document / CRF columns stay `—` until those runs exist.

### Best test F1 (tuned sentence-level)

| Model        | Config / source              | LR    | Epochs | BS  | Test micro F1       | Test macro F1       | Δ micro vs tuned BERT |
| ------------ | ---------------------------- | ----- | ------ | --- | ------------------- | ------------------- | ----------------------- |
| **BERT**     | `old/bert_ner_config_3e5`    | 3e-05 | 5      | 16  | **0.9137 ± 0.0007** | **0.8990 ± 0.0010** | —                       |
| ModernBERT   | B (`old/modernbert_ner_B`)   | 5e-05 | 5      | 16  | 0.8984 ± 0.0021     | 0.8847 ± 0.0022     | -1.53 pp                |
| ModernBERT   | + document context           | —     | —      | —   | —                   | —                   | —                       |
| ModernBERT   | + CRF (sentence)             | —     | —      | —   | —                   | —                   | —                       |
| ModernBERT   | + document + CRF             | —     | —      | —   | —                   | —                   | —                       |

### Per-entity F1 (same columns, best HP per available variant)

Entity order matches `tab:entity-dist` (PER, ORG, LOC, MISC). BERT and ModernBERT (sentence) cells come from the tuned CSVs above.

| Entity | BERT (best HP)      | ModernBERT (sentence, best HP) | ModernBERT (document) | ModernBERT (sentence + CRF) | ModernBERT (document + CRF) |
| ------ | ------------------- | ------------------------------ | --------------------- | --------------------------- | ----------------------------- |
| PER    | **0.9612** ± 0.0010 | 0.9573 ± 0.0016                | —                     | —                           | —                             |
| ORG    | **0.8962** ± 0.0026 | 0.8669 ± 0.0039                | —                     | —                           | —                             |
| LOC    | **0.9314** ± 0.0020 | 0.9178 ± 0.0020                | —                     | —                           | —                             |
| MISC   | **0.8073** ± 0.0041 | 0.7969 ± 0.0049                | —                     | —                           | —                             |

With tuned sentence-level settings, **BERT** still leads on every entity type vs the best ModernBERT sweep config (B). The largest gap is again **ORG** (~2.9 pp). Baseline (matched-HP) sentence ModernBERT widens that ORG gap vs BERT; tuning ModernBERT partially closes it but does not erase it.

## Training diagnostics (from logs)

- **BERT** ([`train_bert.txt`](train_bert.txt)): Best dev F1 occurs at **epoch 5 for all three seeds**; dev F1 is still inching up on the last epoch in some runs. Val loss drifts up slightly while train loss decays—mild overfitting pressure. Consider **trying more than 5 epochs** (or dev-based early stopping) to see if dev F1 improves without hurting test selection.
- **ModernBERT 0** ([`train_mbert.txt`](train_mbert.txt)): Strong **overfitting signal**: train loss approaches ~0 while **val loss rises** after epoch 2–3 and **val F1 is volatile** late in training. Same **epoch-5 best** for all seeds—worth revisiting regularization, LR, or epoch count.
- **ModernBERT G**: Best dev at **epoch 10** (two seeds) and **epoch 9** (one seed). Val loss trends up in later epochs while train loss vanishes—**overfitting** after the mid-run sweet spot; longer training without early stopping on dev would likely hurt.
- **Document run** ([`train_mbert_doc.txt`](train_mbert_doc.txt)): Log **stops after epoch 1 / seed 21**—incomplete; no metrics to report until the run finishes.

## Notes

- **BERT** and **ModernBERT** both evaluate on the test set using the checkpoint with **highest dev F1** for each seed (best-checkpoint selection), then aggregate mean ± std across seeds.
- **BERT** (current log): mean best dev F1 **0.9510 ± 0.0006**; mean best epoch **5.00** (all seeds at epoch 5). Source: [`train_bert.txt`](train_bert.txt).
- **ModernBERT 0** (same log): mean best dev F1 **0.9395 ± 0.0030**; mean best epoch **5.00**. Source: [`train_mbert.txt`](train_mbert.txt) (config 0 block).
- Full hyperparameter grid (17 ModernBERT configs) is in [`results_summary.csv`](results_summary.csv). Rows **BERT** and **0** updated 2026-04-03 to match current CSVs/logs. An archived sweep table is in [`old/modernbert_hp_sweep_summary.csv`](old/modernbert_hp_sweep_summary.csv).
