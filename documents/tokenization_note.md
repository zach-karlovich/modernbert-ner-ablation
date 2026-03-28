# CoNLL-2003 tokenization: BERT vs ModernBERT

For this project, **each encoder is paired with its own pretrained tokenizer** (`bert-base-cased` and `answerdotai/ModernBERT-base`). That pairing is fixed: we do not swap tokenizers across checkpoints. **BERT** appears as a **historical / literature comparison**; any gap versus ModernBERT is **not** a controlled “encoder-only” ablation, because vocabulary, subword statistics, pretraining data, and architecture all differ.

## Which tokenizers (and how they differ)

| | **BERT** (`bert-base-cased`) | **ModernBERT** (`answerdotai/ModernBERT-base`) |
|--|------------------------------|-----------------------------------------------|
| **Subword algorithm** | WordPiece (continuations marked with `##` in decoded pieces) | BPE (byte-pair style; different merge table and surface rules) |
| **Vocabulary** | Original BERT cased vocab, fixed size and id↔token map tied to the checkpoint | Separate vocab learned with ModernBERT; **token ids are not comparable** across the two models |
| **Normalization / specials** | BERT’s defaults (e.g. `[CLS]`, `[SEP]`, `[PAD]`) | ModernBERT’s own special tokens and padding id (must use this tokenizer with this model) |

Same Unicode word string can therefore map to **different numbers of pieces, different piece boundaries, and different ids** under the two pipelines—this is expected and is why we treat tokenizer+encoder as one unit per model.

The script `scripts/conll2003_tokenization_compare.py` measures tokenization on the same CoNLL files and parsing as training (after skipping `-DOCSTART-`). Summary of what it reports:

## Words split into more than one subword

Roughly **twice as many** word tokens are split into multiple subwords under ModernBERT than under BERT on the same surface words.

| Split        | BERT (% multi-subword) | ModernBERT (% multi-subword) |
|-------------|-------------------------|------------------------------|
| Train       | ~16.8%                  | ~30.9%                       |
| Validation  | ~16.0%                  | ~30.5%                       |
| Test        | ~17.9%                  | ~31.0%                       |

Mean subwords per word is higher for ModernBERT (~**1.48** train) than BERT (~**1.34** train); medians stay **1** for both because most words are still a single piece.

## Sequence length (full encoding, no truncation)

For the same sentences, **ModernBERT’s `len(input_ids)` is longer more often** than BERT’s (on the order of **~60%** of sentences longer under ModernBERT vs **~18–21%** longer under BERT, with the rest tied—exact mix varies slightly by split). Mean tokens per sentence is a few positions higher for ModernBERT than BERT on each split. In our data, **no** sentences exceed 512 tokens for either tokenizer, so truncation at 512 is not driving differences from length alone.

## Implication for NER (first-subword labeling)

We use a **first-subword** label strategy with `word_ids()`: only the **first** subpiece of each word gets the gold tag; continuation subwords are masked in the loss. **Heavier subword splitting** means **fewer supervised positions per word** and more reliance on the first piece to carry the whole word’s semantics. That is a plausible **confound** when comparing BERT and ModernBERT numbers—not “extra flexibility” in the head, but **tighter coupling between segmentation and where learning signal is applied**.

Re-run or refresh numbers with:

`uv run python scripts/conll2003_tokenization_compare.py`
