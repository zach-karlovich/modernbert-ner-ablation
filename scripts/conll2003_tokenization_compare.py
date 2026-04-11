from pathlib import Path

from transformers import AutoTokenizer

from conll2003_parse import parse_conll

root = Path.cwd() if (Path.cwd() / "pyproject.toml").exists() else Path.cwd().parent
out = root / "data" / "conll2003"

required = ["eng.train", "eng.testa", "eng.testb"]
missing = [n for n in required if not (out / n).exists()]
assert not missing, (
    f"CoNLL-2003 data not found at {out}. Missing: {missing}. "
    "Run: python scripts/download_data.py to download and save data to data/conll2003/"
)

SPLIT_NAMES = {"eng.train": "train", "eng.testa": "validation", "eng.testb": "test"}
MAX_LENGTH = 512


def subword_counts_per_word(encoding, n_words):
    counts = [0] * n_words
    for wid in encoding.word_ids():
        if wid is not None:
            counts[wid] += 1
    return counts


def summarize_tokenizer(name, tokenizer, sentences):
    total_words = 0
    multi_subword_words = 0
    all_piece_counts = []
    seq_lens = []
    truncated = 0

    for sentence in sentences:
        words = [w for w, _ in sentence]
        if not words:
            continue
        enc = tokenizer(
            words,
            is_split_into_words=True,
            truncation=False,
            padding=False,
        )
        n_tok = len(enc["input_ids"])
        seq_lens.append(n_tok)
        if n_tok > MAX_LENGTH:
            truncated += 1

        for k in subword_counts_per_word(enc, len(words)):
            total_words += 1
            if k > 1:
                multi_subword_words += 1
            all_piece_counts.append(k)

    n_sents = len(seq_lens)
    mean_pieces = sum(all_piece_counts) / total_words if total_words else 0.0
    sorted_pieces = sorted(all_piece_counts)
    if not sorted_pieces:
        median_pieces = 0.0
    else:
        mid = len(sorted_pieces) // 2
        if len(sorted_pieces) % 2 == 1:
            median_pieces = float(sorted_pieces[mid])
        else:
            median_pieces = (sorted_pieces[mid - 1] + sorted_pieces[mid]) / 2

    mean_len = sum(seq_lens) / n_sents if n_sents else 0.0
    max_len = max(seq_lens) if seq_lens else 0

    print(f"  {name}:")
    print(f"    words: {total_words}")
    print(
        f"    words split into >1 token: {multi_subword_words} "
        f"({100.0 * multi_subword_words / total_words:.2f}%)" if total_words else "    words split into >1 token: n/a"
    )
    print(f"    subwords per word (mean / median): {mean_pieces:.4f} / {median_pieces:.4f}")
    print(f"    input_ids length per sentence (mean / max): {mean_len:.2f} / {max_len}")
    print(
        f"    sentences with len(input_ids) > {MAX_LENGTH}: {truncated} "
        f"({100.0 * truncated / n_sents:.2f}%)" if n_sents else f"    sentences with len(input_ids) > {MAX_LENGTH}: n/a"
    )


def compare_pair(bert_tok, modern_tok, sentences):
    bert_longer = 0
    modern_longer = 0
    equal = 0
    best_diff = []

    for sentence in sentences:
        words = [w for w, _ in sentence]
        if not words:
            continue
        eb = bert_tok(words, is_split_into_words=True, truncation=False, padding=False)
        em = modern_tok(words, is_split_into_words=True, truncation=False, padding=False)
        lb = len(eb["input_ids"])
        lm = len(em["input_ids"])
        if lb > lm:
            bert_longer += 1
        elif lm > lb:
            modern_longer += 1
        else:
            equal += 1
        best_diff.append((abs(lm - lb), lm - lb, words))

    best_diff.sort(reverse=True)
    n = bert_longer + modern_longer + equal
    print("  Per sentence (full length, no truncation):")
    print(f"    BERT longer: {bert_longer} ({100.0 * bert_longer / n:.2f}%)")
    print(f"    ModernBERT longer: {modern_longer} ({100.0 * modern_longer / n:.2f}%)")
    print(f"    Same length: {equal} ({100.0 * equal / n:.2f}%)")
    print("  Top 5 sentences by |len_modern - len_bert| (word list truncated to 12):")
    for _, diff, words in best_diff[:5]:
        preview = words[:12]
        more = " ..." if len(words) > 12 else ""
        print(f"    diff={diff:+d}  {' '.join(preview)}{more}")


print("Loading tokenizers (bert-base-cased, answerdotai/ModernBERT-base)...")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
modern_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

print("\nCoNLL-2003 tokenization comparison (data/conll2003/)\n")

for fname in required:
    split = SPLIT_NAMES[fname]
    sents = parse_conll(out / fname)
    print(f"=== {split} ({fname}) — {len(sents)} sentences ===\n")
    summarize_tokenizer("BERT", bert_tokenizer, sents)
    summarize_tokenizer("ModernBERT", modern_tokenizer, sents)
    print("\n  BERT vs ModernBERT sequence lengths:")
    compare_pair(bert_tokenizer, modern_tokenizer, sents)
    print()
