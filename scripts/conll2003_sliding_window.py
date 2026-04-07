"""
CoNLL-2003 sliding-window verification for document-flattened sequences.

Design (document in write-ups / reports):
- Token spans are half-open [start, end). Word j covers tokenizer subwords
  [prefix[j], prefix[j+1]) with prefix[0]=0 and prefix[n]=total subwords.
- Window starts are advanced in subword space by (budget - overlap), then the
  first word in the window is the first word whose end lies past start_t.
  Because windows are word-aligned, actual subword overlap between consecutive
  windows is at most the requested overlap and can be less when start_t snaps
  forward to the next word boundary (quantization gap).

Uses one full-sequence encode (add_special_tokens=False) to derive per-word
subword counts so lengths match training-style tokenization.

Run from repo root: uv run python scripts/conll2003_sliding_window.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from conll2003_labels import id2label, label2id  # noqa: E402
from dense_bio_labels import (  # noqa: E402
    assign_dense_bio_labels,
    continuation_tag,
)
from sliding_window_conll import (  # noqa: E402
    DEFAULT_TOKEN_OVERLAP,
    build_windows_word_ranges,
    overlap_subwords,
    prefix_subwords_per_word,
    token_span,
)

_here = Path.cwd()
root = _here if (_here / "pyproject.toml").exists() else _here.parent
DATA_DIR = root / "data" / "conll2003"
DEFAULT_TRAIN = DATA_DIR / "eng.train"

MODEL_ID = "answerdotai/ModernBERT-base"
MAX_SEQ_LENGTH = 8192
OVERLAP_TOKENS = DEFAULT_TOKEN_OVERLAP
N_DOCS = 10
RNG_SEED = 42


def parse_conll_documents(filepath: Path):
    documents: list[list[list[tuple[str, str]]]] = []
    current_doc: list[list[tuple[str, str]]] = []
    current_sent: list[tuple[str, str]] = []

    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                if current_sent:
                    current_doc.append(current_sent)
                    current_sent = []
                if current_doc:
                    documents.append(current_doc)
                    current_doc = []
                continue
            if line == "":
                if current_sent:
                    current_doc.append(current_sent)
                    current_sent = []
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Malformed line {line_num} in {filepath}: {line!r}"
                )
            current_sent.append((parts[0], parts[-1]))

    if current_sent:
        current_doc.append(current_sent)
    if current_doc:
        documents.append(current_doc)

    return documents


def flatten_doc(doc: list[list[tuple[str, str]]]):
    words: list[str] = []
    tags: list[str] = []
    for sent in doc:
        for w, t in sent:
            words.append(w)
            tags.append(t)
    return words, tags


def check_window_dense_labels(
    word_ids: list, label_ids: list, tags: list[str]
):
    for i, wid in enumerate(word_ids):
        if wid is None:
            if label_ids[i] != label2id["O"]:
                return (
                    f"tok {i}: special/padding expected O, got "
                    f"{id2label[label_ids[i]]}"
                )
            continue
        prev_w = word_ids[i - 1] if i > 0 else None
        is_word_start = prev_w != wid
        if is_word_start:
            exp = label2id[tags[wid]]
        else:
            exp = label2id[continuation_tag(tags[wid])]
        if label_ids[i] != exp:
            return (
                f"tok {i} word {wid}: expected {id2label[exp]}, "
                f"got {id2label[label_ids[i]]}"
            )
    return None


def check_window(
    tokenizer,
    words: list[str],
    tags: list[str],
    max_length: int,
    budget: int,
    special: int,
):
    enc = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=True,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=max_length,
    )
    ids = enc["input_ids"]
    if len(ids) > max_length:
        return f"len(input_ids)={len(ids)} > max_length={max_length}"
    content_tokens = len(ids) - special
    if content_tokens > budget:
        return (
            f"content subwords {content_tokens} > budget {budget} "
            f"(len(ids)={len(ids)}, special={special})"
        )

    word_ids = enc.word_ids()
    label_ids = assign_dense_bio_labels(word_ids, tags)
    return check_window_dense_labels(word_ids, label_ids, tags)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify word-aligned sliding windows and dense BIO alignment."
    )
    ap.add_argument(
        "--stress-budget",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Subword budget per window (content only). Low values force "
            "multiple windows on typical CoNLL docs so overlap logic is tested."
        ),
    )
    args = ap.parse_args()

    train_path = DEFAULT_TRAIN
    if not train_path.exists():
        print(
            f"Missing {train_path}. "
            "Run: uv run python scripts/download_data.py"
        )
        return 1

    random.seed(RNG_SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    special = tokenizer.num_special_tokens_to_add(pair=False)
    train_budget = MAX_SEQ_LENGTH - special
    if args.stress_budget is not None:
        if args.stress_budget <= OVERLAP_TOKENS:
            print(
                "--stress-budget must exceed OVERLAP_TOKENS "
                f"({OVERLAP_TOKENS})"
            )
            return 1
        win_budget = min(train_budget, args.stress_budget)
        max_len = win_budget + special
        stress_note = (
            f"stress_budget={args.stress_budget} -> win_budget={win_budget}"
        )
    else:
        win_budget = train_budget
        max_len = MAX_SEQ_LENGTH
        stress_note = "stress_budget=off"

    stride = win_budget - OVERLAP_TOKENS

    docs = parse_conll_documents(train_path)
    long_indices: list[int] = []
    for di, doc in enumerate(docs):
        words, _ = flatten_doc(doc)
        pref = prefix_subwords_per_word(tokenizer, words)
        if pref[-1] > win_budget:
            long_indices.append(di)

    k = min(N_DOCS, len(long_indices))
    if k > 0:
        chosen = sorted(random.sample(long_indices, k=k))
    else:
        chosen = sorted(random.sample(range(len(docs)), k=min(N_DOCS, len(docs))))

    print(
        f"model={MODEL_ID} max_length={max_len} content_budget={win_budget} "
        f"overlap_request={OVERLAP_TOKENS} stride={stride} ({stress_note})"
    )
    print(
        f"train docs={len(docs)} docs_over_win_budget={len(long_indices)} "
        f"sampled={len(chosen)} seed={RNG_SEED}"
    )
    if not long_indices and args.stress_budget is None:
        print(
            "Note: no train document exceeds content budget; all windows are "
            "single-chunk. Use --stress-budget 400 (or similar) to exercise "
            "multi-window overlap checks."
        )

    failed = False
    for di in chosen:
        doc = docs[di]
        words, tags = flatten_doc(doc)
        prefix = prefix_subwords_per_word(tokenizer, words)
        total = prefix[-1]
        wranges = build_windows_word_ranges(
            prefix, win_budget, OVERLAP_TOKENS
        )
        print(
            f"\ndoc {di}: n_words={len(words)} n_subwords={total} "
            f"windows={len(wranges)}"
        )

        for wi, (a, b) in enumerate(wranges):
            chunk_w = words[a:b]
            chunk_t = tags[a:b]
            err = check_window(
                tokenizer,
                chunk_w,
                chunk_t,
                max_len,
                win_budget,
                special,
            )
            t0, t1 = token_span(prefix, a, b)
            seg = t1 - t0
            if err:
                msg = (
                    f"  window {wi} words [{a},{b}) subwords={seg} FAIL: {err}"
                )
                print(msg)
                failed = True
            else:
                print(f"  window {wi} words [{a},{b}) subwords={seg} PASS")

        for wi in range(len(wranges) - 1):
            ov = overlap_subwords(prefix, wranges[wi], wranges[wi + 1])
            t_cur = token_span(prefix, *wranges[wi])
            t_next = token_span(prefix, *wranges[wi + 1])
            omsg = (
                f"  overlap windows {wi},{wi+1}: subwords={ov} "
                f"(want <= {OVERLAP_TOKENS}; spans [{t_cur[0]},{t_cur[1]}), "
                f"[{t_next[0]},{t_next[1]}))"
            )
            print(omsg)
            if ov > OVERLAP_TOKENS:
                print(
                    f"    FAIL: overlap {ov} exceeds requested "
                    f"{OVERLAP_TOKENS}"
                )
                failed = True
            if ov == 0 and total > win_budget:
                print(
                    "    FAIL: consecutive windows have zero subword overlap"
                )
                failed = True

    if failed:
        print("\nOverall: FAIL")
        return 1
    print("\nOverall: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
