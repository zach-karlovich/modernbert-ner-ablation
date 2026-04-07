"""
Word-aligned sliding windows over CoNLL-style word lists (subword token space).

Used by training datasets and conll2003_sliding_window.py verification.
"""

from __future__ import annotations

DEFAULT_TOKEN_OVERLAP = 128


def prefix_subwords_per_word(tokenizer, words: list[str]) -> list[int]:
    if not words:
        return [0]
    enc = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=False,
    )
    wids = enc.word_ids()
    n = len(words)
    counts = [0] * n
    for wid in wids:
        if wid is not None:
            counts[wid] += 1
    prefix = [0]
    for c in counts:
        prefix.append(prefix[-1] + c)
    return prefix


def _first_word_with_token_after(prefix: list[int], start_t: int) -> int:
    n_words = len(prefix) - 1
    lo, hi = 0, n_words - 1
    ans = n_words
    while lo <= hi:
        mid = (lo + hi) // 2
        if prefix[mid + 1] > start_t:
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans


def _max_exclusive_end(prefix: list[int], w_start: int, budget: int) -> int:
    n = len(prefix) - 1
    lo, hi = w_start + 1, n
    best = w_start
    while lo <= hi:
        mid = (lo + hi + 1) // 2
        if prefix[mid] - prefix[w_start] <= budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best if best > w_start else w_start + 1


def build_windows_word_ranges(
    prefix: list[int], budget: int, overlap: int
) -> list[tuple[int, int]]:
    total = prefix[-1]
    if total <= budget:
        return [(0, len(prefix) - 1)]

    stride = budget - overlap
    if stride <= 0:
        raise ValueError("budget must exceed overlap")

    ranges: list[tuple[int, int]] = []
    start_t = 0
    n_words = len(prefix) - 1

    while start_t < total:
        end_t = min(start_t + budget, total)
        w_start = _first_word_with_token_after(prefix, start_t)
        if w_start >= n_words:
            break
        w_end = _max_exclusive_end(prefix, w_start, budget)
        ranges.append((w_start, w_end))

        if end_t >= total:
            break
        start_t = end_t - overlap

    return ranges


def token_span(
    prefix: list[int], w_lo: int, w_hi_excl: int
) -> tuple[int, int]:
    return prefix[w_lo], prefix[w_hi_excl]


def overlap_subwords(
    prefix: list[int], a: tuple[int, int], b: tuple[int, int]
) -> int:
    ta0, ta1 = token_span(prefix, a[0], a[1])
    tb0, tb1 = token_span(prefix, b[0], b[1])
    return max(0, min(ta1, tb1) - max(ta0, tb0))


def filter_windows_intersecting_target(
    ranges: list[tuple[int, int]],
    target_lo: int,
    target_hi_excl: int,
) -> list[tuple[int, int]]:
    out = []
    for a, b in ranges:
        if b <= target_lo or a >= target_hi_excl:
            continue
        out.append((a, b))
    return out


def pick_best_centered_window(
    prefix: list[int],
    candidates: list[tuple[int, int]],
    target_lo: int,
    target_hi_excl: int,
) -> tuple[int, int]:
    if not candidates:
        raise ValueError("pick_best_centered_window: empty candidates")
    if len(candidates) == 1:
        return candidates[0]
    mid_t = (prefix[target_lo] + prefix[target_hi_excl]) / 2.0

    def score(r: tuple[int, int]) -> tuple[float, int]:
        a, b = r
        mid_w = (prefix[a] + prefix[b]) / 2.0
        return (abs(mid_w - mid_t), a)

    return min(candidates, key=score)


def word_window_from_start(
    prefix: list[int], w_start: int, budget: int
) -> tuple[int, int]:
    """Largest word-aligned window from w_start with at most budget subwords."""
    return (w_start, _max_exclusive_end(prefix, w_start, budget))
