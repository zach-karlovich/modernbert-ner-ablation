"""CoNLL-2003 sentence-level parsing (blank-line sentences, -DOCSTART- boundaries)."""

from __future__ import annotations

from pathlib import Path

Sentence = list[tuple[str, str]]


def parse_conll(filepath: Path | str) -> list[Sentence]:
    """One list entry per CoNLL sentence (blank-line delimited).

    ``-DOCSTART-`` marks a new document. If the previous sentence does not end
    with a blank line before the next ``-DOCSTART-``, the in-progress sentence is
    flushed first so the sentence count matches blank-line / official splits.
    """
    path = Path(filepath)
    sentences: list[Sentence] = []
    current: Sentence = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line == "":
                if current:
                    sentences.append(current)
                    current = []
            else:
                parts = line.split()
                current.append((parts[0], parts[-1]))
        if current:
            sentences.append(current)
    return sentences
