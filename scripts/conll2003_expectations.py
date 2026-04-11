"""Shared CoNLL-2003 (Kaggle / original) corpus checks.

``EXPECTED_SENTENCE_COUNTS`` are **parsed** sentence counts (``len(parse_conll)``):
word-line sequences between blank lines, excluding blank lines that appear only
immediately after ``-DOCSTART-`` (946 / 216 / 231 such lines per split). Raw blank
line totals in the files are 14987 / 3466 / 3684 — higher by exactly one per
document — and are **not** the number of NER training sentences.

Covers sentence counts, -DOCSTART- lines, and B-* span tallies.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Final

REQUIRED_FILES: Final = ("eng.train", "eng.testa", "eng.testb")

EXPECTED_SENTENCE_COUNTS: Final = {
    "eng.train": 14041,
    "eng.testa": 3250,
    "eng.testb": 3453,
}

BLANK_LINES_IN_FILE: Final = {
    "eng.train": 14987,
    "eng.testa": 3466,
    "eng.testb": 3684,
}

EXPECTED_DOCSTART_COUNTS: Final = {
    "eng.train": 946,
    "eng.testa": 216,
    "eng.testb": 231,
}

EXPECTED_B_TAG_COUNTS: Final = {
    "eng.train": {"PER": 6600, "ORG": 6321, "LOC": 7140, "MISC": 3438},
    "eng.testa": {"PER": 1842, "ORG": 1341, "LOC": 1837, "MISC": 922},
    "eng.testb": {"PER": 1617, "ORG": 1661, "LOC": 1668, "MISC": 702},
}


def count_blank_lines(filepath: Path) -> int:
    """All empty lines in the file (includes blanks after ``-DOCSTART-``)."""
    with filepath.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip() == "")


def count_docstart(filepath: Path) -> int:
    with filepath.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.startswith("-DOCSTART-"))


def count_b_tag_entities(filepath: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with filepath.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "" or line.startswith("-DOCSTART-"):
                continue
            ner_tag = line.split()[-1]
            if not ner_tag.startswith("B-"):
                continue
            entity_type = ner_tag.split("-", 1)[1]
            counts[entity_type] += 1
    return counts


def assert_parsed_sentence_counts_match_expected(
    train_n: int, dev_n: int, test_n: int
) -> None:
    """Raise if ``parse_conll`` sentence counts do not match official split sizes."""
    for name, n in (
        ("eng.train", train_n),
        ("eng.testa", dev_n),
        ("eng.testb", test_n),
    ):
        exp = EXPECTED_SENTENCE_COUNTS[name]
        if n != exp:
            raise ValueError(
                f"Parsed {n} sentences from {name}; expected {exp}. "
                "Check conll2003_parse.parse_conll."
            )


def assert_conll2003_dataset(data_dir: Path) -> None:
    """Raise if data_dir is not the expected original-format CoNLL-2003 release."""
    data_dir = data_dir.resolve()
    missing = [n for n in REQUIRED_FILES if not (data_dir / n).is_file()]
    if missing:
        raise FileNotFoundError(
            f"CoNLL-2003 data not found at {data_dir}. Missing: {missing}. "
            "Run: uv run python scripts/download_data.py"
        )
    from conll2003_parse import parse_conll

    for name in REQUIRED_FILES:
        path = data_dir / name
        n_parsed = len(parse_conll(path))
        exp = EXPECTED_SENTENCE_COUNTS[name]
        if n_parsed != exp:
            raise ValueError(
                f"{path.name}: expected {exp} parsed sentences, found {n_parsed}. "
                "Wrong or corrupted CoNLL-2003 copy "
                "(e.g. HuggingFace preprocessing has different split sizes)."
            )
        n_blank = count_blank_lines(path)
        exp_blank = BLANK_LINES_IN_FILE[name]
        if n_blank != exp_blank:
            raise ValueError(
                f"{path.name}: expected {exp_blank} blank lines in file, "
                f"found {n_blank}."
            )
        n_doc = count_docstart(path)
        exp_doc = EXPECTED_DOCSTART_COUNTS[name]
        if n_doc != exp_doc:
            raise ValueError(
                f"{path.name}: expected {exp_doc} -DOCSTART- lines, "
                f"found {n_doc}."
            )
        got = count_b_tag_entities(path)
        exp_bt = EXPECTED_B_TAG_COUNTS[name]
        for et, v in exp_bt.items():
            if got.get(et, 0) != v:
                got_n = got.get(et, 0)
                raise ValueError(
                    f"{path.name}: expected {v} B-{et} spans, found {got_n}."
                )


def print_verification_report(data_dir: Path) -> None:
    """Print the same report as the standalone verification CLI."""
    from conll2003_parse import parse_conll

    data_dir = data_dir.resolve()
    print("CoNLL-2003 verification (data/conll2003/):")
    for name in REQUIRED_FILES:
        path = data_dir / name
        n = len(parse_conll(path))
        exp = EXPECTED_SENTENCE_COUNTS[name]
        status = "PASS" if n == exp else "FAIL"
        nb = count_blank_lines(path)
        eb = BLANK_LINES_IN_FILE[name]
        print(
            f"  {name}: {n} parsed sentences (expected {exp}) {status}; "
            f"blank lines in file {nb} (expected {eb})"
        )

    print("\nEntity spans (B- tags):")
    split_names = ("train", "validation", "test")
    for name, split in zip(REQUIRED_FILES, split_names, strict=True):
        counts = count_b_tag_entities(data_dir / name)
        print(
            f"  {split}: PER={counts['PER']} ORG={counts['ORG']} "
            f"LOC={counts['LOC']} MISC={counts['MISC']}"
        )

    print("\nDOCSTART:")
    for name in REQUIRED_FILES:
        n = count_docstart(data_dir / name)
        exp = EXPECTED_DOCSTART_COUNTS[name]
        print(f"  {name}: {n} (expected {exp})")
