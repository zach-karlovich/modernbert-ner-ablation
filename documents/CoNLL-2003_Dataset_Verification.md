# CoNLL-2003 dataset verification

Two notions of “sentence” appear in the original English `eng.*` files:

|                                          |  Train | Dev (`eng.testa`) | Test (`eng.testb`) |
| ---------------------------------------- | -----: | ----------------: | -----------------: |
| **Blank lines** in the file              | 14,987 |             3,466 |              3,684 |
| **Parsed NER sentences** (`parse_conll`) | 14,041 |             3,250 |              3,453 |

Each `-DOCSTART-` line is followed by **one blank line** before the first token line of that article. That blank is a **file-format** separator, not the end of a word-level sentence, so it is counted in “blank lines” but **not** as a training example.

The gap per split equals the number of documents: **946 / 216 / 231** `-DOCSTART-` lines.

**Checks:** `uv run python scripts/conll2003_dataset_verification.py` — asserts parsed counts, blank-line totals, `-DOCSTART-` counts, and B-tag span tallies. Training scripts call the same logic via `conll2003_expectations` and `conll2003_parse.parse_conll`.

When we cite “number of sentences” in the paper, we specify whether we mean **blank lines** (common in corpus statistics) or **parsed sentences** (what NER training actually uses).
