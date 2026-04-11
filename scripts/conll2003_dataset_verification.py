"""CLI: verify data/conll2003 matches original Kaggle CoNLL-2003 release."""

from __future__ import annotations

import sys
from pathlib import Path

from conll2003_expectations import (
    assert_conll2003_dataset,
    print_verification_report,
)

_here = Path.cwd()
root = _here if (_here / "pyproject.toml").exists() else _here.parent
data_dir = root / "data" / "conll2003"

if __name__ == "__main__":
    try:
        assert_conll2003_dataset(data_dir)
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    print_verification_report(data_dir)
