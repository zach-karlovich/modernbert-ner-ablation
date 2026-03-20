from pathlib import Path
from collections import Counter

root = Path.cwd() if (Path.cwd() / "pyproject.toml").exists() else Path.cwd().parent
out = root / "data" / "conll2003"

required = ["eng.train", "eng.testa", "eng.testb"]
missing = [n for n in required if not (out / n).exists()]
assert not missing, (
    f"CoNLL-2003 data not found at {out}. Missing: {missing}. "
    "Run: python scripts/download_data.py to download and save data to data/conll2003/"
)

EXPECTED = {"eng.train": 14987, "eng.testa": 3466, "eng.testb": 3684}
DOCSTART_EXPECTED = {"eng.train": 946, "eng.testa": 216, "eng.testb": 231}


def count_sentences(filepath):
    return sum(1 for line in filepath.open(encoding="utf-8") if line.strip() == "")


def count_entities(filepath):
    counts = Counter()
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "" or line.startswith("-DOCSTART-"):
                continue
            ner_tag = line.split()[-1]
            if ner_tag == "O":
                continue
            entity_type = ner_tag.split("-")[1]
            counts[entity_type] += 1
    return counts


def count_docstart(filepath):
    return sum(1 for line in filepath.open(encoding="utf-8") if line.startswith("-DOCSTART-"))


print("CoNLL-2003 verification (data/conll2003/):")
for name in required:
    n = count_sentences(out / name)
    exp = EXPECTED[name]
    status = "PASS" if n == exp else "FAIL"
    print(f"  {name}: {n} sentences (expected {exp}) {status}")

print("\nEntity distribution:")
for name, split in [("eng.train", "train"), ("eng.testa", "validation"), ("eng.testb", "test")]:
    counts = count_entities(out / name)
    print(f"  {split}: PER={counts['PER']} ORG={counts['ORG']} LOC={counts['LOC']} MISC={counts['MISC']}")

print("\nDOCSTART:")
for name in required:
    n = count_docstart(out / name)
    exp = DOCSTART_EXPECTED[name]
    print(f"  {name}: {n} (expected {exp})")
