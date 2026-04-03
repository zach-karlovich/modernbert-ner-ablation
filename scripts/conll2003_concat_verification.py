"""
Verify sentence-level vs document-level CoNLL-2003 parsing agree.
"""

from pathlib import Path

_here = Path.cwd()
root = _here if (_here / "pyproject.toml").exists() else _here.parent
out = root / "data" / "conll2003"
required = ["eng.train", "eng.testa", "eng.testb"]


def parse_conll(filepath):
    """Sentence-level: skip -DOCSTART-, split on blank lines."""
    sentences = []
    current = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("-DOCSTART-"):
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


def parse_conll_documents(filepath):
    """Document-level: split on -DOCSTART- then blank-line sentences."""
    documents = []
    current_doc = []
    current_sent = []

    with open(filepath, encoding="utf-8") as f:
        for line in f:
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
            assert len(parts) >= 2, f"Malformed line: {line}"
            current_sent.append((parts[0], parts[-1]))

    if current_sent:
        current_doc.append(current_sent)
    if current_doc:
        documents.append(current_doc)

    return documents


def concat_document_sentences(documents):
    return [sent for doc in documents for sent in doc]


def mismatch_detail(sentence_level, flat_from_docs):
    if len(sentence_level) != len(flat_from_docs):
        return f"sentence count {len(sentence_level)} vs {len(flat_from_docs)}"
    for i, (a, b) in enumerate(zip(sentence_level, flat_from_docs)):
        if a != b:
            if len(a) != len(b):
                return f"sentence index {i}: len {len(a)} vs {len(b)}"
            return f"sentence index {i}: content differs"
    return "unknown"


missing = [n for n in required if not (out / n).exists()]
assert not missing, (
    f"CoNLL-2003 data not found at {out}. Missing: {missing}. "
    "Run: python scripts/download_data.py"
)

print("CoNLL-2003 sentence vs document concat (data/conll2003/):")
for name in required:
    path = out / name
    sents = parse_conll(path)
    docs = parse_conll_documents(path)
    flat = concat_document_sentences(docs)
    ok = sents == flat
    status = "PASS" if ok else "FAIL"
    n_docs = len(docs)
    n_sents_doc = sum(len(d) for d in docs)
    line = (
        f"  {name}: {len(sents)} sents, {n_docs} docs "
        f"({n_sents_doc} sents in docs); concat == sentence-level {status}"
    )
    print(line)
    if not ok:
        print(f"    detail: {mismatch_detail(sents, flat)}")
