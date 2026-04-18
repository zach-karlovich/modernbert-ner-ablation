import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = {
    "bert": "train_bert_ner.py",
    "bert_doc": "train_bert_doc_ner.py",
    "modernbert": "train_modernbert_ner.py",
    "modernbert_doc": "train_modernbert_doc_ner.py",
    "modernbert_crf": "train_modernbert_crf_ner.py",
    "modernbert_doc_crf": "train_modernbert_doc_crf_ner.py",
    "modernbert_doc_crf_doc0": "train_modernbert_doc_crf_ner_doc0.py",
    "modernbert_doc_crf_doc5e5": "train_modernbert_doc_crf_ner_doc5e5.py",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Run NER training scripts.")
    p.add_argument(
        "which",
        choices=[
            "bert",
            "bert_doc",
            "modernbert",
            "modernbert_doc",
            "modernbert_crf",
            "modernbert_doc_crf",
            "modernbert_doc_crf_doc0",
            "modernbert_doc_crf_doc5e5",
            "all",
        ],
        help="Which trainer to run, or all in SCRIPTS order.",
    )
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    order = list(SCRIPTS.keys()) if args.which == "all" else [args.which]

    for key in order:
        script = here / SCRIPTS[key]
        subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    main()
