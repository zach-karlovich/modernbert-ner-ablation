import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = {
    "bert": "train_bert_ner.py",
    "modernbert": "train_modernbert_ner.py",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Run NER training scripts.")
    p.add_argument(
        "which",
        choices=["bert", "modernbert", "all"],
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
