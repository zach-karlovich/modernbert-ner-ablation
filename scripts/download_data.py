import kagglehub
from pathlib import Path
import shutil

root = Path.cwd() if (Path.cwd() / "pyproject.toml").exists() else Path.cwd().parent
out = root / "data" / "conll2003"
out.mkdir(parents=True, exist_ok=True)

path = kagglehub.dataset_download("juliangarratt/conll2003-dataset")
src = Path(path)
for n in ["eng.train", "eng.testa", "eng.testb"]:
    f = src / n if (src / n).exists() else next(src.rglob(n), None)
    if f:
        shutil.copy(f, out / n)

print("Original CoNLL-2003:", out)
