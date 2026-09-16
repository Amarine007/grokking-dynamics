"""Build colab/grokking-dynamics.zip: the code a Colab session needs.

    python colab/make_bundle.py

Upload the zip when a colab/*.ipynb notebook asks for it. Checkpoints are not included
(the notebooks produce them), but the committed CSVs in results/ ARE: without them
regenerate_checkpoints.py has nothing to byte-compare against and silently reports
NO COMMITTED CSV instead of MATCH, which is how a 45-minute run once verified nothing.
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = [
    "src",
    "experiments",
    "configs",
    "tests",
    "results",  # committed CSVs: required for the regenerate_checkpoints byte-comparison
    "figures/make_all.py",
    "requirements.txt",
    "README.md",
]
OUT = ROOT / "colab" / "grokking-dynamics.zip"


def main() -> None:
    files = []
    for item in INCLUDE:
        path = ROOT / item
        files += [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    files = sorted(p for p in files if "__pycache__" not in p.parts and p.suffix != ".pyc")
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, Path("grokking-dynamics") / p.relative_to(ROOT))
    print(f"wrote {OUT} ({len(files)} files, {OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
