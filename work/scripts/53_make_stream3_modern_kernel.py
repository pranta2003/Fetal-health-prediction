# FIELD: Tabular
# MODEL: XGBoost / LightGBM / CatBoost / LR
# STEP: kernel_builder
# USED BY: work/docs/CODE_MAP.md

"""scripts/53_make_stream3_modern_kernel.py

Generates Kaggle CPU notebook kernel for Stream 3 Modern GBDT Campaign (Step 1):
- Notebook: notebooks/kaggle/stream3_modern/stream3-modern-s1/stream3-modern-s1.ipynb
- Metadata: notebooks/kaggle/stream3_modern/stream3-modern-s1/kernel-metadata.json
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNEL_DIR = REPO_ROOT / "notebooks" / "kaggle" / "stream3_modern" / "stream3-modern-s1"


def make_code_cell(source: str, cell_id: str | None = None) -> dict:
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")],
    }
    if cell_id:
        cell["id"] = cell_id
    return cell


def make_markdown_cell(source: str, cell_id: str | None = None) -> dict:
    cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")],
    }
    if cell_id:
        cell["id"] = cell_id
    return cell


def main():
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": "prantakumerpandit390/stream3-modern-s1",
        "title": "stream3-modern-s1",
        "code_file": "stream3-modern-s1.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [
            "prantakumerpandit390/capstone-processed"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }

    cells = []

    # Cell 1: Intro
    cells.append(make_markdown_cell("""# Stream 3 Modern GBDT Campaign (Step 1)
- **Protocol**: Temporal split reproducing E1 (train <= 2021, test >= 2022).
- **Features**: Primary clinical feature set (zero outcome-time variables).
- **Models**: CatBoost, LightGBM, and XGBoost with Optuna hyperparameter tuning (40 trials).
- **Ensemble**: Calibrated probability blend of best GBM + tuned Logistic Regression.
- **Hardware**: Kaggle CPU (consumes zero GPU quota).
""", cell_id="cell_1_intro"))

    # Cell 2: Install packages
    cells.append(make_code_cell("""!pip install --quiet optuna catboost lightgbm xgboost
""", cell_id="cell_2_install"))

    # Cell 3: Data discovery & fail-fast
    cells.append(make_code_cell("""import os
from pathlib import Path

print("=" * 60)
print("INSPECTING INPUT DATASET")
print("=" * 60)

for root, dirs, files in os.walk("/kaggle/input"):
    print(f"{root}: {len(files)} files")

csv_candidates = list(Path("/kaggle/input").rglob("*.csv"))
print(f"Discovered CSV files: {csv_candidates}")
assert len(csv_candidates) > 0, "No CSV dataset found in /kaggle/input"
csv_path = csv_candidates[0]
print(f"Using clinical dataset: {csv_path}")
""", cell_id="cell_3_discovery"))

    # Cell 4: Read and embed runner script execution via base64 for zero-escaping safety
    import base64
    runner_code = (REPO_ROOT / "scripts" / "62_train_stream3_modern.py").read_text(encoding="utf-8")
    b64_code = base64.b64encode(runner_code.encode("utf-8")).decode("ascii")
    run_cell_text = f"""import base64
import subprocess

script_code = base64.b64decode({repr(b64_code)}).decode("utf-8")
with open("/kaggle/working/train_modern.py", "w", encoding="utf-8") as f:
    f.write(script_code)

cmd = f"python /kaggle/working/train_modern.py --csv '{{csv_path}}' --out-dir /kaggle/working/stream3_modern --n-trials 40"
print("Executing command:\\n", cmd)
ret = subprocess.run(cmd, shell=True)
if ret.returncode != 0:
    raise RuntimeError(f"Stream 3 modern training failed with return code {{ret.returncode}}")
"""
    cells.append(make_code_cell(run_cell_text, cell_id="cell_4_run"))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_nb = KERNEL_DIR / "stream3-modern-s1.ipynb"
    with open(out_nb, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    out_meta = KERNEL_DIR / "kernel-metadata.json"
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated {out_nb} and {out_meta}")


if __name__ == "__main__":
    main()
