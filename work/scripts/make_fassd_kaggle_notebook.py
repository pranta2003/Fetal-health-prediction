"""make_fassd_kaggle_notebook.py: Generate notebooks/kaggle/fassd_train_kernel.ipynb.

Creates the training notebook for Kaggle (T4 GPU), orchestrating experiments across:
- Code snapshot mounted from private dataset: prantakumerpandit390/fetal-code-snapshot
- Data mounted from private dataset: prantakumerpandit390/fassd-fetal-abdomen-v2
- Configurable RUNS list: executes (seed 42, rgb) and (seed 42, gray) now
- Outputs saved to /kaggle/working
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "notebooks" / "kaggle"
OUTPUT_NOTEBOOK = OUTPUT_DIR / "fassd_train_kernel.ipynb"


def build_notebook() -> dict:
    cells = []

    def add_markdown(source: str):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().splitlines()],
        })

    def add_code(source: str):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().splitlines()],
        })

    # Header
    add_markdown("""# FASSD Multilabel Segmentation Training Kernel
**Fetal Abdominal Structures Segmentation (artery, liver, stomach, vein)**
- Architecture: U-Net with `efficientnet-b4` encoder (ImageNet weights)
- Input Resolution: $384 \\times 512$ (native aspect-ratio of $768 \\times 1024$)
- Loss: Combined BCE + Soft-Dice multilabel loss (sigmoid activation)
- Optimization: AdamW, cosine annealing, AMP, Model EMA
- Evaluation: Native resolution ($768 \\times 1024$) Dice, IoU, surface HD95, patient bootstrap 95% CI
- Codebase: Mounted from private snapshot dataset `prantakumerpandit390/fetal-code-snapshot`
- Data: Mounted from private dataset `prantakumerpandit390/fassd-fetal-abdomen-v2`
""")

    # Run configuration
    add_markdown("## 1. Run Configuration")
    add_code("""# Select active experiment runs (seed 42 RGB and Gray active; remaining runs commented)
RUNS = [
    {"mode": "rgb", "seed": 42},
    {"mode": "gray", "seed": 42},
    # Future planned runs:
    # {"mode": "rgb", "seed": 43},
    # {"mode": "rgb", "seed": 44},
    # {"mode": "gray", "seed": 43},
    # {"mode": "gray", "seed": 44},
]

print(f"Active runs to execute ({len(RUNS)}):")
for r in RUNS:
    print(f"  - mode: {r['mode']}, seed: {r['seed']}")
""")

    # Environment setup
    add_markdown("## 2. Environment & Codebase Setup")
    add_code("""# Setup codebase from private snapshot dataset (no GitHub PAT / clone required)
import os
import shutil
import sys
from pathlib import Path

print("Scanning /kaggle/input...")
if Path("/kaggle/input").exists():
    for p in Path("/kaggle/input").glob("*"):
        print(" ", p)

code_input = None
candidates = [
    Path("/kaggle/input/fetal-code-snapshot"),
    Path("/kaggle/input/datasets/prantakumerpandit390/fetal-code-snapshot"),
]
for c in candidates:
    if c.exists() and (c / "COMMIT.txt").exists():
        code_input = c
        break

if code_input is None and Path("/kaggle/input").exists():
    for p in Path("/kaggle/input").rglob("COMMIT.txt"):
        code_input = p.parent
        break

if code_input is None:
    raise FileNotFoundError("Could not find fetal-code-snapshot in /kaggle/input")

print(f"Found code snapshot at: {code_input}")
work_dir = Path("/kaggle/working/code")

if not work_dir.exists():
    shutil.copytree(code_input, work_dir)

commit_file = work_dir / "COMMIT.txt"
if commit_file.exists():
    print("Code Snapshot Commit:", commit_file.read_text().strip())

os.chdir(work_dir)
if str(work_dir / "src") not in sys.path:
    sys.path.insert(0, str(work_dir / "src"))
print("Current Working Directory:", os.getcwd())
""")

    add_code("""# Install required dependencies
!pip install --quiet segmentation-models-pytorch
""")

    add_code("""# Verify GPU and Dataset
import torch
from pathlib import Path

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    !nvidia-smi

data_root = None
data_candidates = [
    Path("/kaggle/input/fassd-fetal-abdomen-v2"),
    Path("/kaggle/input/datasets/prantakumerpandit390/fassd-fetal-abdomen-v2"),
]
for c in data_candidates:
    if c.exists() and (c / "meta.csv").exists():
        data_root = c
        break

if data_root is None and Path("/kaggle/input").exists():
    for p in Path("/kaggle/input").rglob("meta.csv"):
        if (p.parent / "images").is_dir() and (p.parent / "masks").is_dir():
            data_root = p.parent
            break

if data_root is None:
    raise FileNotFoundError("Could not find fassd-fetal-abdomen-v2 in /kaggle/input")

print("Dataset directory:", data_root)
n_imgs = len(list((data_root / "images").glob("*.png")))
n_masks = len(list((data_root / "masks").glob("*.png")))
meta_exists = (data_root / "meta.csv").exists()
print(f"Dataset images: {n_imgs}, masks: {n_masks}, meta.csv: {meta_exists}")
""")

    # Execution loop
    add_markdown("## 3. Training Execution")
    add_code("""# Execute configured runs sequentially
import subprocess
import sys

for run_cfg in RUNS:
    mode = run_cfg["mode"]
    seed = run_cfg["seed"]
    out_dir = f"/kaggle/working/fassd_{mode}_seed{seed}"
    print(f"\\n{'='*70}\\nSTARTING RUN: mode={mode}, seed={seed}\\nOutput: {out_dir}\\n{'='*70}")

    cmd = [
        sys.executable,
        "scripts/35_fassd_train.py",
        "--data-root",
        str(data_root),
        "--split-csv",
        str(work_dir / "docs/results/fassd/split_v2.csv"),
        "--output-dir",
        out_dir,
        "--input",
        mode,
        "--encoder",
        "efficientnet-b4",
        "--encoder-weights",
        "imagenet",
        "--epochs",
        "60",
        "--batch-size",
        "8",
        "--lr",
        "3e-4",
        "--seed",
        str(seed),
    ]

    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise RuntimeError(f"Run failed with return code {ret.returncode} for mode={mode}, seed={seed}")
""")

    # Summary
    add_markdown("## 4. Results Consolidation")
    add_code("""# Load summaries and generate comparison table
import json
from pathlib import Path
import pandas as pd

rows = []
for run_cfg in RUNS:
    mode = run_cfg["mode"]
    seed = run_cfg["seed"]
    path_str = f"/kaggle/working/fassd_{mode}_seed{seed}/train_summary.json"
    p = Path(path_str)
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        tr = data["test_results"]
        ci = tr["patient_bootstrap_ci"]
        row = {
            "mode": mode,
            "seed": seed,
            "best_epoch": data["best_epoch"],
            "val_macro_dice": round(data["best_val_macro_dice"], 4),
            "test_macro_dice": round(tr["macro_mean_dice"], 4),
            "test_macro_iou": round(tr["macro_mean_iou"], 4),
            "test_macro_hd95": round(tr["macro_mean_hd95"], 2),
            "test_ci_95": f"[{ci['ci_lower_95']:.4f}, {ci['ci_upper_95']:.4f}]",
        }
        for sname in ("artery", "liver", "stomach", "vein"):
            row[f"{sname}_dice"] = round(tr["by_structure"][sname]["dice_mean"], 4)
            row[f"{sname}_hd95"] = round(tr["by_structure"][sname]["hd95_mean"], 2)
        rows.append(row)

if rows:
    df_results = pd.DataFrame(rows)
    print("=== Consolidated Test Results ===")
    print(df_results.to_string(index=False))
    df_results.to_csv("/kaggle/working/consolidated_fassd_results.csv", index=False)
    print("\\nSaved consolidated table to /kaggle/working/consolidated_fassd_results.csv")
else:
    print("No summary files found to consolidate.")
""")

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    with open(OUTPUT_NOTEBOOK, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated Kaggle notebook at: {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
