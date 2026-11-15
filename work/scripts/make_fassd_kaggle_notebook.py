"""make_fassd_kaggle_notebook.py: Generate notebooks/kaggle/fassd_train_kernel.ipynb.

Creates the training notebook for Kaggle (T4 GPU), orchestrating experiments across:
- seeds: 42, 43, 44
- input modes: rgb and gray
- model: smp.Unet (efficientnet-b4, ImageNet weights)
- outputs saved to /kaggle/working
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
- Target size: $384 \\times 512$ (native aspect-ratio of $768 \\times 1024$)
- Loss: Combined BCE + Soft-Dice multilabel loss (sigmoid activation)
- Optimization: AdamW, cosine annealing, AMP, Model EMA
- Runs: 3 seeds (42, 43, 44) $\\times$ 2 input modes (`rgb` vs `gray`)
- Metrics: Native resolution ($768 \\times 1024$) Dice, IoU, surface HD95, patient bootstrap 95% CI
""")

    # Setup
    add_markdown("## 1. Setup & Environment")
    add_code("""# Clone repository using Kaggle Secret GITHUB_TOKEN
from kaggle_secrets import UserSecretsClient
import os
import subprocess

user_secrets = UserSecretsClient()
github_pat = user_secrets.get_secret("GITHUB_TOKEN")

repo_url = f"https://{github_pat}@github.com/pranta2003/fetal-health-multimodal.git"
repo_dir = "/kaggle/working/repo"

if not os.path.exists(repo_dir):
    subprocess.run(["git", "clone", "--depth", "1", repo_url, repo_dir], check=True)
os.chdir(os.path.join(repo_dir, "work"))
print("Current working directory:", os.getcwd())
""")

    add_code("""# Install required dependencies
!pip install --quiet segmentation-models-pytorch opencv-python-headless
""")

    add_code("""# Verify GPU and Dataset
import torch
from pathlib import Path

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))

data_root = Path("/kaggle/input/fassd-fetal-abdomen-v2")
print("Dataset directory exists:", data_root.exists())
if data_root.exists():
    n_imgs = len(list((data_root / "images").glob("*.png")))
    n_masks = len(list((data_root / "masks").glob("*.png")))
    print(f"Dataset images: {n_imgs}, masks: {n_masks}")
""")

    # RGB experiments
    add_markdown("## 2. Experiments: RGB Input Mode (Seeds 42, 43, 44)")
    add_code("""# --- RGB Mode: Seed 42 ---
!python scripts/35_fassd_train.py \\
    --data-root /kaggle/input/fassd-fetal-abdomen-v2 \\
    --output-dir /kaggle/working/fassd_rgb_seed42 \\
    --input rgb \\
    --encoder efficientnet-b4 \\
    --encoder-weights imagenet \\
    --epochs 60 \\
    --batch-size 8 \\
    --lr 3e-4 \\
    --seed 42
""")

    add_code("""# --- RGB Mode: Seed 43 ---
!python scripts/35_fassd_train.py \\
    --data-root /kaggle/input/fassd-fetal-abdomen-v2 \\
    --output-dir /kaggle/working/fassd_rgb_seed43 \\
    --input rgb \\
    --encoder efficientnet-b4 \\
    --encoder-weights imagenet \\
    --epochs 60 \\
    --batch-size 8 \\
    --lr 3e-4 \\
    --seed 43
""")

    add_code("""# --- RGB Mode: Seed 44 ---
!python scripts/35_fassd_train.py \\
    --data-root /kaggle/input/fassd-fetal-abdomen-v2 \\
    --output-dir /kaggle/working/fassd_rgb_seed44 \\
    --input rgb \\
    --encoder efficientnet-b4 \\
    --encoder-weights imagenet \\
    --epochs 60 \\
    --batch-size 8 \\
    --lr 3e-4 \\
    --seed 44
""")

    # Gray experiments
    add_markdown("## 3. Experiments: Grayscale Input Mode (Seeds 42, 43, 44)")
    add_code("""# --- Gray Mode: Seed 42 ---
!python scripts/35_fassd_train.py \\
    --data-root /kaggle/input/fassd-fetal-abdomen-v2 \\
    --output-dir /kaggle/working/fassd_gray_seed42 \\
    --input gray \\
    --encoder efficientnet-b4 \\
    --encoder-weights imagenet \\
    --epochs 60 \\
    --batch-size 8 \\
    --lr 3e-4 \\
    --seed 42
""")

    add_code("""# --- Gray Mode: Seed 43 ---
!python scripts/35_fassd_train.py \\
    --data-root /kaggle/input/fassd-fetal-abdomen-v2 \\
    --output-dir /kaggle/working/fassd_gray_seed43 \\
    --input gray \\
    --encoder efficientnet-b4 \\
    --encoder-weights imagenet \\
    --epochs 60 \\
    --batch-size 8 \\
    --lr 3e-4 \\
    --seed 43
""")

    add_code("""# --- Gray Mode: Seed 44 ---
!python scripts/35_fassd_train.py \\
    --data-root /kaggle/input/fassd-fetal-abdomen-v2 \\
    --output-dir /kaggle/working/fassd_gray_seed44 \\
    --input gray \\
    --encoder efficientnet-b4 \\
    --encoder-weights imagenet \\
    --epochs 60 \\
    --batch-size 8 \\
    --lr 3e-4 \\
    --seed 44
""")

    # Summary
    add_markdown("## 4. Results Consolidation")
    add_code("""# Load summaries and generate comparison table
import json
from pathlib import Path
import pandas as pd

runs = [
    ("rgb", 42, "/kaggle/working/fassd_rgb_seed42/train_summary.json"),
    ("rgb", 43, "/kaggle/working/fassd_rgb_seed43/train_summary.json"),
    ("rgb", 44, "/kaggle/working/fassd_rgb_seed44/train_summary.json"),
    ("gray", 42, "/kaggle/working/fassd_gray_seed42/train_summary.json"),
    ("gray", 43, "/kaggle/working/fassd_gray_seed43/train_summary.json"),
    ("gray", 44, "/kaggle/working/fassd_gray_seed44/train_summary.json"),
]

rows = []
for mode, seed, path_str in runs:
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
