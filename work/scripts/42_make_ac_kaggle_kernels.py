"""scripts/42_make_ac_kaggle_kernels.py

Generates Kaggle training notebooks and metadata configs for AC Campaign experiments:
1. ac-baseline-s42 (Recipe a: baseline v3_full, 5 folds, seed 42) [PUSHED]
2. ac-baseline-s43 (Seed 43 baseline, 5 folds) [NOT PUSHED]
3. ac-wing-s42     (Recipe b: wing loss, 5 folds, seed 42) [NOT PUSHED]
4. ac-ema-s42      (Recipe c: EMA + 40 epochs / patience 12, 5 folds, seed 42) [NOT PUSHED]
5. ac-fassd-s42    (Recipe e: FASSD U-Net effb4 pretrained encoder, 5 folds, seed 42) [NOT PUSHED]
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "notebooks" / "kaggle" / "ac_campaign"


def make_notebook_dict(cells: list[dict]) -> dict:
    return {
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


def make_code_cell(source: str) -> dict:
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def make_markdown_cell(source: str) -> dict:
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def generate_kernel(
    slug: str,
    title: str,
    extra_train_args: str = "",
    extra_datasets: list[str] | None = None,
    extra_kernels: list[str] | None = None,
) -> Path:
    kernel_dir = OUTPUT_DIR / slug
    kernel_dir.mkdir(parents=True, exist_ok=True)

    nb_filename = f"{slug}.ipynb"
    nb_path = kernel_dir / nb_filename

    cells = []

    # Cell 1: Intro Markdown
    cells.append(make_markdown_cell(f"""# AC Campaign: {title}
Kernel slug: `{slug}`
Part of the AC 5-Fold Cross-Validation Campaign on Kaggle T4 GPU.
"""))

    # Cell 2: GPU and Environment check
    cells.append(make_code_cell("""# Check GPU and environment
import os
import sys
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    !nvidia-smi
"""))

    # Cell 3: Unpack code snapshot
    cells.append(make_code_cell("""# Setup code from fetal-code-snapshot
import os
import shutil
import sys
from pathlib import Path

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

print("Found code snapshot at:", code_input)
work_dir = Path("/kaggle/working/code")
if not work_dir.exists():
    shutil.copytree(code_input, work_dir)

# Unzip nested directories if necessary
for z in work_dir.glob("*.zip"):
    dir_name = z.stem
    if not (work_dir / dir_name).is_dir():
        print(f"Extracting {z.name} -> {dir_name}/")
        shutil.unpack_archive(str(z), str(work_dir / dir_name))

os.chdir(work_dir)
if str(work_dir / "src") not in sys.path:
    sys.path.insert(0, str(work_dir / "src"))

print("Working directory set to:", os.getcwd())
commit_p = work_dir / "COMMIT.txt"
if commit_p.exists():
    print("Code Commit:", commit_p.read_text().strip())
"""))

    # Cell 4: Install packages if needed
    cells.append(make_code_cell("""# Install required dependencies
!pip install --quiet segmentation-models-pytorch timm
"""))

    # Cell 5: Setup AC Data
    cells.append(make_code_cell("""# Setup AC Dataset
import shutil
from pathlib import Path

data_candidate = None
for c in [
    Path("/kaggle/input/ac-fetal-v1"),
    Path("/kaggle/input/datasets/prantakumerpandit390/ac-fetal-v1"),
]:
    if c.exists():
        data_candidate = c
        break

if data_candidate is None and Path("/kaggle/input").exists():
    for p in Path("/kaggle/input").rglob("folds_v1.csv"):
        data_candidate = p.parent
        break

if data_candidate is None:
    raise FileNotFoundError("Could not find ac-fetal-v1 dataset in /kaggle/input")

print("Found AC dataset at:", data_candidate)

# Setup target data directory
data_dir = Path("/kaggle/working/data")
data_dir.mkdir(parents=True, exist_ok=True)
os.environ["FETAL_DATA_DIR"] = str(data_dir)

# Unpack FP and UCL if zipped, or symlink/copy if directory
for src_name in ["FP", "UCL"]:
    zip_p = data_candidate / f"{src_name}.zip"
    folder_p = data_candidate / src_name
    dest_p = data_dir / src_name

    if not dest_p.exists():
        if zip_p.exists():
            print(f"Unpacking {zip_p} -> {dest_p}...")
            shutil.unpack_archive(str(zip_p), str(dest_p))
        elif folder_p.is_dir():
            print(f"Copying {folder_p} -> {dest_p}...")
            shutil.copytree(str(folder_p), str(dest_p))

# Copy folds_v1.csv
folds_src = data_candidate / "folds_v1.csv"
if not (data_dir / "folds_v1.csv").exists() and folds_src.exists():
    shutil.copy2(str(folds_src), str(data_dir / "folds_v1.csv"))

print("Data directory contents:", list(data_dir.glob("*")))
print("FP images count:", len(list((data_dir / "FP" / "images" / "Abdomen").glob("*.png"))))
print("UCL images count:", len(list((data_dir / "UCL" / "images" / "Abdomen").glob("*.*"))))
print("folds_v1.csv exists:", (data_dir / "folds_v1.csv").exists())
"""))

    # Cell 6: Extra setup if FASSD pretrained encoder required
    if "init-encoder" in extra_train_args:
        cells.append(make_code_cell("""# Locate FASSD Pretrained Checkpoint
from pathlib import Path

fassd_ckpt = None
candidates = [
    Path("/kaggle/input/fassd-train-seed42/fassd_gray_seed42/best_model.pt"),
    Path("/kaggle/input/fassd-train-round2/fassd_gray_seed42/best_model.pt"),
]
for c in candidates:
    if c.exists():
        fassd_ckpt = c
        break

if fassd_ckpt is None and Path("/kaggle/input").exists():
    for p in Path("/kaggle/input").rglob("best_model.pt"):
        if "fassd" in str(p).lower():
            fassd_ckpt = p
            break

if fassd_ckpt is None:
    raise FileNotFoundError("Could not find FASSD pretrained checkpoint in /kaggle/input")

print("Found FASSD checkpoint at:", fassd_ckpt)
FASSD_CKPT_PATH = str(fassd_ckpt)
"""))

    # Cell 7: Execute 5-fold training
    run_cmd = f"python scripts/04_train_ac.py --fold all --data-dir /kaggle/working/data --folds-csv /kaggle/working/data/folds_v1.csv --out-dir /kaggle/working/{slug} {extra_train_args}"
    cells.append(make_code_cell(f"""# Run 5-fold AC Cross-Validation
import subprocess

cmd = \"\"\"{run_cmd}\"\"\"
print("Running command:\\n", cmd)
ret = subprocess.run(cmd, shell=True)
if ret.returncode != 0:
    raise RuntimeError(f"Training failed with return code {{ret.returncode}}")
"""))

    # Cell 8: Print and verify summary JSON
    cells.append(make_code_cell(f"""# Display Final OOF Summary
import json
from pathlib import Path
import pandas as pd

summary_p = Path("/kaggle/working/{slug}/oof_summary.json")
if summary_p.exists():
    with open(summary_p, "r") as f:
        summary = json.load(f)
    print("=== Pooled 5-Fold OOF Summary ===")
    print(json.dumps(summary, indent=2))

oof_csv = Path("/kaggle/working/{slug}/oof_predictions_all.csv")
if oof_csv.exists():
    df_oof = pd.read_csv(oof_csv)
    print(f"\\nOOF Predictions CSV: {{oof_csv}} (rows={{len(df_oof)}})")
    print(df_oof.head())
"""))

    # Write notebook
    nb_dict = make_notebook_dict(cells)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=2)

    # Prepare kernel-metadata.json
    dataset_sources = [
        "prantakumerpandit390/ac-fetal-v1",
        "prantakumerpandit390/fetal-code-snapshot",
    ]
    if extra_datasets:
        for d in extra_datasets:
            if d not in dataset_sources:
                dataset_sources.append(d)

    kernel_sources = []
    if extra_kernels:
        for k in extra_kernels:
            if k not in kernel_sources:
                kernel_sources.append(k)

    meta = {
        "id": f"prantakumerpandit390/{slug}",
        "title": slug,
        "code_file": nb_filename,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        "kernel_sources": kernel_sources,
        "model_sources": [],
    }

    meta_path = kernel_dir / "kernel-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Generated kernel '{slug}' at {kernel_dir}")
    return kernel_dir


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        {
            "slug": "ac-baseline-s42",
            "title": "Recipe (a) Unchanged v3_full Baseline (Seed 42)",
            "extra_train_args": "--seed 42 --recipe v3_full",
            "extra_datasets": None,
            "extra_kernels": None,
        },
        {
            "slug": "ac-baseline-s43",
            "title": "Recipe (a) Unchanged v3_full Baseline (Seed 43)",
            "extra_train_args": "--seed 43 --recipe v3_full",
            "extra_datasets": None,
            "extra_kernels": None,
        },
        {
            "slug": "ac-wing-s42",
            "title": "Recipe (b) Wing Loss for Landmark Term (Seed 42)",
            "extra_train_args": "--seed 42 --recipe v3_full --loss wing",
            "extra_datasets": None,
            "extra_kernels": None,
        },
        {
            "slug": "ac-ema-s42",
            "title": "Recipe (c) Model EMA + Longer Schedule 40 ep (Seed 42)",
            "extra_train_args": "--seed 42 --recipe v3_full --ema --epochs 40 --patience 12",
            "extra_datasets": None,
            "extra_kernels": None,
        },
        {
            "slug": "ac-fassd-s42",
            "title": "Recipe (e) FASSD Pretrained U-Net Encoder (Seed 42)",
            "extra_train_args": "--seed 42 --recipe v3_full --init-encoder $FASSD_CKPT_PATH",
            "extra_datasets": None,
            "extra_kernels": ["prantakumerpandit390/fassd-train-seed42"],
        },
    ]

    print("=== Generating AC Campaign Kaggle Kernels ===")
    for c in configs:
        generate_kernel(
            slug=c["slug"],
            title=c["title"],
            extra_train_args=c["extra_train_args"],
            extra_datasets=c["extra_datasets"],
            extra_kernels=c["extra_kernels"],
        )

    print("\nPush command for Kernel 1 (ac-baseline-s42):")
    print(f"kaggle kernels push -p {OUTPUT_DIR / 'ac-baseline-s42'}")
    print("\nPush commands for other kernels (DO NOT RUN NOW):")
    for c in configs[1:]:
        print(f"kaggle kernels push -p {OUTPUT_DIR / c['slug']}")


if __name__ == "__main__":
    main()
