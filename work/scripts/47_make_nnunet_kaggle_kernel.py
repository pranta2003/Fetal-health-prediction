"""scripts/47_make_nnunet_kaggle_kernel.py

Generates Kaggle training notebook and metadata config for HC18 nnU-Net v2 (2D, Fold 0, 250 epochs).
Target: notebooks/kaggle/hc18_nnunet/hc18-nnunet-f0-250ep
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "notebooks" / "kaggle" / "hc18_nnunet" / "hc18-nnunet-f0-250ep"


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


def make_code_cell(source: str, cell_id: str | None = None) -> dict:
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }
    if cell_id:
        cell["id"] = cell_id
    return cell


def make_markdown_cell(source: str, cell_id: str | None = None) -> dict:
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }
    if cell_id:
        cell["id"] = cell_id
    return cell


def build_kernel_files() -> tuple[dict, dict]:
    slug = "hc18-nnunet-f0-250ep"
    metadata = {
        "id": f"prantakumerpandit390/{slug}",
        "title": slug,
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [
            "prantakumerpandit390/dataset501-hc18",
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }

    cells = []

    # Cell 1: Intro Markdown
    cells.append(
        make_markdown_cell(
            """# HC18 nnU-Net v2 Training (Fold 0, 250 Epochs)

- Dataset: `Dataset501_HC18` (packaged 999 HC18 training cases)
- Split: Fixed patient-grouped 5-fold split (`splits_final.json`, Fold 0: 793 train / 206 val)
- Configuration: 2D (`nnUNetPlans__2d`)
- Trainer: `nnUNetTrainer_250epochs`
- Output: Preserves validation masks, `summary.json`, `progress.png`, and execution metrics.
""",
            cell_id="cell_1_intro",
        )
    )

    # Cell 2: System & GPU check, installation
    cells.append(
        make_code_cell(
            """!nvidia-smi
!pip install --upgrade pip
!pip install nnunetv2
""",
            cell_id="cell_2_install",
        )
    )

    # Cell 3: nnU-Net Environment Setup
    cells.append(
        make_code_cell(
            """import os
import shutil
from pathlib import Path

# Setup nnU-Net v2 directory paths and environment variables
os.environ["nnUNet_raw"] = "/kaggle/working/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/kaggle/working/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "/kaggle/working/nnUNet_results"

raw_dir = Path("/kaggle/working/nnUNet_raw")
prep_dir = Path("/kaggle/working/nnUNet_preprocessed")
res_dir = Path("/kaggle/working/nnUNet_results")
out_dir = Path("/kaggle/working/out")

raw_dir.mkdir(parents=True, exist_ok=True)
prep_dir.mkdir(parents=True, exist_ok=True)
res_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)
print("nnU-Net v2 directories created successfully.")
""",
            cell_id="cell_3_env",
        )
    )

    # Cell 4: Unpack dataset
    cells.append(
        make_code_cell(
            """import os
import shutil
import zipfile
from pathlib import Path

dataset_dir = raw_dir / "Dataset501_HC18"

print("Locating dataset in /kaggle/input...")
# Check for dataset.json anywhere in /kaggle/input
dataset_jsons = list(Path("/kaggle/input").rglob("dataset.json"))
if dataset_jsons:
    src_dir = dataset_jsons[0].parent
    print(f"Found dataset directory at: {src_dir}")
    shutil.copytree(src_dir, dataset_dir, dirs_exist_ok=True)
else:
    # Check for zip archive anywhere in /kaggle/input
    zips = list(Path("/kaggle/input").rglob("*.zip"))
    if zips:
        print(f"Extracting {zips[0]} to {raw_dir}...")
        with zipfile.ZipFile(zips[0], "r") as zf:
            zf.extractall(raw_dir)
    else:
        print("Directory listing of /kaggle/input:")
        for root, dirs, files in os.walk("/kaggle/input"):
            print(f"  {root}: dirs={dirs[:5]}, files={files[:5]}")

assert (dataset_dir / "dataset.json").is_file(), f"Missing dataset.json in {dataset_dir}"
assert (dataset_dir / "imagesTr").is_dir(), f"Missing imagesTr in {dataset_dir}"
assert (dataset_dir / "labelsTr").is_dir(), f"Missing labelsTr in {dataset_dir}"
n_images = len(list((dataset_dir / "imagesTr").glob("*.png")))
n_labels = len(list((dataset_dir / "labelsTr").glob("*.png")))
print(f"Dataset unpacked successfully: {n_images} imagesTr, {n_labels} labelsTr.")
assert n_images == 999, f"Expected 999 imagesTr, found {n_images}"
assert n_labels == 999, f"Expected 999 labelsTr, found {n_labels}"
""",
            cell_id="cell_4_unpack",
        )
    )

    # Cell 5: Plan and Preprocess
    cells.append(
        make_code_cell(
            """!nnUNetv2_plan_and_preprocess -d 501 -c 2d --verify_dataset_integrity
""",
            cell_id="cell_5_plan",
        )
    )

    # Cell 6: Copy splits_final.json and Diff Check
    cells.append(
        make_code_cell(
            """import json

src_split = dataset_dir / "splits_final.json"
dst_split = prep_dir / "Dataset501_HC18" / "splits_final.json"

assert src_split.is_file(), f"Source splits_final.json not found in {dataset_dir}"
dst_split.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(src_split, dst_split)
print(f"Copied {src_split} -> {dst_split}")

with open(dst_split, "r") as f:
    splits = json.load(f)

print(f"Total folds in splits_final.json: {len(splits)}")
fold0 = splits[0]
n_train = len(fold0["train"])
n_val = len(fold0["val"])
print(f"Fold 0 Train cases: {n_train}")
print(f"Fold 0 Val cases  : {n_val}")
print(f"Total cases       : {n_train + n_val}")

assert n_train == 793, f"Expected 793 train cases, got {n_train}"
assert n_val == 206, f"Expected 206 val cases, got {n_val}"
overlap = set(fold0["train"]).intersection(set(fold0["val"]))
assert len(overlap) == 0, f"Patient leakage detected: {overlap}"
print("Diff check passed: exactly 793 train / 206 val cases with zero overlap.")
""",
            cell_id="cell_6_splits",
        )
    )

    # Cell 7: Train nnU-Net
    cells.append(
        make_code_cell(
            """import time

start_train = time.time()
print("=" * 70)
print("Starting nnU-Net training: 501 2d 0 -tr nnUNetTrainer_250epochs")
print("=" * 70)

!nnUNetv2_train 501 2d 0 -tr nnUNetTrainer_250epochs

elapsed = time.time() - start_train
print(f"nnU-Net training finished in {elapsed:.1f} s ({elapsed / 60:.1f} min)")
""",
            cell_id="cell_7_train",
        )
    )

    # Cell 8: Preserve validation outputs & create manifest
    cells.append(
        make_code_cell(
            """trainer_dir = res_dir / "Dataset501_HC18" / "nnUNetTrainer_250epochs__nnUNetPlans__2d" / "fold_0"

if trainer_dir.exists():
    print(f"Preserving results from {trainer_dir} to {out_dir}...")
    for pattern in ["summary.json", "progress.png", "training_log_*.txt"]:
        for match in trainer_dir.glob(pattern):
            shutil.copy2(match, out_dir / match.name)
            print(f"  Saved {match.name}")

    val_dir = trainer_dir / "validation"
    if val_dir.exists():
        dst_val = out_dir / "validation_fold0"
        shutil.copytree(val_dir, dst_val, dirs_exist_ok=True)
        n_val_files = len(list(dst_val.glob("*")))
        print(f"  Saved validation folder with {n_val_files} files.")

    manifest = {
        "dataset": "Dataset501_HC18",
        "configuration": "2d",
        "fold": 0,
        "trainer": "nnUNetTrainer_250epochs",
        "train_cases": 793,
        "val_cases": 206,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(out_dir / "nnunet_run_summary.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Summary manifest saved to", out_dir / "nnunet_run_summary.json")
else:
    print(f"WARNING: trainer directory {trainer_dir} does not exist!")
""",
            cell_id="cell_8_preserve",
        )
    )

    nb = make_notebook_dict(cells)
    return metadata, nb


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta, nb = build_kernel_files()

    meta_file = OUTPUT_DIR / "kernel-metadata.json"
    nb_file = OUTPUT_DIR / f"{meta['code_file']}"

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(nb_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Generated {meta_file}")
    print(f"Generated {nb_file}")


if __name__ == "__main__":
    main()
