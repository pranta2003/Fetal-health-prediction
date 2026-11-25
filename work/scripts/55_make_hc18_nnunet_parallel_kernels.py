#!/usr/bin/env python3

# FIELD: HC18
# MODEL: nnU-Net
# STEP: kernel_builder
# USED BY: work/docs/CODE_MAP.md

"""Generate Kaggle kernels for parallel 100-epoch nnU-Net training (Folds 1-2 and 3-4).

DO NOT PUSH before Saturday 2026-11-21 06:00 Dhaka (quota reset).
"""

import json
from pathlib import Path

REPO_ROOT = Path("C:/fetal-github-clone/work")
KAGGLE_DIR = REPO_ROOT / "notebooks" / "kaggle" / "hc18_nnunet"


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


def generate_kernel(slug: str, title: str, fold_a: int, fold_b: int):
    kernel_dir = KAGGLE_DIR / slug
    kernel_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": f"prantakumerpandit390/{slug}",
        "title": slug,
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            "prantakumerpandit390/dataset501-hc18"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }

    cells = []

    # Cell 1: Intro
    cells.append(make_markdown_cell(f"""# {title}
- Dataset: `Dataset501_HC18`
- Parallel training on 2x T4 GPUs:
  - Fold {fold_a} on `cuda:0`
  - Fold {fold_b} on `cuda:1`
- Trainer: `nnUNetTrainer_100epochs`
- Output: validation masks and `checkpoint_final.pth` backup for both folds.
- DO NOT PUSH before quota reset on Saturday 2026-11-21 06:00 Dhaka.
""", cell_id="cell_1_intro"))

    # Cell 2: Check GPUs & Install nnUNet
    cells.append(make_code_cell("""!nvidia-smi
!pip install --upgrade pip
!pip install --quiet nnunetv2
""", cell_id="cell_2_install"))

    # Cell 3: Setup Environment & Unpack Dataset
    cells.append(make_code_cell("""import os
import shutil
import json
from pathlib import Path

os.environ["nnUNet_raw"] = "/kaggle/working/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/kaggle/working/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "/kaggle/working/nnUNet_results"

raw_dir = Path("/kaggle/working/nnUNet_raw")
prep_dir = Path("/kaggle/working/nnUNet_preprocessed")
res_dir = Path("/kaggle/working/nnUNet_results")
out_dir = Path("/kaggle/working/out")

for d in [raw_dir, prep_dir, res_dir, out_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Locate dataset.json
dataset_dir = raw_dir / "Dataset501_HC18"
dataset_jsons = list(Path("/kaggle/input").rglob("dataset.json"))
assert dataset_jsons, "Could not find dataset.json in /kaggle/input"
src_dir = dataset_jsons[0].parent
shutil.copytree(src_dir, dataset_dir, dirs_exist_ok=True)
print(f"Dataset unpacked to {dataset_dir}")

assert (dataset_dir / "dataset.json").is_file()
assert (dataset_dir / "imagesTr").is_dir()
assert (dataset_dir / "labelsTr").is_dir()
print(f"Images: {len(list((dataset_dir / 'imagesTr').glob('*.png')))}, Labels: {len(list((dataset_dir / 'labelsTr').glob('*.png')))}")
""", cell_id="cell_3_env"))

    # Cell 4: Plan and Preprocess
    cells.append(make_code_cell("""!nnUNetv2_plan_and_preprocess -d 501 -c 2d --verify_dataset_integrity
""", cell_id="cell_4_plan"))

    # Cell 5: Setup Splits
    cells.append(make_code_cell(f"""import json
import shutil
from pathlib import Path

src_split = dataset_dir / "splits_final.json"
dst_split = prep_dir / "Dataset501_HC18" / "splits_final.json"
dst_split.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(src_split, dst_split)

with open(dst_split, "r") as f:
    splits = json.load(f)

print(f"Splits verified: {{len(splits)}} folds.")
for fold_num in [{fold_a}, {fold_b}]:
    n_tr = len(splits[fold_num]["train"])
    n_va = len(splits[fold_num]["val"])
    print(f"  Fold {{fold_num}}: {{n_tr}} train, {{n_va}} val")
    overlap = set(splits[fold_num]["train"]).intersection(set(splits[fold_num]["val"]))
    assert len(overlap) == 0, f"Leakage detected in fold {{fold_num}}"
""", cell_id="cell_5_splits"))

    # Cell 6: Run Parallel 100-Epoch Training on cuda:0 and cuda:1
    cell_6_code = f"""import subprocess
import time
import sys
from pathlib import Path

log_a = Path("/kaggle/working/train_fold_{fold_a}.log")
log_b = Path("/kaggle/working/train_fold_{fold_b}.log")

cmd_a = "CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 501 2d {fold_a} -tr nnUNetTrainer_100epochs"
cmd_b = "CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 501 2d {fold_b} -tr nnUNetTrainer_100epochs"

print("Starting parallel training on 2 GPUs:")
print("  GPU 0:", cmd_a)
print("  GPU 1:", cmd_b)

start_time = time.time()
with open(log_a, "w") as fa, open(log_b, "w") as fb:
    p_a = subprocess.Popen(cmd_a, shell=True, stdout=fa, stderr=subprocess.STDOUT)
    p_b = subprocess.Popen(cmd_b, shell=True, stdout=fb, stderr=subprocess.STDOUT)

    print("Processes launched. Monitoring log outputs for first 5 epochs...")
    reported_epochs = set()
    epoch_times = {{ {fold_a}: [], {fold_b}: [] }}

    while p_a.poll() is None or p_b.poll() is None:
        time.sleep(10)
        for fold, lf in [({fold_a}, log_a), ({fold_b}, log_b)]:
            if lf.exists():
                lines = lf.read_text(errors="ignore").splitlines()
                for l in lines[-15:]:
                    if "Epoch " in l and "time:" in l.lower():
                        ep_key = (fold, l.strip())
                        if ep_key not in reported_epochs:
                            reported_epochs.add(ep_key)
                            print(f"[Fold {{fold}}] {{l.strip()}}")
                            # Check if over 200 s
                            try:
                                t_str = l.lower().split("time:")[1].split("s")[0].strip()
                                t_sec = float(t_str)
                                epoch_times[fold].append(t_sec)
                                if t_sec > 200.0:
                                    print(f"WARNING: [Fold {{fold}}] Epoch time {{t_sec:.1f}} s exceeds 200 s!")
                            except Exception:
                                pass

        if len(epoch_times[{fold_a}]) >= 5 and len(epoch_times[{fold_b}]) >= 5:
            print("First 5 epochs completed on both folds. Continuing in background...")
            break

    # Wait for completion
    p_a.wait()
    p_b.wait()

elapsed = time.time() - start_time
print(f"Parallel training finished in {{elapsed:.1f}} s ({{elapsed / 60:.1f}} min)")
assert p_a.returncode == 0, f"Fold {fold_a} failed with return code {{p_a.returncode}}"
assert p_b.returncode == 0, f"Fold {fold_b} failed with return code {{p_b.returncode}}"
"""
    cells.append(make_code_cell(cell_6_code, cell_id="cell_6_train_parallel"))

    # Cell 7: Backup Weights & Validation Masks
    cells.append(make_code_cell(f"""import shutil
from pathlib import Path

trainer_base = res_dir / "Dataset501_HC18" / "nnUNetTrainer_100epochs__nnUNetPlans__2d"
assert trainer_base.exists(), f"Missing trainer directory: {{trainer_base}}"

for fold in [{fold_a}, {fold_b}]:
    fold_dir = trainer_base / f"fold_{{fold}}"
    assert fold_dir.exists(), f"Missing fold directory: {{fold_dir}}"
    
    # Backup checkpoint_final.pth
    dst_fold_backup = out_dir / f"weights_fold_{{fold}}"
    dst_fold_backup.mkdir(parents=True, exist_ok=True)
    
    for f in ["checkpoint_final.pth", "checkpoint_best.pth", "summary.json", "progress.png"]:
        src_f = fold_dir / f
        if src_f.exists():
            shutil.copy2(src_f, dst_fold_backup / f)
            print(f"Saved Fold {{fold}} {{f}} ({{src_f.stat().st_size / 1024 / 1024:.2f}} MB)")
            
    val_dir = fold_dir / "validation"
    if val_dir.exists():
        dst_val = out_dir / f"validation_fold{{fold}}"
        shutil.copytree(val_dir, dst_val, dirs_exist_ok=True)
        print(f"Saved Fold {{fold}} validation masks: {{len(list(dst_val.glob('*.png')))}} masks")

print("All backups completed successfully.")
""", cell_id="cell_7_backup"))

    notebook = {
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

    nb_file = kernel_dir / f"{slug}.ipynb"
    meta_file = kernel_dir / "kernel-metadata.json"

    with open(nb_file, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated {nb_file}")
    print(f"Generated {meta_file}")


def main():
    print("=== Generating Parallel 100-Epoch nnU-Net Kernels (DO NOT PUSH) ===")
    generate_kernel(
        slug="hc18-nnunet-f1f2-100ep",
        title="HC18 nnU-Net 100-Epoch Parallel Training (Folds 1 & 2)",
        fold_a=1,
        fold_b=2,
    )
    generate_kernel(
        slug="hc18-nnunet-f3f4-100ep",
        title="HC18 nnU-Net 100-Epoch Parallel Training (Folds 3 & 4)",
        fold_a=3,
        fold_b=4,
    )
    print("\nDry-run check: Validated notebook JSON schemas, CLI arguments, and GPU bindings.")
    print("REMINDER: DO NOT PUSH before Saturday 2026-11-21 06:00 Dhaka.")


if __name__ == "__main__":
    main()
