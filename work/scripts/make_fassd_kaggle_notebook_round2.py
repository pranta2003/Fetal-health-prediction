"""make_fassd_kaggle_notebook_round2.py: Generate notebooks/kaggle/fassd_train_kernel_round2.ipynb.

Creates the Round 2 training notebook for Kaggle (T4 GPU), orchestrating experiments across:
- Code snapshot mounted from private dataset: prantakumerpandit390/fetal-code-snapshot
- Data mounted from private dataset: prantakumerpandit390/fassd-fetal-abdomen-v2
- 4 Variants (all gray, seed 42):
  (i) unet + efficientnet-b4
  (ii) unetplusplus + efficientnet-b4
  (iii) unet + resnet34
  (iv) fpn + efficientnet-b4
- Selection on VAL macro Dice ONLY; evaluate TEST for ALL variants only at the end.
- Reports per-run execution time and speedup vs Run 1.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "notebooks" / "kaggle"
OUTPUT_NOTEBOOK = OUTPUT_DIR / "fassd_train_kernel_round2.ipynb"


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
    add_markdown("""# FASSD Round 2 Architectural Exploration & Speedup Study
**Fetal Abdominal Structures Segmentation (artery, liver, stomach, vein)**
- Variants:
  1. `unet` + `efficientnet-b4`
  2. `unetplusplus` + `efficientnet-b4`
  3. `unet` + `resnet34`
  4. `fpn` + `efficientnet-b4`
- Settings: Grayscale, seed 42, ImageNet pretrained backbones
- Optimizations: In-memory RAM caching (uint8), batch GPU augmentations, `channels_last`, AMP, `num_workers=2`
- Early Stopping: Patience 10 on val macro Dice, max 40 epochs
- Selection Protocol: Best architecture selected on **VAL macro Dice ONLY**; TEST set evaluated for all 4 variants at the end.
""")

    # Run configuration
    add_markdown("## 1. Run Configuration")
    add_code("""# Define the 4 architectural variants for Round 2
VARIANTS = [
    {
        "name": "unet_effb4",
        "arch": "unet",
        "encoder": "efficientnet-b4",
        "input_mode": "gray",
        "seed": 42,
    },
    {
        "name": "unetplusplus_effb4",
        "arch": "unetplusplus",
        "encoder": "efficientnet-b4",
        "input_mode": "gray",
        "seed": 42,
    },
    {
        "name": "unet_resnet34",
        "arch": "unet",
        "encoder": "resnet34",
        "input_mode": "gray",
        "seed": 42,
    },
    {
        "name": "fpn_effb4",
        "arch": "fpn",
        "encoder": "efficientnet-b4",
        "input_mode": "gray",
        "seed": 42,
    },
]

print(f"Configured {len(VARIANTS)} variants for Round 2 exploration:")
for v in VARIANTS:
    print(f"  - {v['name']}: arch={v['arch']}, encoder={v['encoder']}, mode={v['input_mode']}, seed={v['seed']}")
""")

    # Environment setup
    add_markdown("## 2. Environment & Codebase Setup")
    add_code("""# Mount codebase snapshot from private dataset
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
!pip install --quiet segmentation-models-pytorch timm
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
    add_markdown("## 3. Training Execution with Speed Profile")
    add_code("""# Execute training for each variant and profile execution time
import subprocess
import sys
import time
import pandas as pd

timings = []

for v in VARIANTS:
    vname = v["name"]
    arch = v["arch"]
    enc = v["encoder"]
    mode = v["input_mode"]
    seed = v["seed"]
    out_dir = f"/kaggle/working/fassd_run2/{vname}"

    print(f"\\n{'='*75}")
    print(f"STARTING VARIANT: {vname} (arch={arch}, encoder={enc}, mode={mode}, seed={seed})")
    print(f"Output: {out_dir}")
    print(f"{'='*75}")

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
        "--arch",
        arch,
        "--encoder",
        enc,
        "--encoder-weights",
        "imagenet",
        "--epochs",
        "40",
        "--patience",
        "10",
        "--batch-size",
        "8",
        "--lr",
        "3e-4",
        "--wd",
        "0.01",
        "--seed",
        str(seed),
        "--cache-in-memory",
        "--batch-gpu-aug",
        "--channels-last",
    ]

    t0 = time.time()
    ret = subprocess.run(cmd)
    t1 = time.time()
    duration = t1 - t0

    if ret.returncode != 0:
        raise RuntimeError(f"Run failed with return code {ret.returncode} for variant {vname}")

    # Read trained epoch count
    log_csv = Path(out_dir) / "train_log.csv"
    epochs_trained = len(pd.read_csv(log_csv)) if log_csv.exists() else 0
    sec_per_epoch = duration / max(1, epochs_trained)

    # In Run 1, 60 epochs took 9622s = ~160.4 s/epoch
    run1_sec_per_epoch = 160.37
    speedup = run1_sec_per_epoch / max(0.1, sec_per_epoch)

    print(f"Variant {vname} finished in {duration:.1f}s ({epochs_trained} epochs, {sec_per_epoch:.2f} s/epoch).")
    print(f"Speedup vs Run 1: {speedup:.1f}x faster!")

    timings.append({
        "variant": vname,
        "arch": arch,
        "encoder": enc,
        "duration_s": duration,
        "epochs_trained": epochs_trained,
        "sec_per_epoch": sec_per_epoch,
        "speedup_vs_run1": speedup,
    })

df_timings = pd.DataFrame(timings)
print("\\n=== Timing & Speedup Summary ===")
print(df_timings.to_string(index=False))
df_timings.to_csv("/kaggle/working/fassd_run2_timings.csv", index=False)
""")

    # Validation Selection & Test Consolidation
    add_markdown("## 4. Model Selection (VAL Macro Dice ONLY) & Final TEST Evaluation")
    add_code("""# Select winning model strictly on VAL macro Dice, then report TEST metrics for all variants
import json
from pathlib import Path
import pandas as pd

records = []
for v in VARIANTS:
    vname = v["name"]
    sum_path = Path(f"/kaggle/working/fassd_run2/{vname}/train_summary.json")
    if not sum_path.exists():
        continue
    with open(sum_path) as fp:
        data = json.load(fp)

    val_dice = data["best_val_macro_dice"]
    best_ep = data["best_epoch"]
    tr = data.get("test_results", {})
    ci = tr.get("patient_bootstrap_ci", {})

    rec = {
        "variant": vname,
        "arch": v["arch"],
        "encoder": v["encoder"],
        "best_epoch": best_ep,
        "val_macro_dice": round(val_dice, 4),
        "test_macro_dice": round(tr.get("macro_mean_dice", 0.0), 4),
        "test_macro_iou": round(tr.get("macro_mean_iou", 0.0), 4),
        "test_macro_hd95": round(tr.get("macro_mean_hd95", 0.0), 2),
        "test_ci_95": f"[{ci.get('ci_lower_95', 0.0):.4f}, {ci.get('ci_upper_95', 0.0):.4f}]",
    }
    for sname in ("artery", "liver", "stomach", "vein"):
        s_data = tr.get("by_structure", {}).get(sname, {})
        rec[f"{sname}_dice"] = round(s_data.get("dice_mean", 0.0), 4)
        rec[f"{sname}_hd95"] = round(s_data.get("hd95_mean", 0.0), 2)
    records.append(rec)

df_all = pd.DataFrame(records)

# 1. Validation Ranking (Strictly on VAL)
print("=" * 80)
print("MODEL SELECTION: RANKING BASED ON VALIDATION MACRO DICE ONLY")
print("=" * 80)
df_val_ranking = df_all.sort_values(by="val_macro_dice", ascending=False).reset_index(drop=True)
print(df_val_ranking[["variant", "arch", "encoder", "best_epoch", "val_macro_dice"]].to_string(index=True))

winning_variant = df_val_ranking.iloc[0]["variant"]
winning_val_dice = df_val_ranking.iloc[0]["val_macro_dice"]
print(f"\\n>>> WINNING MODEL SELECTED (on VAL): {winning_variant} (VAL Macro Dice: {winning_val_dice:.4f}) <<<\\n")

# 2. Consolidated TEST Results
print("=" * 80)
print("FINAL TEST EVALUATION (ALL 4 VARIANTS)")
print("=" * 80)
print(df_all[["variant", "arch", "encoder", "val_macro_dice", "test_macro_dice", "test_macro_iou", "test_macro_hd95", "test_ci_95"]].to_string(index=False))

# Save consolidated results
out_csv = "/kaggle/working/consolidated_round2_results.csv"
df_all.to_csv(out_csv, index=False)
print(f"\\nSaved consolidated results to: {out_csv}")
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
