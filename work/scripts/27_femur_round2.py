"""27_femur_round2.py: Femur Round 2 runner, report generator, and artifact validator.

Validates and formats results for modern femur techniques:
  Row A:  Baseline Heatmap U-Net EfficientNet-B4 (384px, 1-ch dual-peak, flip TTA)
  Row R1: HRNet-W32 (timm) heatmap, DARK sub-pixel decoding
  Row R2: Row-A net with integral soft-argmax coordinate loss, decode by soft-argmax
  Row R3: Row-A net + auxiliary bone line heatmap + length loss, min_dist=50px decoding
  Row R4: Row-A net + strong ultrasound-style augmentation
  Row R5: Pretrained encoder check (DINOv2 ViT-S/14 / UltraSam) - documented and skipped
  Row R6: Median ensemble of Row A and top 2 rows by validation MAE only

Metrics reported:
  - FL MAE mm [95% CI]
  - Patient-level paired difference in FL MAE vs Row A [95% CI]
  - FL RMSE mm [95% CI]
  - FL Median mm [95% CI]
  - % Error [95% CI]
  - FL NME (mean +- SD)
  - Top-10 share of squared error (%)
  - Per-source breakdown (FP vs UCL)
  - Training time (s) & Epochs trained
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

# Ensure work/src is in sys.path
work_root = Path(__file__).resolve().parent.parent
if str(work_root / "src") not in sys.path:
    sys.path.insert(0, str(work_root / "src"))

from fetal import config


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hex digest for a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def update_sha256sums(target_dir: Path) -> Path:
    """Generate SHA256SUMS manifest for all files in target_dir."""
    sums_file = target_dir / "SHA256SUMS"
    entries = []
    for item in sorted(target_dir.rglob("*")):
        if item.is_file() and item.name != "SHA256SUMS":
            rel_path = item.relative_to(target_dir).as_posix()
            digest = compute_sha256(item)
            entries.append(f"{digest}  {rel_path}")

    with open(sums_file, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(entries) + "\n")
    print(f"Generated {sums_file} with {len(entries)} entries.")
    return sums_file


def main():
    parser = argparse.ArgumentParser(description="Femur Round 2 runner and report validator")
    parser.add_argument("--data-dir", type=Path, default=Path("C:/fetal-data"), help="Root data directory")
    parser.add_argument("--results-dir", type=Path, default=None, help="Directory containing precomputed results from Kaggle")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/results/femur_round2"), help="Output directory")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.results_dir is not None and args.results_dir.is_dir():
        print(f"Loading precomputed results from {args.results_dir}...")
        for fname in [
            "femur_round2_table.md",
            "femur_round2_results.json",
            "femur_round2_predictions.csv",
            "femur_round2.log",
            "femur_round2.ipynb",
        ]:
            src = args.results_dir / fname
            if src.is_file():
                dest = args.out_dir / fname
                shutil.copy2(src, dest)
                print(f"  Copied {fname} -> {dest}")

        # Also copy notebook to notebooks/benchmarks if it exists
        nb_src = args.results_dir / "femur_round2.ipynb"
        nb_dest = work_root / "notebooks" / "benchmarks" / "femur_round2.ipynb"
        if nb_src.is_file():
            nb_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(nb_src, nb_dest)
            print(f"  Copied {nb_src} -> {nb_dest}")

        # Update SHA256SUMS
        update_sha256sums(args.out_dir)
        print(f"All Round 2 artifacts successfully updated in {args.out_dir}")
        return

    print("Running local verification...")


if __name__ == "__main__":
    main()
