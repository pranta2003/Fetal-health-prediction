#!/usr/bin/env python3

# FIELD: HC18
# MODEL: nnU-Net
# STEP: data_prep
# USED BY: work/docs/CODE_MAP.md

"""45_hc18_to_nnunet.py: Convert HC18 training data to nnU-Net v2 format.

Prepares Dataset501_HC18 in standard nnU-Net v2 2D directory structure:
  Dataset501_HC18/
    ├── dataset.json
    ├── splits_final.json (derived from our patient-grouped 5-fold split)
    ├── imagesTr/
    │     ├── {case_id}_0000.png
    └── labelsTr/
          └── {case_id}.png

Rules:
  - 999 training images and solid filled-ellipse masks ({0, 1} uint8).
  - splits_final.json matches docs/results/hc18/oof_predictions.csv exactly.
  - Assert zero patient leakage across train/val for all 5 folds.
  - Package dataset to zip archive (reports size in MB, NO upload).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
src_dir = REPO_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fetal.config import HC18_DIR


def extract_filled_mask(raw_annotation: np.ndarray) -> np.ndarray:
    """Fill HC18 boundary contour into a solid binary mask with label {0, 1}."""
    contours, _ = cv2.findContours(
        raw_annotation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    filled = np.zeros_like(raw_annotation, dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled, [largest], -1, 1, thickness=cv2.FILLED)
    return filled


def convert_hc18_to_nnunet(
    hc18_dir: Path,
    oof_csv: Path,
    output_dir: Path,
    package_zip: Path | None = None,
    max_cases: int | None = None,
) -> dict:
    """Convert HC18 training dataset to nnU-Net v2 format and write splits_final.json."""
    hc18_dir = Path(hc18_dir).resolve()
    oof_csv = Path(oof_csv).resolve()
    output_dir = Path(output_dir).resolve()

    if not oof_csv.is_file():
        raise FileNotFoundError(f"Missing OOF predictions CSV: {oof_csv}")

    df_oof = pd.read_csv(oof_csv)
    if max_cases is not None:
        df_oof = df_oof.iloc[:max_cases].copy()

    images_tr_dir = output_dir / "imagesTr"
    labels_tr_dir = output_dir / "labelsTr"
    images_tr_dir.mkdir(parents=True, exist_ok=True)
    labels_tr_dir.mkdir(parents=True, exist_ok=True)

    training_set_dir = hc18_dir / "training_set"
    if not training_set_dir.is_dir():
        raise FileNotFoundError(f"HC18 training_set folder not found: {training_set_dir}")

    print(f"Converting {len(df_oof)} HC18 images to nnU-Net v2 format in {output_dir}...")
    converted_cases = []

    for idx, row in df_oof.iterrows():
        img_filename = str(row["image"])
        case_id = Path(img_filename).stem
        converted_cases.append(case_id)

        # 1. Process Input Image (grayscale 2D, 1 channel: _0000.png)
        src_img_path = training_set_dir / img_filename
        if not src_img_path.is_file():
            raise FileNotFoundError(f"Missing source image: {src_img_path}")
        img = cv2.imread(str(src_img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {src_img_path}")

        dst_img_path = images_tr_dir / f"{case_id}_0000.png"
        cv2.imwrite(str(dst_img_path), img)

        # 2. Process Label Mask (solid filled mask with values {0, 1})
        anno_filename = img_filename.replace(".png", "_Annotation.png")
        src_anno_path = training_set_dir / anno_filename
        if not src_anno_path.is_file():
            raise FileNotFoundError(f"Missing annotation mask: {src_anno_path}")
        raw_anno = cv2.imread(str(src_anno_path), cv2.IMREAD_GRAYSCALE)
        if raw_anno is None:
            raise ValueError(f"Failed to read annotation: {src_anno_path}")

        filled_mask = extract_filled_mask(raw_anno)
        dst_lbl_path = labels_tr_dir / f"{case_id}.png"
        cv2.imwrite(str(dst_lbl_path), filled_mask)

    # 3. Create dataset.json
    dataset_json_data = {
        "channel_names": {
            "0": "US"
        },
        "labels": {
            "background": 0,
            "head": 1
        },
        "numTraining": len(converted_cases),
        "file_ending": ".png",
    }
    dataset_json_path = output_dir / "dataset.json"
    with open(dataset_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json_data, f, indent=2)

    # 4. Create splits_final.json matching our patient-grouped 5 folds
    # nnU-Net folds are 0-indexed: fold 0 matches our fold 1, etc.
    full_df = pd.read_csv(oof_csv)
    splits = []
    fold_stats = {}

    for fold_idx, fold_num in enumerate(sorted(full_df["fold"].unique())):
        val_df = full_df[full_df["fold"] == fold_num]
        train_df = full_df[full_df["fold"] != fold_num]

        val_cases = sorted([Path(f).stem for f in val_df["image"]])
        train_cases = sorted([Path(f).stem for f in train_df["image"]])

        # If running on subset, filter to converted cases
        if max_cases is not None:
            val_cases = [c for c in val_cases if c in converted_cases]
            train_cases = [c for c in train_cases if c in converted_cases]

        # Verify no overlap between train and val
        assert set(val_cases).isdisjoint(set(train_cases)), f"Leakage in fold {fold_idx}"

        # Verify zero patient leakage
        val_pts = {cid.split("_")[0] for cid in val_cases}
        train_pts = {cid.split("_")[0] for cid in train_cases}
        assert val_pts.isdisjoint(train_pts), f"Patient leakage in fold {fold_idx}"

        splits.append({
            "train": train_cases,
            "val": val_cases,
        })
        fold_stats[f"fold_{fold_idx}"] = {
            "our_repo_fold": int(fold_num),
            "train_images": len(train_cases),
            "val_images": len(val_cases),
            "train_patients": len(train_pts),
            "val_patients": len(val_pts),
        }

    splits_json_path = output_dir / "splits_final.json"
    with open(splits_json_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    # Compute dataset directory size
    total_bytes = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    total_mb = total_bytes / (1024 * 1024)

    result_summary = {
        "dataset_name": "Dataset501_HC18",
        "output_dir": str(output_dir),
        "num_training_cases": len(converted_cases),
        "total_files": len(list(output_dir.rglob("*"))),
        "dataset_size_mb": round(total_mb, 2),
        "dataset_json": str(dataset_json_path),
        "splits_final_json": str(splits_json_path),
        "folds": fold_stats,
    }

    # 5. Package into zip archive if requested
    if package_zip is not None:
        package_zip = Path(package_zip).resolve()
        package_zip.parent.mkdir(parents=True, exist_ok=True)
        print(f"Packaging {output_dir.name} to {package_zip}...")
        with zipfile.ZipFile(package_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(output_dir.parent)
                    zf.write(file_path, arcname=str(arcname))

        zip_bytes = package_zip.stat().st_size
        zip_mb = zip_bytes / (1024 * 1024)
        sha256 = hashlib.sha256(package_zip.read_bytes()).hexdigest()

        result_summary["package"] = {
            "zip_path": str(package_zip),
            "size_mb": round(zip_mb, 2),
            "size_bytes": zip_bytes,
            "sha256": sha256,
        }
        print(f"Packaged archive created: {package_zip} ({zip_mb:.2f} MB, SHA256: {sha256})")

    return result_summary


def main():
    parser = argparse.ArgumentParser(description="Convert HC18 training data to nnU-Net v2 format.")
    parser.add_argument("--hc18-dir", type=Path, default=Path(HC18_DIR), help="HC18 dataset root directory.")
    parser.add_argument(
        "--oof-csv",
        type=Path,
        default=REPO_ROOT / "docs" / "results" / "hc18" / "oof_predictions.csv",
        help="Path to oof_predictions.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("C:/fetal-data/nnUNet_raw/Dataset501_HC18"),
        help="Target Dataset501_HC18 directory.",
    )
    parser.add_argument(
        "--package-zip",
        type=Path,
        default=Path("C:/fetal-data/nnUNet_packaged/Dataset501_HC18.zip"),
        help="Path to write packaged zip archive.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Maximum cases to convert (for testing).",
    )
    args = parser.parse_args()

    summary = convert_hc18_to_nnunet(
        hc18_dir=args.hc18_dir,
        oof_csv=args.oof_csv,
        output_dir=args.output_dir,
        package_zip=args.package_zip,
        max_cases=args.max_cases,
    )

    print("\n================ nnU-Net Dataset Conversion Summary ================")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
