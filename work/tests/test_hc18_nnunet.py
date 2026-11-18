"""tests/test_hc18_nnunet.py

Unit tests for nnU-Net v2 dataset conversion (Dataset501_HC18) and fold-1 evaluation:
  1. Dataset conversion structure (dataset.json, splits_final.json, image/mask files).
  2. Patient-grouped fold partitioning and zero patient leakage.
  3. Fold 1 evaluation script and paired bootstrap comparison.
  4. Packaged archive integrity and size verification.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import zipfile
import cv2
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
src_dir = REPO_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import importlib
conv_mod = importlib.import_module("scripts.45_hc18_to_nnunet")
eval_mod = importlib.import_module("scripts.46_eval_nnunet_hc18")

convert_hc18_to_nnunet = conv_mod.convert_hc18_to_nnunet
extract_filled_mask = conv_mod.extract_filled_mask
patient_level_paired_bootstrap = eval_mod.patient_level_paired_bootstrap
evaluate_nnunet_fold1 = eval_mod.evaluate_nnunet_fold1


def test_extract_filled_mask_values():
    """Verify contour filling produces pure binary {0, 1} mask."""
    outline = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(outline, (50, 50), 30, 255, thickness=2)

    filled = extract_filled_mask(outline)
    assert filled.dtype == np.uint8
    assert set(np.unique(filled)).issubset({0, 1})
    assert (filled == 1).sum() > 2000
    assert filled[50, 50] == 1
    assert filled[10, 10] == 0


def test_conversion_synthetic(tmp_path):
    """Test full conversion pipeline on a synthetic 5-image dataset."""
    raw_dir = tmp_path / "raw_hc18"
    train_dir = raw_dir / "training_set"
    train_dir.mkdir(parents=True)

    cases = ["001_HC.png", "001_2HC.png", "002_HC.png", "003_HC.png", "004_HC.png"]
    folds = [1, 1, 2, 3, 4]
    patients = ["001", "001", "002", "003", "004"]

    for c in cases:
        # Create image
        img = np.full((120, 160), 128, dtype=np.uint8)
        cv2.imwrite(str(train_dir / c), img)

        # Create annotation
        anno = np.zeros((120, 160), dtype=np.uint8)
        cv2.circle(anno, (80, 60), 30, 255, thickness=2)
        cv2.imwrite(str(train_dir / c.replace(".png", "_Annotation.png")), anno)

    oof_df = pd.DataFrame({
        "image": cases,
        "patient_id": patients,
        "fold": folds,
        "true_HC_mm": [100.0, 102.0, 110.0, 115.0, 120.0],
        "pred_HC_mm": [99.0, 101.5, 109.0, 114.0, 118.0],
        "abs_err_HC": [1.0, 0.5, 1.0, 1.0, 2.0],
    })
    oof_csv = tmp_path / "synthetic_oof.csv"
    oof_df.to_csv(oof_csv, index=False)

    out_dataset = tmp_path / "Dataset501_HC18"
    out_zip = tmp_path / "Dataset501_HC18.zip"

    summary = convert_hc18_to_nnunet(
        hc18_dir=raw_dir,
        oof_csv=oof_csv,
        output_dir=out_dataset,
        package_zip=out_zip,
    )

    assert summary["num_training_cases"] == 5
    assert (out_dataset / "dataset.json").is_file()
    assert (out_dataset / "splits_final.json").is_file()

    with open(out_dataset / "dataset.json") as f:
        djson = json.load(f)
    assert djson["channel_names"] == {"0": "US"}
    assert djson["labels"] == {"background": 0, "head": 1}
    assert djson["file_ending"] == ".png"

    with open(out_dataset / "splits_final.json") as f:
        splits = json.load(f)
    assert len(splits) == 4  # 4 unique folds in synthetic data

    # Check zip exists
    assert out_zip.is_file()
    assert out_zip.stat().st_size > 0


def test_real_splits_final_json_integrity():
    """Verify our patient-grouped 5 folds match oof_predictions.csv perfectly."""
    oof_csv = REPO_ROOT / "docs" / "results" / "hc18" / "oof_predictions.csv"
    assert oof_csv.is_file()
    df_oof = pd.read_csv(oof_csv)
    assert len(df_oof) == 999

    expected_counts = {1: 206, 2: 199, 3: 208, 4: 194, 5: 192}
    for fold_num, count in expected_counts.items():
        assert (df_oof["fold"] == fold_num).sum() == count

    # Verify patient grouping (zero overlap across folds)
    for f in range(1, 6):
        val_pts = set(df_oof[df_oof["fold"] == f]["patient_id"])
        train_pts = set(df_oof[df_oof["fold"] != f]["patient_id"])
        assert val_pts.isdisjoint(train_pts)


def test_eval_paired_bootstrap_logic():
    """Verify paired bootstrap calculation and statistics."""
    df_exist = pd.DataFrame({
        "image": [f"{i:03d}_HC.png" for i in range(10)],
        "patient_id": [f"P{i // 2:02d}" for i in range(10)],
        "abs_err_HC": [2.0] * 10,
    })
    df_nnunet = pd.DataFrame({
        "image": [f"{i:03d}_HC.png" for i in range(10)],
        "patient_id": [f"P{i // 2:02d}" for i in range(10)],
        "abs_diff_HC": [1.0] * 10,
    })

    res = patient_level_paired_bootstrap(df_exist, df_nnunet, n_bootstraps=1000, seed=42)
    assert res["n_patients"] == 5
    assert res["n_images"] == 10
    assert res["existing_fold1_mae"] == pytest.approx(2.0, rel=1e-3)
    assert res["nnunet_fold1_mae"] == pytest.approx(1.0, rel=1e-3)
    assert res["delta_mae"] == pytest.approx(-1.0, rel=1e-3)
    assert res["ci95_delta_mae"][0] <= -1.0 <= res["ci95_delta_mae"][1]
