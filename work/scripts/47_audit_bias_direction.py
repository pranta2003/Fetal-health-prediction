#!/usr/bin/env python3

# FIELD: General
# MODEL: None
# STEP: utility
# USED BY: work/docs/CODE_MAP.md

"""Audit bias direction discrepancy between validation and test sets for HC18.

Compares:
1. On the 206 nnU-Net fold 0 validation masks:
   - Pipeline A (OOF mask-to-HC): cv2.fitEllipseAMS, semi-axes in mm, Ramanujan HC
   - Pipeline B (Submission pipeline): fit_submission_ellipse (cv2.fitEllipseAMS, angle fixed, Ramanujan HC)
   - Annotation-mask ellipse: cv2.fitEllipseAMS on the ground truth training annotation masks vs CSV HC
   Both signed differences reported under both conventions:
     - (pred - true)  [Grand Challenge platform convention]
     - (true - pred)  [Internal Convention A]
2. Pixel size distributions:
   - Training set (training_set_pixel_size_and_HC.csv, N=999)
   - Test set (test_set_pixel_size.csv, N=335)
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2
import numpy as np
import pandas as pd

from fetal.eval.ellipse_fitting import fit_ellipse_robust, hc_bpd_from_ellipse_params
from fetal.eval.hc18_eval import fit_submission_ellipse

REPO_ROOT = Path(__file__).resolve().parent.parent
HC18_DATA_DIR = Path("C:/fetal-data/HC18")
VAL_MASK_DIR = Path("E:/BUBT-CSE-4th yr/Capstone/kaggle_outputs/hc18_nnunet_f0/out/validation_fold0")
OUT_JSON = REPO_ROOT / "docs/results/hc18_nnunet/bias_audit.json"

def main():
    train_csv_path = HC18_DATA_DIR / "training_set_pixel_size_and_HC.csv"
    test_csv_path = HC18_DATA_DIR / "test_set_pixel_size.csv"

    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # 1. Pixel size comparison
    train_ps = df_train["pixel size(mm)"].to_numpy()
    test_ps = df_test["pixel size(mm)"].to_numpy()

    pixel_size_stats = {
        "train": {
            "n": int(len(train_ps)),
            "mean": float(np.mean(train_ps)),
            "std": float(np.std(train_ps, ddof=1)),
            "median": float(np.median(train_ps)),
            "min": float(np.min(train_ps)),
            "max": float(np.max(train_ps)),
        },
        "test": {
            "n": int(len(test_ps)),
            "mean": float(np.mean(test_ps)),
            "std": float(np.std(test_ps, ddof=1)),
            "median": float(np.median(test_ps)),
            "min": float(np.min(test_ps)),
            "max": float(np.max(test_ps)),
        },
    }

    # 2. Validation set evaluation (206 cases)
    # Map filename -> row
    train_map = {r["filename"]: r for _, r in df_train.iterrows()}

    val_files = sorted(list(VAL_MASK_DIR.glob("*.png")))
    assert len(val_files) == 206, f"Expected 206 validation masks, found {len(val_files)}"

    records = []
    for mf in val_files:
        fn = mf.name
        if fn not in train_map:
            raise KeyError(f"Filename {fn} not in training CSV")

        row = train_map[fn]
        ps = float(row["pixel size(mm)"])
        csv_hc = float(row["head circumference (mm)"])

        # Load predicted mask
        pred_mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
        prob_map = (pred_mask > 0).astype(np.float32)

        # Pipeline A: OOF mask-to-HC
        contours_pred, _ = cv2.findContours((prob_map > 0.5).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pred_largest = max(contours_pred, key=cv2.contourArea)
        (_, _), (p_ax1, p_ax2), _ = fit_ellipse_robust(pred_largest)
        major_mm = (max(p_ax1, p_ax2) / 2.0) * ps
        minor_mm = (min(p_ax1, p_ax2) / 2.0) * ps
        oof_hc, _ = hc_bpd_from_ellipse_params(major_mm, minor_mm)

        # Pipeline B: Submission pipeline
        sub_dict = fit_submission_ellipse(prob_map, ps, fn)
        sub_hc, _ = hc_bpd_from_ellipse_params(sub_dict["semi_axes_a_mm"], sub_dict["semi_axes_b_mm"])

        # Ground truth annotation mask
        ann_fn = fn.replace(".png", "_Annotation.png")
        ann_path = HC18_DATA_DIR / "training_set" / ann_fn
        ann_mask = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)
        contours_ann, _ = cv2.findContours(ann_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ann_largest = max(contours_ann, key=cv2.contourArea)
        (_, _), (a_ax1, a_ax2), _ = fit_ellipse_robust(ann_largest)
        ann_major_mm = (max(a_ax1, a_ax2) / 2.0) * ps
        ann_minor_mm = (min(a_ax1, a_ax2) / 2.0) * ps
        ann_hc, _ = hc_bpd_from_ellipse_params(ann_major_mm, ann_minor_mm)

        records.append({
            "filename": fn,
            "pixel_size": ps,
            "csv_hc": csv_hc,
            "oof_hc": float(oof_hc),
            "sub_hc": float(sub_hc),
            "ann_hc": float(ann_hc),
        })

    df_eval = pd.DataFrame(records)

    # Compute comparison summaries
    def get_stats(pred_col, target_col):
        diff_pred_minus_true = df_eval[pred_col] - df_eval[target_col]
        diff_true_minus_pred = df_eval[target_col] - df_eval[pred_col]
        abs_diff = np.abs(diff_pred_minus_true)
        return {
            "n": int(len(df_eval)),
            "mae": float(np.mean(abs_diff)),
            "std_ae": float(np.std(abs_diff, ddof=1)),
            "median_ae": float(np.median(abs_diff)),
            "bias_pred_minus_true_mean": float(np.mean(diff_pred_minus_true)),
            "bias_pred_minus_true_std": float(np.std(diff_pred_minus_true, ddof=1)),
            "bias_true_minus_pred_mean": float(np.mean(diff_true_minus_pred)),
            "bias_true_minus_pred_std": float(np.std(diff_true_minus_pred, ddof=1)),
        }

    results = {
        "dataset": "HC18 Fold 0 Validation (206 cases)",
        "pixel_size_distributions": pixel_size_stats,
        "pipeline_a_oof_vs_csv": get_stats("oof_hc", "csv_hc"),
        "pipeline_b_submission_vs_csv": get_stats("sub_hc", "csv_hc"),
        "pipeline_b_vs_pipeline_a": {
            "max_abs_diff_between_pipelines": float(np.max(np.abs(df_eval["sub_hc"] - df_eval["oof_hc"]))),
            "mean_abs_diff_between_pipelines": float(np.mean(np.abs(df_eval["sub_hc"] - df_eval["oof_hc"]))),
        },
        "gt_annotation_mask_vs_csv": get_stats("ann_hc", "csv_hc"),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=== AUDIT RESULTS ===")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
