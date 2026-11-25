#!/usr/bin/env python3

# FIELD: General
# MODEL: None
# STEP: utility
# USED BY: work/docs/CODE_MAP.md

"""HC18 Measurement-Fix Study.

Investigates and models the systematic measurement discrepancy between
OpenCV ellipse fitting on annotation masks and official CSV Head Circumference.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from fetal.eval.ellipse_fitting import fit_ellipse_robust, hc_bpd_from_ellipse_params
from fetal.eval.hc18_eval import fit_submission_ellipse

HC18_DIR = Path("C:/fetal-data/HC18")
TRAIN_CSV = HC18_DIR / "training_set_pixel_size_and_HC.csv"
TRAIN_DIR = HC18_DIR / "training_set"
OUT_JSON = REPO_ROOT / "docs/results/hc18_official/measfix_study.json"


def get_patient_id(filename: str) -> str:
    # Pattern: 000_HC.png -> 000, 010_2HC.png -> 010
    base = filename.split(".")[0]
    return base.split("_")[0]


def bootstrap_paired_delta(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    diffs = a - b
    mean_diff = float(np.mean(diffs))
    indices = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    boot_means = np.mean(diffs[indices], axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    p_val = float(2 * min(np.mean(boot_means <= 0), np.mean(boot_means >= 0)))
    return {
        "mean_delta": mean_diff,
        "ci_95": [float(ci_low), float(ci_high)],
        "p_value": p_val,
    }


def main():
    print("Loading training set...")
    df_train = pd.read_csv(TRAIN_CSV)
    df_train["patient_id"] = df_train["filename"].apply(get_patient_id)

    records = []
    print(f"Processing {len(df_train)} training annotation masks...")
    for idx, row in df_train.iterrows():
        fn = row["filename"]
        ps = float(row["pixel size(mm)"])
        csv_hc = float(row["head circumference (mm)"])
        pid = row["patient_id"]

        ann_path = TRAIN_DIR / fn.replace(".png", "_Annotation.png")
        ann_img = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)
        if ann_img is None:
            raise FileNotFoundError(f"Missing {ann_path}")

        # Measure ring thickness: nonzero pixels / perimeter of contour
        n_nonzero = int(np.count_nonzero(ann_img))
        contours_all, _ = cv2.findContours(ann_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_ext, _ = cv2.findContours(ann_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        largest = max(contours_ext, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(largest, True))
        thickness_px = (n_nonzero / perimeter) if perimeter > 0 else 0.0

        (_, _), (ax1, ax2), _ = fit_ellipse_robust(largest)
        semi_major_mm = (max(ax1, ax2) / 2.0) * ps
        semi_minor_mm = (min(ax1, ax2) / 2.0) * ps
        fit_hc, fit_bpd = hc_bpd_from_ellipse_params(semi_major_mm, semi_minor_mm)

        # Trimester binning
        if csv_hc < 110.0:
            trimester = 1
        elif csv_hc < 260.0:
            trimester = 2
        else:
            trimester = 3

        records.append({
            "filename": fn,
            "patient_id": pid,
            "pixel_size": ps,
            "csv_hc": csv_hc,
            "fit_hc": fit_hc,
            "semi_major_mm": semi_major_mm,
            "semi_minor_mm": semi_minor_mm,
            "thickness_px": thickness_px,
            "thickness_mm": thickness_px * ps,
            "trimester": trimester,
            "diff_fit_minus_csv": fit_hc - csv_hc,
        })

    df = pd.DataFrame(records)

    # 1. Summary of raw fit vs CSV HC
    raw_diff = df["diff_fit_minus_csv"].to_numpy()
    mean_offset = float(np.mean(raw_diff))
    std_offset = float(np.std(raw_diff, ddof=1))
    median_offset = float(np.median(raw_diff))
    mean_delta_r = mean_offset / (2.0 * math.pi)

    # 2. Patient-grouped 5-fold CV to compare forms of relation
    patients = np.unique(df["patient_id"])
    rng = np.random.default_rng(42)
    shuffled_patients = rng.permutation(patients)
    folds = np.array_split(shuffled_patients, 5)

    cv_results = {
        "none": [],
        "constant_hc_offset": [],
        "constant_radius_shrink": [],
        "ratio_scale": [],
        "linear_regression": [],
    }

    for fold_idx, val_pids in enumerate(folds):
        train_mask = ~df["patient_id"].isin(val_pids)
        val_mask = df["patient_id"].isin(val_pids)

        train_sub = df[train_mask]
        val_sub = df[val_mask]

        # (a) None
        pred_none = val_sub["fit_hc"].to_numpy()
        cv_results["none"].extend(np.abs(pred_none - val_sub["csv_hc"].to_numpy()))

        # (b) Constant HC offset
        delta_c = float(np.mean(train_sub["diff_fit_minus_csv"]))
        pred_const_c = val_sub["fit_hc"].to_numpy() - delta_c
        cv_results["constant_hc_offset"].extend(np.abs(pred_const_c - val_sub["csv_hc"].to_numpy()))

        # (c) Constant radius shrink delta_r:
        # a_new = a - delta_r, b_new = b - delta_r
        # delta_r = delta_c / (2 * pi)
        delta_r = delta_c / (2.0 * math.pi)
        pred_const_r = []
        for _, r in val_sub.iterrows():
            a_s = max(0.1, r["semi_major_mm"] - delta_r)
            b_s = max(0.1, r["semi_minor_mm"] - delta_r)
            hc_s, _ = hc_bpd_from_ellipse_params(a_s, b_s)
            pred_const_r.append(hc_s)
        cv_results["constant_radius_shrink"].extend(np.abs(np.array(pred_const_r) - val_sub["csv_hc"].to_numpy()))

        # (d) Ratio scale
        ratio = float(np.mean(train_sub["csv_hc"] / train_sub["fit_hc"]))
        pred_ratio = val_sub["fit_hc"].to_numpy() * ratio
        cv_results["ratio_scale"].extend(np.abs(pred_ratio - val_sub["csv_hc"].to_numpy()))

        # (e) Linear regression
        reg = LinearRegression().fit(train_sub[["fit_hc"]], train_sub["csv_hc"])
        pred_lin = reg.predict(val_sub[["fit_hc"]])
        cv_results["linear_regression"].extend(np.abs(pred_lin - val_sub["csv_hc"].to_numpy()))

    cv_summary = {
        model: {
            "mae": float(np.mean(errs)),
            "std": float(np.std(errs, ddof=1)),
            "median": float(np.median(errs)),
        }
        for model, errs in cv_results.items()
    }

    # 3. Trimester breakdown of raw offset
    trimester_stats = {}
    for t in [1, 2, 3]:
        sub_t = df[df["trimester"] == t]
        d_t = sub_t["diff_fit_minus_csv"].to_numpy()
        trimester_stats[f"trimester_{t}"] = {
            "n": int(len(sub_t)),
            "mean_hc": float(np.mean(sub_t["csv_hc"])),
            "diff_mean": float(np.mean(d_t)),
            "diff_std": float(np.std(d_t, ddof=1)),
            "diff_median": float(np.median(d_t)),
            "thickness_px_mean": float(np.mean(sub_t["thickness_px"])),
            "pixel_size_mean": float(np.mean(sub_t["pixel_size"])),
        }

    # Global fit parameters on ALL 999 training annotations:
    global_delta_c = float(np.mean(df["diff_fit_minus_csv"]))
    global_delta_r = global_delta_c / (2.0 * math.pi)
    global_ratio = float(np.mean(df["csv_hc"] / df["fit_hc"]))
    global_lin = LinearRegression().fit(df[["fit_hc"]], df["csv_hc"])

    print("\n=== MODEL COMPARISON ON 999 TRAINING ANNOTATIONS (5-Fold Grouped CV) ===")
    for m, res in cv_summary.items():
        print(f"  {m:25s}: MAE = {res['mae']:.4f} +- {res['std']:.4f} mm, Median = {res['median']:.4f} mm")

    print(f"\nFitted Global Parameters (from 999 training annotations ONLY):")
    print(f"  delta_C (HC offset) : {global_delta_c:.4f} mm")
    print(f"  delta_r (axis shrink): {global_delta_r:.4f} mm")
    print(f"  Ratio               : {global_ratio:.6f}")
    print(f"  Linear slope/int    : slope={global_lin.coef_[0]:.6f}, intercept={global_lin.intercept_:.4f}")

    # 4. Apply to Validation Predictions
    # (A) Single-model OOF (docs/results/hc18/oof_predictions.csv)
    oof_csv_path = REPO_ROOT / "docs/results/hc18/oof_predictions.csv"
    df_oof = pd.read_csv(oof_csv_path)
    # Check if df_oof has true_HC_mm
    oof_true = df_oof["true_HC_mm"].to_numpy()
    oof_pred_raw = df_oof["pred_HC_mm"].to_numpy()

    # Raw metrics
    raw_oof_ae = np.abs(oof_pred_raw - oof_true)
    raw_oof_signed = oof_pred_raw - oof_true  # pred - true

    # Corrected metrics using constant radius shrink delta_r
    # Since OOF predictions stored only pred_HC_mm, we apply delta_C = 2*pi*delta_r
    oof_pred_fix = oof_pred_raw - global_delta_c
    fix_oof_ae = np.abs(oof_pred_fix - oof_true)
    fix_oof_signed = oof_pred_fix - oof_true

    oof_bootstrap = bootstrap_paired_delta(fix_oof_ae, raw_oof_ae, seed=42)

    # (B) nnU-Net fold-0 206 validation masks
    # Use bias_audit.json records or recompute from masks
    bias_audit_path = REPO_ROOT / "docs/results/hc18_nnunet/bias_audit.json"
    with open(bias_audit_path, "r", encoding="utf-8") as f:
        bias_audit = json.load(f)

    # We evaluate nnunet predictions against CSV HC
    # From scripts/47, we have the records of sub_hc and csv_hc
    # Let's re-run for all 206 cases to get individual paired arrays
    train_map = {r["filename"]: r for _, r in df_train.iterrows()}
    val_files = sorted(list(Path("E:/BUBT-CSE-4th yr/Capstone/kaggle_outputs/hc18_nnunet_f0/out/validation_fold0").glob("*.png")))

    nnunet_raw_ae = []
    nnunet_raw_signed = []
    nnunet_fix_ae = []
    nnunet_fix_signed = []

    for mf in val_files:
        fn = mf.name
        row = train_map[fn]
        ps = float(row["pixel size(mm)"])
        csv_hc = float(row["head circumference (mm)"])

        pred_mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
        prob_map = (pred_mask > 0).astype(np.float32)
        sub_dict = fit_submission_ellipse(prob_map, ps, fn)

        a_raw = sub_dict["semi_axes_a_mm"]
        b_raw = sub_dict["semi_axes_b_mm"]
        hc_raw, _ = hc_bpd_from_ellipse_params(a_raw, b_raw)

        # Apply measurement fix to semi-axes
        a_fix = max(0.1, a_raw - global_delta_r)
        b_fix = max(0.1, b_raw - global_delta_r)
        hc_fix, _ = hc_bpd_from_ellipse_params(a_fix, b_fix)

        nnunet_raw_ae.append(abs(hc_raw - csv_hc))
        nnunet_raw_signed.append(hc_raw - csv_hc)
        nnunet_fix_ae.append(abs(hc_fix - csv_hc))
        nnunet_fix_signed.append(hc_fix - csv_hc)

    nnunet_raw_ae = np.array(nnunet_raw_ae)
    nnunet_raw_signed = np.array(nnunet_raw_signed)
    nnunet_fix_ae = np.array(nnunet_fix_ae)
    nnunet_fix_signed = np.array(nnunet_fix_signed)

    nnunet_bootstrap = bootstrap_paired_delta(nnunet_fix_ae, nnunet_raw_ae, seed=42)

    val_application = {
        "single_model_oof_999": {
            "n": int(len(df_oof)),
            "raw_vs_csv": {
                "mae": float(np.mean(raw_oof_ae)),
                "std": float(np.std(raw_oof_ae, ddof=1)),
                "signed_bias_pred_minus_true": float(np.mean(raw_oof_signed)),
                "signed_bias_true_minus_pred": float(-np.mean(raw_oof_signed)),
            },
            "measfix_vs_csv": {
                "mae": float(np.mean(fix_oof_ae)),
                "std": float(np.std(fix_oof_ae, ddof=1)),
                "signed_bias_pred_minus_true": float(np.mean(fix_oof_signed)),
                "signed_bias_true_minus_pred": float(-np.mean(fix_oof_signed)),
            },
            "paired_bootstrap_delta_mae": oof_bootstrap,
        },
        "nnunet_fold0_206": {
            "n": int(len(val_files)),
            "raw_vs_csv": {
                "mae": float(np.mean(nnunet_raw_ae)),
                "std": float(np.std(nnunet_raw_ae, ddof=1)),
                "signed_bias_pred_minus_true": float(np.mean(nnunet_raw_signed)),
                "signed_bias_true_minus_pred": float(-np.mean(nnunet_raw_signed)),
            },
            "measfix_vs_csv": {
                "mae": float(np.mean(nnunet_fix_ae)),
                "std": float(np.std(nnunet_fix_ae, ddof=1)),
                "signed_bias_pred_minus_true": float(np.mean(nnunet_fix_signed)),
                "signed_bias_true_minus_pred": float(-np.mean(nnunet_fix_signed)),
            },
            "paired_bootstrap_delta_mae": nnunet_bootstrap,
        },
    }

    # 5. Build submission_ensemble_5fold_corrected_measfix.csv
    corrected_csv_path = REPO_ROOT / "docs/results/hc18_official/submission_ensemble_5fold_corrected.csv"
    df_sub = pd.read_csv(corrected_csv_path)

    # Shrink BOTH semi-axes by delta_r
    # Keep center and angle intact
    df_sub_fix = df_sub.copy()

    # Calculate expected HC before and after
    hc_before = []
    hc_after = []
    for _, r in df_sub.iterrows():
        a = float(r["semi_axes_a_mm"])
        b = float(r["semi_axes_b_mm"])
        hc_b, _ = hc_bpd_from_ellipse_params(a, b)
        hc_a, _ = hc_bpd_from_ellipse_params(max(0.1, a - global_delta_r), max(0.1, b - global_delta_r))
        hc_before.append(hc_b)
        hc_after.append(hc_a)

    hc_before = np.array(hc_before)
    hc_after = np.array(hc_after)
    mean_hc_change = float(np.mean(hc_after - hc_before))

    df_sub_fix["semi_axes_a_mm"] = (df_sub["semi_axes_a_mm"] - global_delta_r).clip(lower=0.1).round(6)
    df_sub_fix["semi_axes_b_mm"] = (df_sub["semi_axes_b_mm"] - global_delta_r).clip(lower=0.1).round(6)

    # Save to candidates folder
    candidates_dir = REPO_ROOT / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path = candidates_dir / "submission_ensemble_5fold_corrected_measfix.csv"

    # Write with strict LF line endings
    csv_str = df_sub_fix.to_csv(index=False, lineterminator="\n")
    out_csv_path.write_bytes(csv_str.encode("utf-8"))

    import hashlib
    out_sha256 = hashlib.sha256(out_csv_path.read_bytes()).hexdigest()

    submission_info = {
        "file": str(out_csv_path),
        "n_rows": int(len(df_sub_fix)),
        "columns": list(df_sub_fix.columns),
        "delta_r_mm": global_delta_r,
        "mean_hc_change_test_mm": mean_hc_change,
        "sha256": out_sha256,
    }

    full_report = {
        "raw_training_annotation_discrepancy": {
            "n": int(len(df)),
            "mean_diff_mm": mean_offset,
            "std_diff_mm": std_offset,
            "median_diff_mm": median_offset,
            "delta_r_mm": global_delta_r,
        },
        "model_comparison_cv": cv_summary,
        "trimester_breakdown": trimester_stats,
        "validation_application": val_application,
        "submission_measfix": submission_info,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nWrote full report to {OUT_JSON}")
    print(f"Generated submission file: {out_csv_path}")
    print(f"  Rows: {len(df_sub_fix)}, SHA-256: {out_sha256}")
    print(f"  Mean HC change on test set: {mean_hc_change:.4f} mm")


if __name__ == "__main__":
    main()
