#!/usr/bin/env python3
"""68_eval_nnunet_f1_f2.py: Comprehensive evaluation of nnU-Net Fold 1 and Fold 2 predictions.

Evaluates validation masks against official CSV ground truth and annotation masks:
  1. Fits submission ellipse (fit_submission_ellipse).
  2. Computes raw HC and measurement-fix HC (delta_r = 0.1134215 mm).
  3. Computes MAE, RMSE, std, median, signed bias (both pred-true and true-pred).
  4. Computes binary Dice and contour Hausdorff distance (mm).
  5. Computes per-size-band metrics (small <110mm, medium 110-250mm, large >250mm).
  6. Patient-level paired bootstrap (5000 resamples, seed 42) vs matching single-model OOF.
  7. Outputs docs/results/hc18_nnunet/f1_eval.json and f2_eval.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff

REPO_ROOT = Path(__file__).resolve().parent.parent
src_dir = REPO_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fetal.eval.ellipse_fitting import hc_bpd_from_ellipse_params
from fetal.eval.hc18_eval import fit_submission_ellipse

HC18_DIR = Path("C:/fetal-data/HC18")
TRAIN_CSV = HC18_DIR / "training_set_pixel_size_and_HC.csv"
TRAIN_DIR = HC18_DIR / "training_set"
OOF_CSV = REPO_ROOT / "docs" / "results" / "hc18" / "oof_predictions.csv"
DELTA_R = 0.11342152722878392


def compute_mask_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a_bin = mask_a > 0
    b_bin = mask_b > 0
    inter = np.logical_and(a_bin, b_bin).sum()
    total = a_bin.sum() + b_bin.sum()
    if total == 0:
        return 1.0
    return float(2.0 * inter / total)


def compute_contour_hausdorff_mm(mask_a: np.ndarray, mask_b: np.ndarray, pixel_size: float) -> float:
    a_bin = (mask_a > 0).astype(np.uint8)
    b_bin = (mask_b > 0).astype(np.uint8)

    cnts_a, _ = cv2.findContours(a_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts_b, _ = cv2.findContours(b_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not cnts_a or not cnts_b:
        return float("nan")

    pts_a = np.vstack(cnts_a).squeeze().reshape(-1, 2).astype(np.float64) * pixel_size
    pts_b = np.vstack(cnts_b).squeeze().reshape(-1, 2).astype(np.float64) * pixel_size

    d_ab = directed_hausdorff(pts_a, pts_b)[0]
    d_ba = directed_hausdorff(pts_b, pts_a)[0]
    return float(max(d_ab, d_ba))


def extract_ground_truth_mask(anno_path: Path) -> np.ndarray:
    raw = cv2.imread(str(anno_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Missing annotation: {anno_path}")
    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(raw, dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled, [largest], -1, 1, thickness=cv2.FILLED)
    return filled


def patient_level_paired_bootstrap(
    patients: Sequence[str],
    err_a_by_patient: dict[str, np.ndarray],
    err_b_by_patient: dict[str, np.ndarray],
    n_bootstraps: int = 5000,
    seed: int = 42,
) -> dict:
    """Bootstrap delta = B - A grouped by patient."""
    unique_pts = np.unique(patients)
    n_pts = len(unique_pts)

    obs_a = float(np.mean(np.concatenate([err_a_by_patient[p] for p in unique_pts])))
    obs_b = float(np.mean(np.concatenate([err_b_by_patient[p] for p in unique_pts])))
    obs_delta = obs_b - obs_a

    rng = np.random.default_rng(seed)
    boot_deltas = np.empty(n_bootstraps, dtype=np.float64)

    for i in range(n_bootstraps):
        sampled = rng.choice(unique_pts, size=n_pts, replace=True)
        sample_a = np.concatenate([err_a_by_patient[p] for p in sampled])
        sample_b = np.concatenate([err_b_by_patient[p] for p in sampled])
        boot_deltas[i] = np.mean(sample_b) - np.mean(sample_a)

    ci_low, ci_high = np.percentile(boot_deltas, [2.5, 97.5])
    p_val = float(2.0 * min(np.mean(boot_deltas <= 0.0), np.mean(boot_deltas >= 0.0)))
    p_val = min(1.0, p_val)

    return {
        "n_patients": int(n_pts),
        "mean_a": round(obs_a, 4),
        "mean_b": round(obs_b, 4),
        "delta": round(obs_delta, 4),
        "ci95": [round(float(ci_low), 4), round(float(ci_high), 4)],
        "p_value": round(p_val, 4),
    }


def evaluate_nnunet_fold(
    fold_idx: int,
    mask_dir: Path,
    out_json_path: Path,
    n_bootstraps: int = 5000,
    seed: int = 42,
) -> dict:
    print(f"\n================ Evaluating nnU-Net Fold {fold_idx} ================")
    print(f"Mask directory: {mask_dir}")

    df_train = pd.read_csv(TRAIN_CSV)
    df_train_map = {r["filename"]: r for _, r in df_train.iterrows()}

    # In our single-model OOF, fold mapping:
    # nnU-Net fold 0 -> OOF fold 1
    # nnU-Net fold 1 -> OOF fold 2
    # nnU-Net fold 2 -> OOF fold 3
    # nnU-Net fold 3 -> OOF fold 4
    # nnU-Net fold 4 -> OOF fold 5
    oof_fold_idx = fold_idx + 1
    df_oof = pd.read_csv(OOF_CSV)
    df_oof_fold = df_oof[df_oof["fold"] == oof_fold_idx].copy()
    oof_err_map = {}
    for _, r in df_oof_fold.iterrows():
        # raw absolute error vs CSV
        oof_err_map[str(r["image"])] = abs(float(r["pred_HC_mm"]) - float(r["true_HC_mm"]))

    mask_files = sorted(list(mask_dir.glob("*.png")))
    if not mask_files:
        raise FileNotFoundError(f"No PNG masks found in {mask_dir}")

    print(f"Found {len(mask_files)} mask files.")

    records = []
    for mf in mask_files:
        fn = mf.name
        if fn not in df_train_map:
            raise KeyError(f"Filename {fn} not in training CSV")

        row = df_train_map[fn]
        ps = float(row["pixel size(mm)"])
        csv_hc = float(row["head circumference (mm)"])
        pid = str(fn.split(".")[0].split("_")[0])

        # Load pred mask
        pred_mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
        prob_map = (pred_mask > 0).astype(np.float32)

        # Fit submission ellipse
        sub_dict = fit_submission_ellipse(prob_map, pixel_size=ps, filename=fn)
        a_raw = float(sub_dict["semi_axes_a_mm"])
        b_raw = float(sub_dict["semi_axes_b_mm"])
        hc_raw, _ = hc_bpd_from_ellipse_params(a_raw, b_raw)

        # Measurement fix
        a_fix = max(0.1, a_raw - DELTA_R)
        b_fix = max(0.1, b_raw - DELTA_R)
        hc_fix, _ = hc_bpd_from_ellipse_params(a_fix, b_fix)

        # Ground truth solid mask for Dice and HD
        anno_path = TRAIN_DIR / fn.replace(".png", "_Annotation.png")
        gt_mask = extract_ground_truth_mask(anno_path)
        dice = compute_mask_dice(pred_mask, gt_mask)
        hd_mm = compute_contour_hausdorff_mm(pred_mask, gt_mask, pixel_size=ps)

        # Size bands
        if csv_hc < 110.0:
            band = "small_under_110mm"
            trimester = 1
        elif csv_hc <= 250.0:
            band = "medium_110_to_250mm"
            trimester = 2
        else:
            band = "large_over_250mm"
            trimester = 3

        abs_diff_raw = abs(hc_raw - csv_hc)
        abs_diff_fix = abs(hc_fix - csv_hc)
        bias_raw = hc_raw - csv_hc  # pred - true
        bias_fix = hc_fix - csv_hc

        oof_ae = oof_err_map.get(fn, np.nan)

        records.append({
            "image": fn,
            "patient_id": pid,
            "pixel_size": ps,
            "csv_hc": csv_hc,
            "band": band,
            "trimester": trimester,
            "pred_hc_raw": round(hc_raw, 4),
            "abs_diff_raw": round(abs_diff_raw, 4),
            "bias_raw_pred_minus_true": round(bias_raw, 4),
            "pred_hc_measfix": round(hc_fix, 4),
            "abs_diff_measfix": round(abs_diff_fix, 4),
            "bias_fix_pred_minus_true": round(bias_fix, 4),
            "dice": round(dice, 4),
            "hausdorff_mm": round(hd_mm, 4),
            "oof_abs_err": round(oof_ae, 4) if not np.isnan(oof_ae) else None,
        })

    df_res = pd.DataFrame(records)
    n_images = len(df_res)
    n_patients = int(df_res["patient_id"].nunique())

    # Raw metrics
    raw_ae = df_res["abs_diff_raw"].to_numpy()
    raw_bias = df_res["bias_raw_pred_minus_true"].to_numpy()
    raw_metrics = {
        "mae": round(float(np.mean(raw_ae)), 4),
        "rmse": round(float(np.sqrt(np.mean(raw_ae ** 2))), 4),
        "std": round(float(np.std(raw_ae, ddof=1)), 4),
        "median": round(float(np.median(raw_ae)), 4),
        "signed_bias_pred_minus_true_mean": round(float(np.mean(raw_bias)), 4),
        "signed_bias_pred_minus_true_std": round(float(np.std(raw_bias, ddof=1)), 4),
        "signed_bias_true_minus_pred_mean": round(float(-np.mean(raw_bias)), 4),
        "signed_bias_true_minus_pred_std": round(float(np.std(raw_bias, ddof=1)), 4),
    }

    # Measfix metrics
    fix_ae = df_res["abs_diff_measfix"].to_numpy()
    fix_bias = df_res["bias_fix_pred_minus_true"].to_numpy()
    fix_metrics = {
        "mae": round(float(np.mean(fix_ae)), 4),
        "rmse": round(float(np.sqrt(np.mean(fix_ae ** 2))), 4),
        "std": round(float(np.std(fix_ae, ddof=1)), 4),
        "median": round(float(np.median(fix_ae)), 4),
        "signed_bias_pred_minus_true_mean": round(float(np.mean(fix_bias)), 4),
        "signed_bias_pred_minus_true_std": round(float(np.std(fix_bias, ddof=1)), 4),
        "signed_bias_true_minus_pred_mean": round(float(-np.mean(fix_bias)), 4),
        "signed_bias_true_minus_pred_std": round(float(np.std(fix_bias, ddof=1)), 4),
    }

    # Mask segmentation metrics
    dice_vals = df_res["dice"].to_numpy()
    hd_vals = df_res["hausdorff_mm"].dropna().to_numpy()
    mask_metrics = {
        "dice_mean": round(float(np.mean(dice_vals)), 4),
        "dice_std": round(float(np.std(dice_vals, ddof=1)), 4),
        "dice_median": round(float(np.median(dice_vals)), 4),
        "hausdorff_mean_mm": round(float(np.mean(hd_vals)), 4),
        "hausdorff_std_mm": round(float(np.std(hd_vals, ddof=1)), 4),
        "hausdorff_median_mm": round(float(np.median(hd_vals)), 4),
    }

    # Size-band breakdown
    band_metrics = {}
    for b_name in ["small_under_110mm", "medium_110_to_250mm", "large_over_250mm"]:
        sub_b = df_res[df_res["band"] == b_name]
        b_raw_ae = sub_b["abs_diff_raw"].to_numpy()
        b_fix_ae = sub_b["abs_diff_measfix"].to_numpy()
        b_raw_bias = sub_b["bias_raw_pred_minus_true"].to_numpy()
        b_fix_bias = sub_b["bias_fix_pred_minus_true"].to_numpy()
        b_dice = sub_b["dice"].to_numpy()
        b_hd = sub_b["hausdorff_mm"].dropna().to_numpy()

        band_metrics[b_name] = {
            "n": int(len(sub_b)),
            "raw_mae": round(float(np.mean(b_raw_ae)), 4),
            "raw_rmse": round(float(np.sqrt(np.mean(b_raw_ae ** 2))), 4),
            "raw_bias_pred_minus_true": round(float(np.mean(b_raw_bias)), 4),
            "measfix_mae": round(float(np.mean(b_fix_ae)), 4),
            "measfix_rmse": round(float(np.sqrt(np.mean(b_fix_ae ** 2))), 4),
            "measfix_bias_pred_minus_true": round(float(np.mean(b_fix_bias)), 4),
            "dice_mean": round(float(np.mean(b_dice)), 4),
            "hausdorff_mean_mm": round(float(np.mean(b_hd)), 4),
        }

    # Paired comparisons (patient-clustered bootstrap)
    patient_ids = df_res["patient_id"].to_numpy()
    raw_by_pt = {p: df_res[df_res["patient_id"] == p]["abs_diff_raw"].to_numpy() for p in df_res["patient_id"].unique()}
    fix_by_pt = {p: df_res[df_res["patient_id"] == p]["abs_diff_measfix"].to_numpy() for p in df_res["patient_id"].unique()}

    # 1. Measfix vs Raw
    boot_measfix_vs_raw = patient_level_paired_bootstrap(
        patients=patient_ids,
        err_a_by_patient=raw_by_pt,
        err_b_by_patient=fix_by_pt,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    # 2. vs Single-model OOF on matching images
    matching_oof = df_res[df_res["oof_abs_err"].notnull()]
    oof_pts = matching_oof["patient_id"].to_numpy()
    oof_err_by_pt = {p: matching_oof[matching_oof["patient_id"] == p]["oof_abs_err"].to_numpy() for p in np.unique(oof_pts)}
    raw_matching_by_pt = {p: matching_oof[matching_oof["patient_id"] == p]["abs_diff_raw"].to_numpy() for p in np.unique(oof_pts)}
    fix_matching_by_pt = {p: matching_oof[matching_oof["patient_id"] == p]["abs_diff_measfix"].to_numpy() for p in np.unique(oof_pts)}

    boot_raw_vs_oof = patient_level_paired_bootstrap(
        patients=oof_pts,
        err_a_by_patient=oof_err_by_pt,
        err_b_by_patient=raw_matching_by_pt,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    boot_fix_vs_oof = patient_level_paired_bootstrap(
        patients=oof_pts,
        err_a_by_patient=oof_err_by_pt,
        err_b_by_patient=fix_matching_by_pt,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    output = {
        "fold": int(fold_idx),
        "oof_matching_fold": int(oof_fold_idx),
        "n_images": int(n_images),
        "n_patients": int(n_patients),
        "delta_r_mm": DELTA_R,
        "raw_vs_csv": raw_metrics,
        "measfix_vs_csv": fix_metrics,
        "mask_metrics": mask_metrics,
        "per_size_band": band_metrics,
        "paired_bootstrap_measfix_vs_raw": boot_measfix_vs_raw,
        "paired_bootstrap_nnunet_raw_vs_oof": boot_raw_vs_oof,
        "paired_bootstrap_nnunet_measfix_vs_oof": boot_fix_vs_oof,
        "per_image": records,
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved evaluation JSON: {out_json_path}")
    print(f"  Raw MAE: {raw_metrics['mae']:.4f} mm, RMSE: {raw_metrics['rmse']:.4f} mm, Bias [pred-true]: {raw_metrics['signed_bias_pred_minus_true_mean']:+.4f} mm")
    print(f"  Measfix MAE: {fix_metrics['mae']:.4f} mm, RMSE: {fix_metrics['rmse']:.4f} mm, Bias [pred-true]: {fix_metrics['signed_bias_pred_minus_true_mean']:+.4f} mm")
    print(f"  Dice: {mask_metrics['dice_mean']:.4f}, HD: {mask_metrics['hausdorff_mean_mm']:.4f} mm")
    print(f"  Bootstrap Measfix vs Raw Delta: {boot_measfix_vs_raw['delta']:.4f} mm, 95% CI: {boot_measfix_vs_raw['ci95']}, p={boot_measfix_vs_raw['p_value']}")
    print(f"  Bootstrap Measfix vs OOF Delta: {boot_fix_vs_oof['delta']:.4f} mm, 95% CI: {boot_fix_vs_oof['ci95']}, p={boot_fix_vs_oof['p_value']}")

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=[1, 2])
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--n-bootstraps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate_nnunet_fold(
        fold_idx=args.fold,
        mask_dir=args.mask_dir,
        out_json_path=args.out_json,
        n_bootstraps=args.n_bootstraps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
