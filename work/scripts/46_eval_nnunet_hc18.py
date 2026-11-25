#!/usr/bin/env python3

# FIELD: HC18
# MODEL: nnU-Net
# STEP: evaluation
# USED BY: work/docs/CODE_MAP.md, work/docs/report/EVIDENCE_MAP.md

"""46_eval_nnunet_hc18.py: Evaluate nnU-Net predicted masks for HC18 fold 1.

Takes nnU-Net's predicted masks for fold 1:
  1. Fits the ellipse using fit_submission_ellipse (same code as submission).
  2. Computes HC in mm (Ramanujan formula) and absolute difference vs ground truth.
  3. Computes Dice similarity coefficient and Hausdorff distance (mm) on fold-1 images.
  4. Compares with our existing fold-1 OOF rows using patient-level paired bootstrap
     (5000 resamples, seed 42).
"""

from __future__ import annotations

import argparse
import json
import os
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

from fetal.config import HC18_DIR
from fetal.eval.ellipse_fitting import hc_bpd_from_ellipse_params
from fetal.eval.hc18_eval import fit_submission_ellipse


def compute_mask_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute binary Dice similarity coefficient between two 2D masks."""
    a_bin = mask_a > 0
    b_bin = mask_b > 0
    inter = np.logical_and(a_bin, b_bin).sum()
    total = a_bin.sum() + b_bin.sum()
    if total == 0:
        return 1.0
    return float(2.0 * inter / total)


def compute_contour_hausdorff_mm(mask_a: np.ndarray, mask_b: np.ndarray, pixel_size: float) -> float:
    """Compute bidirectional Hausdorff distance in millimeters between mask boundaries."""
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
    """Read HC18 outline annotation and rasterize solid filled mask."""
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
    df_existing: pd.DataFrame,
    df_nnunet: pd.DataFrame,
    n_bootstraps: int = 5000,
    seed: int = 42,
) -> dict:
    """Patient-clustered paired bootstrap comparing nnU-Net vs Existing model on Fold 1.

    Delta is defined as: nnU-Net - Existing.
    A negative delta indicates nnU-Net achieved lower error (improvement).
    """
    df_a = df_existing[["image", "patient_id", "abs_err_HC"]].copy()
    df_b = df_nnunet[["image", "patient_id", "abs_diff_HC"]].copy()
    df_a["patient_id"] = df_a["patient_id"].astype(str)
    df_b["patient_id"] = df_b["patient_id"].astype(str)

    merged = pd.merge(
        df_a,
        df_b,
        on=["image", "patient_id"],
        how="inner",
    )

    if len(merged) == 0:
        raise ValueError("No common images found for paired bootstrap evaluation.")

    patients = merged["patient_id"].unique()
    n_patients = len(patients)

    patient_data = {}
    for p in patients:
        sub = merged[merged["patient_id"] == p]
        patient_data[p] = {
            "err_exist": sub["abs_err_HC"].to_numpy(dtype=np.float64),
            "err_nnunet": sub["abs_diff_HC"].to_numpy(dtype=np.float64),
        }

    obs_mae_exist = float(np.mean(np.concatenate([patient_data[p]["err_exist"] for p in patients])))
    obs_mae_nnunet = float(np.mean(np.concatenate([patient_data[p]["err_nnunet"] for p in patients])))
    obs_delta_mae = obs_mae_nnunet - obs_mae_exist

    rng = np.random.default_rng(seed)
    boot_delta_mae = np.empty(n_bootstraps, dtype=np.float64)

    for b in range(n_bootstraps):
        sampled_p = rng.choice(patients, size=n_patients, replace=True)
        exist_concat = np.concatenate([patient_data[p]["err_exist"] for p in sampled_p])
        nnunet_concat = np.concatenate([patient_data[p]["err_nnunet"] for p in sampled_p])
        boot_delta_mae[b] = np.mean(nnunet_concat) - np.mean(exist_concat)

    ci_low = float(np.percentile(boot_delta_mae, 2.5))
    ci_high = float(np.percentile(boot_delta_mae, 97.5))
    p_val = float(2.0 * min(np.mean(boot_delta_mae <= 0.0), np.mean(boot_delta_mae >= 0.0)))
    p_val = min(1.0, p_val)

    return {
        "n_patients": int(n_patients),
        "n_images": int(len(merged)),
        "existing_fold1_mae": round(obs_mae_exist, 4),
        "nnunet_fold1_mae": round(obs_mae_nnunet, 4),
        "delta_mae": round(obs_delta_mae, 4),
        "ci95_delta_mae": [round(ci_low, 4), round(ci_high, 4)],
        "p_val_mae": round(p_val, 4),
    }


def evaluate_nnunet_fold1(
    pred_dir: Path | None,
    hc18_dir: Path,
    oof_csv: Path,
    fold: int = 1,
    n_bootstraps: int = 5000,
    seed: int = 42,
    synthetic_noise_std: float | None = None,
) -> dict:
    """Evaluate fold-1 predicted masks against HC18 ground truth and existing model."""
    hc18_dir = Path(hc18_dir).resolve()
    oof_csv = Path(oof_csv).resolve()

    df_oof = pd.read_csv(oof_csv)
    df_fold = df_oof[df_oof["fold"] == fold].copy().reset_index(drop=True)

    training_set_dir = hc18_dir / "training_set"
    px_csv = hc18_dir / "training_set_pixel_size_and_HC.csv"
    if px_csv.is_file():
        df_px = pd.read_csv(px_csv)
        px_map = dict(zip(df_px["filename"], df_px["pixel size(mm)"]))
    else:
        px_map = {}

    results = []
    print(f"Evaluating {len(df_fold)} images for fold {fold}...")

    rng = np.random.default_rng(seed)

    for idx, row in df_fold.iterrows():
        img_name = str(row["image"])
        case_id = Path(img_name).stem
        patient_id = str(row["patient_id"])
        true_hc = float(row["true_HC_mm"])
        ps = px_map.get(img_name, 0.1)

        # Ground truth solid mask
        anno_path = training_set_dir / f"{case_id}_Annotation.png"
        gt_mask = extract_ground_truth_mask(anno_path)

        # Load predicted mask
        if pred_dir is not None and (pred_dir / f"{case_id}.png").is_file():
            pred_path = pred_dir / f"{case_id}.png"
            pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
            prob_map = (pred_mask > 0).astype(np.float32)
        elif synthetic_noise_std is not None:
            # Synthetic mask evaluation for unit testing / dry run
            # Perturb ground truth slightly
            prob_map = gt_mask.astype(np.float32)
            noise = rng.normal(0, synthetic_noise_std, size=prob_map.shape)
            prob_map = np.clip(prob_map + noise, 0.0, 1.0)
            pred_mask = (prob_map > 0.5).astype(np.uint8)
        else:
            raise FileNotFoundError(f"Predicted mask not found for {case_id} in {pred_dir}")

        # Fit submission ellipse
        sub = fit_submission_ellipse(prob_map, pixel_size=ps, filename=img_name)
        pred_hc, pred_bpd = hc_bpd_from_ellipse_params(
            float(sub["semi_axes_a_mm"]), float(sub["semi_axes_b_mm"])
        )
        abs_diff_hc = abs(pred_hc - true_hc)

        # Dice and Hausdorff
        dice = compute_mask_dice(pred_mask, gt_mask)
        hd_mm = compute_contour_hausdorff_mm(pred_mask, gt_mask, pixel_size=ps)

        results.append({
            "image": img_name,
            "patient_id": patient_id,
            "true_HC_mm": true_hc,
            "pred_HC_mm": round(float(pred_hc), 4),
            "abs_diff_HC": round(float(abs_diff_hc), 4),
            "dice": round(float(dice), 4),
            "hausdorff_mm": round(float(hd_mm), 4),
        })

    df_nnunet = pd.DataFrame(results)

    # Compute aggregate metrics
    agg_metrics = {
        "fold": int(fold),
        "n_images": len(df_nnunet),
        "hc_mae": round(float(df_nnunet["abs_diff_HC"].mean()), 4),
        "hc_rmse": round(float(np.sqrt((df_nnunet["abs_diff_HC"] ** 2).mean())), 4),
        "dice_mean": round(float(df_nnunet["dice"].mean()), 4),
        "hausdorff_mean_mm": round(float(df_nnunet["hausdorff_mm"].dropna().mean()), 4),
    }

    # Paired bootstrap vs existing fold-1 model
    paired_comp = patient_level_paired_bootstrap(
        df_existing=df_fold,
        df_nnunet=df_nnunet,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    agg_metrics["paired_comparison"] = paired_comp
    return {
        "aggregates": agg_metrics,
        "per_image": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate nnU-Net predicted masks on HC18 fold 1.")
    parser.add_argument("--pred-dir", type=Path, default=None, help="Directory containing predicted PNG masks.")
    parser.add_argument("--hc18-dir", type=Path, default=Path(HC18_DIR), help="HC18 dataset root directory.")
    parser.add_argument(
        "--oof-csv",
        type=Path,
        default=REPO_ROOT / "docs" / "results" / "hc18" / "oof_predictions.csv",
        help="Path to oof_predictions.csv.",
    )
    parser.add_argument("--fold", type=int, default=1, help="Cross-validation fold to evaluate (default 1).")
    parser.add_argument("--n-bootstraps", type=int, default=5000, help="Bootstrap iterations (default 5000).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    parser.add_argument("--synthetic", action="store_true", help="Run in synthetic dry-run mode.")
    parser.add_argument("--out-json", type=Path, default=None, help="Path to write evaluation JSON.")
    args = parser.parse_args()

    eval_out = evaluate_nnunet_fold1(
        pred_dir=args.pred_dir,
        hc18_dir=args.hc18_dir,
        oof_csv=args.oof_csv,
        fold=args.fold,
        n_bootstraps=args.n_bootstraps,
        seed=args.seed,
        synthetic_noise_std=0.05 if args.synthetic else None,
    )

    print("\n================ nnU-Net Fold 1 Evaluation Summary ================")
    print(json.dumps(eval_out["aggregates"], indent=2))

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(eval_out, f, indent=2)
        print(f"Saved evaluation results to {args.out_json}")


if __name__ == "__main__":
    main()
