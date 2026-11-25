#!/usr/bin/env python3

# FIELD: AC
# MODEL: nnU-Net
# STEP: evaluation
# USED BY: work/docs/CODE_MAP.md, work/docs/results/RESULTS_LEDGER.md

"""scripts/85_eval_ac_nnunet_f0.py

Evaluates AC nnU-Net v2 (Fold 0, 100 epochs) validation predictions on the 142 validation images.
Pipeline:
1. Predict/load the 142 validation masks of Fold 0.
2. Keep the largest connected component.
3. Fit an ellipse with cv2.fitEllipseAMS.
4. Calculate AC in mm with per-image pixel size using Ramanujan perimeter formula.
5. Compute: MAE, RMSE, median, errors over 10 mm, signed bias.
6. Compare on the SAME 142 images with ac-baseline-s42, ac-baseline-s43, and their 2-seed average
   using paired patient-level cluster bootstrap (5,000 resamples, seed 42) for MAE difference and p-value.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
src_dir = REPO_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fetal.data.ac_dataset import load_ac_folds
from fetal.eval.ac_evaluator import patient_level_paired_bootstrap


def ramanujan_perimeter(a: float, b: float) -> float:
    """Ramanujan's first approximation for ellipse perimeter.
    a, b are semi-major and semi-minor axes (half diameters).
    """
    return math.pi * (3.0 * (a + b) - math.sqrt((3.0 * a + b) * (a + 3.0 * b)))


def extract_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a binary mask."""
    mask_binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    if num_labels <= 1:
        return mask_binary
    # stats[:, 4] is area, ignore index 0 (background)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8)


def fit_ellipse_and_measure_ac(mask: np.ndarray, mmpp: float) -> float:
    """Fit ellipse with cv2.fitEllipseAMS on largest component and calculate AC in mm."""
    largest_mask = extract_largest_component(mask)
    contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return 0.0

    try:
        ellipse = cv2.fitEllipseAMS(cnt)
    except cv2.error:
        ellipse = cv2.fitEllipse(cnt)

    (cx, cy), (axis1, axis2), angle = ellipse
    a = max(axis1, axis2) / 2.0
    b = min(axis1, axis2) / 2.0
    perimeter_px = ramanujan_perimeter(a, b)
    return perimeter_px * mmpp


def compute_cohort_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute summary metrics for AC predictions."""
    err = np.abs(y_pred - y_true)
    sq_err = (y_pred - y_true) ** 2
    signed = y_pred - y_true
    return {
        "n_samples": int(len(y_true)),
        "mae": round(float(np.mean(err)), 4),
        "rmse": round(float(np.sqrt(np.mean(sq_err))), 4),
        "median": round(float(np.median(err)), 4),
        "bias": round(float(np.mean(signed)), 4),
        "errors_gt10_rate": round(float(np.mean(err > 10.0)), 4),
        "errors_gt10_count": int(np.sum(err > 10.0)),
        "nme": round(float(np.mean(err / np.maximum(y_true, 1e-6))), 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AC nnU-Net Fold 0 Validation Outputs")
    parser.add_argument(
        "--nnunet-val-dir",
        type=Path,
        default=Path(r"C:\fetal-data\kaggle_outputs\ac_nnunet_f0\validation_fold0"),
        help="Path to downloaded validation masks folder",
    )
    parser.add_argument(
        "--s42-oof-csv",
        type=Path,
        default=Path(r"E:\BUBT-CSE-4th yr\Capstone\kaggle_outputs\ac_baseline_s42\ac-baseline-s42\oof_predictions_pooled.csv"),
        help="Path to baseline s42 OOF predictions CSV",
    )
    parser.add_argument(
        "--s43-oof-csv",
        type=Path,
        default=Path(r"C:\fetal-data\kaggle_outputs\ac_baseline_s43\ac-baseline-s43\oof_predictions_pooled.csv"),
        help="Path to baseline s43 OOF predictions CSV",
    )
    parser.add_argument(
        "--folds-csv",
        type=Path,
        default=REPO_ROOT / "docs" / "results" / "ac_campaign" / "folds_v1.csv",
        help="Path to folds_v1.csv",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"C:\fetal-data"),
        help="Path to base data directory containing raw images/annotations",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "docs" / "results" / "ac_nnunet" / "f0_eval.json",
        help="Path to save evaluation JSON results",
    )
    return parser.parse_args()


def run_evaluation(
    nnunet_val_dir: Path,
    s42_oof_csv: Path,
    s43_oof_csv: Path,
    folds_csv: Path,
    data_dir: Path,
    output_json: Path,
) -> dict:
    nnunet_val_dir = Path(nnunet_val_dir)
    s42_oof_csv = Path(s42_oof_csv)
    s43_oof_csv = Path(s43_oof_csv)
    folds_csv = Path(folds_csv)
    data_dir = Path(data_dir)
    output_json = Path(output_json)

    if not nnunet_val_dir.exists():
        raise FileNotFoundError(
            f"nnU-Net validation directory not found: {nnunet_val_dir}. "
            "Please ensure kernel output has finished and downloaded."
        )

    # 1. Load fold annotations
    folds_df = load_ac_folds(folds_csv=folds_csv, base=data_dir)
    # Fold 1 in folds_v1.csv corresponds to Fold 0 of Dataset502_AC (142 cases, 101 patients)
    val_df = folds_df[folds_df["fold"] == 1].copy().reset_index(drop=True)
    assert len(val_df) == 142, f"Expected 142 validation cases, got {len(val_df)}"

    # 2. Load s42 and s43 OOF predictions
    df_s42 = pd.read_csv(s42_oof_csv)
    df_s43 = pd.read_csv(s43_oof_csv)

    p_map = dict(zip(folds_df["image_name"], folds_df["patient"]))
    df_s42["patient"] = df_s42["image_name"].map(p_map)
    df_s43["patient"] = df_s43["image_name"].map(p_map)

    # Filter s42 & s43 to the exact same 142 cases
    val_images = set(val_df["image_name"])
    sub_s42 = df_s42[df_s42["image_name"].isin(val_images)].sort_values("image_name").reset_index(drop=True)
    sub_s43 = df_s43[df_s43["image_name"].isin(val_images)].sort_values("image_name").reset_index(drop=True)
    val_df = val_df.sort_values("image_name").reset_index(drop=True)

    assert len(sub_s42) == 142, f"Expected 142 rows in s42, got {len(sub_s42)}"
    assert len(sub_s43) == 142, f"Expected 142 rows in s43, got {len(sub_s43)}"
    assert (sub_s42["image_name"] == val_df["image_name"]).all(), "Image ordering mismatch"
    assert (sub_s43["image_name"] == val_df["image_name"]).all(), "Image ordering mismatch"

    # 3. Read nnU-Net predicted masks and compute AC
    nnunet_preds = []
    for idx, row in val_df.iterrows():
        img_name = str(row["image_name"])
        case_id = Path(img_name).stem
        mmpp = float(row["px_to_mm_rate"])

        # Try possible filenames (.png, .nii.gz, etc.)
        mask_candidates = [
            nnunet_val_dir / f"{case_id}.png",
            nnunet_val_dir / f"{case_id}.nii.gz",
            nnunet_val_dir / f"{case_id}_pred.png",
        ]
        mask_file = None
        for cand in mask_candidates:
            if cand.is_file():
                mask_file = cand
                break

        if mask_file is None:
            raise FileNotFoundError(f"Could not find validation mask for case {case_id} in {nnunet_val_dir}")

        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        ac_val = fit_ellipse_and_measure_ac(mask, mmpp)
        nnunet_preds.append(ac_val)

    nnunet_preds = np.array(nnunet_preds, dtype=float)
    y_true = sub_s42["ac_true"].to_numpy(dtype=float)
    s42_preds = sub_s42["ac_pred"].to_numpy(dtype=float)
    s43_preds = sub_s43["ac_pred"].to_numpy(dtype=float)
    avg_preds = (s42_preds + s43_preds) / 2.0

    # 4. Metrics
    metrics_nnunet = compute_cohort_metrics(y_true, nnunet_preds)
    metrics_s42 = compute_cohort_metrics(y_true, s42_preds)
    metrics_s43 = compute_cohort_metrics(y_true, s43_preds)
    metrics_avg = compute_cohort_metrics(y_true, avg_preds)

    # 5. Paired bootstrap comparisons
    # nnU-Net vs baseline s42
    m_nnunet_vs_s42 = pd.DataFrame({
        "patient": val_df["patient"],
        "ac_true": y_true,
        "ac_pred_a": s42_preds,
        "ac_pred_b": nnunet_preds,
    })
    boot_nnunet_vs_s42 = patient_level_paired_bootstrap(m_nnunet_vs_s42, n_bootstraps=5000, seed=42)

    # nnU-Net vs baseline s43
    m_nnunet_vs_s43 = pd.DataFrame({
        "patient": val_df["patient"],
        "ac_true": y_true,
        "ac_pred_a": s43_preds,
        "ac_pred_b": nnunet_preds,
    })
    boot_nnunet_vs_s43 = patient_level_paired_bootstrap(m_nnunet_vs_s43, n_bootstraps=5000, seed=42)

    # nnU-Net vs 2-seed average
    m_nnunet_vs_avg = pd.DataFrame({
        "patient": val_df["patient"],
        "ac_true": y_true,
        "ac_pred_a": avg_preds,
        "ac_pred_b": nnunet_preds,
    })
    boot_nnunet_vs_avg = patient_level_paired_bootstrap(m_nnunet_vs_avg, n_bootstraps=5000, seed=42)

    results = {
        "cohort": "AC Campaign Fold 0 Validation (Fold 1 of folds_v1.csv)",
        "n_images": 142,
        "n_patients": int(val_df["patient"].nunique()),
        "models": {
            "ac_nnunet_f0": metrics_nnunet,
            "ac_baseline_s42": metrics_s42,
            "ac_baseline_s43": metrics_s43,
            "ac_baseline_2seed_avg": metrics_avg,
        },
        "paired_bootstrap_5000_seed42": {
            "nnunet_vs_s42": boot_nnunet_vs_s42,
            "nnunet_vs_s43": boot_nnunet_vs_s43,
            "nnunet_vs_2seed_avg": boot_nnunet_vs_avg,
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results successfully saved to {output_json}")
    return results


def main():
    args = parse_args()
    print("=" * 70)
    print("AC nnU-Net Fold 0 Evaluation Script")
    print(f"nnU-Net validation dir: {args.nnunet_val_dir}")
    print(f"Target output JSON    : {args.output_json}")
    print("=" * 70)

    if not args.nnunet_val_dir.exists():
        print(f"NOTICE: Validation directory {args.nnunet_val_dir} does not exist yet.")
        print("Script prepared and ready to run upon output download.")
        return

    run_evaluation(
        nnunet_val_dir=args.nnunet_val_dir,
        s42_oof_csv=args.s42_oof_csv,
        s43_oof_csv=args.s43_oof_csv,
        folds_csv=args.folds_csv,
        data_dir=args.data_dir,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
