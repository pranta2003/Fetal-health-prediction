"""Out-Of-Fold (OOF) evaluation report for HC18 5-fold segmentation pipeline.

Loads 5 fold checkpoints (fm_B), reconstructs patient-grouped folds, runs TTA inference
on each fold's outer validation images, writes `oof_predictions.csv`, asserts pooled metrics
against target (HC 2.30/3.58, BPD 0.75/1.09, tol=0.02), produces tercile analysis table,
and calculates patient-level bootstrap 95% confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from fetal.data.hc18_dataset import HC18SegDataset
from fetal.data.hc18_split import make_grouped_folds, patient_id_from_name
from fetal.eval.hc18_eval import (
    compute_metrics,
    compute_patient_bootstrap_ci,
    compute_tercile_analysis,
    fit_ellipse_robust,
    measure_biometrics_from_prob,
    predict_prob_with_tta,
)
from fetal.models.hc18_unet import build_segmentation_model


def parse_args():
    parser = argparse.ArgumentParser(description="HC18 Out-Of-Fold Report and Analysis")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="C:/fetal-data/HC18",
        help="Path to HC18 dataset directory",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="E:/BUBT-CSE-4th yr/Capstone/fm_B",
        help="Directory containing 5-fold checkpoints",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="docs/results",
        help="Directory to save report and predictions",
    )
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default=None,
        help="Optional path to existing oof_predictions.csv to analyze directly",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke test mode (5 images per fold)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Metric tolerance vs expected target (default: 0.02)",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        help="Number of patient-level bootstrap iterations (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def run_oof_inference(
    data_dir: str,
    model_dir: str,
    device: str | None = None,
    smoke: bool = False,
) -> pd.DataFrame:
    """Run TTA inference across all 5 folds on outer validation sets."""
    csv_path = Path(data_dir) / "training_set_pixel_size_and_HC.csv"
    img_dir = Path(data_dir) / "training_set"

    if not csv_path.exists() or not img_dir.exists():
        raise FileNotFoundError(f"HC18 training data not found at {data_dir}")

    dataset = HC18SegDataset(csv_path=str(csv_path), image_dir=str(img_dir), augment=False)
    folds = make_grouped_folds(dataset.names, n_splits=5, seed=42)

    dev = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {dev}")

    all_rows: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        fold_num = fold_idx + 1
        ckpt_path = Path(model_dir) / f"hc18_seg_fold{fold_num}_eff_b4_384.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint for fold {fold_num} missing: {ckpt_path}")

        print(f"\n--- Loading Fold {fold_num}/5 Checkpoint: {ckpt_path.name} ---")
        model = build_segmentation_model(
            encoder_name="efficientnet-b4",
            encoder_weights=None,
            classes=1,
        ).to(dev)
        state_dict = torch.load(str(ckpt_path), map_location=dev)
        model.load_state_dict(state_dict)
        model.eval()

        eval_indices = val_idx[:5] if smoke else val_idx
        print(f"Evaluating fold {fold_num} on {len(eval_indices)} outer val images...")

        for count, idx in enumerate(eval_indices, 1):
            name = dataset.names[idx]
            patient_id = patient_id_from_name(name)
            img_tensor, mask_tensor, pixel_size, scale = dataset[idx]

            prob_map = predict_prob_with_tta(model, img_tensor, device=dev)
            bio = measure_biometrics_from_prob(prob_map, mask_tensor, pixel_size, scale)

            if bio is None:
                print(f"Warning: could not fit ellipse for {name}, skipping")
                continue

            all_rows.append(
                {
                    "image": name,
                    "patient_id": patient_id,
                    "fold": fold_num,
                    "true_HC_mm": round(bio["true_hc"], 4),
                    "pred_HC_mm": round(bio["pred_hc"], 4),
                    "abs_err_HC": round(bio["abs_err_hc"], 4),
                    "true_BPD": round(bio["true_bpd"], 4),
                    "pred_BPD": round(bio["pred_bpd"], 4),
                    "abs_err_BPD": round(bio["abs_err_bpd"], 4),
                }
            )
            if count % 50 == 0 or count == len(eval_indices):
                print(f"  [{count}/{len(eval_indices)}] processed")

    return pd.DataFrame(all_rows)


def build_markdown_tercile_table(tercile_df: pd.DataFrame) -> str:
    """Format tercile analysis DataFrame into Markdown table."""
    lines = [
        "# HC18 Out-Of-Fold Error Analysis by Head-Size Terciles",
        "",
        "> Head-size proxy for gestational age (terciles of true ground-truth HC mm).",
        "",
        "| Tercile | Samples | True HC Range (mm) | HC MAE (mm) | HC RMSE (mm) | BPD MAE (mm) | BPD RMSE (mm) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for _, row in tercile_df.iterrows():
        t_name = row["tercile"]
        n = row["n_samples"]
        hc_range = f"{row['min_true_hc_mm']:.1f} - {row['max_true_hc_mm']:.1f}"
        hc_mae = f"{row['hc_mae']:.2f}"
        hc_rmse = f"{row['hc_rmse']:.2f}"
        bpd_mae = f"{row['bpd_mae']:.2f}"
        bpd_rmse = f"{row['bpd_rmse']:.2f}"
        lines.append(f"| {t_name} | {n} | {hc_range} | {hc_mae} | {hc_rmse} | {bpd_mae} | {bpd_rmse} |")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_out_path = out_dir / "oof_predictions.csv"
    metrics_out_path = out_dir / "metrics.json"
    tercile_out_path = out_dir / "tercile_analysis.md"

    if args.predictions_csv and Path(args.predictions_csv).exists():
        print(f"Loading predictions directly from {args.predictions_csv}...")
        df = pd.read_csv(args.predictions_csv)
    elif csv_out_path.exists() and not args.smoke:
        print(f"Found existing OOF predictions at {csv_out_path}, loading...")
        df = pd.read_csv(csv_out_path)
    else:
        print("Running out-of-fold inference across 5 folds...")
        df = run_oof_inference(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            device=args.device,
            smoke=args.smoke,
        )
        if not args.smoke:
            df.to_csv(csv_out_path, index=False)
            print(f"Saved OOF predictions to {csv_out_path}")

    # Compute per-fold metrics
    per_fold_summary = {}
    for f in sorted(df["fold"].unique()):
        f_df = df[df["fold"] == f]
        f_hc_mae = float(f_df["abs_err_HC"].mean())
        f_hc_rmse = float(np.sqrt(np.mean(f_df["abs_err_HC"] ** 2)))
        f_bpd_mae = float(f_df["abs_err_BPD"].mean())
        f_bpd_rmse = float(np.sqrt(np.mean(f_df["abs_err_BPD"] ** 2)))
        per_fold_summary[f"fold{f}"] = {
            "n_samples": len(f_df),
            "hc_mae": round(f_hc_mae, 4),
            "hc_rmse": round(f_hc_rmse, 4),
            "bpd_mae": round(f_bpd_mae, 4),
            "bpd_rmse": round(f_bpd_rmse, 4),
        }
        print(f"Fold {f}: HC MAE={f_hc_mae:.2f}mm, BPD MAE={f_bpd_mae:.2f}mm (n={len(f_df)})")

    # Pooled metrics
    hc_mae = float(df["abs_err_HC"].mean())
    hc_rmse = float(np.sqrt(np.mean(df["abs_err_HC"] ** 2)))
    bpd_mae = float(df["abs_err_BPD"].mean())
    bpd_rmse = float(np.sqrt(np.mean(df["abs_err_BPD"] ** 2)))

    print("\n" + "=" * 60)
    print("===== HC18 POOLED OUT-OF-FOLD METRICS =====")
    print(f"Total Evaluated: {len(df)} images")
    print(f"Pooled HC  MAE:  {hc_mae:.2f} mm (RMSE: {hc_rmse:.2f} mm)")
    print(f"Pooled BPD MAE:  {bpd_mae:.2f} mm (RMSE: {bpd_rmse:.2f} mm)")
    print("=" * 60)

    # Check against expected targets
    target = {
        "hc_mae": (2.30, hc_mae),
        "hc_rmse": (3.58, hc_rmse),
        "bpd_mae": (0.75, bpd_mae),
        "bpd_rmse": (1.09, bpd_rmse),
    }

    print("\nTolerance Check (target vs actual, tolerance ±" f"{args.tolerance:.2f}):")
    checks_passed = True
    for metric_name, (exp, act) in target.items():
        diff = abs(act - exp)
        status = "PASS" if diff <= args.tolerance else "FAIL"
        print(f"  {metric_name:<10}: target={exp:.2f}, actual={act:.2f}, diff={diff:.4f} -> [{status}]")
        if status == "FAIL" and not args.smoke:
            checks_passed = False

    if not checks_passed and not args.smoke:
        raise ValueError("OOF metrics deviate from target beyond tolerance!")

    # Tercile analysis
    print("\n--- Error Analysis by True-HC Terciles ('head-size proxy for gestational age') ---")
    tercile_df = compute_tercile_analysis(df)
    print(tercile_df.to_string(index=False))

    md_table = build_markdown_tercile_table(tercile_df)
    with open(tercile_out_path, "w", encoding="utf-8") as f:
        f.write(md_table)
    print(f"Wrote tercile analysis table to {tercile_out_path}")

    # Patient-level Bootstrap 95% CI
    print(f"\nComputing patient-level bootstrap 95% CI ({args.n_boot} resamples, seed={args.seed})...")
    ci_low, ci_high = compute_patient_bootstrap_ci(df, n_boot=args.n_boot, seed=args.seed)
    print(f"Pooled HC MAE 95% CI: [{ci_low:.3f}, {ci_high:.3f}] mm")

    # Metrics JSON
    metrics_dict = {
        "hc_mae": round(hc_mae, 4),
        "hc_rmse": round(hc_rmse, 4),
        "bpd_mae": round(bpd_mae, 4),
        "bpd_rmse": round(bpd_rmse, 4),
        "n_samples": int(len(df)),
        "bootstrap_95ci_hc_mae": [round(ci_low, 4), round(ci_high, 4)],
        "target_hc_mae": 2.30,
        "target_hc_rmse": 3.58,
        "target_bpd_mae": 0.75,
        "target_bpd_rmse": 1.09,
        "tolerance": args.tolerance,
        "per_fold": per_fold_summary,
    }

    with open(metrics_out_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Saved summary metrics to {metrics_out_path}")


if __name__ == "__main__":
    main()
