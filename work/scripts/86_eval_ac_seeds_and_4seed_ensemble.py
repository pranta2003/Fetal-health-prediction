#!/usr/bin/env python3
"""scripts/86_eval_ac_seeds_and_4seed_ensemble.py

Evaluates ac-baseline-s44 and ac-baseline-s45 5-fold out-of-fold predictions against
baseline ac-baseline-s42 across identical 642 images (501 patients), constructs the
4-seed average ensemble (s42, s43, s44, s45), and performs paired patient-level
cluster bootstrap (5,000 resamples, seed 42) for:
  a) 4-seed average vs baseline s42
  b) 4-seed average vs 2-seed average (s42 + s43)
  c) baseline s44 vs baseline s42
  d) baseline s45 vs baseline s42
  (and baseline s43 vs baseline s42 for reference)

Outputs:
- docs/results/ac_campaign/ac_baseline_s44.json
- docs/results/ac_campaign/ac_baseline_s45.json
- docs/results/ac_campaign/ac_4seed_comparison.json
- docs/results/ac_campaign/oof_predictions_4seed_avg.csv
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fetal.eval.ac_evaluator import patient_level_paired_bootstrap

# Input paths
S42_DIR = Path(r"E:\BUBT-CSE-4th yr\Capstone\kaggle_outputs\ac_baseline_s42\ac-baseline-s42")
S43_DIR = Path(r"C:\fetal-data\kaggle_outputs\ac_baseline_s43\ac-baseline-s43")
S44_DIR = Path(r"C:\fetal-data\kaggle_outputs\ac_baseline_s44\ac-baseline-s44")
S45_DIR = Path(r"C:\fetal-data\kaggle_outputs\ac_baseline_s45\ac-baseline-s45")
FOLDS_CSV = REPO_ROOT / "docs" / "results" / "ac_campaign" / "folds_v1.csv"
OUT_DIR = REPO_ROOT / "docs" / "results" / "ac_campaign"


def load_oof(oof_csv_path: Path, folds_path: Path) -> pd.DataFrame:
    df = pd.read_csv(oof_csv_path)
    folds_df = pd.read_csv(folds_path)
    p_map = dict(zip(folds_df["filename"], folds_df["patient"]))
    df["patient"] = df["image_name"].map(p_map)
    if df["patient"].isna().any():
        raise ValueError(f"Missing patient mapping in {oof_csv_path}")
    return df


def compute_metrics(pred: pd.Series, true: pd.Series) -> dict:
    err = np.abs(pred - true)
    signed = pred - true
    return {
        "ac_mae": float(np.mean(err)),
        "ac_rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "ac_median": float(np.median(err)),
        "ac_bias": float(np.mean(signed)),
        "nme": float(np.mean(err / np.maximum(true, 1e-6))),
        "over10": float(np.mean(err > 10.0)),
        "over10_pct": float(np.mean(err > 10.0) * 100.0),
    }


def main():
    print("[1/5] Loading OOF predictions from all 4 seeds and patient folds...")
    df_s42 = load_oof(S42_DIR / "oof_predictions_pooled.csv", FOLDS_CSV)
    df_s43 = load_oof(S43_DIR / "oof_predictions_pooled.csv", FOLDS_CSV)
    df_s44 = load_oof(S44_DIR / "oof_predictions_pooled.csv", FOLDS_CSV)
    df_s45 = load_oof(S45_DIR / "oof_predictions_pooled.csv", FOLDS_CSV)

    for name, df in [("s42", df_s42), ("s43", df_s43), ("s44", df_s44), ("s45", df_s45)]:
        assert len(df) == 642, f"Expected 642 rows in {name}, got {len(df)}"
        assert (df["image_name"] == df_s42["image_name"]).all(), f"Image ordering mismatch in {name}"
        assert np.allclose(df["ac_true"], df_s42["ac_true"]), f"Ground truth mismatch in {name}"

    with open(S44_DIR / "oof_summary.json", "r", encoding="utf-8") as f:
        summary_s44 = json.load(f)
    with open(S45_DIR / "oof_summary.json", "r", encoding="utf-8") as f:
        summary_s45 = json.load(f)
    with open(S43_DIR / "oof_summary.json", "r", encoding="utf-8") as f:
        summary_s43 = json.load(f)

    # Predictions
    pred_2seed = (df_s42["ac_pred"] + df_s43["ac_pred"]) / 2.0
    pred_4seed = (df_s42["ac_pred"] + df_s43["ac_pred"] + df_s44["ac_pred"] + df_s45["ac_pred"]) / 4.0

    # Save 4-seed pooled predictions
    df_4seed_out = df_s42[["image_name", "patient", "source", "ac_true"]].copy()
    df_4seed_out["ac_pred_s42"] = df_s42["ac_pred"]
    df_4seed_out["ac_pred_s43"] = df_s43["ac_pred"]
    df_4seed_out["ac_pred_s44"] = df_s44["ac_pred"]
    df_4seed_out["ac_pred_s45"] = df_s45["ac_pred"]
    df_4seed_out["ac_pred_2seed"] = pred_2seed
    df_4seed_out["ac_pred_4seed"] = pred_4seed
    df_4seed_out["ac_err_4seed"] = np.abs(pred_4seed - df_s42["ac_true"])
    df_4seed_out["ac_signed_4seed"] = pred_4seed - df_s42["ac_true"]
    pred_4seed_csv = OUT_DIR / "oof_predictions_4seed_avg.csv"
    df_4seed_out.to_csv(pred_4seed_csv, index=False)
    print(f"  Saved 4-seed OOF predictions to {pred_4seed_csv}")

    # Metrics
    m_s42 = compute_metrics(df_s42["ac_pred"], df_s42["ac_true"])
    m_s43 = compute_metrics(df_s43["ac_pred"], df_s43["ac_true"])
    m_s44 = compute_metrics(df_s44["ac_pred"], df_s44["ac_true"])
    m_s45 = compute_metrics(df_s45["ac_pred"], df_s45["ac_true"])
    m_2seed = compute_metrics(pred_2seed, df_s42["ac_true"])
    m_4seed = compute_metrics(pred_4seed, df_s42["ac_true"])

    print("\n[2/5] Summary of 5-Fold OOF Predictions (N=642 images, 501 patients):")
    print(f"{'Model / Ensemble':<25} | {'AC MAE (mm)':<12} | {'AC RMSE (mm)':<12} | {'Median (mm)':<12} | {'Bias (mm)':<10} | {'NME':<8} | {'>10mm (%)':<10}")
    print("-" * 105)
    print(f"{'ac-baseline-s42':<25} | {m_s42['ac_mae']:<12.4f} | {m_s42['ac_rmse']:<12.4f} | {m_s42['ac_median']:<12.4f} | {m_s42['ac_bias']:<10.4f} | {m_s42['nme']:<8.4f} | {m_s42['over10_pct']:<10.2f}%")
    print(f"{'ac-baseline-s43':<25} | {m_s43['ac_mae']:<12.4f} | {m_s43['ac_rmse']:<12.4f} | {m_s43['ac_median']:<12.4f} | {m_s43['ac_bias']:<10.4f} | {m_s43['nme']:<8.4f} | {m_s43['over10_pct']:<10.2f}%")
    print(f"{'ac-baseline-s44':<25} | {m_s44['ac_mae']:<12.4f} | {m_s44['ac_rmse']:<12.4f} | {m_s44['ac_median']:<12.4f} | {m_s44['ac_bias']:<10.4f} | {m_s44['nme']:<8.4f} | {m_s44['over10_pct']:<10.2f}%")
    print(f"{'ac-baseline-s45':<25} | {m_s45['ac_mae']:<12.4f} | {m_s45['ac_rmse']:<12.4f} | {m_s45['ac_median']:<12.4f} | {m_s45['ac_bias']:<10.4f} | {m_s45['nme']:<8.4f} | {m_s45['over10_pct']:<10.2f}%")
    print(f"{'2-seed avg (s42+s43)':<25} | {m_2seed['ac_mae']:<12.4f} | {m_2seed['ac_rmse']:<12.4f} | {m_2seed['ac_median']:<12.4f} | {m_2seed['ac_bias']:<10.4f} | {m_2seed['nme']:<8.4f} | {m_2seed['over10_pct']:<10.2f}%")
    print(f"{'4-seed avg (s42..s45)':<25} | {m_4seed['ac_mae']:<12.4f} | {m_4seed['ac_rmse']:<12.4f} | {m_4seed['ac_median']:<12.4f} | {m_4seed['ac_bias']:<10.4f} | {m_4seed['nme']:<8.4f} | {m_4seed['over10_pct']:<10.2f}%")

    print("\n[3/5] Running Paired Patient-Level Bootstrap (5,000 resamples, seed 42)...")
    # Comparison 1: 4-seed avg vs s42
    df_4_vs_42 = pd.DataFrame({
        "patient": df_s42["patient"],
        "ac_true": df_s42["ac_true"],
        "ac_pred_a": df_s42["ac_pred"],
        "ac_pred_b": pred_4seed,
    })
    boot_4_vs_42 = patient_level_paired_bootstrap(df_4_vs_42, n_bootstraps=5000, seed=42)

    # Comparison 2: 4-seed avg vs 2-seed avg
    df_4_vs_2 = pd.DataFrame({
        "patient": df_s42["patient"],
        "ac_true": df_s42["ac_true"],
        "ac_pred_a": pred_2seed,
        "ac_pred_b": pred_4seed,
    })
    boot_4_vs_2 = patient_level_paired_bootstrap(df_4_vs_2, n_bootstraps=5000, seed=42)

    # Comparison 3: s44 vs s42
    df_44_vs_42 = pd.DataFrame({
        "patient": df_s42["patient"],
        "ac_true": df_s42["ac_true"],
        "ac_pred_a": df_s42["ac_pred"],
        "ac_pred_b": df_s44["ac_pred"],
    })
    boot_44_vs_42 = patient_level_paired_bootstrap(df_44_vs_42, n_bootstraps=5000, seed=42)

    # Comparison 4: s45 vs s42
    df_45_vs_42 = pd.DataFrame({
        "patient": df_s42["patient"],
        "ac_true": df_s42["ac_true"],
        "ac_pred_a": df_s42["ac_pred"],
        "ac_pred_b": df_s45["ac_pred"],
    })
    boot_45_vs_42 = patient_level_paired_bootstrap(df_45_vs_42, n_bootstraps=5000, seed=42)

    # Comparison 5: s43 vs s42 (for consistency check)
    df_43_vs_42 = pd.DataFrame({
        "patient": df_s42["patient"],
        "ac_true": df_s42["ac_true"],
        "ac_pred_a": df_s42["ac_pred"],
        "ac_pred_b": df_s43["ac_pred"],
    })
    boot_43_vs_42 = patient_level_paired_bootstrap(df_43_vs_42, n_bootstraps=5000, seed=42)

    print(f"  [1] 4-seed avg vs s42:     delta MAE = {boot_4_vs_42['delta_mae']:+.4f} mm, 95% CI [{boot_4_vs_42['ci95_delta_mae'][0]:+.4f}, {boot_4_vs_42['ci95_delta_mae'][1]:+.4f}], p = {boot_4_vs_42['p_val_mae']:.4f}")
    print(f"  [2] 4-seed avg vs 2-seed:    delta MAE = {boot_4_vs_2['delta_mae']:+.4f} mm, 95% CI [{boot_4_vs_2['ci95_delta_mae'][0]:+.4f}, {boot_4_vs_2['ci95_delta_mae'][1]:+.4f}], p = {boot_4_vs_2['p_val_mae']:.4f}")
    print(f"  [3] s44 vs s42:            delta MAE = {boot_44_vs_42['delta_mae']:+.4f} mm, 95% CI [{boot_44_vs_42['ci95_delta_mae'][0]:+.4f}, {boot_44_vs_42['ci95_delta_mae'][1]:+.4f}], p = {boot_44_vs_42['p_val_mae']:.4f}")
    print(f"  [4] s45 vs s42:            delta MAE = {boot_45_vs_42['delta_mae']:+.4f} mm, 95% CI [{boot_45_vs_42['ci95_delta_mae'][0]:+.4f}, {boot_45_vs_42['ci95_delta_mae'][1]:+.4f}], p = {boot_45_vs_42['p_val_mae']:.4f}")
    print(f"  [5] s43 vs s42 (ref):      delta MAE = {boot_43_vs_42['delta_mae']:+.4f} mm, 95% CI [{boot_43_vs_42['ci95_delta_mae'][0]:+.4f}, {boot_43_vs_42['ci95_delta_mae'][1]:+.4f}], p = {boot_43_vs_42['p_val_mae']:.4f}")

    print("\n[4/5] Saving JSON outputs...")
    # 1. ac_baseline_s44.json
    s44_payload = dict(summary_s44)
    s44_payload["oof_summary"] = dict(summary_s44)
    s44_payload["paired_bootstrap_vs_baseline_s42"] = {
        "delta_mae": boot_44_vs_42["delta_mae"],
        "ci95_delta_mae": boot_44_vs_42["ci95_delta_mae"],
        "p_val_mae": boot_44_vs_42["p_val_mae"],
        "delta_rmse": boot_44_vs_42["delta_rmse"],
        "ci95_delta_rmse": boot_44_vs_42["ci95_delta_rmse"],
        "p_val_rmse": boot_44_vs_42["p_val_rmse"],
    }
    s44_path = OUT_DIR / "ac_baseline_s44.json"
    with open(s44_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(s44_payload, f, indent=2)
    print(f"  Saved {s44_path}")

    # 2. ac_baseline_s45.json
    s45_payload = dict(summary_s45)
    s45_payload["oof_summary"] = dict(summary_s45)
    s45_payload["paired_bootstrap_vs_baseline_s42"] = {
        "delta_mae": boot_45_vs_42["delta_mae"],
        "ci95_delta_mae": boot_45_vs_42["ci95_delta_mae"],
        "p_val_mae": boot_45_vs_42["p_val_mae"],
        "delta_rmse": boot_45_vs_42["delta_rmse"],
        "ci95_delta_rmse": boot_45_vs_42["ci95_delta_rmse"],
        "p_val_rmse": boot_45_vs_42["p_val_rmse"],
    }
    s45_path = OUT_DIR / "ac_baseline_s45.json"
    with open(s45_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(s45_payload, f, indent=2)
    print(f"  Saved {s45_path}")

    # 3. ac_4seed_comparison.json
    comp_payload = {
        "description": "Patient-level paired cluster bootstrap comparison (5000 resamples, seed 42) for 4-seed AC baseline models and ensembles",
        "cohort": {"n_images": 642, "n_patients": 501, "protocol": "5-fold cross-validation on folds_v1.csv"},
        "models": {
            "ac_baseline_s42": {
                "ac_mae": round(m_s42["ac_mae"], 4),
                "ac_rmse": round(m_s42["ac_rmse"], 4),
                "ac_median": round(m_s42["ac_median"], 4),
                "ac_bias": round(m_s42["ac_bias"], 4),
                "nme": round(m_s42["nme"], 4),
                "over10_pct": round(m_s42["over10_pct"], 2),
                "seed": 42,
            },
            "ac_baseline_s43": {
                "ac_mae": round(m_s43["ac_mae"], 4),
                "ac_rmse": round(m_s43["ac_rmse"], 4),
                "ac_median": round(m_s43["ac_median"], 4),
                "ac_bias": round(m_s43["ac_bias"], 4),
                "nme": round(m_s43["nme"], 4),
                "over10_pct": round(m_s43["over10_pct"], 2),
                "seed": 43,
            },
            "ac_baseline_s44": {
                "ac_mae": round(m_s44["ac_mae"], 4),
                "ac_rmse": round(m_s44["ac_rmse"], 4),
                "ac_median": round(m_s44["ac_median"], 4),
                "ac_bias": round(m_s44["ac_bias"], 4),
                "nme": round(m_s44["nme"], 4),
                "over10_pct": round(m_s44["over10_pct"], 2),
                "seed": 44,
            },
            "ac_baseline_s45": {
                "ac_mae": round(m_s45["ac_mae"], 4),
                "ac_rmse": round(m_s45["ac_rmse"], 4),
                "ac_median": round(m_s45["ac_median"], 4),
                "ac_bias": round(m_s45["ac_bias"], 4),
                "nme": round(m_s45["nme"], 4),
                "over10_pct": round(m_s45["over10_pct"], 2),
                "seed": 45,
            },
            "ac_2seed_avg_s42_s43": {
                "ac_mae": round(m_2seed["ac_mae"], 4),
                "ac_rmse": round(m_2seed["ac_rmse"], 4),
                "ac_median": round(m_2seed["ac_median"], 4),
                "ac_bias": round(m_2seed["ac_bias"], 4),
                "nme": round(m_2seed["nme"], 4),
                "over10_pct": round(m_2seed["over10_pct"], 2),
                "seeds": [42, 43],
            },
            "ac_4seed_avg_s42_s45": {
                "ac_mae": round(m_4seed["ac_mae"], 4),
                "ac_rmse": round(m_4seed["ac_rmse"], 4),
                "ac_median": round(m_4seed["ac_median"], 4),
                "ac_bias": round(m_4seed["ac_bias"], 4),
                "nme": round(m_4seed["nme"], 4),
                "over10_pct": round(m_4seed["over10_pct"], 2),
                "seeds": [42, 43, 44, 45],
            },
        },
        "comparisons": {
            "4seed_avg_vs_s42": {
                "comparison": "4-seed average (s42..s45) vs baseline s42",
                "delta_mae": boot_4_vs_42["delta_mae"],
                "ci95_delta_mae": boot_4_vs_42["ci95_delta_mae"],
                "p_val_mae": boot_4_vs_42["p_val_mae"],
                "delta_rmse": boot_4_vs_42["delta_rmse"],
                "ci95_delta_rmse": boot_4_vs_42["ci95_delta_rmse"],
                "p_val_rmse": boot_4_vs_42["p_val_rmse"],
            },
            "4seed_avg_vs_2seed_avg": {
                "comparison": "4-seed average vs 2-seed average (s42 + s43)",
                "delta_mae": boot_4_vs_2["delta_mae"],
                "ci95_delta_mae": boot_4_vs_2["ci95_delta_mae"],
                "p_val_mae": boot_4_vs_2["p_val_mae"],
                "delta_rmse": boot_4_vs_2["delta_rmse"],
                "ci95_delta_rmse": boot_4_vs_2["ci95_delta_rmse"],
                "p_val_rmse": boot_4_vs_2["p_val_rmse"],
            },
            "s44_vs_s42": {
                "comparison": "baseline s44 vs baseline s42",
                "delta_mae": boot_44_vs_42["delta_mae"],
                "ci95_delta_mae": boot_44_vs_42["ci95_delta_mae"],
                "p_val_mae": boot_44_vs_42["p_val_mae"],
                "delta_rmse": boot_44_vs_42["delta_rmse"],
                "ci95_delta_rmse": boot_44_vs_42["ci95_delta_rmse"],
                "p_val_rmse": boot_44_vs_42["p_val_rmse"],
            },
            "s45_vs_s42": {
                "comparison": "baseline s45 vs baseline s42",
                "delta_mae": boot_45_vs_42["delta_mae"],
                "ci95_delta_mae": boot_45_vs_42["ci95_delta_mae"],
                "p_val_mae": boot_45_vs_42["p_val_mae"],
                "delta_rmse": boot_45_vs_42["delta_rmse"],
                "ci95_delta_rmse": boot_45_vs_42["ci95_delta_rmse"],
                "p_val_rmse": boot_45_vs_42["p_val_rmse"],
            },
            "s43_vs_s42_reference": {
                "comparison": "baseline s43 vs baseline s42",
                "delta_mae": boot_43_vs_42["delta_mae"],
                "ci95_delta_mae": boot_43_vs_42["ci95_delta_mae"],
                "p_val_mae": boot_43_vs_42["p_val_mae"],
                "delta_rmse": boot_43_vs_42["delta_rmse"],
                "ci95_delta_rmse": boot_43_vs_42["ci95_delta_rmse"],
                "p_val_rmse": boot_43_vs_42["p_val_rmse"],
            },
        },
    }
    comp_path = OUT_DIR / "ac_4seed_comparison.json"
    with open(comp_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(comp_payload, f, indent=2)
    print(f"  Saved {comp_path}")
    print("\n[5/5] All evaluations and bootstrap comparisons complete.")


if __name__ == "__main__":
    main()
