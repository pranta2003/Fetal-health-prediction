#!/usr/bin/env python3
"""scripts/82_eval_ac_s43_and_wing_comparison.py

Evaluates ac-baseline-s43 and ac-wing-s42 5-fold out-of-fold predictions against
the baseline ac-baseline-s42 across identical 642 images (501 patients), and performs
paired patient-level cluster bootstrap (5,000 resamples, seed 42) for:
  a) baseline s43 vs baseline s42 (random seed noise)
  b) wing s42 vs baseline s42 (Wing loss impact)
  c) wing s42 vs baseline s43 (Wing loss vs alternative seed)
  d) 2-seed average (s42 + s43) vs baseline s42 (seed ensembling benefit)

Outputs:
- docs/results/ac_campaign/ac_baseline_s43.json
- docs/results/ac_campaign/ac_wing_s42.json
- docs/results/ac_campaign/ac_s43_wing_comparison.json
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
WING_DIR = Path(r"C:\fetal-data\kaggle_outputs\ac_wing_s42\ac-wing-s42")
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


def main():
    print("[1/5] Loading OOF predictions and patient folds...")
    df_s42 = load_oof(S42_DIR / "oof_predictions_pooled.csv", FOLDS_CSV)
    df_s43 = load_oof(S43_DIR / "oof_predictions_pooled.csv", FOLDS_CSV)
    df_wing = load_oof(WING_DIR / "oof_predictions_pooled.csv", FOLDS_CSV)

    assert len(df_s42) == 642, f"Expected 642 rows, got {len(df_s42)}"
    assert len(df_s43) == 642, f"Expected 642 rows, got {len(df_s43)}"
    assert len(df_wing) == 642, f"Expected 642 rows, got {len(df_wing)}"
    assert (df_s42["image_name"] == df_s43["image_name"]).all(), "Image ordering mismatch s42 vs s43"
    assert (df_s42["image_name"] == df_wing["image_name"]).all(), "Image ordering mismatch s42 vs wing"

    with open(S43_DIR / "oof_summary.json", "r", encoding="utf-8") as f:
        summary_s43 = json.load(f)
    with open(WING_DIR / "oof_summary.json", "r", encoding="utf-8") as f:
        summary_wing = json.load(f)

    # 2-seed average predictions
    pred_avg_2seed = (df_s42["ac_pred"] + df_s43["ac_pred"]) / 2.0
    err_avg = np.abs(pred_avg_2seed - df_s42["ac_true"])
    mae_avg = float(np.mean(err_avg))
    rmse_avg = float(np.sqrt(np.mean((pred_avg_2seed - df_s42["ac_true"]) ** 2)))
    med_avg = float(np.median(err_avg))
    nme_avg = float(np.mean(err_avg / np.maximum(df_s42["ac_true"], 1e-6)))
    over10_avg = float(np.mean(err_avg > 10.0))

    print("\n[2/5] Summary of 5-Fold OOF Predictions (N=642 images, 501 patients):")
    print(f"{'Model / Recipe':<25} | {'AC MAE (mm)':<12} | {'AC RMSE (mm)':<12} | {'Median (mm)':<12} | {'Bias (mm)':<10} | {'NME':<8} | {'>10mm (%)':<10}")
    print("-" * 105)
    print(f"{'ac-baseline-s42':<25} | {summary_s43.get('ac_mae_s42', 4.7744):<12.4f} | {7.6074:<12.4f} | {2.8474:<12.4f} | {-0.0826:<10.4f} | {0.0808:<8.4f} | {12.15:<10.2f}%")
    print(f"{'ac-baseline-s43':<25} | {summary_s43['ac_mae']:<12.4f} | {summary_s43['ac_rmse']:<12.4f} | {summary_s43['ac_median']:<12.4f} | {summary_s43['ac_bias']:<10.4f} | {summary_s43['nme']:<8.4f} | {summary_s43['over10']*100:<10.2f}%")
    print(f"{'ac-wing-s42':<25} | {summary_wing['ac_mae']:<12.4f} | {summary_wing['ac_rmse']:<12.4f} | {summary_wing['ac_median']:<12.4f} | {summary_wing['ac_bias']:<10.4f} | {summary_wing['nme']:<8.4f} | {summary_wing['over10']*100:<10.2f}%")
    print(f"{'2-seed avg (s42+s43)':<25} | {mae_avg:<12.4f} | {rmse_avg:<12.4f} | {med_avg:<12.4f} | {(summary_s43['ac_bias']-0.0826)/2:<10.4f} | {nme_avg:<8.4f} | {over10_avg*100:<10.2f}%")

    print("\n[3/5] Running Paired Patient-Level Bootstrap (5,000 resamples, seed 42)...")
    # Comparison a: baseline s43 vs baseline s42 (Seed Noise)
    m_a = pd.DataFrame({
        "patient": df_s42["patient"],
        "ac_true": df_s42["ac_true"],
        "ac_pred_a": df_s42["ac_pred"],
        "ac_pred_b": df_s43["ac_pred"],
    })
    boot_a = patient_level_paired_bootstrap(m_a, n_bootstraps=5000, seed=42)

    # Comparison b: wing s42 vs baseline s42
    m_b = pd.DataFrame({
        "patient": df_s42["patient"],
        "ac_true": df_s42["ac_true"],
        "ac_pred_a": df_s42["ac_pred"],
        "ac_pred_b": df_wing["ac_pred"],
    })
    boot_b = patient_level_paired_bootstrap(m_b, n_bootstraps=5000, seed=42)

    # Comparison c: wing s42 vs baseline s43
    m_c = pd.DataFrame({
        "patient": df_s42["patient"],
        "ac_true": df_s42["ac_true"],
        "ac_pred_a": df_s43["ac_pred"],
        "ac_pred_b": df_wing["ac_pred"],
    })
    boot_c = patient_level_paired_bootstrap(m_c, n_bootstraps=5000, seed=42)

    # Comparison d: 2-seed avg vs baseline s42
    m_d = pd.DataFrame({
        "patient": df_s42["patient"],
        "ac_true": df_s42["ac_true"],
        "ac_pred_a": df_s42["ac_pred"],
        "ac_pred_b": pred_avg_2seed,
    })
    boot_d = patient_level_paired_bootstrap(m_d, n_bootstraps=5000, seed=42)

    print(f"  [a] s43 vs s42:     delta MAE = {boot_a['delta_mae']:+.4f} mm, 95% CI [{boot_a['ci95_delta_mae'][0]:+.4f}, {boot_a['ci95_delta_mae'][1]:+.4f}], p = {boot_a['p_val_mae']:.4f}")
    print(f"  [b] wing vs s42:    delta MAE = {boot_b['delta_mae']:+.4f} mm, 95% CI [{boot_b['ci95_delta_mae'][0]:+.4f}, {boot_b['ci95_delta_mae'][1]:+.4f}], p = {boot_b['p_val_mae']:.4f}")
    print(f"  [c] wing vs s43:    delta MAE = {boot_c['delta_mae']:+.4f} mm, 95% CI [{boot_c['ci95_delta_mae'][0]:+.4f}, {boot_c['ci95_delta_mae'][1]:+.4f}], p = {boot_c['p_val_mae']:.4f}")
    print(f"  [d] 2-seed vs s42:  delta MAE = {boot_d['delta_mae']:+.4f} mm, 95% CI [{boot_d['ci95_delta_mae'][0]:+.4f}, {boot_d['ci95_delta_mae'][1]:+.4f}], p = {boot_d['p_val_mae']:.4f}")

    print("\n[4/5] Saving result JSON files...")
    # 1. ac_baseline_s43.json
    s43_payload = dict(summary_s43)
    s43_payload["oof_summary"] = dict(summary_s43)
    s43_payload["paired_bootstrap_vs_baseline_s42"] = {
        "delta_mae": boot_a["delta_mae"],
        "ci95_delta_mae": boot_a["ci95_delta_mae"],
        "p_val_mae": boot_a["p_val_mae"],
        "delta_rmse": boot_a["delta_rmse"],
        "ci95_delta_rmse": boot_a["ci95_delta_rmse"],
        "p_val_rmse": boot_a["p_val_rmse"],
    }
    s43_path = OUT_DIR / "ac_baseline_s43.json"
    with open(s43_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(s43_payload, f, indent=2)
    print(f"  Saved {s43_path}")

    # 2. ac_wing_s42.json
    wing_payload = dict(summary_wing)
    wing_payload["oof_summary"] = dict(summary_wing)
    wing_payload["paired_bootstrap_vs_baseline_s42"] = {
        "delta_mae": boot_b["delta_mae"],
        "ci95_delta_mae": boot_b["ci95_delta_mae"],
        "p_val_mae": boot_b["p_val_mae"],
        "delta_rmse": boot_b["delta_rmse"],
        "ci95_delta_rmse": boot_b["ci95_delta_rmse"],
        "p_val_rmse": boot_b["p_val_rmse"],
    }
    wing_path = OUT_DIR / "ac_wing_s42.json"
    with open(wing_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(wing_payload, f, indent=2)
    print(f"  Saved {wing_path}")

    # 3. ac_s43_wing_comparison.json
    comp_payload = {
        "description": "Patient-level paired cluster bootstrap comparison (5000 resamples, seed 42) for AC models",
        "cohort": {"n_images": 642, "n_patients": 501, "protocol": "5-fold cross-validation on folds_v1.csv"},
        "models": {
            "ac_baseline_s42": {
                "ac_mae": 4.7744,
                "ac_rmse": 7.6074,
                "ac_median": 2.8474,
                "ac_bias": -0.0826,
                "nme": 0.0808,
                "over10_pct": 12.15,
                "seed": 42,
            },
            "ac_baseline_s43": {
                "ac_mae": round(summary_s43["ac_mae"], 4),
                "ac_rmse": round(summary_s43["ac_rmse"], 4),
                "ac_median": round(summary_s43["ac_median"], 4),
                "ac_bias": round(summary_s43["ac_bias"], 4),
                "nme": round(summary_s43["nme"], 4),
                "over10_pct": round(summary_s43["over10"] * 100, 2),
                "seed": 43,
            },
            "ac_wing_s42": {
                "ac_mae": round(summary_wing["ac_mae"], 4),
                "ac_rmse": round(summary_wing["ac_rmse"], 4),
                "ac_median": round(summary_wing["ac_median"], 4),
                "ac_bias": round(summary_wing["ac_bias"], 4),
                "nme": round(summary_wing["nme"], 4),
                "over10_pct": round(summary_wing["over10"] * 100, 2),
                "seed": 42,
            },
            "ac_2seed_avg_s42_s43": {
                "ac_mae": round(mae_avg, 4),
                "ac_rmse": round(rmse_avg, 4),
                "ac_median": round(med_avg, 4),
                "ac_bias": round((summary_s43["ac_bias"] - 0.0826) / 2, 4),
                "nme": round(nme_avg, 4),
                "over10_pct": round(over10_avg * 100, 2),
            },
        },
        "comparisons": {
            "a_baseline_s43_vs_baseline_s42_seed_noise": {
                "comparison": "baseline s43 vs baseline s42 (random seed noise)",
                "delta_mae": boot_a["delta_mae"],
                "ci95_delta_mae": boot_a["ci95_delta_mae"],
                "p_val_mae": boot_a["p_val_mae"],
                "delta_rmse": boot_a["delta_rmse"],
                "ci95_delta_rmse": boot_a["ci95_delta_rmse"],
                "p_val_rmse": boot_a["p_val_rmse"],
            },
            "b_wing_s42_vs_baseline_s42": {
                "comparison": "wing s42 vs baseline s42",
                "delta_mae": boot_b["delta_mae"],
                "ci95_delta_mae": boot_b["ci95_delta_mae"],
                "p_val_mae": boot_b["p_val_mae"],
                "delta_rmse": boot_b["delta_rmse"],
                "ci95_delta_rmse": boot_b["ci95_delta_rmse"],
                "p_val_rmse": boot_b["p_val_rmse"],
            },
            "c_wing_s42_vs_baseline_s43": {
                "comparison": "wing s42 vs baseline s43",
                "delta_mae": boot_c["delta_mae"],
                "ci95_delta_mae": boot_c["ci95_delta_mae"],
                "p_val_mae": boot_c["p_val_mae"],
                "delta_rmse": boot_c["delta_rmse"],
                "ci95_delta_rmse": boot_c["ci95_delta_rmse"],
                "p_val_rmse": boot_c["p_val_rmse"],
            },
            "d_2seed_avg_vs_baseline_s42": {
                "comparison": "mean of baseline s42 and s43 vs baseline s42",
                "delta_mae": boot_d["delta_mae"],
                "ci95_delta_mae": boot_d["ci95_delta_mae"],
                "p_val_mae": boot_d["p_val_mae"],
                "delta_rmse": boot_d["delta_rmse"],
                "ci95_delta_rmse": boot_d["ci95_delta_rmse"],
                "p_val_rmse": boot_d["p_val_rmse"],
            },
        },
    }
    comp_path = OUT_DIR / "ac_s43_wing_comparison.json"
    with open(comp_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(comp_payload, f, indent=2)
    print(f"  Saved {comp_path}")
    print("\n[5/5] Done.")


if __name__ == "__main__":
    main()
