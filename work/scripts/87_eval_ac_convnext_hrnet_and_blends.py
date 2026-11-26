#!/usr/bin/env python3

# FIELD: AC
# MODEL: ConvNeXt-Tiny, HRNet-W32, and Heterogeneous Landmark U-Net Blends
# STEP: evaluation
# USED BY: work/docs/MODEL_SELECTION_TABLES.md, work/docs/report/PROOF_LOCATIONS.md

"""scripts/87_eval_ac_convnext_hrnet_and_blends.py

Evaluates ac-convnext-s42 (tu-convnext_tiny) and ac-hrnet-s42 (tu-hrnet_w32)
on folds_v1.csv (642 images, 501 patients) with paired patient-level cluster
bootstrap (5,000 resamples, seed 42) vs ac-baseline-s42 and vs the 4-seed
EfficientNet-B4 average (4.4028 mm).

Also evaluates three pre-specified heterogeneous OOF blends (fixed weights,
no weight search):
  a) 5-model equal mean: (s42 + s43 + s44 + s45 + hrnet) / 5
  b) 50/50 mean: (4seed_avg + hrnet) / 2
  c) 6-model equal mean: (s42 + s43 + s44 + s45 + hrnet + convnext) / 6
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fetal.eval.ac_evaluator import patient_level_paired_bootstrap

OUT_DIR = REPO_ROOT / "docs" / "results" / "ac_campaign"
FOUR_SEED_CSV = OUT_DIR / "oof_predictions_4seed_avg.csv"
CONVNEXT_DIR = Path(r"C:\fetal-data\kaggle_outputs\ac_convnext_s42_v2\ac-convnext-s42")
HRNET_DIR = Path(r"C:\fetal-data\kaggle_outputs\ac_hrnet_s42_v2\ac-hrnet-s42")


def compute_metrics(pred: pd.Series, true: pd.Series) -> dict:
    err = np.abs(pred - true)
    signed = pred - true
    # Note: oof_predictions_pooled.csv lacks 2D landmark coordinates, so Landmark NME
    # cannot be computed for averaged predictions; this is scalar AC relative error.
    return {
        "ac_mae": float(np.mean(err)),
        "ac_rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "ac_median": float(np.median(err)),
        "ac_bias": float(np.mean(signed)),
        "ac_rel_err": float(np.mean(err / np.maximum(true, 1e-6))),
        "over10": float(np.mean(err > 10.0)),
        "over10_pct": float(np.mean(err > 10.0) * 100.0),
    }


def run_boot(patient: pd.Series, true: pd.Series, pred_ref: pd.Series, pred_cand: pd.Series) -> dict:
    df = pd.DataFrame({
        "patient": patient,
        "ac_true": true,
        "ac_pred_a": pred_ref,
        "ac_pred_b": pred_cand,
    })
    res = patient_level_paired_bootstrap(df, n_bootstraps=5000, seed=42)
    ci_lo, ci_hi = res["ci95_delta_mae"]
    excludes_zero = bool((ci_hi < 0.0) or (ci_lo > 0.0))
    return {
        "delta_mae": float(res["delta_mae"]),
        "ci95_delta_mae": [float(ci_lo), float(ci_hi)],
        "p_val_mae": float(res["p_val_mae"]),
        "delta_rmse": float(res["delta_rmse"]),
        "ci95_delta_rmse": [float(res["ci95_delta_rmse"][0]), float(res["ci95_delta_rmse"][1])],
        "p_val_rmse": float(res["p_val_rmse"]),
        "ci_excludes_zero": excludes_zero,
    }


def main():
    df_4seed = pd.read_csv(FOUR_SEED_CSV)
    df_conv = pd.read_csv(CONVNEXT_DIR / "oof_predictions_pooled.csv")
    df_hrnet = pd.read_csv(HRNET_DIR / "oof_predictions_pooled.csv")

    with open(CONVNEXT_DIR / "oof_summary.json", "r", encoding="utf-8") as f:
        sum_conv = json.load(f)
    with open(HRNET_DIR / "oof_summary.json", "r", encoding="utf-8") as f:
        sum_hrnet = json.load(f)

    # Align by image_name
    df_conv = df_4seed[["image_name"]].merge(df_conv, on="image_name", how="left")
    df_hrnet = df_4seed[["image_name"]].merge(df_hrnet, on="image_name", how="left")

    assert len(df_4seed) == 642
    assert not df_conv["ac_pred"].isna().any()
    assert not df_hrnet["ac_pred"].isna().any()
    assert np.allclose(df_conv["ac_true"], df_4seed["ac_true"])
    assert np.allclose(df_hrnet["ac_true"], df_4seed["ac_true"])

    patient = df_4seed["patient"]
    ac_true = df_4seed["ac_true"]
    pred_s42 = df_4seed["ac_pred_s42"]
    pred_s43 = df_4seed["ac_pred_s43"]
    pred_s44 = df_4seed["ac_pred_s44"]
    pred_s45 = df_4seed["ac_pred_s45"]
    pred_4seed = df_4seed["ac_pred_4seed"]
    pred_conv = df_conv["ac_pred"]
    pred_hrnet = df_hrnet["ac_pred"]

    # Single models
    m_conv = compute_metrics(pred_conv, ac_true)
    m_hrnet = compute_metrics(pred_hrnet, ac_true)
    m_4seed = compute_metrics(pred_4seed, ac_true)

    boot_conv_vs_s42 = run_boot(patient, ac_true, pred_s42, pred_conv)
    boot_conv_vs_4seed = run_boot(patient, ac_true, pred_4seed, pred_conv)
    boot_hrnet_vs_s42 = run_boot(patient, ac_true, pred_s42, pred_hrnet)
    boot_hrnet_vs_4seed = run_boot(patient, ac_true, pred_4seed, pred_hrnet)

    # Heterogeneous blends (pre-specified fixed weights)
    pred_blend_a = (pred_s42 + pred_s43 + pred_s44 + pred_s45 + pred_hrnet) / 5.0
    pred_blend_b = (pred_4seed + pred_hrnet) / 2.0
    pred_blend_c = (pred_s42 + pred_s43 + pred_s44 + pred_s45 + pred_hrnet + pred_conv) / 6.0

    m_blend_a = compute_metrics(pred_blend_a, ac_true)
    m_blend_b = compute_metrics(pred_blend_b, ac_true)
    m_blend_c = compute_metrics(pred_blend_c, ac_true)

    boot_a_vs_4seed = run_boot(patient, ac_true, pred_4seed, pred_blend_a)
    boot_b_vs_4seed = run_boot(patient, ac_true, pred_4seed, pred_blend_b)
    boot_c_vs_4seed = run_boot(patient, ac_true, pred_4seed, pred_blend_c)

    print("=== Single Models (N=642, 501 patients) ===")
    for name, m in [("ac-4seed-avg", m_4seed), ("ac-convnext-s42", m_conv), ("ac-hrnet-s42", m_hrnet)]:
        print(f"{name:<18} | MAE {m['ac_mae']:.4f} | RMSE {m['ac_rmse']:.4f} | Med {m['ac_median']:.4f} | Bias {m['ac_bias']:+.4f} | AC RelErr {m['ac_rel_err']:.4f} | >10mm {m['over10_pct']:.2f}%")

    print("\n=== Single Models Paired Bootstrap ===")
    for label, b in [
        ("convnext vs s42", boot_conv_vs_s42),
        ("convnext vs 4seed", boot_conv_vs_4seed),
        ("hrnet vs s42", boot_hrnet_vs_s42),
        ("hrnet vs 4seed", boot_hrnet_vs_4seed),
    ]:
        print(f"{label:<20} | delta {b['delta_mae']:+.4f} mm | 95% CI [{b['ci95_delta_mae'][0]:+.4f}, {b['ci95_delta_mae'][1]:+.4f}] | p={b['p_val_mae']:.4f} | excl0={b['ci_excludes_zero']}")

    print("\n=== Heterogeneous Blends vs 4-Seed Average (4.4028 mm) ===")
    for name, m, b in [
        ("Blend a (5-model 1/5)", m_blend_a, boot_a_vs_4seed),
        ("Blend b (50/50 4seed+hrnet)", m_blend_b, boot_b_vs_4seed),
        ("Blend c (6-model 1/6)", m_blend_c, boot_c_vs_4seed),
    ]:
        print(f"{name:<28} | MAE {m['ac_mae']:.4f} | Med {m['ac_median']:.4f} | >10mm {m['over10_pct']:.2f}% | AC RelErr {m['ac_rel_err']:.4f} | delta {b['delta_mae']:+.4f} [{b['ci95_delta_mae'][0]:+.4f}, {b['ci95_delta_mae'][1]:+.4f}] | p={b['p_val_mae']:.4f} | excl0={b['ci_excludes_zero']}")

    # Save JSONs
    conv_payload = dict(sum_conv)
    conv_payload["oof_summary"] = dict(sum_conv)
    conv_payload["ac_rel_err"] = m_conv["ac_rel_err"]
    conv_payload["paired_bootstrap_vs_baseline_s42"] = boot_conv_vs_s42
    conv_payload["paired_bootstrap_vs_4seed_avg"] = boot_conv_vs_4seed
    with open(OUT_DIR / "ac_convnext_s42.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(conv_payload, f, indent=2)

    hrnet_payload = dict(sum_hrnet)
    hrnet_payload["oof_summary"] = dict(sum_hrnet)
    hrnet_payload["ac_rel_err"] = m_hrnet["ac_rel_err"]
    hrnet_payload["paired_bootstrap_vs_baseline_s42"] = boot_hrnet_vs_s42
    hrnet_payload["paired_bootstrap_vs_4seed_avg"] = boot_hrnet_vs_4seed
    with open(OUT_DIR / "ac_hrnet_s42.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(hrnet_payload, f, indent=2)

    blends_payload = {
        "description": "Pre-specified equal-weight heterogeneous OOF blends (no weight search) vs 4-seed EfficientNet-B4 average",
        "cohort": {"n_images": 642, "n_patients": 501, "protocol": "5-fold CV (folds_v1.csv)"},
        "reference_4seed_avg": m_4seed,
        "blends": {
            "blend_a_5model_equal": {
                "formula": "(s42 + s43 + s44 + s45 + hrnet) / 5",
                "metrics": m_blend_a,
                "vs_4seed_avg": boot_a_vs_4seed,
            },
            "blend_b_50_50": {
                "formula": "(4seed_avg + hrnet) / 2",
                "metrics": m_blend_b,
                "vs_4seed_avg": boot_b_vs_4seed,
            },
            "blend_c_6model_equal": {
                "formula": "(s42 + s43 + s44 + s45 + hrnet + convnext) / 6",
                "metrics": m_blend_c,
                "vs_4seed_avg": boot_c_vs_4seed,
            },
        },
    }
    with open(OUT_DIR / "ac_heterogeneous_blends.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(blends_payload, f, indent=2)

    print(f"\nSaved JSONs to {OUT_DIR}")


if __name__ == "__main__":
    main()
