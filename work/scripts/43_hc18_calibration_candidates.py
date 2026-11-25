# FIELD: HC18
# MODEL: EfficientNet-B4 U-Net / Post-hoc Transform
# STEP: calibration
# USED BY: work/docs/CODE_MAP.md

"""scripts/43_hc18_calibration_candidates.py

Computes damped multiplicative post-hoc calibration candidates (strengths 0.25, 0.5, 1.0)
on HC18 OOF predictions and generates EXPLORATORY test submission CSVs from the
corrected-angle ensemble submission file.

DO NOT SUBMIT ANY GENERATED CANDIDATE CSV TO ANY CHALLENGE.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def compute_damped_calibration(
    oof_csv: Path | str,
    submission_csv: Path | str,
    out_dir: Path | str,
    strengths: list[float] | None = None,
) -> dict:
    if strengths is None:
        strengths = [0.0, 0.25, 0.5, 0.75, 1.0]

    oof_path = Path(oof_csv)
    sub_path = Path(submission_csv)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_oof = pd.read_csv(oof_path)
    df_sub = pd.read_csv(sub_path)

    # 1. Full-dataset median ratio: true / pred
    full_ratio = float(np.median(df_oof["true_HC_mm"] / df_oof["pred_HC_mm"]))
    delta_full = full_ratio - 1.0

    results = {
        "dataset": "HC18",
        "n_oof_images": len(df_oof),
        "n_oof_patients": int(df_oof["patient_id"].nunique()),
        "full_median_ratio": round(full_ratio, 6),
        "delta_full": round(delta_full, 6),
        "strengths": {},
        "generated_csvs": {},
    }

    print("\n================================================================================")
    print("HC18 Damped Multiplicative Calibration Candidates (Nested 5-Fold OOF)")
    print("================================================================================")
    print(f"{'Strength (alpha)':<18} | {'Multiplicative Ratio R':<22} | {'Nested OOF MAE (mm)':<20} | {'MAE Delta vs Raw'}")
    print("-" * 80)

    for alpha in strengths:
        # Nested 5-fold CV evaluation
        preds_nested = np.zeros(len(df_oof), dtype=np.float64)
        for fold in range(1, 6):
            train_mask = df_oof["fold"] != fold
            val_mask = df_oof["fold"] == fold
            r_fold = float(np.median(df_oof.loc[train_mask, "true_HC_mm"] / df_oof.loc[train_mask, "pred_HC_mm"]))
            r_damped = 1.0 + alpha * (r_fold - 1.0)
            preds_nested[val_mask] = df_oof.loc[val_mask, "pred_HC_mm"] * r_damped

        err = np.abs(preds_nested - df_oof["true_HC_mm"])
        sq_err = (preds_nested - df_oof["true_HC_mm"]) ** 2
        signed_err_a = df_oof["true_HC_mm"] - preds_nested  # Convention A: True - Pred
        signed_err_b = preds_nested - df_oof["true_HC_mm"]  # Convention B: Pred - True (Official van den Heuvel DF)

        mae = float(np.mean(err))
        rmse = float(np.sqrt(np.mean(sq_err)))
        med_ae = float(np.median(err))

        # Full-dataset damped ratio for test application
        r_test = 1.0 + alpha * delta_full

        results["strengths"][f"alpha_{alpha:.2f}"] = {
            "alpha": alpha,
            "r_test": round(r_test, 6),
            "nested_oof_mae": round(mae, 4),
            "nested_oof_rmse": round(rmse, 4),
            "nested_oof_median_ae": round(med_ae, 4),
            "signed_bias_convention_a_mean": round(float(np.mean(signed_err_a)), 4),
            "signed_bias_convention_b_mean": round(float(np.mean(signed_err_b)), 4),
        }

        raw_mae = results["strengths"]["alpha_0.00"]["nested_oof_mae"] if "alpha_0.00" in results["strengths"] else mae
        delta_str = f"{mae - raw_mae:+.4f} mm" if alpha > 0 else "Baseline"
        print(f"alpha = {alpha:<10.2f} | R = {r_test:<20.6f} | {mae:<20.4f} | {delta_str}")

    print("================================================================================\n")

    # 2. Generate candidate test submission CSVs for alpha = 0.50 and alpha = 1.00
    for alpha in [0.50, 1.00]:
        r_scale = 1.0 + alpha * delta_full
        df_candidate = df_sub.copy()
        # Scale semi-axes a and b; keep center_x, center_y, and angle_rad intact
        df_candidate["semi_axes_a_mm"] = df_candidate["semi_axes_a_mm"] * r_scale
        df_candidate["semi_axes_b_mm"] = df_candidate["semi_axes_b_mm"] * r_scale

        pct_label = int(round(alpha * 100))
        fname = f"submission_ensemble_5fold_corrected_mult{pct_label:03d}_EXPLORATORY.csv"
        out_csv_path = out_path / fname
        df_candidate.to_csv(out_csv_path, index=False, lineterminator="\n")

        # Compute SHA-256 (normalized LF)
        content = out_csv_path.read_bytes().replace(b"\r\n", b"\n")
        sha = hashlib.sha256(content).hexdigest()
        results["generated_csvs"][fname] = {
            "path": str(out_csv_path),
            "alpha": alpha,
            "scale_factor_R": round(r_scale, 6),
            "sha256": sha,
            "rows": len(df_candidate),
        }
        print(f"Generated candidate: {fname} (R={r_scale:.6f}, SHA256={sha})")

    # Write candidates SHA256SUMS
    sha_lines = [f"{info['sha256']}  {fname}" for fname, info in results["generated_csvs"].items()]
    sha_path = out_path / "SHA256SUMS"
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sha_lines) + "\n")
    print(f"Saved checksums to {sha_path}")

    # Write results summary JSON
    summary_path = out_path / "candidates_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved candidate summary JSON to {summary_path}")

    return results


if __name__ == "__main__":
    compute_damped_calibration(
        oof_csv=REPO_ROOT / "docs" / "results" / "hc18" / "oof_predictions.csv",
        submission_csv=REPO_ROOT / "docs" / "results" / "hc18_official" / "submission_ensemble_5fold_corrected.csv",
        out_dir=REPO_ROOT / "docs" / "results" / "hc18_calibration" / "candidates",
    )
