"""scripts/40_ac_compare_recipes.py

Compares two AC prediction sets (OOF or Test) with patient-level paired bootstrap
(5,000 resamples, seed 42) for AC MAE and RMSE delta.

Usage:
    python scripts/40_ac_compare_recipes.py --rec-a <path_or_dir_a> --rec-b <path_or_dir_b> [--name-a A] [--name-b B]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
src_dir = REPO_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def load_prediction_df(path_or_dir: Path | str, folds_csv: Path | str | None = None) -> pd.DataFrame:
    p = Path(path_or_dir)
    if p.is_dir():
        candidates = [
            p / "oof_predictions_all.csv",
            p / "oof_predictions.csv",
            p / "test_predictions.csv",
            p / "predictions.csv",
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break
        if found is None:
            csvs = list(p.glob("*.csv"))
            if csvs:
                found = csvs[0]
            else:
                raise FileNotFoundError(f"No CSV predictions found in directory: {p}")
        p = found

    df = pd.read_csv(p)
    # Standardize column names
    col_map = {
        "image": "image_name",
        "filename": "image_name",
        "pred": "ac_pred",
        "true": "ac_true",
    }
    for old_col, new_col in col_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df = df.rename(columns={old_col: new_col})

    if "image_name" not in df.columns:
        raise ValueError(f"Missing image identifier column in {p}. Available: {list(df.columns)}")

    # Add patient column if missing
    if "patient" not in df.columns and "patient_id" in df.columns:
        df["patient"] = df["patient_id"]

    if "patient" not in df.columns:
        if folds_csv is not None and Path(folds_csv).exists():
            f_df = pd.read_csv(folds_csv)
            f_col = "filename" if "filename" in f_df.columns else "image_name"
            if f_col in f_df.columns and "patient" in f_df.columns:
                m = dict(zip(f_df[f_col], f_df["patient"]))
                df["patient"] = df["image_name"].map(m)

    # Fallback patient extraction if still missing
    if "patient" not in df.columns or df["patient"].isna().any():
        def extract_p(img: str) -> str:
            if str(img).startswith("Patient"):
                return "FP_" + str(img).split("_")[0]
            elif "HC" in str(img):
                return "UCL_" + str(img).split(".")[0]
            return "P_" + str(img)

        df["patient"] = df["patient"].fillna(df["image_name"].map(extract_p))

    return df


from fetal.eval.ac_evaluator import compute_metrics, patient_level_paired_bootstrap


def compare_recipes(
    path_a: Path | str,
    path_b: Path | str,
    name_a: str = "Recipe A",
    name_b: str = "Recipe B",
    folds_csv: Path | str | None = None,
    n_bootstraps: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    df_a = load_prediction_df(path_a, folds_csv=folds_csv)
    df_b = load_prediction_df(path_b, folds_csv=folds_csv)

    merged = pd.merge(
        df_a[["image_name", "patient", "ac_true", "ac_pred"]].rename(columns={"ac_pred": "ac_pred_a"}),
        df_b[["image_name", "ac_pred"]].rename(columns={"ac_pred": "ac_pred_b"}),
        on="image_name",
    )

    if len(merged) == 0:
        raise ValueError("No common images found between the two prediction sets.")

    metrics_a = compute_metrics(merged["ac_true"].to_numpy(), merged["ac_pred_a"].to_numpy())
    metrics_b = compute_metrics(merged["ac_true"].to_numpy(), merged["ac_pred_b"].to_numpy())
    boot_res = patient_level_paired_bootstrap(merged, n_bootstraps=n_bootstraps, seed=seed)

    result = {
        "name_a": name_a,
        "name_b": name_b,
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "bootstrap_comparison": boot_res,
    }
    return result


def print_comparison_table(res: dict[str, Any]) -> None:
    na = res["name_a"]
    nb = res["name_b"]
    ma = res["metrics_a"]
    mb = res["metrics_b"]
    bc = res["bootstrap_comparison"]

    print("\n================================================================================")
    print(f"AC Recipe Comparison: '{nb}' vs '{na}' (N={bc['n_images']} images, {bc['n_patients']} patients)")
    print("================================================================================")
    print(f"{'Metric':<20} | {na:<18} | {nb:<18} | {'Delta (' + nb + ' - ' + na + ')':<25}")
    print("-" * 88)
    print(f"{'AC MAE (mm)':<20} | {ma['mae']:<18.4f} | {mb['mae']:<18.4f} | {bc['delta_mae']:+.4f} (95% CI: [{bc['ci95_delta_mae'][0]:+.4f}, {bc['ci95_delta_mae'][1]:+.4f}], p={bc['p_val_mae']:.4f})")
    print(f"{'AC RMSE (mm)':<20} | {ma['rmse']:<18.4f} | {mb['rmse']:<18.4f} | {bc['delta_rmse']:+.4f} (95% CI: [{bc['ci95_delta_rmse'][0]:+.4f}, {bc['ci95_delta_rmse'][1]:+.4f}], p={bc['p_val_rmse']:.4f})")
    print(f"{'Median Err (mm)':<20} | {ma['median']:<18.4f} | {mb['median']:<18.4f} | {mb['median'] - ma['median']:+.4f}")
    print(f"{'NME':<20} | {ma['nme']:<18.4f} | {mb['nme']:<18.4f} | {mb['nme'] - ma['nme']:+.4f}")
    print(f"{'>10 mm Outliers':<20} | {ma['gt10_rate_pct']:<5.2f}% ({ma['gt10_count']}){'':<7} | {mb['gt10_rate_pct']:<5.2f}% ({mb['gt10_count']}){'':<7} | {mb['gt10_rate_pct'] - ma['gt10_rate_pct']:+.2f}%")
    print("================================================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two AC prediction recipes with patient-level paired bootstrap.")
    parser.add_argument("--rec-a", required=True, help="Path to Recipe A predictions CSV or directory")
    parser.add_argument("--rec-b", required=True, help="Path to Recipe B predictions CSV or directory")
    parser.add_argument("--name-a", default="Recipe A", help="Name/Label for Recipe A")
    parser.add_argument("--name-b", default="Recipe B", help="Name/Label for Recipe B")
    parser.add_argument("--folds-csv", default=str(REPO_ROOT / "docs" / "results" / "ac_campaign" / "folds_v1.csv"), help="Path to folds_v1.csv for patient lookup")
    parser.add_argument("--n-bootstrap", type=int, default=5000, help="Number of bootstrap resamples (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap (default: 42)")
    parser.add_argument("--out-json", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    res = compare_recipes(
        path_a=args.rec_a,
        path_b=args.rec_b,
        name_a=args.name_a,
        name_b=args.name_b,
        folds_csv=args.folds_csv,
        n_bootstraps=args.n_bootstrap,
        seed=args.seed,
    )
    print_comparison_table(res)

    if args.out_json:
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Saved results JSON to {out_p}")


if __name__ == "__main__":
    main()
