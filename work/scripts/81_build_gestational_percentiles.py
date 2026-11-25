#!/usr/bin/env python3

# FIELD: Fusion
# MODEL: Hadlock Formula / CDC Growth Charts
# STEP: data_prep
# USED BY: work/docs/CODE_MAP.md, work/docs/report/PROOF_LOCATIONS.md

"""scripts/81_build_gestational_percentiles.py

Builds gestational-age-specific empirical birth weight percentiles (p10, p50, p90)
from CDC NVSS vital statistics for the Biometric Growth Bridge (Plan A).

Rules & Inclusion Criteria:
1. Data source: CDC NVSS natality records (C:/fetal-data/cdc/clinical_training_data.csv).
2. Training cohort: birth years <= 2021 only (2016-2021). Years >= 2022 are preserved
   as untouched held-out temporal evaluation data.
3. Outcome status: LIVE births only (OUTCOME == 0). Stillbirths and fetal deaths
   (OUTCOME == 1) are strictly excluded from normal growth reference curves.
4. Gestational age: COMBGEST in completed weeks [20, 42]. Sentinel 99 (unknown)
   and out-of-range (<20 or >42) are excluded.
5. Birth weight: DBWT in grams. Implausible/missing values excluded:
   - DBWT != 9999 (CDC sentinel for unknown / not stated)
   - 227 <= DBWT <= 8165 g (standard CDC NVSS plausible birth weight range).

Outputs:
- docs/results/fusion/gestational_percentiles.csv
- docs/results/fusion/gestational_percentiles_summary.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_CSV = Path("C:/fetal-data/cdc/clinical_training_data.csv")
OUTPUT_DIR = REPO_ROOT / "docs" / "results" / "fusion"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Build CDC gestational age birth weight percentiles.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_DATA_CSV),
        help="Path to clinical_training_data.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory to save percentiles CSV and summary JSON",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CDC CSV not found: {input_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading CDC natality dataset from {input_path}...")
    usecols = ["YEAR", "OUTCOME", "COMBGEST", "DBWT"]
    df = pd.read_csv(input_path, usecols=usecols, low_memory=False)
    total_raw_rows = len(df)
    print(f"      Total raw rows loaded: {total_raw_rows:,}")

    # Step 1: Filter training cohort (YEAR <= 2021)
    train_mask = df["YEAR"] <= 2021
    n_train_cohort = int(train_mask.sum())

    # Step 2: Filter live births only (OUTCOME == 0)
    live_mask = train_mask & (df["OUTCOME"] == 0)
    n_live_pre2022 = int(live_mask.sum())

    # Step 3: Filter gestational age range (20 <= COMBGEST <= 42)
    # Ensure COMBGEST is numeric
    combgest_numeric = pd.to_numeric(df["COMBGEST"], errors="coerce")
    ga_mask = live_mask & combgest_numeric.between(20, 42)
    n_ga_valid = int(ga_mask.sum())

    # Step 4: Filter plausible birth weights (DBWT != 9999 and 227 <= DBWT <= 8165)
    dbwt_numeric = pd.to_numeric(df["DBWT"], errors="coerce")
    wt_valid_mask = ga_mask & (dbwt_numeric != 9999) & dbwt_numeric.between(227, 8165)
    n_final_valid = int(wt_valid_mask.sum())

    dropped_missing_wt = n_ga_valid - n_final_valid

    print(f"[2/4] Cohort filtering breakdown:")
    print(f"      - Training years (<= 2021): {n_train_cohort:,} (dropped {total_raw_rows - n_train_cohort:,} test records >= 2022)")
    print(f"      - Live births (OUTCOME == 0): {n_live_pre2022:,} (dropped {n_train_cohort - n_live_pre2022:,} stillbirths)")
    print(f"      - Valid GA (20 <= COMBGEST <= 42): {n_ga_valid:,} (dropped {n_live_pre2022 - n_ga_valid:,} invalid/out-of-range GA)")
    print(f"      - Plausible weight (227 <= DBWT <= 8165 g, not 9999): {n_final_valid:,} (dropped {dropped_missing_wt:,} missing weights)")

    # Prepare analysis subset
    sub = pd.DataFrame({
        "COMBGEST": combgest_numeric[wt_valid_mask].astype(int),
        "DBWT": dbwt_numeric[wt_valid_mask].astype(float),
    })

    # Group by week 20 to 42
    print(f"[3/4] Computing percentiles per completed week (20 to 42)...")
    records = []
    summary_per_week = {}

    for week in range(20, 43):
        w_df = sub[sub["COMBGEST"] == week]["DBWT"]
        n_w = len(w_df)
        if n_w == 0:
            raise ValueError(f"No records found for gestational week {week}")

        p10_val = round(float(np.percentile(w_df, 10)), 1)
        p50_val = round(float(np.percentile(w_df, 50)), 1)
        p90_val = round(float(np.percentile(w_df, 90)), 1)
        mean_val = round(float(w_df.mean()), 1)
        std_val = round(float(w_df.std()), 1)

        records.append({
            "ga_weeks": week,
            "COMBGEST": week,
            "n": n_w,
            "p10": p10_val,
            "p50": p50_val,
            "p90": p90_val,
            "mean": mean_val,
            "std": std_val,
        })

        summary_per_week[str(week)] = {
            "n": n_w,
            "p10_g": p10_val,
            "p50_g": p50_val,
            "p90_g": p90_val,
            "mean_g": mean_val,
            "std_g": std_val,
        }

    pct_df = pd.DataFrame(records)

    # Save CSV
    csv_path = out_dir / "gestational_percentiles.csv"
    pct_df.to_csv(csv_path, index=False, lineterminator="\n")
    csv_sha = sha256_file(csv_path)
    print(f"      Saved CSV -> {csv_path} (SHA-256: {csv_sha})")

    # Build summary JSON
    summary_data = {
        "dataset_source": str(input_path),
        "dataset_sha256": sha256_file(input_path) if input_path.stat().st_size < 300_000_000 else None,
        "rules": {
            "years": "<= 2021 (training years 2016-2021; >= 2022 held-out)",
            "live_births_only": "OUTCOME == 0 (stillbirths and fetal deaths OUTCOME == 1 excluded)",
            "gestational_age": "COMBGEST in [20, 42] completed weeks (unknown code 99 and extreme terms excluded)",
            "birth_weight_rule": "DBWT != 9999 and 227 <= DBWT <= 8165 (grams; CDC NVSS plausible valid birth weight range)",
        },
        "row_counts": {
            "total_raw_rows": total_raw_rows,
            "train_cohort_rows": n_train_cohort,
            "live_births_pre2022": n_live_pre2022,
            "valid_ga_rows": n_ga_valid,
            "valid_weight_rows": n_final_valid,
            "dropped_missing_weight": dropped_missing_wt,
            "final_reference_rows": n_final_valid,
        },
        "weeks_covered": {"min_week": 20, "max_week": 42, "num_weeks": 23},
        "csv_sha256": csv_sha,
        "per_week": summary_per_week,
    }

    json_path = out_dir / "gestational_percentiles_summary.json"
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary_data, f, indent=2)
    json_sha = sha256_file(json_path)
    print(f"      Saved summary JSON -> {json_path} (SHA-256: {json_sha})")

    print("\n[4/4] Gestational Age Reference Distribution (Weeks 20-42):")
    print(f"{'GA (wk)':<8} {'N':<8} {'p10 (g)':<10} {'p50 (g)':<10} {'p90 (g)':<10} {'Mean (g)':<10} {'Std (g)':<10}")
    print("-" * 68)
    for r in records:
        print(f"{r['ga_weeks']:<8} {r['n']:<8} {r['p10']:<10.1f} {r['p50']:<10.1f} {r['p90']:<10.1f} {r['mean']:<10.1f} {r['std']:<10.1f}")


if __name__ == "__main__":
    main()
