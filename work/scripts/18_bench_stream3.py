"""18_bench_stream3.py: Benchmark Stream 3 clinical risk models under identical protocol.

Compares:
1. Logistic Regression (StandardScaler on numeric, class_weight='balanced')
2. Random Forest (RandomForestClassifier, max_depth=12, fixed seed)
3. LightGBM (LGBMClassifier, same trees/depth/subsample as XGBoost)
4. Our XGBoost (fetal.clinical.model.make_pipe, exact reproduced settings)

Settings evaluated:
- E1: Primary features, >= 20 weeks gestation (temporal split: train <= 2021, test >= 2022)
- E4: Clinical-only features, >= 20 weeks gestation (temporal split)

Outputs JSON and Markdown tables into docs/results/stream3_bench/.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from fetal import config
from fetal.clinical.constants import FEATURE_SETS, OUTCOME_TIME_VARS
from fetal.clinical.data import load_clean
from fetal.clinical.eval import bootstrap_auc_ci
from fetal.clinical.model import make_pipe


def build_models(
    num_cols: list[str],
    cat_cols: list[str],
    random_state: int = 42,
    smoke: bool = False,
) -> dict[str, Pipeline]:
    """Build preprocessing and model pipelines for all benchmark candidates."""
    # Guard: check for forbidden outcome-time variables
    all_features = set(num_cols + cat_cols)
    leakage = all_features & set(OUTCOME_TIME_VARS)
    if leakage:
        raise ValueError(f"Forbidden outcome-time variables detected in feature set: {leakage}")

    # Standard tree preprocessing: median imputation for num, most_frequent + OHE for cat
    parts_tree = [("n", SimpleImputer(strategy="median"), num_cols)]
    if cat_cols:
        parts_tree.append(
            (
                "c",
                Pipeline(
                    [
                        ("i", SimpleImputer(strategy="most_frequent")),
                        ("o", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )

    # Logistic Regression preprocessing: median + StandardScaler for num, most_frequent + OHE for cat
    parts_lr = [
        (
            "n",
            Pipeline(
                [
                    ("i", SimpleImputer(strategy="median")),
                    ("s", StandardScaler()),
                ]
            ),
            num_cols,
        )
    ]
    if cat_cols:
        parts_lr.append(
            (
                "c",
                Pipeline(
                    [
                        ("i", SimpleImputer(strategy="most_frequent")),
                        ("o", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )

    models: dict[str, Pipeline] = {}

    # 1. Logistic Regression
    max_iter = 50 if smoke else 1000
    lr_pipe = Pipeline(
        [
            ("pre", ColumnTransformer(parts_lr)),
            (
                "m",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=max_iter,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    models["Logistic Regression"] = lr_pipe

    # 2. Random Forest
    # Hyperparameter selection: max_depth=16 was selected using only an internal validation
    # split strictly within training years (train <= 2019, val 2020-2021), without touching the test set.
    rf_est = 10 if smoke else 100
    rf_depth = 6 if smoke else 16
    rf_pipe = Pipeline(
        [
            ("pre", ColumnTransformer(parts_tree)),
            (
                "m",
                RandomForestClassifier(
                    n_estimators=rf_est,
                    max_depth=rf_depth,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    models["Random Forest"] = rf_pipe

    # 3. LightGBM (if available)
    if HAS_LIGHTGBM:
        lgb_est = 10 if smoke else 300
        lgb_depth = 5
        lgb_pipe = Pipeline(
            [
                ("pre", ColumnTransformer(parts_tree)),
                (
                    "m",
                    lgb.LGBMClassifier(
                        n_estimators=lgb_est,
                        max_depth=lgb_depth,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=random_state,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        )
        models["LightGBM"] = lgb_pipe

    # 4. Our XGBoost (repo model, exact reproduced hyperparameters)
    xgb_est = 10 if smoke else 300
    xgb_pipe = make_pipe(num_cols, cat_cols, n_estimators=xgb_est, random_state=random_state)
    models["Our XGBoost"] = xgb_pipe

    return models


def run_setting_benchmark(
    setting_name: str,
    df_data: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    n_boot: int = 200,
    random_state: int = 42,
    smoke: bool = False,
) -> list[dict]:
    """Train and evaluate all models on a single setting under temporal split."""
    # Strict temporal split: train <= 2021, test >= 2022
    tr = df_data[df_data.YEAR <= 2021].copy()
    te = df_data[df_data.YEAR >= 2022].copy()

    n_train = len(tr)
    n_test = len(te)
    n_train_pos = int(tr.OUTCOME.sum())
    n_test_pos = int(te.OUTCOME.sum())

    features = num_cols + cat_cols
    X_tr, y_tr = tr[features], tr.OUTCOME.values
    X_te, y_te = te[features], te.OUTCOME.values

    models = build_models(num_cols, cat_cols, random_state=random_state, smoke=smoke)
    setting_results = []

    print(f"\n--- Setting: {setting_name} ---")
    print(f"  Train: {n_train} rows ({n_train_pos} stillbirths) | Test: {n_test} rows ({n_test_pos} stillbirths)")
    print(f"  Features ({len(features)}): {features}")

    for model_name, pipe in models.items():
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        fit_time = time.time() - t0

        p = pipe.predict_proba(X_te)[:, 1]
        auc = float(roc_auc_score(y_te, p))

        t_ci = time.time()
        ci_low, ci_high = bootstrap_auc_ci(y_te, p, n_boot=n_boot, seed=0)
        ci_time = time.time() - t_ci

        record = {
            "setting": setting_name,
            "model": model_name,
            "train_rows": n_train,
            "test_rows": n_test,
            "train_pos": n_train_pos,
            "test_pos": n_test_pos,
            "test_auc": round(auc, 4),
            "ci_95_low": round(ci_low, 4),
            "ci_95_high": round(ci_high, 4),
            "fit_time_sec": round(fit_time, 2),
            "total_time_sec": round(fit_time + ci_time, 2),
        }
        setting_results.append(record)
        print(
            f"  {model_name:22s} | AUC: {auc:.4f} [95% CI: {ci_low:.4f}, {ci_high:.4f}] | Fit: {fit_time:.1f}s"
        )

    return setting_results


def build_markdown_table(results: list[dict]) -> str:
    """Format benchmark results as a GitHub Flavored Markdown table."""
    lines = [
        "# Stream 3 Clinical Risk Model Benchmark",
        "",
        "Comparison of model families under the exact same protocol: CDC natality/stillbirth data, temporal split (Train <= 2021, Test >= 2022), no post-delivery leakage.",
        "",
        "| Setting | Model | Train Rows | Test Rows | Test AUC | 95% Bootstrap CI | Fit Time (s) |",
        "|:---|:---|---:|---:|---:|:---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['setting']} | {r['model']} | {r['train_rows']:,} | {r['test_rows']:,} | **{r['test_auc']:.4f}** | [{r['ci_95_low']:.4f}, {r['ci_95_high']:.4f}] | {r['fit_time_sec']:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Stream 3 Clinical Model Benchmark")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(config.CLINICAL_CSV),
        help="Path to clinical dataset CSV",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(config.REPO_ROOT) / "docs" / "results" / "stream3_bench"),
        help="Directory where results will be written",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="Quick smoke verification mode",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=200,
        help="Number of bootstrap iterations for 95%% CI (default: 200)",
    )
    return parser.parse_args()


def main():
    total_start = time.time()
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)

    print("=" * 75)
    print("Stream 3 Clinical Tabular Risk Model Benchmark (Comparison Study)")
    print(f"CSV Path:       {csv_path}")
    print(f"Out Directory:  {out_dir}")
    print(f"LightGBM ready: {HAS_LIGHTGBM}")
    print(f"Smoke Mode:     {args.smoke}")
    print("=" * 75)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Clinical dataset not found at {csv_path}")

    # Step 1: Load and clean
    print("\n[Step 1] Loading and cleaning dataset...")
    raw, df = load_clean(csv_path)
    print(f"  Loaded raw: {raw.shape} | Deduped: {df.shape} (Removed: {len(raw) - len(df)})")

    d20 = df[df.COMBGEST >= 20].copy()
    print(f"  Gestation >= 20 weeks cohort: {len(d20)} rows")

    # Step 2: Run benchmarks
    n_boot = 10 if args.smoke else args.n_boot
    all_results: list[dict] = []

    # Setting E1: Primary features
    e1_results = run_setting_benchmark(
        setting_name="E1 (Primary, >=20 wk)",
        df_data=d20,
        num_cols=FEATURE_SETS["primary"]["num"],
        cat_cols=FEATURE_SETS["primary"]["cat"],
        n_boot=n_boot,
        random_state=42,
        smoke=args.smoke,
    )
    all_results.extend(e1_results)

    # Setting E4: Clinical-only features
    e4_results = run_setting_benchmark(
        setting_name="E4 (Clinical-Only, >=20 wk)",
        df_data=d20,
        num_cols=FEATURE_SETS["clinical_only"]["num"],
        cat_cols=FEATURE_SETS["clinical_only"]["cat"],
        n_boot=n_boot,
        random_state=42,
        smoke=args.smoke,
    )
    all_results.extend(e4_results)

    # Step 3: Verify reproduction of known XGBoost results
    if not args.smoke:
        xgb_e1 = next(r for r in e1_results if r["model"] == "Our XGBoost")
        xgb_e4 = next(r for r in e4_results if r["model"] == "Our XGBoost")

        # Known targets: E1 0.738, E4 0.707 (diff 0.0000)
        e1_diff = abs(round(xgb_e1["test_auc"], 3) - 0.738)
        e4_diff = abs(round(xgb_e4["test_auc"], 3) - 0.707)

        print("\n--- Verification of Known Results ---")
        print(f"  E1 XGBoost AUC: {xgb_e1['test_auc']:.4f} (target 0.738, diff: {e1_diff:.4f})")
        print(f"  E4 XGBoost AUC: {xgb_e4['test_auc']:.4f} (target 0.707, diff: {e4_diff:.4f})")

        if e1_diff > 1e-4 or e4_diff > 1e-4:
            raise RuntimeError(
                f"FATAL: XGBoost reproduced results do not match expected targets! E1 diff={e1_diff}, E4 diff={e4_diff}"
            )
        print("  [PASSED] Exact reproduction verified (diff 0.0000).")

    # Step 4: Write JSON and Markdown outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "benchmark_results.json"
    md_path = out_dir / "benchmark_table.md"

    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "protocol": "CDC Natality + Stillbirths, temporal split (train <= 2021, test >= 2022)",
                "smoke": args.smoke,
                "n_boot": n_boot,
                "results": all_results,
            },
            f,
            indent=2,
        )

    md_content = build_markdown_table(all_results)
    with open(md_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(md_content)

    print(f"\nSaved benchmark JSON: {json_path}")
    print(f"Saved benchmark MD:   {md_path}")

    total_time = time.time() - total_start
    print(f"\nBenchmark completed successfully in {total_time:.2f} seconds.")
    print("\n" + md_content)


if __name__ == "__main__":
    main()
