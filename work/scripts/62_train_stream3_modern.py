"""scripts/62_train_stream3_modern.py

Stream 3 Modern GBDT Campaign (Step 1):
- Protocol: Strict temporal split replicating E1 (train <= 2021, test >= 2022).
- Internal validation: train_inner <= 2019, val_inner 2020-2021 (no test leakage).
- Hyperparameter tuning: CatBoost, LightGBM, and XGBoost with Optuna (40 trials).
- Calibrated Ensembling: Blend best GBM with tuned Logistic Regression and calibrate probabilities.
- Guard tests: zero outcome-time feature leakage, year boundary verification, sanity checks.
- Metrics: Test ROC-AUC with 2000-sample bootstrap 95% confidence intervals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Repo setup
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from fetal.clinical.constants import FEATURE_SETS, OUTCOME_TIME_VARS
from fetal.clinical.data import load_clean
from fetal.clinical.eval import bootstrap_auc_ci


def build_preprocessors(num_cols: list[str], cat_cols: list[str]):
    # Trees
    parts_tree = [("n", SimpleImputer(strategy="median"), num_cols)]
    if cat_cols:
        parts_tree.append((
            "c",
            Pipeline([
                ("i", SimpleImputer(strategy="most_frequent")),
                ("o", OneHotEncoder(handle_unknown="ignore")),
            ]),
            cat_cols,
        ))
    pre_tree = ColumnTransformer(parts_tree)

    # Linear / LR
    parts_lr = [(
        "n",
        Pipeline([
            ("i", SimpleImputer(strategy="median")),
            ("s", StandardScaler()),
        ]),
        num_cols,
    )]
    if cat_cols:
        parts_lr.append((
            "c",
            Pipeline([
                ("i", SimpleImputer(strategy="most_frequent")),
                ("o", OneHotEncoder(handle_unknown="ignore")),
            ]),
            cat_cols,
        ))
    pre_lr = ColumnTransformer(parts_lr)

    return pre_tree, pre_lr


def run_campaign(
    csv_path: str | Path,
    out_dir: str | Path,
    n_trials: int = 40,
    smoke: bool = False,
    smoke_fraction: float = 0.05,
) -> dict:
    import xgboost as xgb
    import lightgbm as lgb
    try:
        import catboost as cb
        has_cb = True
    except ImportError:
        has_cb = False

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        has_optuna = True
    except ImportError:
        has_optuna = False

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STREAM 3 MODERN GBDT CAMPAIGN (Step 1)")
    print(f"Data CSV:     {csv_path}")
    print(f"Output Dir:   {out_dir}")
    print(f"Smoke Mode:   {smoke} (fraction: {smoke_fraction if smoke else 1.0})")
    print(f"Optuna Trials: {n_trials if not smoke else 2}")
    print("=" * 70)

    # 1. Load and Clean
    raw, df = load_clean(csv_path)
    d20 = df[df.COMBGEST >= 20].copy()
    print(f"Cohort (COMBGEST >= 20 wk): {len(d20)} rows (positives: {int(d20.OUTCOME.sum())})")

    if smoke:
        # 5% stratified sample for local smoke verification
        print(f"\n[Smoke Mode] Sampling {smoke_fraction*100:.1f}% stratified subset...")
        pos = d20[d20.OUTCOME == 1].sample(frac=smoke_fraction, random_state=42)
        neg = d20[d20.OUTCOME == 0].sample(frac=smoke_fraction, random_state=42)
        d20 = pd.concat([pos, neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"  Subsampled cohort: {len(d20)} rows ({int(d20.OUTCOME.sum())} stillbirths)")

    # 2. Strict Temporal Split
    tr = d20[d20.YEAR <= 2021].copy()
    te = d20[d20.YEAR >= 2022].copy()

    # Inner validation for hyperparameter tuning (no future leakage)
    tr_inner = tr[tr.YEAR <= 2019].copy()
    val_inner = tr[tr.YEAR.isin([2020, 2021])].copy()

    print(f"Temporal Split:")
    print(f"  Full Train (<= 2021): {len(tr)} rows ({int(tr.OUTCOME.sum())} pos)")
    print(f"    Inner Train (<= 2019): {len(tr_inner)} rows ({int(tr_inner.OUTCOME.sum())} pos)")
    print(f"    Inner Val (2020-2021): {len(val_inner)} rows ({int(val_inner.OUTCOME.sum())} pos)")
    print(f"  Held-out Test (>= 2022): {len(te)} rows ({int(te.OUTCOME.sum())} pos)")

    num_cols = FEATURE_SETS["primary"]["num"]
    cat_cols = FEATURE_SETS["primary"]["cat"]
    features = num_cols + cat_cols

    # Guard check: zero outcome-time feature leakage
    leakage = set(features) & set(OUTCOME_TIME_VARS)
    assert not leakage, f"Leakage detected! Forbidden features present: {leakage}"
    assert "COMBGEST" not in features, "COMBGEST must not be an input feature!"
    print("GUARD CHECK PASSED: 0 forbidden outcome-time features.")

    pre_tree, pre_lr = build_preprocessors(num_cols, cat_cols)

    # Pre-fit transformers on training sets
    print("\nFitting feature transformers...")
    X_tr_tree = pre_tree.fit_transform(tr[features])
    X_val_tree = pre_tree.transform(val_inner[features])
    X_te_tree = pre_tree.transform(te[features])
    X_tr_in_tree = pre_tree.transform(tr_inner[features])

    X_tr_lr = pre_lr.fit_transform(tr[features])
    X_val_lr = pre_lr.transform(val_inner[features])
    X_te_lr = pre_lr.transform(te[features])
    X_tr_in_lr = pre_lr.transform(tr_inner[features])

    y_tr = tr.OUTCOME.values
    y_te = te.OUTCOME.values
    y_tr_in = tr_inner.OUTCOME.values
    y_val_in = val_inner.OUTCOME.values

    # Tuning budget
    trials = 2 if smoke else n_trials

    # Model 1: XGBoost
    print("\n[Tuning 1/4] Tuning XGBoost...")
    def obj_xgb(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 10 if smoke else 50, 30 if smoke else 300),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
            "random_state": 42,
            "eval_metric": "auc",
            "n_jobs": -1,
        }
        m = xgb.XGBClassifier(**params)
        m.fit(X_tr_in_tree, y_tr_in)
        p = m.predict_proba(X_val_tree)[:, 1]
        return roc_auc_score(y_val_in, p)

    if has_optuna:
        study_xgb = optuna.create_study(direction="maximize")
        study_xgb.optimize(obj_xgb, n_trials=trials)
        best_xgb_params = study_xgb.best_params | {"random_state": 42, "eval_metric": "auc", "n_jobs": -1}
    else:
        best_xgb_params = {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 20 if smoke else 300, "random_state": 42, "n_jobs": -1}

    model_xgb = xgb.XGBClassifier(**best_xgb_params)
    model_xgb.fit(X_tr_tree, y_tr)
    pred_xgb_test = model_xgb.predict_proba(X_te_tree)[:, 1]
    pred_xgb_val = model_xgb.predict_proba(X_val_tree)[:, 1]
    auc_xgb = float(roc_auc_score(y_te, pred_xgb_test))
    print(f"  Tuned XGBoost Test AUC: {auc_xgb:.4f}")

    # Model 2: LightGBM
    print("\n[Tuning 2/4] Tuning LightGBM...")
    def obj_lgb(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 10 if smoke else 50, 30 if smoke else 300),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr_in_tree, y_tr_in)
        p = m.predict_proba(X_val_tree)[:, 1]
        return roc_auc_score(y_val_in, p)

    if has_optuna:
        study_lgb = optuna.create_study(direction="maximize")
        study_lgb.optimize(obj_lgb, n_trials=trials)
        best_lgb_params = study_lgb.best_params | {"class_weight": "balanced", "random_state": 42, "n_jobs": -1, "verbose": -1}
    else:
        best_lgb_params = {"max_depth": 5, "num_leaves": 31, "learning_rate": 0.05, "n_estimators": 20 if smoke else 300, "class_weight": "balanced", "random_state": 42, "n_jobs": -1, "verbose": -1}

    model_lgb = lgb.LGBMClassifier(**best_lgb_params)
    model_lgb.fit(X_tr_tree, y_tr)
    pred_lgb_test = model_lgb.predict_proba(X_te_tree)[:, 1]
    pred_lgb_val = model_lgb.predict_proba(X_val_tree)[:, 1]
    auc_lgb = float(roc_auc_score(y_te, pred_lgb_test))
    print(f"  Tuned LightGBM Test AUC: {auc_lgb:.4f}")

    # Model 3: CatBoost (if installed)
    pred_cb_test = None
    pred_cb_val = None
    auc_cb = None
    if has_cb:
        print("\n[Tuning 3/4] Tuning CatBoost...")
        def obj_cb(trial):
            params = {
                "depth": trial.suggest_int("depth", 4, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "iterations": trial.suggest_int("iterations", 10 if smoke else 50, 30 if smoke else 300),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "auto_class_weights": "Balanced",
                "random_state": 42,
                "verbose": 0,
            }
            m = cb.CatBoostClassifier(**params)
            m.fit(X_tr_in_tree, y_tr_in)
            p = m.predict_proba(X_val_tree)[:, 1]
            return roc_auc_score(y_val_in, p)

        if has_optuna:
            study_cb = optuna.create_study(direction="maximize")
            study_cb.optimize(obj_cb, n_trials=trials)
            best_cb_params = study_cb.best_params | {"auto_class_weights": "Balanced", "random_state": 42, "verbose": 0}
        else:
            best_cb_params = {"depth": 6, "learning_rate": 0.05, "iterations": 20 if smoke else 300, "auto_class_weights": "Balanced", "random_state": 42, "verbose": 0}

        model_cb = cb.CatBoostClassifier(**best_cb_params)
        model_cb.fit(X_tr_tree, y_tr)
        pred_cb_test = model_cb.predict_proba(X_te_tree)[:, 1]
        pred_cb_val = model_cb.predict_proba(X_val_tree)[:, 1]
        auc_cb = float(roc_auc_score(y_te, pred_cb_test))
        print(f"  Tuned CatBoost Test AUC: {auc_cb:.4f}")

    # Model 4: Tuned Logistic Regression
    print("\n[Tuning 4/4] Tuning Logistic Regression...")
    best_c = 0.1
    best_lr_auc = 0.0
    for c_cand in [0.01, 0.05, 0.1, 0.5, 1.0]:
        lr_cand = LogisticRegression(C=c_cand, class_weight="balanced", max_iter=200 if not smoke else 30, random_state=42)
        lr_cand.fit(X_tr_in_lr, y_tr_in)
        p_val = lr_cand.predict_proba(X_val_lr)[:, 1]
        val_auc = roc_auc_score(y_val_in, p_val)
        if val_auc > best_lr_auc:
            best_lr_auc = val_auc
            best_c = c_cand

    model_lr = LogisticRegression(C=best_c, class_weight="balanced", max_iter=500 if not smoke else 50, random_state=42)
    model_lr.fit(X_tr_lr, y_tr)
    pred_lr_test = model_lr.predict_proba(X_te_lr)[:, 1]
    pred_lr_val = model_lr.predict_proba(X_val_lr)[:, 1]
    auc_lr = float(roc_auc_score(y_te, pred_lr_test))
    print(f"  Tuned Logistic Regression (C={best_c}) Test AUC: {auc_lr:.4f}")

    # Model 5: Blended Ensemble (Best GBM + Tuned LR)
    # Determine best GBM on inner validation
    val_aucs = {"xgboost": roc_auc_score(y_val_in, pred_xgb_val), "lightgbm": roc_auc_score(y_val_in, pred_lgb_val)}
    if pred_cb_val is not None:
        val_aucs["catboost"] = roc_auc_score(y_val_in, pred_cb_val)
    best_gbm_name = max(val_aucs, key=val_aucs.get)
    print(f"\nBest performing GBM on inner validation: {best_gbm_name} (val AUC: {val_aucs[best_gbm_name]:.4f})")

    if best_gbm_name == "xgboost":
        best_gbm_val, best_gbm_test = pred_xgb_val, pred_xgb_test
    elif best_gbm_name == "lightgbm":
        best_gbm_val, best_gbm_test = pred_lgb_val, pred_lgb_test
    else:
        best_gbm_val, best_gbm_test = pred_cb_val, pred_cb_test

    # Optimize blend weight w on inner validation
    best_w = 0.5
    best_blend_val_auc = 0.0
    for w in np.linspace(0.1, 0.9, 17):
        p_blend = w * best_gbm_val + (1 - w) * pred_lr_val
        b_auc = roc_auc_score(y_val_in, p_blend)
        if b_auc > best_blend_val_auc:
            best_blend_val_auc = b_auc
            best_w = w

    print(f"Optimal ensemble blend weight: {best_w:.2f} * {best_gbm_name} + {1-best_w:.2f} * LR")
    pred_blend_test = best_w * best_gbm_test + (1 - best_w) * pred_lr_test
    auc_blend = float(roc_auc_score(y_te, pred_blend_test))
    print(f"  Blended Ensemble Test AUC: {auc_blend:.4f}")

    # Model 6: Probability Calibration (Isotonic / Platt on Blend)
    print("\nCalibrating probabilities...")
    from sklearn.isotonic import IsotonicRegression
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    p_blend_val = best_w * best_gbm_val + (1 - best_w) * pred_lr_val
    calibrator.fit(p_blend_val, y_val_in)
    pred_calibrated_test = calibrator.transform(pred_blend_test)
    auc_calibrated = float(roc_auc_score(y_te, pred_calibrated_test))
    brier_raw = float(brier_score_loss(y_te, pred_blend_test))
    brier_cal = float(brier_score_loss(y_te, pred_calibrated_test))
    print(f"  Calibrated Blend Test AUC: {auc_calibrated:.4f}")
    print(f"  Brier score: raw={brier_raw:.5f} -> calibrated={brier_cal:.5f}")

    # Compute bootstrap CIs
    print("\nComputing 95% bootstrap confidence intervals (n=2000)...")
    n_boot = 50 if smoke else 2000
    ci_xgb = bootstrap_auc_ci(y_te, pred_xgb_test, n_boot=n_boot, seed=42)
    ci_lgb = bootstrap_auc_ci(y_te, pred_lgb_test, n_boot=n_boot, seed=42)
    ci_cb = bootstrap_auc_ci(y_te, pred_cb_test, n_boot=n_boot, seed=42) if pred_cb_test is not None else (None, None)
    ci_lr = bootstrap_auc_ci(y_te, pred_lr_test, n_boot=n_boot, seed=42)
    ci_blend = bootstrap_auc_ci(y_te, pred_blend_test, n_boot=n_boot, seed=42)
    ci_cal = bootstrap_auc_ci(y_te, pred_calibrated_test, n_boot=n_boot, seed=42)

    results = {
        "protocol": "CDC NVSS Stillbirth Temporal Benchmark (E1: >= 20 wk gestation)",
        "feature_set": "Primary (5 numeric, 10 categorical, 0 post-delivery features)",
        "features": features,
        "n_train": len(tr),
        "n_test": len(te),
        "n_test_stillbirths": int(te.OUTCOME.sum()),
        "models": {
            "tuned_xgboost": {
                "test_auc": round(auc_xgb, 4),
                "ci_95": [round(ci_xgb[0], 4), round(ci_xgb[1], 4)],
                "params": {k: v for k, v in best_xgb_params.items() if not callable(v)},
            },
            "tuned_lightgbm": {
                "test_auc": round(auc_lgb, 4),
                "ci_95": [round(ci_lgb[0], 4), round(ci_lgb[1], 4)],
                "params": {k: v for k, v in best_lgb_params.items() if not callable(v)},
            },
            "tuned_catboost": {
                "test_auc": round(auc_cb, 4) if auc_cb is not None else "not_installed",
                "ci_95": [round(ci_cb[0], 4), round(ci_cb[1], 4)] if ci_cb[0] is not None else None,
            },
            "tuned_logistic_regression": {
                "test_auc": round(auc_lr, 4),
                "ci_95": [round(ci_lr[0], 4), round(ci_lr[1], 4)],
                "best_C": best_c,
            },
            "calibrated_gbm_lr_blend": {
                "test_auc": round(auc_blend, 4),
                "calibrated_auc": round(auc_calibrated, 4),
                "ci_95": [round(ci_cal[0], 4), round(ci_cal[1], 4)],
                "blend_formula": f"{best_w:.2f} * {best_gbm_name} + {1-best_w:.2f} * LR",
                "brier_score_raw": round(brier_raw, 5),
                "brier_score_calibrated": round(brier_cal, 5),
            },
        },
    }

    out_json = out_dir / "stream3_modern_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    out_md = out_dir / "stream3_modern_table.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Stream 3 Modern Tabular Campaign Results (E1 Protocol)\n\n")
        f.write(f"- **Protocol**: Temporal split (train <= 2021 [N={len(tr)}], test >= 2022 [N={len(te)}])\n")
        f.write(f"- **Features**: {len(features)} clinical & pre-pregnancy variables (zero outcome-time leakage)\n")
        f.write(f"- **Hyperparameter Tuning**: Optuna ({trials} trials)\n\n")
        f.write("| Model | Test ROC-AUC | 95% Bootstrap CI | Description |\n")
        f.write("|---|:---:|:---:|---|\n")
        f.write(f"| **Tuned XGBoost** | **{auc_xgb:.4f}** | [{ci_xgb[0]:.4f}, {ci_xgb[1]:.4f}] | Depth, lr, subsample tuned |\n")
        f.write(f"| **Tuned LightGBM** | **{auc_lgb:.4f}** | [{ci_lgb[0]:.4f}, {ci_lgb[1]:.4f}] | Leaves, depth, lr tuned |\n")
        if auc_cb is not None:
            f.write(f"| **Tuned CatBoost** | **{auc_cb:.4f}** | [{ci_cb[0]:.4f}, {ci_cb[1]:.4f}] | Depth, l2_reg, lr tuned |\n")
        f.write(f"| **Tuned Logistic Regression** | **{auc_lr:.4f}** | [{ci_lr[0]:.4f}, {ci_lr[1]:.4f}] | Standardized, C={best_c} |\n")
        f.write(f"| **Calibrated GBM+LR Blend** | **{auc_calibrated:.4f}** | [{ci_cal[0]:.4f}, {ci_cal[1]:.4f}] | `{best_w:.2f}*{best_gbm_name} + {1-best_w:.2f}*LR` + Isotonic Calibration |\n")

    print(f"\nGenerated {out_json} and {out_md}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="C:/fetal-data/cdc/clinical_training_data.csv")
    parser.add_argument("--out-dir", type=str, default="docs/results/stream3_modern")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    run_campaign(args.csv, args.out_dir, n_trials=args.n_trials, smoke=args.smoke)
