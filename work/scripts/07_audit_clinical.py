"""07_audit_clinical.py: CLI runner for Stream 3 clinical risk audit and temporal model training."""

import argparse
import json
import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fetal import config
from fetal.clinical.constants import FEATURE_SETS, TRUE_PREVALENCE
from fetal.clinical.data import load_clean
from fetal.clinical.model import fit_eval
from fetal.clinical.eval import bootstrap_auc_ci, band_aucs, prior_shift, contrib_by_feature
from fetal.clinical.audit import run_full_audit


def parse_args():
    parser = argparse.ArgumentParser(
        description="07_audit_clinical: Run clinical risk model specification audit and train primary temporal model."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=config.CLINICAL_CSV,
        help=f"Path to clinical dataset CSV (default: {config.CLINICAL_CSV})",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=config.CLINICAL_OUT_DIR,
        help=f"Directory to save outputs and checkpoints (default: {config.CLINICAL_OUT_DIR})",
    )
    parser.add_argument(
        "--n-est",
        type=int,
        default=300,
        help="Number of estimators for XGBClassifier (default: 300)",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        default=False,
        help="Skip the full audit table and run only the final primary model",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="Run smoke mode with reduced estimators and bootstrap iterations for quick verification",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("07_audit_clinical: Stream 3 Clinical Tabular Risk Assessment")
    print(f"  CSV path : {args.csv}")
    print(f"  Out dir  : {args.out_dir}")
    print(f"  n_est    : {args.n_est} (smoke={args.smoke}, no_audit={args.no_audit})")
    print("=" * 70)

    if not os.path.isfile(args.csv):
        print(f"\n[Notice] Clinical training CSV not found at: {args.csv}")
        print("Real data lives outside this machine. Exiting cleanly.")
        return

    n_est = 10 if args.smoke else args.n_est
    n_boot = 10 if args.smoke else 200

    print("\n[Step 1] Loading and cleaning dataset...")
    raw, df = load_clean(args.csv)
    print(f"  raw shape: {raw.shape} | dedup shape: {df.shape} | duplicates removed: {len(raw) - len(df)}")

    audit_table = None
    if not args.no_audit:
        print("\n[Step 2] Executing specification and leakage audit table (Rows A-D5, E1-E5)...")
        audit_table = run_full_audit(raw, df, out_dir=args.out_dir, n_estimators=n_est)

    print("\n[Step 3] Training PRIMARY model with temporal split (train <= 2021, test >= 2022)...")
    d20 = df[df.COMBGEST >= 20].copy()
    num_feats = FEATURE_SETS["primary"]["num"]
    cat_feats = FEATURE_SETS["primary"]["cat"]
    feats = num_feats + cat_feats
    print(f"  Features used ({len(feats)}): {feats}")

    pipe, tr, te, p = fit_eval(d20, num_feats, cat_feats, mode="temporal", n_estimators=n_est)
    y = te.OUTCOME.values
    auc = float(roc_auc_score(y, p))
    lo, hi = bootstrap_auc_ci(y, p, n_boot=n_boot)
    print(f"  PRIMARY temporal test: AUC={auc:.4f}  95% CI [{lo:.4f}, {hi:.4f}] (train n={len(tr)}, test n={len(te)})")

    print("\n[Step 4] Gestational age band evaluation...")
    bands = band_aucs(te, y, p)
    for b in bands:
        print(f"  {b['band']:10s} n={b['n']:6d}  stillbirth_share={b['stillbirth_share']:.3f}  AUC={b['auc']:.4f}")

    print("\n[Step 5] Prior-shift recalibration to population prevalence...")
    pi_train = float(tr.OUTCOME.mean())
    p_corr = prior_shift(p, pi_train=pi_train, true_prevalence=TRUE_PREVALENCE)
    print(f"  Sample prevalence (train): {pi_train:.4f} -> Assumed population prevalence: {TRUE_PREVALENCE:.4f}")
    print(f"  Brier score on sample: {brier_score_loss(y, p):.4f}")

    dec = pd.qcut(p, 10, labels=False, duplicates="drop")
    tab = pd.DataFrame({"decile": dec, "sample_rate": y, "corrected_risk": p_corr}).groupby("decile").mean().round(4)
    tab["n"] = pd.Series(dec).value_counts().sort_index().values
    print("\nRisk by decile (0 = lowest risk):")
    print(tab)

    print("\n[Step 6] Saving models, predictions, and results...")
    xgb_model_path = os.path.join(args.out_dir, "stream3_clinical_xgb_v2.json")
    pipe.named_steps["m"].save_model(xgb_model_path)
    joblib_path = os.path.join(args.out_dir, "stream3_pipeline_v2.joblib")
    joblib.dump(pipe, joblib_path)

    preds_path = os.path.join(args.out_dir, "temporal_test_predictions.csv")
    pd.DataFrame({"year": te.YEAR.values, "outcome": y, "p_sample": p, "p_corrected": p_corr}).to_csv(preds_path, index=False)

    results_data = {
        "features": feats,
        "scope": ">=20 weeks, COMBGEST not a feature, temporal split 2016-2021 / 2022-2023",
        "n_train": len(tr),
        "n_test": len(te),
        "auc": round(auc, 4),
        "auc_ci95": [round(lo, 4), round(hi, 4)],
        "bands": bands,
        "true_prevalence_assumed": TRUE_PREVALENCE,
        "audit": audit_table.to_dict(orient="records") if audit_table is not None else [],
    }
    results_path = os.path.join(args.out_dir, "results_v2.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=1)

    print(f"\nAll artifacts saved to {args.out_dir} successfully.")


if __name__ == "__main__":
    main()
