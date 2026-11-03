"""Automated regression test suite for Stream 3 clinical tabular risk models.

Checks exact reproduction of specification audit table rows and temporal models
against canonical baseline AUC values reported in Capstone_fetalDataset.ipynb.
"""

import os
import pytest
from sklearn.metrics import roc_auc_score

from fetal import config
from fetal.clinical.data import load_clean
from fetal.clinical.constants import NUM_FEATURES, CAT_FEATURES
from fetal.clinical.model import fit_eval
from fetal.clinical.audit import run_audit_rows_a_to_d5, run_audit_rows_e1_to_e5


TOLERANCE = 0.003

EXPECTED_AUCS = {
    "A": 0.9684,
    "B": 0.8832,
    "C": 0.9382,
    "D": 0.7862,
    "D2": 0.7703,
    "D3": 0.7464,
    "D4": 0.7131,
    "D5": 0.7434,
    "final_temporal_all": 0.7865,
    "E1": 0.7381,
    "E2": 0.7117,
    "E3": 0.7530,
    "E4": 0.7069,
    "E5": 0.6917,
}


@pytest.mark.slow
def test_clinical_audit_regression_against_baseline():
    """Verify Stream 3 audit AUC values match published baseline within +/-0.003."""
    if not os.path.isfile(config.CLINICAL_CSV):
        pytest.skip(f"Clinical CSV dataset missing at {config.CLINICAL_CSV}")

    raw, df = load_clean(config.CLINICAL_CSV)

    results = {}

    # 1. Audit rows A through D5
    rows_ad = run_audit_rows_a_to_d5(raw, df, n_estimators=300)
    for r in rows_ad:
        prefix = r["setting"].split()[0]
        results[prefix] = r["auc"]

    # 2. Final temporal model with ALL features (FEATS)
    d20 = df[df.COMBGEST >= 20].copy()
    _, _, te, p = fit_eval(d20, NUM_FEATURES, CAT_FEATURES, mode="temporal", n_estimators=300)
    act_temp_all = float(roc_auc_score(te.OUTCOME, p))
    results["final_temporal_all"] = act_temp_all

    # 3. Audit rows E1 through E5
    rows_e = run_audit_rows_e1_to_e5(df, n_estimators=300)
    for r in rows_e:
        prefix = r["setting"].split()[0]
        results[prefix] = r["auc"]

    # Build comparison table and collect mismatches
    print("\n" + "=" * 65)
    print(f"{'Row':<20} | {'Actual':<10} | {'Expected':<10} | {'Diff':<10} | {'Status'}")
    print("-" * 65)

    failures = []
    for key, exp in EXPECTED_AUCS.items():
        if key not in results:
            failures.append(f"{key}: MISSING from results")
            print(f"{key:<20} | {'MISSING':<10} | {exp:<10.4f} | {'N/A':<10} | FAIL")
            continue
        act = results[key]
        diff = act - exp
        status = "PASS" if abs(diff) <= TOLERANCE else "FAIL"
        print(f"{key:<20} | {act:<10.4f} | {exp:<10.4f} | {diff:<+10.4f} | {status}")
        if status == "FAIL":
            failures.append(f"{key}: actual={act:.4f}, expected={exp:.4f}, diff={diff:+.4f} (tol={TOLERANCE})")

    print("=" * 65)
    assert not failures, f"{len(failures)} row(s) failed regression tolerance:\n" + "\n".join(failures)
