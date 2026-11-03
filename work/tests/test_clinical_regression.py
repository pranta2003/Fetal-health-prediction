"""Automated regression test suite for Stream 3 clinical tabular risk models.

Checks exact reproduction of specification audit table rows and temporal models
against canonical baseline AUC values reported in Capstone_fetalDataset.ipynb.
"""

import os
import pytest
from fetal import config
from fetal.clinical.data import load_clean
from fetal.clinical.constants import NUM_FEATURES, CAT_FEATURES, FEATURE_SETS
from fetal.clinical.model import fit_eval
from fetal.clinical.audit import run_audit_rows_a_to_d5, run_audit_rows_e1_to_e4


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
}


@pytest.mark.slow
def test_clinical_audit_regression_against_baseline():
    """Verify Stream 3 audit AUC values match published baseline within +/-0.003."""
    if not os.path.isfile(config.CLINICAL_CSV):
        pytest.skip(f"Clinical CSV dataset missing at {config.CLINICAL_CSV}")

    raw, df = load_clean(config.CLINICAL_CSV)

    # 1. Audit rows A through D5
    rows_ad = run_audit_rows_a_to_d5(raw, df, n_estimators=300)
    audit_map = {}
    for r in rows_ad:
        prefix = r["setting"].split()[0]
        audit_map[prefix] = r["auc"]

    for key in ["A", "B", "C", "D", "D2", "D3", "D4", "D5"]:
        exp = EXPECTED_AUCS[key]
        act = audit_map[key]
        assert abs(act - exp) <= TOLERANCE, f"Row {key} AUC {act:.4f} differs from expected {exp:.4f} by > {TOLERANCE}"

    # 2. Final temporal model with ALL features (FEATS)
    d20 = df[df.COMBGEST >= 20].copy()
    _, _, te, p = fit_eval(d20, NUM_FEATURES, CAT_FEATURES, mode="temporal", n_estimators=300)
    from sklearn.metrics import roc_auc_score
    act_temp_all = float(roc_auc_score(te.OUTCOME, p))
    exp_temp_all = EXPECTED_AUCS["final_temporal_all"]
    assert abs(act_temp_all - exp_temp_all) <= TOLERANCE, (
        f"Final temporal with ALL features AUC {act_temp_all:.4f} differs from expected {exp_temp_all:.4f}"
    )

    # 3. Audit rows E1 through E4
    rows_e = run_audit_rows_e1_to_e4(df, n_estimators=300)
    for r in rows_e:
        prefix = r["setting"].split()[0]
        exp = EXPECTED_AUCS[prefix]
        act = r["auc"]
        assert abs(act - exp) <= TOLERANCE, f"Row {prefix} AUC {act:.4f} differs from expected {exp:.4f} by > {TOLERANCE}"
