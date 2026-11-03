"""Leakage and specification audit routines for Stream 3 clinical model."""

import os
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from fetal.clinical.constants import NUM_FEATURES, CAT_FEATURES, FEATURE_SETS
from fetal.clinical.model import fit_eval


def record(name, d, num, cat, mode="random", n_estimators=300):
    """Fit model on given data/features and record audit row verbatim from notebook cell 5."""
    _, tr, te, p = fit_eval(d, num, cat, mode=mode, n_estimators=n_estimators)
    row = dict(
        setting=name,
        split=mode,
        n=len(d),
        stillbirths=int(d.OUTCOME.sum()),
        auc=round(float(roc_auc_score(te.OUTCOME, p)), 4),
        pr_auc_not_clinical=round(float(average_precision_score(te.OUTCOME, p)), 4),
    )
    print(f"{name:52s} [{mode:8s}] n={row['n']:7d} AUC={row['auc']:.4f}")
    return row


def run_audit_rows_a_to_d5(raw, df, n_estimators=300):
    """Execute audit rows A through D5 verbatim from notebook cell 7.

    Rows:
    - A: all rows, WITH COMBGEST (original 0.9684) [mode='random', raw]
    - B: all rows, no COMBGEST [mode='random', raw]
    - C: >=20 wk, WITH COMBGEST [mode='random', raw >= 20]
    - D: >=20 wk, no COMBGEST [mode='random', raw >= 20]
    - D2: >=20 wk dedup, minus PRECARE [mode='random', df >= 20]
    - D3: >=20 wk, complete cases only [mode='random', complete cases]
    - D4: >=37 wk, complete cases only [mode='random', complete cases >= 37]
    - D5: >=20 wk, numeric features only [mode='random', df >= 20]
    """
    audit = []
    raw20 = raw[raw.COMBGEST >= 20]
    audit.append(
        record(
            "A all rows, WITH COMBGEST (original 0.9684)",
            raw,
            NUM_FEATURES + ["COMBGEST"],
            CAT_FEATURES,
            mode="random",
            n_estimators=n_estimators,
        )
    )
    audit.append(
        record(
            "B all rows, no COMBGEST",
            raw,
            NUM_FEATURES,
            CAT_FEATURES,
            mode="random",
            n_estimators=n_estimators,
        )
    )
    audit.append(
        record(
            "C >=20 wk, WITH COMBGEST",
            raw20,
            NUM_FEATURES + ["COMBGEST"],
            CAT_FEATURES,
            mode="random",
            n_estimators=n_estimators,
        )
    )
    audit.append(
        record(
            "D >=20 wk, no COMBGEST",
            raw20,
            NUM_FEATURES,
            CAT_FEATURES,
            mode="random",
            n_estimators=n_estimators,
        )
    )

    d20 = df[df.COMBGEST >= 20].copy()
    audit.append(
        record(
            "D2 >=20 wk dedup, minus PRECARE",
            d20,
            [c for c in NUM_FEATURES if c != "PRECARE"],
            CAT_FEATURES,
            mode="random",
            n_estimators=n_estimators,
        )
    )

    cc = d20.dropna(subset=NUM_FEATURES + CAT_FEATURES)
    audit.append(
        record(
            "D3 >=20 wk, complete cases only",
            cc,
            NUM_FEATURES,
            CAT_FEATURES,
            mode="random",
            n_estimators=n_estimators,
        )
    )
    audit.append(
        record(
            "D4 >=37 wk, complete cases only",
            cc[cc.COMBGEST >= 37],
            NUM_FEATURES,
            CAT_FEATURES,
            mode="random",
            n_estimators=n_estimators,
        )
    )
    audit.append(
        record(
            "D5 >=20 wk, numeric features only",
            d20,
            NUM_FEATURES,
            [],
            mode="random",
            n_estimators=n_estimators,
        )
    )

    return audit
