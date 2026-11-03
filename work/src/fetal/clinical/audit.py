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


def run_audit_rows_e1_to_e5(df, n_estimators=300):
    """Execute audit rows E1 through E5.

    Rows:
    - E1: >=20 wk primary features (temporal) [0.7381]
    - E2: >=37 wk primary features (temporal) [0.7117]
    - E3: >=37 wk ALL features, reference (temporal) [0.7530]
    - E4: >=20 wk clinical-only features (temporal) [0.7069]
    - E5: >=37 wk clinical-only features (temporal) [0.6917]
    """
    audit = []
    d20 = df[df.COMBGEST >= 20].copy()
    d37 = d20[d20.COMBGEST >= 37].copy()

    audit.append(
        record(
            "E1 >=20 wk primary features (temporal)",
            d20,
            FEATURE_SETS["primary"]["num"],
            FEATURE_SETS["primary"]["cat"],
            mode="temporal",
            n_estimators=n_estimators,
        )
    )
    audit.append(
        record(
            "E2 >=37 wk primary features (temporal)",
            d37,
            FEATURE_SETS["primary"]["num"],
            FEATURE_SETS["primary"]["cat"],
            mode="temporal",
            n_estimators=n_estimators,
        )
    )
    audit.append(
        record(
            "E3 >=37 wk ALL features, reference (temporal)",
            d37,
            FEATURE_SETS["full"]["num"],
            FEATURE_SETS["full"]["cat"],
            mode="temporal",
            n_estimators=n_estimators,
        )
    )
    audit.append(
        record(
            "E4 >=20 wk clinical-only features (temporal)",
            d20,
            FEATURE_SETS["clinical_only"]["num"],
            FEATURE_SETS["clinical_only"]["cat"],
            mode="temporal",
            n_estimators=n_estimators,
        )
    )
    audit.append(
        record(
            "E5 >=37 wk clinical-only features (temporal)",
            d37,
            FEATURE_SETS["clinical_only"]["num"],
            FEATURE_SETS["clinical_only"]["cat"],
            mode="temporal",
            n_estimators=n_estimators,
        )
    )

    return audit


run_audit_rows_e1_to_e4 = run_audit_rows_e1_to_e5  # backward-compatibility alias


def run_full_audit(raw, df, out_dir=None, n_estimators=300):
    """Run full audit (rows A-D5 and E1-E5) and optionally export audit_table.csv."""
    rows_a_d = run_audit_rows_a_to_d5(raw, df, n_estimators=n_estimators)
    rows_e = run_audit_rows_e1_to_e5(df, n_estimators=n_estimators)
    all_rows = rows_a_d + rows_e
    audit_df = pd.DataFrame(all_rows)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "audit_table.csv")
        audit_df.to_csv(csv_path, index=False)
        print(f"[audit] Saved audit table to {csv_path}")

    return audit_df
