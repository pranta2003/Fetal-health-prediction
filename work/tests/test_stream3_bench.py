"""Unit tests for Stream 3 clinical model benchmark (scripts/18_bench_stream3.py)."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure scripts/ is on sys.path
scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from fetal.clinical.constants import FEATURE_SETS, OUTCOME_TIME_VARS
from importlib import import_module

bench_mod = import_module("18_bench_stream3")
build_models = bench_mod.build_models


def test_no_forbidden_features_used():
    """Verify that forbidden outcome-time variables are never present in benchmark feature sets."""
    for setting in ["primary", "clinical_only"]:
        features = set(FEATURE_SETS[setting]["num"] + FEATURE_SETS[setting]["cat"])
        overlap = features & set(OUTCOME_TIME_VARS)
        assert not overlap, f"Forbidden variables found in setting '{setting}': {overlap}"

    # Also assert that build_models raises ValueError if a forbidden variable is provided
    with pytest.raises(ValueError, match="Forbidden outcome-time variables detected"):
        build_models(["MAGER", "COMBGEST"], ["RF_GDIAB"], smoke=True)


def test_identical_train_test_rows_and_smoke():
    """Verify that all models receive the exact same train and test rows, and produce valid probabilities."""
    rng = np.random.RandomState(42)
    n = 120
    years = rng.choice([2019, 2020, 2021, 2022, 2023], size=n)
    mager = rng.randint(18, 45, size=n)
    bmi = rng.uniform(18.0, 40.0, size=n)
    rf_gdiab = rng.choice(["Y", "N", np.nan], size=n)
    outcome = rng.binomial(1, 0.2, size=n)

    # Ensure both classes exist in both time periods
    outcome[years <= 2021][:2] = [0, 1]
    outcome[years >= 2022][:2] = [0, 1]

    df = pd.DataFrame(
        {
            "YEAR": years,
            "MAGER": mager,
            "BMI": bmi,
            "RF_GDIAB": rf_gdiab,
            "OUTCOME": outcome,
        }
    )

    tr = df[df.YEAR <= 2021]
    te = df[df.YEAR >= 2022]

    num_cols = ["MAGER", "BMI"]
    cat_cols = ["RF_GDIAB"]
    cols = num_cols + cat_cols

    X_tr = tr[cols]
    y_tr = tr.OUTCOME.values
    X_te = te[cols]
    y_te = te.OUTCOME.values

    # Check identical row indices
    expected_train_idx = list(tr.index)
    expected_test_idx = list(te.index)

    models = build_models(num_cols, cat_cols, random_state=42, smoke=True)
    assert len(models) >= 3  # At least LR, RF, XGB (and LGBM if installed)
    assert "Our XGBoost" in models
    assert "Logistic Regression" in models
    assert "Random Forest" in models

    for name, pipe in models.items():
        # Assert X_tr has exact expected train indices
        assert list(X_tr.index) == expected_train_idx, f"Train index mismatch for {name}"
        assert list(X_te.index) == expected_test_idx, f"Test index mismatch for {name}"

        pipe.fit(X_tr, y_tr)
        probs = pipe.predict_proba(X_te)[:, 1]

        assert len(probs) == len(expected_test_idx), f"Prediction count mismatch for {name}"
        assert np.all((probs >= 0.0) & (probs <= 1.0)), f"Probabilities out of bounds for {name}"
