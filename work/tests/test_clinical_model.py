import numpy as np
import pandas as pd
import pytest

from fetal.clinical.model import make_pipe, fit_eval


@pytest.fixture
def synthetic_clinical_df():
    """Create a minimal synthetic dataset with YEAR, OUTCOME, and feature columns."""
    rng = np.random.RandomState(42)
    n = 100
    years = rng.choice([2019, 2020, 2021, 2022, 2023], size=n)
    mager = rng.randint(18, 45, size=n)
    bmi = rng.uniform(18.0, 40.0, size=n)
    rf_gdiab = rng.choice(["Y", "N", np.nan], size=n)
    outcome = rng.binomial(1, 0.2, size=n)

    # Ensure both classes exist in both time periods
    outcome[years <= 2021][:2] = [0, 1]
    outcome[years >= 2022][:2] = [0, 1]

    return pd.DataFrame(
        {
            "YEAR": years,
            "MAGER": mager,
            "BMI": bmi,
            "RF_GDIAB": rf_gdiab,
            "OUTCOME": outcome,
        }
    )


def test_fit_eval_random_split(synthetic_clinical_df):
    """Test fit_eval with 80/20 random stratified split."""
    df = synthetic_clinical_df
    num = ["MAGER", "BMI"]
    cat = ["RF_GDIAB"]

    pipe, tr, te, p = fit_eval(df, num, cat, mode="random", n_estimators=10)

    assert len(tr) + len(te) == len(df)
    assert len(te) == 20  # 20% test size
    assert len(p) == len(te)
    assert np.all((p >= 0.0) & (p <= 1.0))
    # Stratified balance check
    assert tr.OUTCOME.sum() > 0
    assert te.OUTCOME.sum() > 0


def test_fit_eval_temporal_split(synthetic_clinical_df):
    """Test fit_eval with temporal split (train <= 2021, test >= 2022)."""
    df = synthetic_clinical_df
    num = ["MAGER", "BMI"]
    cat = ["RF_GDIAB"]

    pipe, tr, te, p = fit_eval(df, num, cat, mode="temporal", n_estimators=10)

    # Verify no test row has YEAR <= 2021
    assert (te.YEAR <= 2021).sum() == 0
    assert (te.YEAR >= 2022).all()

    # Verify train rows are only <= 2021
    assert (tr.YEAR <= 2021).all()
    assert (tr.YEAR >= 2022).sum() == 0

    assert len(p) == len(te)
    assert np.all((p >= 0.0) & (p <= 1.0))
