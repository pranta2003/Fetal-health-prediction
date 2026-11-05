import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from fetal.clinical.eval import bootstrap_auc_ci, band_aucs, prior_shift, contrib_by_feature
from fetal.clinical.constants import TRUE_PREVALENCE
from fetal.clinical.model import make_pipe


def test_bootstrap_auc_ci_bounds():
    """Verify bootstrap CI generates valid lower and upper percentiles containing the point estimate."""
    rng = np.random.RandomState(42)
    n = 200
    y = rng.binomial(1, 0.3, size=n)
    # Generate probabilities correlated with y
    p = np.clip(y * 0.4 + rng.uniform(0.1, 0.5, size=n), 0.0, 1.0)

    point_auc = roc_auc_score(y, p)
    lo, hi = bootstrap_auc_ci(y, p, n_boot=100, seed=42)

    assert 0.0 <= lo <= hi <= 1.0
    assert lo <= point_auc <= hi


def test_band_aucs_filtering():
    """Verify band_aucs groups by gestational intervals and enforces minimum sample thresholds."""
    rng = np.random.RandomState(42)
    n = 300
    combgest = np.concatenate(
        [
            rng.uniform(22, 27, size=80),   # 20-28 wk band (n=80 > 50)
            rng.uniform(30, 35, size=120),  # 28-37 wk band (n=120 > 50)
            rng.uniform(38, 41, size=100),  # 37-50 wk band (n=100 > 50)
        ]
    )
    y = rng.binomial(1, 0.25, size=n)
    # Ensure both classes in all bands
    y[0:2] = [0, 1]
    y[80:82] = [0, 1]
    y[200:202] = [0, 1]
    p = np.clip(y * 0.3 + rng.uniform(0.1, 0.6, size=n), 0.0, 1.0)

    te = pd.DataFrame({"COMBGEST": combgest})
    bands = band_aucs(te, y, p)

    assert len(bands) == 3
    band_names = [b["band"] for b in bands]
    assert band_names == ["20-28 wk", "28-37 wk", "37-50 wk"]
    for b in bands:
        assert b["n"] >= 50
        assert 0.0 < b["stillbirth_share"] < 1.0
        assert 0.0 <= b["auc"] <= 1.0


def test_prior_shift_calibration_and_monotonicity():
    """Verify prior_shift maps training prevalence to TRUE_PREVALENCE and is strictly monotonic."""
    pi_train = 0.25
    true_prev = TRUE_PREVALENCE  # 0.0055

    # 1. Exact mapping of sample prevalence to population prevalence
    p_at_sample_rate = prior_shift(pi_train, pi_train, true_prev)
    assert np.isclose(p_at_sample_rate, true_prev, atol=1e-7)

    # 2. Strict monotonicity across probability range
    p_grid = np.linspace(0.001, 0.999, 100)
    p_corr = prior_shift(p_grid, pi_train, true_prev)

    assert np.all(np.diff(p_corr) > 0), "prior_shift must be strictly monotonically increasing"
    assert np.all((p_corr >= 0.0) & (p_corr <= 1.0))


def test_contrib_by_feature_groups_one_hot():
    """Verify contrib_by_feature collapses one-hot categories back to exactly one value per original feature."""
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame(
        {
            "MAGER": rng.randint(20, 40, size=n),
            "BMI": rng.uniform(20.0, 35.0, size=n),
            "RF_GDIAB": rng.choice(["Y", "N", "U"], size=n),
            "SEX": rng.choice(["M", "F"], size=n),
            "OUTCOME": rng.binomial(1, 0.3, size=n),
        }
    )
    num = ["MAGER", "BMI"]
    cat = ["RF_GDIAB", "SEX"]
    features = num + cat

    pipe = make_pipe(num, cat, n_estimators=10, random_state=42)
    pipe.fit(df[features], df.OUTCOME)

    contrib = contrib_by_feature(pipe, df, features, n_sub=50)

    # Must return exactly one aggregated importance value per original feature
    assert isinstance(contrib, pd.Series)
    assert len(contrib) == len(features)
    assert set(contrib.index) == set(features)
    assert (contrib >= 0.0).all()


