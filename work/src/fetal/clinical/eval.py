"""Evaluation metrics, bootstrap confidence intervals, and gestational band evaluations for Stream 3."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from fetal.clinical.constants import TRUE_PREVALENCE


def bootstrap_auc_ci(y, p, n_boot=200, seed=0):
    """Compute empirical 95% bootstrap confidence interval for ROC-AUC verbatim from notebook cell 9.

    Parameters
    ----------
    y : np.ndarray
        Binary ground truth labels.
    p : np.ndarray
        Predicted probabilities.
    n_boot : int
        Number of bootstrap iterations (default 200).
    seed : int
        Random seed for bootstrap resampling (default 0).

    Returns
    -------
    lo : float
        2.5th percentile AUC.
    hi : float
        97.5th percentile AUC.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], p[i]))
    if not boots:
        return float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def band_aucs(te, y, p, bands=None):
    """Compute AUC and outcome share partitioned across gestational age bands verbatim from notebook cell 9.

    Parameters
    ----------
    te : pd.DataFrame
        Test set containing the COMBGEST column.
    y : np.ndarray
        Binary ground truth labels.
    p : np.ndarray
        Predicted probabilities.
    bands : list of (low, high) tuples, optional
        Gestation intervals in weeks (default: [(20, 28), (28, 37), (37, 50)]).

    Returns
    -------
    list of dict
        Summary statistics per band with at least 50 cases and non-trivial outcomes.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    if bands is None:
        bands = [(20, 28), (28, 37), (37, 50)]
    band_rows = []
    for a, b in bands:
        m = ((te.COMBGEST >= a) & (te.COMBGEST < b)).values
        if m.sum() > 50 and 0 < y[m].mean() < 1:
            band_rows.append(
                dict(
                    band=f"{a}-{b} wk",
                    n=int(m.sum()),
                    stillbirth_share=round(float(y[m].mean()), 3),
                    auc=round(roc_auc_score(y[m], p[m]), 4),
                )
            )
    return band_rows


def prior_shift(p, pi_train, true_prevalence=TRUE_PREVALENCE):
    """Rescale predicted probabilities from training sample prevalence to true population prevalence.

    Uses odds ratio correction verbatim from notebook cell 11:
        odds = p / (1 - p)
        odds_corr = odds * (true_prevalence / (1 - true_prevalence)) / (pi_train / (1 - pi_train))
        p_corr = odds_corr / (1 + odds_corr)

    Parameters
    ----------
    p : array-like or float
        Predicted probabilities from model trained on enriched sample.
    pi_train : float
        Observed prevalence (positive class share) in training sample.
    true_prevalence : float, optional
        Target population prevalence (default TRUE_PREVALENCE = 0.0055).

    Returns
    -------
    p_corr : np.ndarray or float
        Recalibrated probabilities aligned with population prevalence.
    """
    p_arr = np.asarray(p, dtype=float)
    odds = p_arr / np.clip(1.0 - p_arr, 1e-9, None)
    odds_c = odds * (true_prevalence / (1.0 - true_prevalence)) / (pi_train / (1.0 - pi_train))
    p_corr = odds_c / (1.0 + odds_c)
    if np.ndim(p) == 0:
        return float(p_corr)
    return p_corr


def contrib_by_feature(pipe, te, features, n_sub=5000):
    """Compute mean |contribution| (SHAP / TreeSHAP) per original input feature verbatim from notebook cell 13.

    Groups one-hot encoded indicator columns back to their original parent categorical features.

    Parameters
    ----------
    pipe : sklearn.pipeline.Pipeline
        Fitted pipeline with 'pre' (ColumnTransformer) and 'm' (XGBClassifier) steps.
    te : pd.DataFrame
        Test dataset containing the feature columns.
    features : list of str
        List of original feature names (numerical and categorical).
    n_sub : int, optional
        Maximum number of samples to evaluate for speed (default 5000).

    Returns
    -------
    pd.Series
        Mean absolute feature contribution values indexed by original feature name.
    """
    pre = pipe.named_steps["pre"]
    booster = pipe.named_steps["m"].get_booster()
    n_sub = min(n_sub, len(te))
    Xs = pre.transform(te[features].iloc[:n_sub])
    contrib = booster.predict(xgb.DMatrix(Xs), pred_contribs=True)[:, :-1]
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]

    def group_of(n):
        for c in sorted(features, key=len, reverse=True):
            if n == c or n.startswith(c + "_"):
                return c
        return n

    groups = (
        pd.Series(np.abs(contrib).mean(0), index=[group_of(n) for n in names])
        .groupby(level=0)
        .sum()
        .sort_values(ascending=False)
    )
    return groups


