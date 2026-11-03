"""Evaluation metrics, bootstrap confidence intervals, and gestational band evaluations for Stream 3."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


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
