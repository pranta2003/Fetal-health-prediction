"""Data loading, sentinel cleaning, and deduplication for Stream 3 clinical records."""

import numpy as np
import pandas as pd
from fetal.clinical.constants import CAT_FEATURES, UNKNOWN_CODES


def load_clean(csv_path):
    """Load clinical CDC dataset, replace sentinel missing codes with NaN, and deduplicate.

    Parameters
    ----------
    csv_path : str or Path
        Path to the clinical training data CSV.

    Returns
    -------
    raw : pd.DataFrame
        The cleaned dataset containing all rows (including duplicates).
    df : pd.DataFrame
        The deduplicated dataset copy (drop_duplicates().reset_index(drop=True)).
    """
    raw = pd.read_csv(csv_path, low_memory=False)

    for c in CAT_FEATURES:
        if c in raw.columns:
            raw[c] = raw[c].replace(UNKNOWN_CODES, np.nan)

    if "COMBGEST" in raw.columns:
        raw["COMBGEST"] = raw["COMBGEST"].replace([99], np.nan)

    df = raw.drop_duplicates().reset_index(drop=True)
    return raw, df
