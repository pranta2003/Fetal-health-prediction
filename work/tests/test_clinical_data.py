import io
import numpy as np
import pandas as pd
import pytest

from fetal.clinical.data import load_clean
from fetal.clinical.constants import CAT_FEATURES, NUM_FEATURES


def test_load_clean_sentinels_and_deduplication(tmp_path):
    """Test that CAT_FEATURES sentinel values and COMBGEST 99 are converted to NaN, and deduplication works."""
    # Synthetic dataframe with CDC column names and sentinel values
    data = {
        "YEAR": [2020, 2020, 2021, 2022],
        "MAGER": [28, 28, 32, 25],
        "BMI": [24.5, 24.5, 29.1, 22.0],
        "PRECARE": [2, 2, 3, 1],
        "PRIORDEAD": [0, 0, 1, 0],
        "PRIORLIVE": [1, 1, 0, 2],
        "MEDUC": [4, 4, 3, 5],
        # CAT features with sentinel values
        "RF_GDIAB": ["N", "N", "U", "Y"],
        "RF_GHYPE": ["N", "N", "9", "N"],
        "RF_EHYPE": ["N", "N", "99", "N"],
        "RF_CESAR": ["Unknown", "Unknown", "N", "Y"],
        "RF_ARTEC": ["N", "N", "N", "N"],
        "RF_INFTR": ["N", "N", "N", "N"],
        "RF_FEDRG": ["N", "N", "N", "N"],
        "CIG_REC": ["N", "N", "N", "Y"],
        "WIC": ["N", "N", "Y", "N"],
        "DPLURAL": [1, 1, 1, 1],
        "SEX": ["M", "M", "F", "M"],
        "MRACE6": [10, 10, 20, 10],
        "MBSTATE_REC": [1, 1, 1, 2],
        # Outcome-time vars with sentinel values
        "COMBGEST": [39, 39, 99, 36],
        "OUTCOME": [0, 0, 1, 0],
    }

    df_synth = pd.DataFrame(data)
    # Row 0 and Row 1 are exact duplicates
    csv_file = tmp_path / "synthetic_cdc.csv"
    df_synth.to_csv(csv_file, index=False)

    raw, dedup = load_clean(csv_file)

    # Raw should retain duplicate rows
    assert len(raw) == 4
    # Dedup should have 1 duplicate removed -> 3 rows
    assert len(dedup) == 3
    assert list(dedup.index) == [0, 1, 2]

    # Verify sentinel conversions in raw and dedup
    # RF_GDIAB: 'U' -> NaN
    assert pd.isna(raw.loc[2, "RF_GDIAB"])
    assert pd.isna(dedup.loc[1, "RF_GDIAB"])

    # RF_GHYPE: '9' -> NaN
    assert pd.isna(raw.loc[2, "RF_GHYPE"])

    # RF_EHYPE: '99' -> NaN
    assert pd.isna(raw.loc[2, "RF_EHYPE"])

    # RF_CESAR: 'Unknown' -> NaN
    assert pd.isna(raw.loc[0, "RF_CESAR"])
    assert pd.isna(raw.loc[1, "RF_CESAR"])

    # COMBGEST: 99 -> NaN
    assert pd.isna(raw.loc[2, "COMBGEST"])
    assert pd.isna(dedup.loc[1, "COMBGEST"])

    # Valid values preserved
    assert raw.loc[0, "COMBGEST"] == 39
    assert raw.loc[3, "RF_GDIAB"] == "Y"
