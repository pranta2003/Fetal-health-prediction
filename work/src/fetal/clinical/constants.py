"""Constants and feature specifications for Stream 3 (CDC clinical risk model)."""

# Numeric features from CDC birth records
NUM_FEATURES = ["MAGER", "BMI", "PRECARE", "PRIORDEAD", "PRIORLIVE", "MEDUC"]

# Categorical risk factors and demographic features
CAT_FEATURES = [
    "RF_GDIAB",
    "RF_GHYPE",
    "RF_EHYPE",
    "RF_CESAR",
    "RF_ARTEC",
    "RF_INFTR",
    "RF_FEDRG",
    "CIG_REC",
    "WIC",
    "DPLURAL",
    "SEX",
    "MRACE6",
    "MBSTATE_REC",
]

# Sentinel values indicating unknown or missing entries in CDC birth records
UNKNOWN_CODES = ["U", "9", "99", "Unknown"]

# Variables only known at/after delivery -> must NEVER be used as model inputs during pregnancy
OUTCOME_TIME_VARS = ["COMBGEST", "OEGest_Comb", "OBGEST_FLG", "DBWT", "BWTR4"]

# PLACEHOLDER: approximate real-world stillbirth prevalence (20+ weeks). VERIFY against CDC/NVSS before the paper.
TRUE_PREVALENCE = 0.006

# Defined feature sets for audit and production
FEATURE_SETS = {
    # Audit-only upper bound (contains socioeconomic & reporting-sensitive fields)
    "full": {
        "num": list(NUM_FEATURES),
        "cat": list(CAT_FEATURES),
    },
    # Primary model: full minus reporting-sensitive fields (WIC, MEDUC, MBSTATE_REC, MRACE6)
    "primary": {
        "num": [c for c in NUM_FEATURES if c != "MEDUC"],
        "cat": [c for c in CAT_FEATURES if c not in ("WIC", "MBSTATE_REC", "MRACE6")],
    },
    # Clinical-only: strictly clinical history & maternal risk factors
    "clinical_only": {
        "num": ["MAGER", "BMI", "PRIORDEAD", "PRIORLIVE"],
        "cat": [
            "RF_GDIAB",
            "RF_GHYPE",
            "RF_EHYPE",
            "RF_CESAR",
            "RF_ARTEC",
            "RF_INFTR",
            "RF_FEDRG",
            "CIG_REC",
            "DPLURAL",
        ],
    },
}


def _validate_feature_sets():
    """Import-time guard to ensure outcome-time variables are never included in feature sets."""
    for name, feats in FEATURE_SETS.items():
        combined = set(feats["num"] + feats["cat"])
        overlap = combined & set(OUTCOME_TIME_VARS)
        assert not overlap, f"Feature set '{name}' contains outcome-time variables: {overlap}"


_validate_feature_sets()
