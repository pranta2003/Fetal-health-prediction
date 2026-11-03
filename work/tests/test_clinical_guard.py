import pytest
from fetal.clinical import constants


def test_outcome_time_vars_not_in_predefined_sets():
    """Verify that none of the predefined feature sets contain outcome-time variables."""
    outcome_vars = set(constants.OUTCOME_TIME_VARS)
    for name, fset in constants.FEATURE_SETS.items():
        used = set(fset["num"] + fset["cat"])
        overlap = used & outcome_vars
        assert not overlap, f"Feature set '{name}' contains outcome-time variables: {overlap}"


def test_guard_triggers_assertion_when_outcome_var_added(monkeypatch):
    """Verify that adding COMBGEST or any outcome-time variable triggers the guard assertion."""
    # Test adding COMBGEST to numerical features of a temporary set
    bad_feature_sets = {
        "bad_set": {
            "num": ["MAGER", "COMBGEST"],
            "cat": ["RF_GDIAB"],
        }
    }
    monkeypatch.setattr(constants, "FEATURE_SETS", bad_feature_sets)

    with pytest.raises(AssertionError, match="contains outcome-time variables"):
        constants._validate_feature_sets()


def test_guard_triggers_assertion_for_all_outcome_time_vars(monkeypatch):
    """Verify each outcome-time variable is caught by the validation guard."""
    for var in constants.OUTCOME_TIME_VARS:
        bad_feature_sets = {
            "test_bad": {
                "num": ["MAGER", var],
                "cat": [],
            }
        }
        monkeypatch.setattr(constants, "FEATURE_SETS", bad_feature_sets)
        with pytest.raises(AssertionError):
            constants._validate_feature_sets()
