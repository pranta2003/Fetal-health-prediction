"""tests/test_three_plane.py: Unit tests for three-plane subject and leakage check."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from fetal import config

WORK_ROOT = Path(__file__).resolve().parent.parent
RESULT_FILE = WORK_ROOT / "docs" / "results" / "three_plane" / "three_plane_check.json"


@pytest.fixture(scope="module")
def results():
    """Load pre-computed results or run check if missing."""
    if not RESULT_FILE.exists():
        pytest.skip(f"Result file not found: {RESULT_FILE}. Run scripts/23_three_plane_check.py first.")
    with open(RESULT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def test_fp_counts_match_paper(results):
    """FP counts must exactly match the benchmark paper: Head 909, Abdomen 586, Femur 629, Union 1047."""
    comp = results["subject_counts"]["paper_comparison_fp"]
    assert comp["exact_match_all"] is True
    assert comp["head"]["ours"] == 909
    assert comp["abdomen"]["ours"] == 586
    assert comp["femur"]["ours"] == 629
    assert comp["total_union"]["ours"] == 1047


def test_fp_three_plane_intersection(results):
    """Check FP three-plane overlap and split distribution."""
    fp_counts = results["subject_counts"]["fp"]
    assert fp_counts["three_plane_intersection"] == 361

    fp_leak = results["splits_and_leakage_audit"]["fp"]
    assert fp_leak["three_plane_subjects_total"] == 361
    assert fp_leak["three_plane_subjects_all_test"] == 17
    assert fp_leak["three_plane_subjects_all_train"] == 136
    assert fp_leak["cross_plane_split_mismatch_count"] == 208


def test_ucl_counts_match_paper(results):
    """UCL raw union must match paper's 51 subjects."""
    ucl_raw = results["subject_counts"]["ucl_raw"]
    assert ucl_raw["subjects_per_plane"]["total_union"] == 51

    ucl_px = results["subject_counts"]["ucl_px_to_mm"]
    assert ucl_px["subjects_per_plane"]["total_union"] == 35
    assert ucl_px["three_plane_intersection"] == 24

    ucl_leak = results["splits_and_leakage_audit"]["ucl_px_to_mm"]
    assert ucl_leak["three_plane_subjects_all_test"] == 5
    assert ucl_leak["three_plane_subjects_all_train"] == 19
    assert ucl_leak["cross_plane_split_mismatch_count"] == 0


def test_no_gestational_age_or_outcome_columns(results):
    """Confirm no gestational age, weight, or clinical outcome metadata exists in any CSV."""
    cols_audit = results["metadata_columns_audit"]
    for dataset_name, planes in cols_audit.items():
        for plane, info in planes.items():
            assert info["has_gestational_age"] is False, f"Unexpected GA column in {dataset_name} {plane}"
            assert info["has_estimated_fetal_weight"] is False, f"Unexpected EFW column in {dataset_name} {plane}"
            assert info["has_birth_weight"] is False, f"Unexpected BW column in {dataset_name} {plane}"
            assert info["has_outcome"] is False, f"Unexpected outcome column in {dataset_name} {plane}"
            assert len(info["matched_clinical_or_outcome_columns"]) == 0
