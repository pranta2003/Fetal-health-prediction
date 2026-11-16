"""Unit tests for FASSD Multilabel Segmentation Ensembling module."""

import numpy as np
import pandas as pd
import pytest

from src.fetal.seg.fassd_ensemble import (
    average_probabilities,
    evaluate_ensemble_slice,
    select_candidate_models,
)
from src.fetal.seg.fassd_postproc import run_patient_paired_bootstrap


def test_average_probabilities_identical():
    """Verify average of identical maps equals the original map."""
    arr1 = np.full((4, 32, 32), 0.75, dtype=np.float16)
    arr2 = np.full((4, 32, 32), 0.75, dtype=np.float16)
    arr3 = np.full((4, 32, 32), 0.75, dtype=np.float16)

    avg = average_probabilities([arr1, arr2, arr3])
    assert avg.shape == (4, 32, 32)
    assert np.allclose(avg, 0.75, atol=1e-3)


def test_average_probabilities_values():
    """Verify numeric averaging across different probability maps."""
    arr1 = np.zeros((4, 16, 16), dtype=np.float16)
    arr2 = np.ones((4, 16, 16), dtype=np.float16)

    avg = average_probabilities([arr1, arr2])
    assert np.allclose(avg, 0.5, atol=1e-3)


def test_average_probabilities_empty_raises():
    """Ensure empty list raises ValueError."""
    with pytest.raises(ValueError):
        average_probabilities([])


def test_select_candidate_models():
    """Verify candidate selection rules: all, top3, top2."""
    val_scores = {
        "m_a": 0.75,
        "m_b": 0.82,
        "m_c": 0.79,
        "m_d": 0.85,
        "m_e": 0.71,
    }

    all_models = select_candidate_models(val_scores, "all")
    assert all_models == ["m_d", "m_b", "m_c", "m_a", "m_e"]

    top3 = select_candidate_models(val_scores, "top3")
    assert top3 == ["m_d", "m_b", "m_c"]

    top2 = select_candidate_models(val_scores, "top2")
    assert top2 == ["m_d", "m_b"]

    with pytest.raises(ValueError):
        select_candidate_models(val_scores, "invalid_rule")


def test_evaluate_ensemble_slice_synthetic():
    """Test evaluation slice on synthetic perfect and zero predictions."""
    h, w = 64, 64
    gts_dict = {}
    probs_dict = {}
    stems = ["sample_1", "sample_2"]

    # Ground truth: circle in center for structure 0 (artery)
    gt1 = np.zeros((4, h, w), dtype=np.uint8)
    gt1[0, 20:40, 20:40] = 1
    gt1[1, 10:30, 10:30] = 1
    gts_dict["sample_1"] = gt1

    gt2 = np.zeros((4, h, w), dtype=np.uint8)
    gt2[2, 15:35, 15:35] = 1
    gts_dict["sample_2"] = gt2

    # Probs: match ground truth exactly
    prob1 = np.zeros((4, h, w), dtype=np.float16)
    prob1[0, 20:40, 20:40] = 0.95
    prob1[1, 10:30, 10:30] = 0.95
    probs_dict["sample_1"] = prob1

    prob2 = np.zeros((4, h, w), dtype=np.float16)
    prob2[2, 15:35, 15:35] = 0.95
    probs_dict["sample_2"] = prob2

    df_sub = pd.DataFrame({
        "filename": ["sample_1.png", "sample_2.png"],
        "patient": ["P001", "P002"],
    })

    res = evaluate_ensemble_slice(
        probs_dict=probs_dict,
        gts_dict=gts_dict,
        df_subset=df_sub,
        threshold=0.5,
        morph_type="none",
        compute_hd=True,
    )

    assert res["by_structure"]["artery"]["dice_mean"] == pytest.approx(1.0, abs=1e-3)
    assert res["by_structure"]["artery"]["hd95_mean"] == pytest.approx(0.0, abs=1e-1)
    assert res["by_structure"]["liver"]["dice_mean"] == pytest.approx(1.0, abs=1e-3)
    assert res["by_structure"]["stomach"]["dice_mean"] == pytest.approx(1.0, abs=1e-3)


def test_run_patient_paired_bootstrap_synthetic():
    """Verify paired bootstrap correctly detects positive delta."""
    diffs = {f"P{i:03d}": 0.05 for i in range(20)}
    obs_mean, ci_low, ci_high = run_patient_paired_bootstrap(diffs, n_bootstraps=500, seed=42)
    assert obs_mean == pytest.approx(0.05, abs=1e-4)
    assert ci_low == pytest.approx(0.05, abs=1e-4)
    assert ci_high == pytest.approx(0.05, abs=1e-4)
