"""Tests for FASSD post-processing module."""

from __future__ import annotations

import numpy as np
import pytest

from src.fetal.seg.fassd_postproc import (
    apply_morphology,
    apply_postprocessing,
    evaluate_structure_predictions,
    run_patient_paired_bootstrap,
    tune_val_grid,
)
from src.fetal.seg.fassd_train import STRUCTURE_NAMES


def test_apply_morphology_none():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    mask[10:12, 10:12] = 1

    out = apply_morphology(mask, morph_type="none")
    assert np.array_equal(out, mask)


def test_apply_morphology_largest():
    mask = np.zeros((30, 30), dtype=np.uint8)
    # Small component (area 4)
    mask[2:4, 2:4] = 1
    # Large component (area 25)
    mask[10:15, 10:15] = 1

    out = apply_morphology(mask, morph_type="largest")
    assert out[2:4, 2:4].sum() == 0
    assert out[10:15, 10:15].sum() == 25
    assert out.sum() == 25


def test_apply_morphology_min_area():
    mask = np.zeros((40, 40), dtype=np.uint8)
    # Blob 1: area 9
    mask[2:5, 2:5] = 1
    # Blob 2: area 36
    mask[10:16, 10:16] = 1
    # Blob 3: area 4
    mask[25:27, 25:27] = 1

    # Filter out blobs with area < 10
    out = apply_morphology(mask, morph_type="min_area", min_area=10)
    assert out[2:5, 2:5].sum() == 0
    assert out[25:27, 25:27].sum() == 0
    assert out[10:16, 10:16].sum() == 36


def test_apply_postprocessing_threshold():
    prob = np.zeros((10, 10), dtype=np.float32)
    prob[2:5, 2:5] = 0.6
    prob[6:8, 6:8] = 0.4

    # Threshold 0.5: only [2:5, 2:5] passes
    out1 = apply_postprocessing(prob, threshold=0.5, morph_type="none")
    assert out1[2:5, 2:5].sum() == 9
    assert out1[6:8, 6:8].sum() == 0

    # Threshold 0.35: both pass
    out2 = apply_postprocessing(prob, threshold=0.35, morph_type="none")
    assert out2[2:5, 2:5].sum() == 9
    assert out2[6:8, 6:8].sum() == 4


def test_run_patient_paired_bootstrap():
    # Empty case
    m, lo, hi = run_patient_paired_bootstrap({}, n_bootstraps=100)
    assert m == 0.0 and lo == 0.0 and hi == 0.0

    # Constant diff
    diffs = {f"P{i:03d}": 0.05 for i in range(20)}
    m, lo, hi = run_patient_paired_bootstrap(diffs, n_bootstraps=500, seed=42)
    assert pytest.approx(m, abs=1e-5) == 0.05
    assert pytest.approx(lo, abs=1e-5) == 0.05
    assert pytest.approx(hi, abs=1e-5) == 0.05


def test_evaluate_structure_predictions_empty_and_non_empty():
    # Image 1: Non-empty GT and perfect prediction
    p1 = np.zeros((20, 20), dtype=np.float32)
    p1[5:10, 5:10] = 0.9
    g1 = np.zeros((20, 20), dtype=np.uint8)
    g1[5:10, 5:10] = 1

    # Image 2: Empty GT and empty prediction (true negative)
    p2 = np.zeros((20, 20), dtype=np.float32)
    g2 = np.zeros((20, 20), dtype=np.uint8)

    # Image 3: Empty GT but false positive prediction
    p3 = np.zeros((20, 20), dtype=np.float32)
    p3[2:4, 2:4] = 0.8
    g3 = np.zeros((20, 20), dtype=np.uint8)

    res = evaluate_structure_predictions(
        [p1, p2, p3], [g1, g2, g3], threshold=0.5, morph_type="none", min_area=0
    )

    assert res["n_non_empty"] == 1
    assert pytest.approx(res["dice_mean"], abs=1e-4) == 1.0
    assert res["n_empty_gt"] == 2
    assert res["empty_gt_detected"] == 1
    assert pytest.approx(res["empty_gt_detection_rate"], abs=1e-4) == 0.5
    assert pytest.approx(res["non_empty_pred_rate_on_empty_gt"], abs=1e-4) == 0.5


def test_tune_val_grid():
    # Synthetic validation dataset
    val_probs = {s: [] for s in STRUCTURE_NAMES}
    val_gts = {s: [] for s in STRUCTURE_NAMES}

    for s in STRUCTURE_NAMES:
        p = np.zeros((30, 30), dtype=np.float32)
        p[5:15, 5:15] = 0.65  # Strong signal
        p[20:23, 20:23] = 0.55  # Small noisy blob

        g = np.zeros((30, 30), dtype=np.uint8)
        g[5:15, 5:15] = 1  # Ground truth only has the large component

        val_probs[s].append(p)
        val_gts[s].append(g)

    tuning = tune_val_grid(
        val_probs,
        val_gts,
        threshold_grid=[0.5, 0.6],
        min_area_grid=[50],
    )

    assert len(tuning) == 4
    for s in STRUCTURE_NAMES:
        best = tuning[s]["best"]
        assert "threshold" in best
        assert "morph_type" in best
        assert best["val_dice"] >= 0.95
