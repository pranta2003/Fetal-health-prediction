"""Fast unit tests for HC18 challenge submission ellipse formatting and pipeline."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from fetal.eval.hc18_eval import fit_submission_ellipse


def _create_synthetic_prob_map(
    shape: tuple[int, int] = (540, 800),
    center: tuple[int, int] = (400, 270),
    axes: tuple[int, int] = (150, 100),
    angle_deg: float = 30.0,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(
        img=mask,
        center=center,
        axes=axes,
        angle=angle_deg,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=cv2.FILLED,
    )
    return mask.astype(np.float32) / 255.0


def test_submission_format_columns_and_ranges():
    prob_map = _create_synthetic_prob_map(shape=(600, 800), center=(400, 300), axes=(180, 120), angle_deg=25.0)
    pixel_size = 0.15
    filename = "007_HC.png"

    row = fit_submission_ellipse(prob_map, pixel_size=pixel_size, filename=filename)

    expected_cols = ["filename", "center_x_mm", "center_y_mm", "semi_axes_a_mm", "semi_axes_b_mm", "angle_rad"]
    assert list(row.keys()) == expected_cols

    assert row["filename"] == filename
    assert row["center_x_mm"] > 0
    assert row["center_y_mm"] > 0
    assert row["semi_axes_a_mm"] > 0
    assert row["semi_axes_b_mm"] > 0
    assert row["semi_axes_a_mm"] >= row["semi_axes_b_mm"]
    assert 0.0 <= row["angle_rad"] <= 2 * np.pi


def test_submission_empty_fallback():
    empty_map = np.zeros((400, 400), dtype=np.float32)
    row = fit_submission_ellipse(empty_map, pixel_size=0.1, filename="empty.png")

    assert row["filename"] == "empty.png"
    assert row["center_x_mm"] == 0.0
    assert row["center_y_mm"] == 0.0
    assert row["semi_axes_a_mm"] == 0.0
    assert row["semi_axes_b_mm"] == 0.0
    assert row["angle_rad"] == 0.0


def test_ensemble_probability_averaging_fits_valid_ellipse():
    # Simulate 5 fold probability maps with slight jitter
    h, w = 500, 700
    base_center = (350, 250)
    base_axes = (140, 90)
    pixel_size = 0.18

    maps = []
    for shift in [-2, -1, 0, 1, 2]:
        p = _create_synthetic_prob_map(
            shape=(h, w),
            center=(base_center[0] + shift, base_center[1] - shift),
            axes=base_axes,
            angle_deg=15.0 + shift,
        )
        maps.append(p)

    ensemble_prob = np.mean(maps, axis=0)
    assert ensemble_prob.shape == (h, w)
    assert 0.0 <= ensemble_prob.min() and ensemble_prob.max() <= 1.0

    res = fit_submission_ellipse(ensemble_prob, pixel_size=pixel_size, filename="ens_test.png")
    assert res["semi_axes_a_mm"] > 0
    assert res["semi_axes_b_mm"] > 0
    assert res["semi_axes_a_mm"] >= res["semi_axes_b_mm"]
    assert pytest.approx(res["center_x_mm"], abs=2.0) == base_center[0] * pixel_size
    assert pytest.approx(res["center_y_mm"], abs=2.0) == base_center[1] * pixel_size
