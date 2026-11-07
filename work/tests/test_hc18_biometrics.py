"""Unit tests for HC18 biometric evaluation, submission formatting, tercile bins, and bootstrap CI."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from fetal.eval.hc18_eval import (
    compute_patient_bootstrap_ci,
    compute_tercile_analysis,
    fit_submission_ellipse,
    measure_biometrics_from_prob,
    measure_from_prob,
)


def _create_synthetic_ellipse_mask(
    height: int = 384,
    width: int = 384,
    center: tuple[int, int] = (192, 192),
    axes: tuple[int, int] = (120, 80),
    angle_deg: float = 30.0,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
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
    return mask


def test_measure_biometrics_matches_legacy():
    true_mask = _create_synthetic_ellipse_mask(384, 384, (192, 192), (100, 70), 20.0)
    pred_mask = _create_synthetic_ellipse_mask(384, 384, (194, 190), (102, 69), 22.0)

    prob_map = pred_mask.astype(np.float32) / 255.0
    mask_tensor = torch.from_numpy(true_mask.astype(np.float32) / 255.0).unsqueeze(0)
    pixel_size = 0.15
    scale = 0.75

    legacy_res = measure_from_prob(prob_map, mask_tensor, pixel_size, scale)
    detailed_res = measure_biometrics_from_prob(prob_map, mask_tensor, pixel_size, scale)

    assert legacy_res is not None
    assert detailed_res is not None
    assert pytest.approx(legacy_res[0], rel=1e-5) == detailed_res["abs_err_hc"]
    assert pytest.approx(legacy_res[1], rel=1e-5) == detailed_res["abs_err_bpd"]
    assert detailed_res["true_hc"] > 0
    assert detailed_res["pred_hc"] > 0
    assert detailed_res["true_bpd"] > 0
    assert detailed_res["pred_bpd"] > 0


def test_fit_submission_ellipse_synthetic():
    h, w = 540, 800
    cx_px, cy_px = 400, 270
    axis_major, axis_minor = 150, 100
    angle_deg = 45.0
    pixel_size = 0.20

    mask = _create_synthetic_ellipse_mask(h, w, (cx_px, cy_px), (axis_major, axis_minor), angle_deg)
    prob_native = mask.astype(np.float32) / 255.0

    res = fit_submission_ellipse(prob_native, pixel_size, filename="042_HC.png")

    assert res["filename"] == "042_HC.png"
    assert pytest.approx(res["center_x_mm"], abs=1.0) == cx_px * pixel_size
    assert pytest.approx(res["center_y_mm"], abs=1.0) == cy_px * pixel_size
    assert res["semi_axes_a_mm"] >= res["semi_axes_b_mm"]
    assert pytest.approx(res["semi_axes_a_mm"], abs=1.0) == axis_major * pixel_size
    assert pytest.approx(res["semi_axes_b_mm"], abs=1.0) == axis_minor * pixel_size
    assert 0.0 <= res["angle_rad"] <= 2 * np.pi


def test_compute_tercile_analysis_partition():
    df = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(12)],
            "true_HC_mm": [100.0 + i * 20.0 for i in range(12)],
            "abs_err_HC": [1.5 + (i % 3) * 0.5 for i in range(12)],
            "abs_err_BPD": [0.6 + (i % 2) * 0.2 for i in range(12)],
        }
    )

    summary = compute_tercile_analysis(df)
    assert len(summary) == 3
    assert set(summary["tercile"]) == {"Small (< T1)", "Medium (T1-T2)", "Large (> T2)"}
    assert summary["n_samples"].sum() == 12
    for _, row in summary.iterrows():
        assert row["min_true_hc_mm"] <= row["max_true_hc_mm"]
        assert row["hc_rmse"] >= row["hc_mae"]
        assert row["bpd_rmse"] >= row["bpd_mae"]


def test_compute_patient_bootstrap_ci_deterministic():
    df = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p3", "p4", "p5", "p6"],
            "true_HC_mm": [120.0, 125.0, 180.0, 220.0, 260.0, 310.0, 340.0],
            "abs_err_HC": [2.1, 2.3, 1.8, 3.2, 2.0, 2.5, 1.9],
            "abs_err_BPD": [0.7, 0.8, 0.5, 0.9, 0.6, 0.7, 0.5],
        }
    )

    ci1 = compute_patient_bootstrap_ci(df, n_boot=200, seed=42)
    ci2 = compute_patient_bootstrap_ci(df, n_boot=200, seed=42)

    assert ci1 == ci2
    ci_low, ci_high = ci1
    assert ci_low < ci_high
    mean_mae = df["abs_err_HC"].mean()
    assert ci_low <= mean_mae <= ci_high
