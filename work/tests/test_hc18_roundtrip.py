"""Unit tests for HC18 ellipse roundtrip decoding and rasterization conventions."""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from fetal.eval.ellipse_fitting import hc_bpd_from_ellipse_params
from fetal.eval.hc18_eval import fit_submission_ellipse

REPO_ROOT = Path(__file__).resolve().parent.parent
ROUNDTRIP_JSON = REPO_ROOT / "docs" / "results" / "hc18_official" / "roundtrip_dice.json"


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = np.logical_and(mask1 > 0, mask2 > 0).sum()
    total = (mask1 > 0).sum() + (mask2 > 0).sum()
    return float(2.0 * inter / total) if total > 0 else 1.0


@pytest.mark.parametrize("angle_deg", [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
def test_synthetic_ellipses_at_many_angles_roundtrip_dice_above_99(angle_deg: int):
    """Verify that eccentric ellipses at many angles round-trip to Dice > 0.99

    under the corrected submission convention (major axis a along angle_rad).
    """
    h, w = 400, 400
    center = (200, 200)
    axes = (120, 70)  # Distinct major/minor axes: ratio 1.71
    pixel_size = 0.15

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        img=mask,
        center=center,
        axes=axes,
        angle=float(angle_deg),
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=cv2.FILLED,
    )

    # 1. Fit submission ellipse with corrected convention
    prob_map = (mask / 255.0).astype(np.float32)
    sub = fit_submission_ellipse(prob_map, pixel_size=pixel_size, filename=f"syn_{angle_deg}.png")

    # 2. Rasterize back under corrected submission convention
    cx_px = sub["center_x_mm"] / pixel_size
    cy_px = sub["center_y_mm"] / pixel_size
    a_px = sub["semi_axes_a_mm"] / pixel_size
    b_px = sub["semi_axes_b_mm"] / pixel_size
    recon_angle_deg = np.rad2deg(sub["angle_rad"])

    m_round = np.zeros_like(mask)
    cv2.ellipse(
        img=m_round,
        center=(int(round(cx_px)), int(round(cy_px))),
        axes=(int(round(a_px)), int(round(b_px))),
        angle=recon_angle_deg,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=cv2.FILLED,
    )

    dice = compute_dice(mask, m_round)
    assert dice > 0.99, f"Angle {angle_deg}°: expected Dice > 0.99, got {dice:.4f}"


def test_hc_value_identical_before_after_for_random_inputs():
    """Verify that Ramanujan HC calculation depends only on semi-axes (a, b) and is identical before and after angle fix."""
    rng = np.random.RandomState(42)
    for _ in range(50):
        a = rng.uniform(20.0, 120.0)
        b = rng.uniform(10.0, a)  # Ensure b <= a

        hc_val, bpd_val = hc_bpd_from_ellipse_params(a, b)

        # Ramanujan formula: pi * [ 3(a+b) - sqrt((3a+b)(a+3b)) ]
        expected_hc = np.pi * (3.0 * (a + b) - np.sqrt((3.0 * a + b) * (a + 3.0 * b)))
        expected_bpd = 2.0 * b

        assert pytest.approx(hc_val, abs=1e-8) == expected_hc
        assert pytest.approx(bpd_val, abs=1e-8) == expected_bpd


def test_roundtrip_dice_json_file_contents():
    assert ROUNDTRIP_JSON.is_file(), f"Missing {ROUNDTRIP_JSON}"
    with open(ROUNDTRIP_JSON, "r", encoding="utf-8") as f:
        res = json.load(f)

    assert res["n_masks"] == 999
    assert res["best_candidate"] == "c1"
    assert res["candidates"]["c1"]["mean"] > 0.99
    assert res["candidates"]["legacy"]["mean"] == pytest.approx(0.8542, abs=0.01)
    assert "c1" in res["result_statement"]
