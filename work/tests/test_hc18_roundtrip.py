"""Unit tests for HC18 ellipse roundtrip decoding and rasterization conventions."""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from fetal.eval.hc18_eval import fit_submission_ellipse

REPO_ROOT = Path(__file__).resolve().parent.parent
ROUNDTRIP_JSON = REPO_ROOT / "docs" / "results" / "hc18_official" / "roundtrip_dice.json"


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = np.logical_and(mask1 > 0, mask2 > 0).sum()
    total = (mask1 > 0).sum() + (mask2 > 0).sum()
    return float(2.0 * inter / total) if total > 0 else 1.0


def test_synthetic_ellipse_roundtrip_dice_above_99():
    """Verify that a small synthetic near-circular ellipse round-trips to Dice > 0.99

    under the submission convention (angle as written, a along rotated x).
    """
    h, w = 300, 300
    center = (150, 150)
    axes = (60, 60)  # Circular head proxy
    pixel_size = 0.1

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        img=mask,
        center=center,
        axes=axes,
        angle=0.0,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=cv2.FILLED,
    )

    # 1. Fit submission ellipse
    prob_map = (mask / 255.0).astype(np.float32)
    sub = fit_submission_ellipse(prob_map, pixel_size=pixel_size, filename="synthetic.png")

    # 2. Rasterize back under submission convention
    cx_px = sub["center_x_mm"] / pixel_size
    cy_px = sub["center_y_mm"] / pixel_size
    a_px = sub["semi_axes_a_mm"] / pixel_size
    b_px = sub["semi_axes_b_mm"] / pixel_size
    angle_deg = np.rad2deg(sub["angle_rad"])

    m_round = np.zeros_like(mask)
    cv2.ellipse(
        img=m_round,
        center=(int(round(cx_px)), int(round(cy_px))),
        axes=(int(round(a_px)), int(round(b_px))),
        angle=angle_deg,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=cv2.FILLED,
    )

    dice = compute_dice(mask, m_round)
    assert dice > 0.99, f"Expected roundtrip Dice > 0.99, got {dice:.4f}"


def test_synthetic_eccentric_ellipse_roundtrip_under_best_convention():
    """Verify that an eccentric ellipse round-trips to Dice > 0.99 under convention c3 (swapped axes)."""
    h, w = 400, 500
    center = (250, 200)
    axes = (120, 80)
    pixel_size = 0.12

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        img=mask,
        center=center,
        axes=axes,
        angle=35.0,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=cv2.FILLED,
    )

    prob_map = (mask / 255.0).astype(np.float32)
    sub = fit_submission_ellipse(prob_map, pixel_size=pixel_size, filename="eccentric.png")

    cx_px = sub["center_x_mm"] / pixel_size
    cy_px = sub["center_y_mm"] / pixel_size
    a_px = sub["semi_axes_a_mm"] / pixel_size
    b_px = sub["semi_axes_b_mm"] / pixel_size
    angle_deg = np.rad2deg(sub["angle_rad"])

    # Convention c3: swap a and b
    m_c3 = np.zeros_like(mask)
    cv2.ellipse(
        img=m_c3,
        center=(int(round(cx_px)), int(round(cy_px))),
        axes=(int(round(b_px)), int(round(a_px))),
        angle=angle_deg,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=cv2.FILLED,
    )

    dice_c3 = compute_dice(mask, m_c3)
    assert dice_c3 > 0.99, f"Expected convention c3 Dice > 0.99, got {dice_c3:.4f}"


def test_roundtrip_dice_json_file_contents():
    assert ROUNDTRIP_JSON.is_file(), f"Missing {ROUNDTRIP_JSON}"
    with open(ROUNDTRIP_JSON, "r", encoding="utf-8") as f:
        res = json.load(f)

    assert res["n_masks"] == 999
    assert res["best_candidate"] in ["c3", "c4"]
    assert res["candidates"]["c3"]["mean"] > 0.99
    assert res["candidates"]["c1"]["mean"] == pytest.approx(0.8542, abs=0.01)
    assert "highest mean Dice" in res["result_statement"]
