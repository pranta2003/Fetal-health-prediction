#!/usr/bin/env python3
"""70_build_blend_candidate.py: Build and verify the heterogeneous blend candidate.

Averages ellipse parameters of:
  - candidates/submission_nnunet_f0_measfix.csv
  - candidates/submission_ensemble_5fold_corrected_measfix.csv
Outputs:
  - candidates/submission_blend_nnunetf0_ens5_measfix.csv
Checks:
  - Exactly 335 rows
  - Matching column order and filenames
  - Angles averaged on the circle modulo pi
  - Assertion that no image has |delta_angle| > 0.5 rad
  - Strict LF line endings
  - SHA-256
  - Mean HC difference vs each parent
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
src_dir = REPO_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fetal.eval.ellipse_fitting import hc_bpd_from_ellipse_params

NNUNET_CSV = REPO_ROOT / "candidates/submission_nnunet_f0_measfix.csv"
ENS5_CSV = REPO_ROOT / "candidates/submission_ensemble_5fold_corrected_measfix.csv"
OUT_CSV = REPO_ROOT / "candidates/submission_blend_nnunetf0_ens5_measfix.csv"


def circular_mean_pi(theta1: np.ndarray, theta2: np.ndarray) -> np.ndarray:
    """Compute circular average of two ellipse angles modulo pi."""
    # Double the angles to place them on the standard 2*pi circle
    s = np.sin(2.0 * theta1) + np.sin(2.0 * theta2)
    c = np.cos(2.0 * theta1) + np.cos(2.0 * theta2)
    mean_2theta = np.arctan2(s, c)
    return mean_2theta / 2.0


def main():
    print(f"Reading parent 1: {NNUNET_CSV}")
    df_nn = pd.read_csv(NNUNET_CSV)
    print(f"Reading parent 2: {ENS5_CSV}")
    df_ens = pd.read_csv(ENS5_CSV)

    assert len(df_nn) == 335, f"Expected 335 rows, got {len(df_nn)}"
    assert len(df_ens) == 335, f"Expected 335 rows, got {len(df_ens)}"
    assert (df_nn["filename"] == df_ens["filename"]).all(), "Filenames do not match!"

    # Semi-axes validation
    assert (df_nn["semi_axes_a_mm"] >= df_nn["semi_axes_b_mm"]).all(), "df_nn has a < b"
    assert (df_ens["semi_axes_a_mm"] >= df_ens["semi_axes_b_mm"]).all(), "df_ens has a < b"

    th_nn = df_nn["angle_rad"].to_numpy()
    th_ens = df_ens["angle_rad"].to_numpy()

    # Angular difference modulo pi
    diff_angle = (th_ens - th_nn + math.pi / 2.0) % math.pi - math.pi / 2.0
    max_diff = float(np.max(np.abs(diff_angle)))
    print(f"Angle difference modulo pi: max={max_diff:.4f} rad, mean={np.mean(np.abs(diff_angle)):.4f} rad")
    assert max_diff <= 0.5, f"Found angle difference > 0.5 rad: {max_diff}"

    # Blend ellipse parameters
    blend_cx = (df_nn["center_x_mm"].to_numpy() + df_ens["center_x_mm"].to_numpy()) / 2.0
    blend_cy = (df_nn["center_y_mm"].to_numpy() + df_ens["center_y_mm"].to_numpy()) / 2.0
    blend_a = (df_nn["semi_axes_a_mm"].to_numpy() + df_ens["semi_axes_a_mm"].to_numpy()) / 2.0
    blend_b = (df_nn["semi_axes_b_mm"].to_numpy() + df_ens["semi_axes_b_mm"].to_numpy()) / 2.0
    blend_angle = circular_mean_pi(th_nn, th_ens)

    # Ensure a >= b for the blend
    assert (blend_a >= blend_b).all(), "Blend semi-axes violate a >= b"

    df_blend = pd.DataFrame({
        "filename": df_nn["filename"],
        "center_x_mm": np.round(blend_cx, 6),
        "center_y_mm": np.round(blend_cy, 6),
        "semi_axes_a_mm": np.round(blend_a, 6),
        "semi_axes_b_mm": np.round(blend_b, 6),
        "angle_rad": np.round(blend_angle, 6),
    })

    # Compare HC
    hc_nn = np.array([hc_bpd_from_ellipse_params(a, b)[0] for a, b in zip(df_nn["semi_axes_a_mm"], df_nn["semi_axes_b_mm"])])
    hc_ens = np.array([hc_bpd_from_ellipse_params(a, b)[0] for a, b in zip(df_ens["semi_axes_a_mm"], df_ens["semi_axes_b_mm"])])
    hc_blend = np.array([hc_bpd_from_ellipse_params(a, b)[0] for a, b in zip(df_blend["semi_axes_a_mm"], df_blend["semi_axes_b_mm"])])

    diff_blend_vs_nn = hc_blend - hc_nn
    diff_blend_vs_ens = hc_blend - hc_ens

    print("\n=== Head Circumference Comparisons (mm) ===")
    print(f"Mean HC nnU-Net:      {np.mean(hc_nn):.4f} mm")
    print(f"Mean HC 5-fold Ens:   {np.mean(hc_ens):.4f} mm")
    print(f"Mean HC Blend:        {np.mean(hc_blend):.4f} mm")
    print(f"Blend vs nnU-Net:     mean diff = {np.mean(diff_blend_vs_nn):+.4f} mm, mean abs diff = {np.mean(np.abs(diff_blend_vs_nn)):.4f} mm")
    print(f"Blend vs 5-fold Ens:  mean diff = {np.mean(diff_blend_vs_ens):+.4f} mm, mean abs diff = {np.mean(np.abs(diff_blend_vs_ens)):.4f} mm")

    # Strict LF output
    csv_str = df_blend.to_csv(index=False, lineterminator="\n")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.write_bytes(csv_str.encode("utf-8"))

    raw_bytes = OUT_CSV.read_bytes()
    assert b"\r\n" not in raw_bytes, "Found CRLF in output file!"
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    print(f"\nWrote {len(df_blend)} rows to {OUT_CSV}")
    print(f"SHA-256: {sha256}")


if __name__ == "__main__":
    main()
