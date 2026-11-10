"""Tests for Hadlock Demo Step 1: Usable subjects audit and head external validation."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from fetal.eval.ellipse_fitting import hc_bpd_from_ellipse_params


def ramanujan_hc(semi_major_mm: float, semi_minor_mm: float) -> float:
    a, b = float(semi_major_mm), float(semi_minor_mm)
    return float(np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))))


def test_ramanujan_hc_circle():
    """For a circle (a = b = r), Ramanujan formula must equal 2 * pi * r."""
    r = 25.0
    hc = ramanujan_hc(r, r)
    expected = 2.0 * np.pi * r
    assert abs(hc - expected) < 1e-6


def test_hc_bpd_consistency():
    """Ellipse params should yield valid HC and BPD."""
    a, b = 40.0, 30.0
    hc, bpd = hc_bpd_from_ellipse_params(a, b)
    assert hc > bpd
    assert abs(bpd - 60.0) < 1e-6


def test_usable_subjects_counts():
    """Verify exact counts of usable subjects for Hadlock Demo 1(a) and 1(b)."""
    data_dir = Path("C:/fetal-data")
    if not (data_dir / "FP" / "annotations" / "Head.csv").exists():
        pytest.skip("Local fetal-data directory not available")

    import pandas as pd
    fp_h = pd.read_csv(data_dir / "FP" / "annotations" / "Head.csv")
    fp_a = pd.read_csv(data_dir / "FP" / "annotations" / "Abdomen.csv")
    fp_f = pd.read_csv(data_dir / "FP" / "annotations" / "Femur.csv")
    fp_h["SubjectID"] = fp_h["SubjectID"].astype(int)
    fp_a["SubjectID"] = fp_a["SubjectID"].astype(int)
    fp_f["SubjectID"] = fp_f["Patient"].astype(int)

    ucl_h = pd.read_csv(data_dir / "UCL" / "annotations" / "px_to_mm" / "Head.csv")
    ucl_a = pd.read_csv(data_dir / "UCL" / "annotations" / "px_to_mm" / "Abdomen.csv")
    ucl_f = pd.read_csv(data_dir / "UCL" / "annotations" / "px_to_mm" / "Femur.csv")
    for df in [ucl_h, ucl_a, ucl_f]:
        df["SubjectID"] = df["image_name"].str.extract(r"^(\d+)_")[0].astype(int)

    # FP
    fp_a_test = set(fp_a[fp_a["Split"] == "Test"]["SubjectID"])
    fp_f_test = set(fp_f[fp_f["Split"] == "Test"]["SubjectID"])
    fp_h_test = set(fp_h[fp_h["Split"] == "Test"]["SubjectID"])
    fp_h_all = set(fp_h["SubjectID"])

    fp_1a = fp_a_test & fp_f_test & fp_h_all
    fp_1b = fp_a_test & fp_f_test & fp_h_test
    assert len(fp_1a) == 29, f"Expected 29 FP subjects for 1(a), got {len(fp_1a)}"
    assert len(fp_1b) == 17, f"Expected 17 FP subjects for 1(b), got {len(fp_1b)}"

    # UCL
    ucl_a_test = set(ucl_a[ucl_a["Split"] == "Test"]["SubjectID"])
    ucl_f_test = set(ucl_f[ucl_f["Split"] == "Test"]["SubjectID"])
    ucl_h_test = set(ucl_h[ucl_h["Split"] == "Test"]["SubjectID"])
    ucl_h_all = set(ucl_h["SubjectID"])

    ucl_1a = ucl_a_test & ucl_f_test & ucl_h_all
    ucl_1b = ucl_a_test & ucl_f_test & ucl_h_test
    assert len(ucl_1a) == 5, f"Expected 5 UCL subjects for 1(a), got {len(ucl_1a)}"
    assert len(ucl_1b) == 5, f"Expected 5 UCL subjects for 1(b), got {len(ucl_1b)}"
