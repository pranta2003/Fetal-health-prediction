import cv2
import numpy as np
import pytest
import torch

import ac_v3_original as orig_ac
from fetal.eval import ac_geometry as new_ac
from fetal.eval import ellipse_fitting


def test_perms_exact():
    """Verify permutation tensor is strictly identical to original."""
    assert torch.equal(orig_ac.PERMS_T, new_ac.PERMS_T)
    assert orig_ac.PERMS == new_ac.PERMS


def test_ac_t_random_inputs():
    """Verify ac_t produces identical outputs to original on random inputs."""
    torch.manual_seed(42)
    tad_mm = torch.rand(100, dtype=torch.float32) * 100.0 + 10.0
    apad_mm = torch.rand(100, dtype=torch.float32) * 100.0 + 10.0

    orig_out = orig_ac.ac_t(tad_mm, apad_mm)
    new_out = new_ac.ac_t(tad_mm, apad_mm)

    assert torch.allclose(orig_out, new_out, atol=1e-7, rtol=1e-7)


def test_lengths_mm_random_inputs():
    """Verify lengths_mm produces identical outputs to original on random inputs."""
    torch.manual_seed(42)
    B = 25
    pts = torch.rand(B, 4, 2, dtype=torch.float32) * 200.0
    mmpp = torch.rand(B, dtype=torch.float32) * 0.5 + 0.1

    orig_tp, orig_ap = orig_ac.lengths_mm(pts, mmpp)
    new_tp, new_ap = new_ac.lengths_mm(pts, mmpp)

    assert torch.allclose(orig_tp, new_tp, atol=1e-7, rtol=1e-7)
    assert torch.allclose(orig_ap, new_ap, atol=1e-7, rtol=1e-7)


def test_match_random_inputs():
    """Verify match produces identical outputs and indices to original on random inputs."""
    torch.manual_seed(42)
    B = 30
    pred = torch.rand(B, 4, 2, dtype=torch.float32) * 256.0
    gt = torch.rand(B, 4, 2, dtype=torch.float32) * 256.0

    orig_matched, orig_order = orig_ac.match(pred, gt)
    new_matched, new_order = new_ac.match(pred, gt)

    assert torch.allclose(orig_matched, new_matched, atol=1e-7, rtol=1e-7)
    assert torch.equal(orig_order, new_order)


def test_ellipse_fitting_hc_bpd():
    """Verify Ramanujan HC and BPD ellipse formula on synthetic semi-axes."""
    semi_major = 50.0
    semi_minor = 35.0
    hc, bpd = ellipse_fitting.hc_bpd_from_ellipse_params(semi_major, semi_minor)

    # Reference manual calculation:
    a, b = semi_major, semi_minor
    expected_hc = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
    expected_bpd = 2 * min(a, b)

    assert np.isclose(hc, expected_hc, atol=1e-7)
    assert np.isclose(bpd, expected_bpd, atol=1e-7)
    assert bpd == 70.0


def test_fit_ellipse_robust():
    """Verify fit_ellipse_robust fits an ellipse to a synthetic contour."""
    # Generate an ellipse contour
    center = (100, 100)
    axes = (40, 20)
    angle = 30
    pts = cv2.ellipse2Poly(center, axes, angle, 0, 360, 5)
    contour = pts.reshape(-1, 1, 2)

    fitted = ellipse_fitting.fit_ellipse_robust(contour)
    (f_center_x, f_center_y), (f_axis1, f_axis2), f_angle = fitted

    assert np.isclose(f_center_x, 100, atol=1.0)
    assert np.isclose(f_center_y, 100, atol=1.0)
    # OpenCV ellipse axes lengths are full diameters: 2*axes
    assert np.isclose(min(f_axis1, f_axis2), 40, atol=1.0)
    assert np.isclose(max(f_axis1, f_axis2), 80, atol=1.0)

