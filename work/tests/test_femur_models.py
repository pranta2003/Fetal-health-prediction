"""Tests for Femur (FL) landmark models, decoders, losses, and splits."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
import torch

from fetal import config
from fetal.data.femur_dataset import build_femur_data, split_femur_data
from fetal.models.femur_landmark import (
    decode_femur_peaks,
    femur_order_invariant_loss,
    generate_femur_heatmap,
    match_endpoints,
    predict_femur_batch,
)
from fetal.eval.femur_evaluator import FemurDS, compute_femur_metrics, letterbox_image


def test_heatmap_encode_decode_roundtrip():
    """Verify heatmap encode -> decode round-trip error is strictly under 0.5 px."""
    S = 384
    sigma = 4.0
    rng = np.random.RandomState(42)

    errors = []
    # Test 20 synthetic endpoint pairs across canvas
    for _ in range(20):
        # Pick endpoints with reasonable separation (e.g. 50-250 px)
        p1 = rng.uniform(50, 330, size=2)
        angle = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(60, 220)
        p2 = p1 + np.array([length * np.cos(angle), length * np.sin(angle)])
        p2 = np.clip(p2, 30, 350)

        pts = torch.tensor(np.stack([p1, p2]), dtype=torch.float32).unsqueeze(0)  # (1, 2, 2)
        prob, _ = generate_femur_heatmap(pts, S=S, sigma=sigma)

        decoded = decode_femur_peaks(prob, S=S, k=9, r=8, min_dist=10.0)  # (1, 2, 2)
        matched, _ = match_endpoints(decoded, pts)

        d = (matched - pts).norm(dim=-1).squeeze(0)  # (2,)
        errors.append(d[0].item())
        errors.append(d[1].item())

    max_err = max(errors)
    mean_err = np.mean(errors)
    print(f"Heatmap round-trip: max error = {max_err:.4f} px, mean error = {mean_err:.4f} px")
    assert max_err < 0.5, f"Expected round-trip error < 0.5 px, got {max_err:.4f} px"


def test_hflip_endpoints_consistency():
    """Verify that horizontal flip reflects endpoints symmetrically and preserves FL."""
    S = 384
    pt1 = torch.tensor([120.5, 180.2])
    pt2 = torch.tensor([260.8, 290.4])
    pts = torch.stack([pt1, pt2]).unsqueeze(0)  # (1, 2, 2)

    # Flip horizontally: x' = S - 1 - x, y' = y
    pts_flipped = pts.clone()
    pts_flipped[:, :, 0] = (S - 1) - pts_flipped[:, :, 0]

    # Generate heatmaps
    prob0, _ = generate_femur_heatmap(pts, S=S)
    prob1, _ = generate_femur_heatmap(pts_flipped, S=S)

    # Decode flipped heatmap
    dec1 = decode_femur_peaks(prob1, S=S)
    # Map decoded back: x = S - 1 - x'
    dec1_back = dec1.clone()
    dec1_back[:, :, 0] = (S - 1) - dec1_back[:, :, 0]

    matched, _ = match_endpoints(dec1_back, pts)
    diff = (matched - pts).norm(dim=-1).max().item()
    assert diff < 0.5, f"Flip consistency error too large: {diff:.4f} px"

    # FL length should be identically preserved
    len_orig = (pts[:, 1] - pts[:, 0]).norm(dim=-1).item()
    len_flipped = (pts_flipped[:, 1] - pts_flipped[:, 0]).norm(dim=-1).item()
    assert pytest.approx(len_orig, rel=1e-5) == len_flipped


def test_fl_mm_equals_dataset_fl_mm():
    """Verify that FL mm from true endpoints equals the dataset computed fl_length_mm."""
    # Build synthetic test dataframe
    df = pd.DataFrame([
        {
            "image_name": "test_img.png",
            "image_path": "non_existent.png",
            "patient": "FP_Patient001",
            "source": "FP",
            "official_split": "test",
            "px_to_mm_rate": 0.15,
            "fl_1_x": 100.0,
            "fl_1_y": 120.0,
            "fl_2_x": 220.0,
            "fl_2_y": 280.0,
            "fl_length_mm": np.hypot(220.0 - 100.0, 280.0 - 120.0) * 0.15,
        }
    ])

    scale = 0.5
    px, py = 10, 20
    p1 = torch.tensor([100.0 * scale + px, 120.0 * scale + py])
    p2 = torch.tensor([220.0 * scale + px, 280.0 * scale + py])
    gt_pts = torch.stack([p1, p2]).unsqueeze(0)  # (1, 2, 2)
    mmpp = torch.tensor([0.15 / scale])

    # Compute metrics when pred equals gt
    metrics = compute_femur_metrics(gt_pts, gt_pts, mmpp)
    fl_pred = metrics["fl_pred"].item()
    fl_true = metrics["fl_true"].item()
    fl_dataset = df.iloc[0]["fl_length_mm"]

    assert pytest.approx(fl_pred, rel=1e-5) == fl_dataset
    assert pytest.approx(fl_true, rel=1e-5) == fl_dataset
    assert metrics["fl_err"].item() == pytest.approx(0.0, abs=1e-6)


def test_order_invariant_loss_swapped_endpoints():
    """Verify that order-invariant loss gives exactly the same value when ground-truth endpoints are swapped."""
    pred = torch.tensor([[[100.0, 150.0], [250.0, 300.0]]], dtype=torch.float32)  # (1, 2, 2)
    gt_order1 = torch.tensor([[[98.0, 152.0], [253.0, 298.0]]], dtype=torch.float32)  # (1, 2, 2)
    gt_order2 = torch.tensor([[[253.0, 298.0], [98.0, 152.0]]], dtype=torch.float32)  # swapped

    loss1 = femur_order_invariant_loss(pred, gt_order1)
    loss2 = femur_order_invariant_loss(pred, gt_order2)

    assert loss1.item() == pytest.approx(loss2.item(), rel=1e-6)

    # Also verify for batch of random endpoints
    rng = torch.Generator().manual_seed(42)
    pred_batch = torch.randn(8, 2, 2, generator=rng) * 100 + 200
    gt_batch1 = torch.randn(8, 2, 2, generator=rng) * 100 + 200
    gt_batch2 = torch.stack([gt_batch1[:, 1], gt_batch1[:, 0]], dim=1)

    loss_b1 = femur_order_invariant_loss(pred_batch, gt_batch1)
    loss_b2 = femur_order_invariant_loss(pred_batch, gt_batch2)
    assert loss_b1.item() == pytest.approx(loss_b2.item(), rel=1e-6)


def test_split_zero_patient_overlap():
    """Verify that femur train, dev, and test splits have 0 patient overlap on real dataset."""
    try:
        data = build_femur_data(drop_missing_rate=True)
    except FileNotFoundError:
        pytest.skip("Local raw dataset not found in this environment")

    tr, dv, te = split_femur_data(data, dev_fraction=0.12, seed=0)

    tr_pats = set(tr["patient"])
    dv_pats = set(dv["patient"])
    te_pats = set(te["patient"])

    assert len(tr_pats & dv_pats) == 0, "Train and Val share patients!"
    assert len(tr_pats & te_pats) == 0, "Train and Test share patients!"
    assert len(dv_pats & te_pats) == 0, "Val and Test share patients!"

    # Also verify split counts match follow-up report
    assert len(tr) == 451, f"Expected 451 train images, got {len(tr)}"
    assert len(dv) == 65, f"Expected 65 validation images, got {len(dv)}"
    assert len(te) == 348, f"Expected 348 test images, got {len(te)}"
    assert len(data) == 864, f"Expected 864 total images, got {len(data)}"
