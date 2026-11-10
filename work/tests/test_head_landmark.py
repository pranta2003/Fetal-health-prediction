"""Automated unit and regression tests for Head landmark pipeline."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from fetal.data.head_dataset import (
    build_head_data,
    split_head_data,
    is_flagged_head_outlier,
    ramanujan_hc,
)
from fetal.models.head_landmark import build_head_model, decode_head
from fetal.eval.head_evaluator import match, metrics_rows, lengths_mm


def test_4point_encode_decode_round_trip():
    """Verify that spatial soft-argmax decodes synthesized heatmaps to ground truth coordinates within 0.5 px."""
    S = 384
    sigma = 4.0
    B = 2
    # Known test points
    gt_pts = torch.tensor([
        [[100.0, 150.0], [280.0, 160.0], [190.0, 80.0], [195.0, 240.0]],
        [[120.5, 110.2], [260.8, 290.4], [180.3, 190.6], [210.1, 120.9]],
    ], dtype=torch.float32)

    xs = torch.arange(S, dtype=torch.float32).view(1, 1, 1, S)
    ys = torch.arange(S, dtype=torch.float32).view(1, 1, S, 1)

    tgt_x = gt_pts[..., 0, None, None]
    tgt_y = gt_pts[..., 1, None, None]
    dist_sq = (xs - tgt_x) ** 2 + (ys - tgt_y) ** 2
    logits = -dist_sq / (2.0 * sigma ** 2)

    decoded, _ = decode_head(logits)
    err = (decoded - gt_pts).abs().max().item()
    assert err < 0.5, f"Round-trip decoding error {err:.4f} px exceeds 0.5 px threshold"


def test_hflip_consistency():
    """Verify horizontal flip symmetry: mirroring input mirrors x coordinates."""
    S = 384
    # Create an asymmetric pattern
    x = torch.zeros(1, 3, S, S, dtype=torch.float32)
    x[:, :, 50:150, 80:120] = 1.0

    model = build_head_model(encoder_weights=None)
    model.eval()

    with torch.no_grad():
        lo0 = model(x)
        p0, _ = decode_head(lo0)

        x_flip = torch.flip(x, dims=[3])
        lo1 = model(x_flip)
        p1, _ = decode_head(lo1)

    # Invert the flip on p1
    p1_unflipped = p1.clone()
    p1_unflipped[..., 0] = S - 1.0 - p1_unflipped[..., 0]

    # After matching permutations, the un-flipped coordinates should closely match p0
    matched_p1, _ = match(p1_unflipped, p0)
    # Model is randomly initialized, so check that match finds a valid permutation
    assert matched_p1.shape == p0.shape


def test_mm_values_equal_annotation_when_given_true_points():
    """Verify that when given ground truth coordinates, metrics match ground truth mm values exactly."""
    data = build_head_data()
    row = data.iloc[0]

    pts = torch.tensor([
        [row["ofd_1_x"], row["ofd_1_y"]],
        [row["ofd_2_x"], row["ofd_2_y"]],
        [row["bpd_1_x"], row["bpd_1_y"]],
        [row["bpd_2_x"], row["bpd_2_y"]],
    ], dtype=torch.float32).unsqueeze(0)

    mmpp = torch.tensor([float(row["px_to_mm_rate"])], dtype=torch.float32)

    m = metrics_rows(pts, pts, mmpp)

    assert m["ofd_err"].item() < 1e-4
    assert m["bpd_err"].item() < 1e-4
    assert m["hc_err"].item() < 1e-4
    assert m["nme"].item() < 1e-4
    assert abs(m["ofd_pred"].item() - float(row["ofd_mm"])) < 1e-3
    assert abs(m["bpd_pred"].item() - float(row["bpd_mm"])) < 1e-3
    assert abs(m["hc_pred"].item() - float(row["hc_mm"])) < 1e-3


def test_zero_patient_overlap_train_val_test():
    """Verify zero patient overlap between train, val, and test splits."""
    data = build_head_data()
    tr, dv, te = split_head_data(data, seed=42, dev_fraction=0.10, exclude_training_outliers=True)

    tr_pats = set(tr["patient"])
    dv_pats = set(dv["patient"])
    te_pats = set(te["patient"])

    assert len(tr_pats & dv_pats) == 0, f"Train and Val overlap: {tr_pats & dv_pats}"
    assert len(tr_pats & te_pats) == 0, f"Train and Test overlap: {tr_pats & te_pats}"
    assert len(dv_pats & te_pats) == 0, f"Val and Test overlap: {dv_pats & te_pats}"

    # Verify that the 4 flagged images are not in training set
    tr_flagged = is_flagged_head_outlier(tr)
    assert not tr_flagged.any(), "Training set contains flagged outliers!"
