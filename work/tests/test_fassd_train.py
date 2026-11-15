"""Unit tests for FASSD multilabel segmentation training and metrics.

Tests use purely synthetic data (no external file dependencies):
- Dataset shape, normalization, and RGB vs grayscale mode.
- FASSDMultiLabelLoss (BCE + Soft Dice) computation and backward pass.
- ModelEMA parameter updating.
- HD95 on known geometry and identical masks.
- Metric edge cases: all-zero GT, true negative empty detection, partial overlap.
- Patient-level bootstrap confidence interval sanity.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from fetal.data.fassd import STRUCTURE_BITFLAGS, STRUCTURE_NAMES
from fetal.seg.fassd_train import (
    FASSDKaggleDataset,
    FASSDMultiLabelLoss,
    ModelEMA,
    build_fassd_model,
    compute_hd95,
    compute_structure_metrics,
    run_patient_bootstrap_ci,
)


@pytest.fixture
def synthetic_dataset_dir(tmp_path: Path):
    """Create a temporary directory with synthetic images, bit-flag masks, and metadata."""
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    # Create 3 synthetic samples
    records = []
    for i in range(3):
        fname = f"sample_{i:02d}.png"
        img = (np.random.default_rng(i).uniform(0, 255, (768, 1024, 3))).astype(np.uint8)
        cv2.imwrite(str(images_dir / fname), img)

        # Bit-flag mask: artery (1) in quadrant 1, liver (2) in quadrant 2
        mask = np.zeros((768, 1024), dtype=np.uint8)
        if i == 0:
            mask[100:300, 100:300] |= STRUCTURE_BITFLAGS["artery"]
            mask[400:600, 400:600] |= STRUCTURE_BITFLAGS["liver"]
            mask[200:400, 200:400] |= STRUCTURE_BITFLAGS["stomach"]
            mask[500:700, 100:300] |= STRUCTURE_BITFLAGS["vein"]
        elif i == 1:
            mask[100:200, 100:200] |= STRUCTURE_BITFLAGS["artery"]
            # liver empty
            mask[300:500, 300:500] |= STRUCTURE_BITFLAGS["stomach"]
            mask[100:300, 500:700] |= STRUCTURE_BITFLAGS["vein"]
        else:
            # all empty
            pass

        cv2.imwrite(str(masks_dir / fname), mask)
        records.append({
            "filename": fname,
            "patient": f"P{i:03d}",
            "group": f"G{i:03d}",
            "split": "train" if i < 2 else "val",
        })

    df = pd.DataFrame(records)
    return images_dir, masks_dir, df


def test_fassd_dataset_shapes_and_rgb_vs_gray(synthetic_dataset_dir):
    images_dir, masks_dir, df = synthetic_dataset_dir

    # 1. RGB Mode
    ds_rgb = FASSDKaggleDataset(images_dir, masks_dir, df, input_mode="rgb", is_train=False)
    item_rgb = ds_rgb[0]
    assert item_rgb["image"].shape == (3, 384, 512)
    assert item_rgb["mask"].shape == (4, 384, 512)
    assert item_rgb["image"].dtype == torch.float32
    assert item_rgb["mask"].dtype == torch.float32

    # 2. Gray Mode
    ds_gray = FASSDKaggleDataset(images_dir, masks_dir, df, input_mode="gray", is_train=False)
    item_gray = ds_gray[0]
    assert item_gray["image"].shape == (3, 384, 512)
    # In grayscale replicated mode before per-channel ImageNet normalization, channels were equal.
    # Check that channel 0 and 1 are non-trivial tensors
    assert item_gray["image"][0].std() > 0


def test_multilabel_loss_and_backward():
    loss_fn = FASSDMultiLabelLoss()
    logits = torch.randn(2, 4, 64, 64, requires_grad=True)
    targets = torch.randint(0, 2, (2, 4, 64, 64)).float()

    total_loss, bce, dice_loss = loss_fn(logits, targets)

    assert total_loss.item() > 0
    assert bce.item() >= 0
    assert dice_loss.item() >= 0

    total_loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_model_ema():
    # Simple linear model to test EMA mechanics
    model = nn.Sequential(nn.Linear(10, 10, bias=False))
    with torch.no_grad():
        model[0].weight.fill_(1.0)

    ema = ModelEMA(model, decay=0.9)
    assert torch.allclose(ema.module[0].weight, torch.full((10, 10), 1.0))

    # Update model weight to 2.0
    with torch.no_grad():
        model[0].weight.fill_(2.0)

    ema.update(model)
    # Expected: 0.9 * 1.0 + 0.1 * 2.0 = 1.1
    expected = torch.full((10, 10), 1.1)
    assert torch.allclose(ema.module[0].weight, expected, atol=1e-5)


def test_hd95_known_shapes():
    # Test identical shapes -> HD95 == 0.0
    mask1 = np.zeros((100, 100), dtype=bool)
    mask1[20:60, 20:60] = True
    assert compute_hd95(mask1, mask1) == 0.0

    # Test shifted box by exactly 10 pixels horizontally
    mask2 = np.zeros((100, 100), dtype=bool)
    mask2[20:60, 30:70] = True
    hd95 = compute_hd95(mask1, mask2)
    assert pytest.approx(hd95, abs=0.5) == 10.0

    # Test empty vs non-empty -> returns diagonal
    empty_mask = np.zeros((100, 100), dtype=bool)
    diag = math.hypot(100, 100)
    assert pytest.approx(compute_hd95(empty_mask, mask1), abs=1e-3) == diag
    assert compute_hd95(empty_mask, empty_mask) == 0.0


def test_compute_structure_metrics_edge_cases():
    pred = np.zeros((4, 50, 50), dtype=np.uint8)
    gt = np.zeros((4, 50, 50), dtype=np.uint8)

    # Channel 0: artery - Both empty
    # Channel 1: liver - GT empty, pred non-empty
    pred[1, 10:20, 10:20] = 1
    # Channel 2: stomach - Perfect match
    gt[2, 10:30, 10:30] = 1
    pred[2, 10:30, 10:30] = 1
    # Channel 3: vein - Half overlap
    gt[3, 10:30, 10:30] = 1
    pred[3, 10:30, 20:40] = 1

    metrics = compute_structure_metrics(pred, gt)

    # Artery (both empty)
    assert metrics["artery"]["gt_empty"] is True
    assert metrics["artery"]["pred_empty"] is True
    assert metrics["artery"]["empty_detected"] is True
    assert metrics["artery"]["dice"] is None

    # Liver (GT empty, Pred false positive)
    assert metrics["liver"]["gt_empty"] is True
    assert metrics["liver"]["pred_empty"] is False
    assert metrics["liver"]["empty_detected"] is False
    assert metrics["liver"]["dice"] is None

    # Stomach (Perfect match)
    assert metrics["stomach"]["gt_empty"] is False
    assert metrics["stomach"]["dice"] == pytest.approx(1.0)
    assert metrics["stomach"]["iou"] == pytest.approx(1.0)
    assert metrics["stomach"]["hd95"] == pytest.approx(0.0)

    # Vein (Half overlap)
    # Area GT = 20*20 = 400, Area Pred = 20*20 = 400.
    # Overlap = 20 * 10 = 200.
    # Dice = 2*200 / (400 + 400) = 0.5
    # IoU = 200 / (400 + 400 - 200) = 200 / 600 = 1/3
    assert metrics["vein"]["gt_empty"] is False
    assert metrics["vein"]["dice"] == pytest.approx(0.5)
    assert metrics["vein"]["iou"] == pytest.approx(1.0 / 3.0)


def test_patient_bootstrap_ci():
    patient_scores = {f"P{i:03d}": 0.8 + 0.05 * (i % 3) for i in range(20)}
    mean, ci_low, ci_high = run_patient_bootstrap_ci(patient_scores, n_bootstraps=500, seed=42)

    assert ci_low <= mean <= ci_high
    assert 0.75 <= mean <= 0.95
