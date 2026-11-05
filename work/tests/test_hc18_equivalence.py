"""Equivalence and isolation tests for HC18 extraction against reference implementation."""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
import torch.nn as nn
import cv2

from fetal.data.hc18_dataset import HC18SegDataset, LetterboxResize
from fetal.data.hc18_split import make_grouped_folds, patient_id_from_name
from fetal.eval.hc18_eval import measure_from_prob
from fetal.models.hc18_unet import HC18Loss, build_segmentation_model
import hc18_original as ref


def test_model_architecture_and_keys_equivalence():
    """Verify build_segmentation_model produces identical architecture and state_dict keys."""
    ref_model = ref.build_segmentation_model()
    new_model = build_segmentation_model(encoder_name="efficientnet-b4", encoder_weights="imagenet")

    ref_keys = list(ref_model.state_dict().keys())
    new_keys = list(new_model.state_dict().keys())
    assert ref_keys == new_keys

    ref_params = sum(p.numel() for p in ref_model.parameters())
    new_params = sum(p.numel() for p in new_model.parameters())
    assert ref_params == new_params

    # Verify forward pass equivalence with transferred weights
    new_model.load_state_dict(ref_model.state_dict())
    ref_model.eval()
    new_model.eval()

    torch.manual_seed(42)
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out_ref = ref_model(x)
        out_new = new_model(x)

    assert torch.allclose(out_ref, out_new, atol=1e-5)


def test_loss_function_equivalence():
    """Verify HC18Loss computes identical output to ref DiceLoss + BCEWithLogitsLoss."""
    dice_ref = ref.smp.losses.DiceLoss(mode="binary")
    bce_ref = nn.BCEWithLogitsLoss()
    loss_new = HC18Loss(mode="binary")

    torch.manual_seed(42)
    preds = torch.randn(4, 1, 64, 64)
    targets = (torch.rand(4, 1, 64, 64) > 0.5).float()

    ref_val = dice_ref(preds, targets) + bce_ref(preds, targets)
    new_val = loss_new(preds, targets)

    assert torch.allclose(ref_val, new_val, atol=1e-6)


def test_letterbox_resize_equivalence():
    """Verify LetterboxResize produces identical transformed images, masks, and scale."""
    ref_letterbox = ref.LetterboxResize(target_size=256)
    new_letterbox = LetterboxResize(target_size=256)

    # Synthetic non-square image and mask
    np.random.seed(42)
    img_arr = np.random.randint(0, 255, (300, 450, 3), dtype=np.uint8)
    mask_arr = np.random.randint(0, 2, (300, 450), dtype=np.uint8) * 255

    img_pil = Image.fromarray(img_arr)
    mask_pil = Image.fromarray(mask_arr)

    ref_img, ref_scale, ref_px, ref_py = ref_letterbox(img_pil)
    ref_mask, _, _, _ = ref_letterbox(mask_pil)

    new_img, new_scale, new_px, new_py = new_letterbox(img_pil)
    new_mask, _, _, _ = new_letterbox(mask_pil)

    assert ref_scale == new_scale
    assert ref_px == new_px
    assert ref_py == new_py
    assert np.array_equal(np.array(ref_img), np.array(new_img))
    assert np.array_equal(np.array(ref_mask), np.array(new_mask))


def test_ellipse_biometrics_measurement_equivalence():
    """Verify ellipse measurement produces identical HC and BPD errors."""
    h, w = 384, 384
    # Draw a synthetic ellipse for ground truth
    mask_gt = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask_gt, (190, 190), (80, 50), 30, 0, 360, 255, -1)

    # Draw slightly displaced ellipse for predicted probabilities
    mask_pred = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask_pred, (192, 188), (82, 49), 28, 0, 360, 255, -1)
    prob_map = (mask_pred.astype(np.float32) / 255.0) * 0.95

    mask_tensor = torch.from_numpy(mask_gt.astype(np.float32) / 255.0).unsqueeze(0)
    pixel_size = 0.125
    scale = 0.75

    ref_res = ref.measure_from_prob(prob_map, mask_tensor, pixel_size, scale)
    new_res = measure_from_prob(prob_map, mask_tensor, pixel_size, scale)

    assert ref_res is not None
    assert new_res is not None
    assert pytest.approx(ref_res[0], rel=1e-5) == new_res[0]
    assert pytest.approx(ref_res[1], rel=1e-5) == new_res[1]


def test_patient_isolation_grouped_folds():
    """Verify no patient ID is ever shared between train and validation in any fold."""
    synthetic_names = [
        "001_HC.png", "001_2HC.png",
        "002_HC.png",
        "003_HC.png", "003_HC_2.png", "003_3HC.png",
        "004_HC.png",
        "005_HC.png", "005_2HC.png",
        "006_HC.png",
        "007_HC.png", "007_2HC.png",
        "008_HC.png",
        "009_HC.png",
        "010_HC.png", "010_2HC.png",
    ]

    folds = make_grouped_folds(synthetic_names, n_splits=5, seed=42)
    assert len(folds) == 5

    all_val_indices = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_patients = {patient_id_from_name(synthetic_names[i]) for i in train_idx}
        val_patients = {patient_id_from_name(synthetic_names[i]) for i in val_idx}

        # Strict isolation check
        overlap = train_patients.intersection(val_patients)
        assert len(overlap) == 0, f"Fold {fold_idx + 1} shares patients: {overlap}"

        all_val_indices.extend(val_idx.tolist())

    # Every sample must appear in validation exactly once
    assert sorted(all_val_indices) == list(range(len(synthetic_names)))

    # Verify with real HC18 CSV if available
    real_csv = Path("C:/fetal-data/HC18/training_set_pixel_size_and_HC.csv")
    if real_csv.exists():
        df = pd.read_csv(real_csv)
        names = df["filename"].tolist()
        real_folds = make_grouped_folds(names, n_splits=5, seed=42)
        assert len(real_folds) == 5
        for fold_idx, (train_idx, val_idx) in enumerate(real_folds):
            train_p = {patient_id_from_name(names[i]) for i in train_idx}
            val_p = {patient_id_from_name(names[i]) for i in val_idx}
            overlap = train_p.intersection(val_p)
            assert len(overlap) == 0, f"Real fold {fold_idx + 1} shares patients: {overlap}"
