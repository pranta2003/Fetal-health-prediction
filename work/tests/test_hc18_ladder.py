"""test_hc18_ladder.py: Tests for HC18 historical approaches and ladder fold-1 setup."""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from fetal.data.hc18_split import make_grouped_folds, split_patient_inner_holdout, patient_id_from_name
from fetal.models.hc18_old_approaches import (
    build_model_row_a,
    build_model_row_b,
    build_model_row_c,
    build_model_row_d,
    hc_bpd_from_ellipse_params,
)


def test_fold1_images_equal_oof_predictions():
    """Assert that fold 1 of make_grouped_folds equals the fold-1 rows of oof_predictions.csv."""
    work_dir = Path(__file__).resolve().parent.parent
    oof_csv = work_dir / "docs" / "results" / "hc18" / "oof_predictions.csv"
    assert oof_csv.exists(), f"Missing oof_predictions.csv at {oof_csv}"

    oof_df = pd.read_csv(oof_csv)
    expected_f1_names = sorted(oof_df[oof_df["fold"] == 1]["image"].tolist())
    assert len(expected_f1_names) == 206, f"Expected 206 fold-1 images, got {len(expected_f1_names)}"

    # Reconstruct the original 999 image names in their canonical order
    # oof_df contains all 999 images. We can sort them or use them directly if dataset names match
    # Let's verify by sorting or grouping
    all_names = oof_df["image"].tolist()
    # If we sort all unique names or use dataset order:
    # Let's check if oof_df['image'] was ordered as in training_set_pixel_size_and_HC.csv:
    folds = make_grouped_folds(all_names, n_splits=5, seed=42)
    # The first fold should contain exactly the 206 images
    val_idx = folds[0][1]
    f1_generated_names = sorted([all_names[i] for i in val_idx])

    assert f1_generated_names == expected_f1_names, "Generated fold 1 names do not match oof_predictions.csv fold 1!"


def test_inner_holdout_zero_patient_leakage():
    """Assert 10% inner patient holdout has zero patient overlap with inner train or outer val."""
    work_dir = Path(__file__).resolve().parent.parent
    oof_csv = work_dir / "docs" / "results" / "hc18" / "oof_predictions.csv"
    oof_df = pd.read_csv(oof_csv)
    all_names = oof_df["image"].tolist()

    folds = make_grouped_folds(all_names, n_splits=5, seed=42)
    outer_train_idx, outer_val_idx = folds[0]

    inner_train_idx, inner_val_idx = split_patient_inner_holdout(
        outer_train_idx, all_names, holdout_fraction=0.10, seed=42
    )

    p_inner_tr = set(patient_id_from_name(all_names[i]) for i in inner_train_idx)
    p_inner_val = set(patient_id_from_name(all_names[i]) for i in inner_val_idx)
    p_outer_val = set(patient_id_from_name(all_names[i]) for i in outer_val_idx)

    assert len(p_inner_tr & p_inner_val) == 0, "Patient leakage between inner train and inner val!"
    assert len(p_inner_val & p_outer_val) == 0, "Patient leakage between inner val and outer val!"
    assert len(p_inner_tr & p_outer_val) == 0, "Patient leakage between inner train and outer val!"
    assert len(inner_val_idx) > 0, "Inner val slice must be non-empty"


def test_models_forward_pass_and_shapes():
    """Test model creation and forward pass without pretrained weights."""
    model_a = build_model_row_a(pretrained=False)
    x_a = torch.randn(2, 3, 224, 224)
    out_a = model_a(x_a)
    assert out_a.shape == (2, 2), f"Row A output shape {out_a.shape} != (2, 2)"

    model_b = build_model_row_b(pretrained=False)
    x_b = torch.randn(2, 3, 224, 224)
    out_b = model_b(x_b)
    assert out_b.shape == (2, 5), f"Row B output shape {out_b.shape} != (2, 5)"

    model_c = build_model_row_c(pretrained=False)
    x_c = torch.randn(2, 3, 224, 224)
    mask_c, geo_c = model_c(x_c)
    assert mask_c.shape == (2, 1, 224, 224), f"Row C mask shape {mask_c.shape} != (2, 1, 224, 224)"
    assert geo_c.shape == (2, 5), f"Row C geo shape {geo_c.shape} != (2, 5)"

    model_d = build_model_row_d(pretrained=False)
    x_d = torch.randn(2, 3, 224, 224)
    mask_d = model_d(x_d)
    assert mask_d.shape == (2, 1, 224, 224), f"Row D mask shape {mask_d.shape} != (2, 1, 224, 224)"


def test_ellipse_to_hc_bpd():
    """Test Ramanujan HC and BPD conversion formula."""
    # Circle of radius 50 mm: circumference = 2 * pi * 50 = 314.159 mm, diameter = 100 mm
    hc, bpd = hc_bpd_from_ellipse_params(50.0, 50.0)
    assert np.isclose(hc, 2 * np.pi * 50.0, atol=0.01)
    assert np.isclose(bpd, 100.0, atol=0.01)
