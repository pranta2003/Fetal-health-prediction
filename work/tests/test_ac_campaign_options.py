"""tests/test_ac_campaign_options.py

Tests for AC campaign options:
- 5-fold data partition and patient grouping
- loss functions (default, smooth_l1, wing)
- ModelEMA
- Pretrained encoder loading
- Paired bootstrap recipe comparison
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
src_dir = REPO_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fetal.data.ac_dataset import load_ac_folds, get_fold_data
from fetal.models.losses import coord_loss_term
from fetal.train.ema import ModelEMA
from fetal.models.ac_landmark import build_model, load_pretrained_encoder
from fetal.eval.ac_evaluator import patient_level_paired_bootstrap, compute_metrics


def test_folds_v1_integrity():
    folds_csv = REPO_ROOT / "docs" / "results" / "ac_campaign" / "folds_v1.csv"
    assert folds_csv.exists(), f"Missing folds file: {folds_csv}"

    df = load_ac_folds(folds_csv)
    assert len(df) == 642, f"Expected 642 train images, got {len(df)}"
    assert df["patient"].nunique() == 501, f"Expected 501 unique patients, got {df['patient'].nunique()}"

    unique_folds = sorted(df["fold"].unique().tolist())
    assert unique_folds == [1, 2, 3, 4, 5], f"Expected folds 1..5, got {unique_folds}"

    # Verify no patient across multiple folds
    patient_folds = df.groupby("patient")["fold"].nunique()
    leakage = patient_folds[patient_folds > 1]
    assert len(leakage) == 0, f"Found patients in multiple folds: {leakage}"

    # Test fold train/val partition
    for fold in range(1, 6):
        train_df, val_df, oof_df = get_fold_data(df, fold=fold)
        assert len(train_df) + len(val_df) + len(oof_df) == 642
        train_p = set(train_df["patient"])
        val_p = set(val_df["patient"])
        oof_p = set(oof_df["patient"])
        assert len(train_p.intersection(val_p)) == 0, f"Patient overlap train-val in fold {fold}"
        assert len(train_p.intersection(oof_p)) == 0, f"Patient overlap train-oof in fold {fold}"
        assert len(val_p.intersection(oof_p)) == 0, f"Patient overlap val-oof in fold {fold}"


def test_coord_loss_types():
    pred = torch.tensor([[10.0, 20.0], [30.0, 40.0]], requires_grad=True)
    gt = torch.tensor([[11.0, 19.0], [28.0, 42.0]])

    for loss_type in ["default", "smooth_l1", "wing"]:
        loss = coord_loss_term(pred, gt, loss_type=loss_type)
        assert loss.ndim == 0, f"Loss should be scalar for {loss_type}"
        assert loss.item() > 0.0, f"Loss should be positive for {loss_type}"
        loss.backward()
        assert pred.grad is not None
        pred.grad.zero_()


def test_model_ema():
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    ema = ModelEMA(model, decay=0.9, warmup=False)

    # Save original weights
    orig_w = model[0].weight.data.clone()

    # Change model weights
    with torch.no_grad():
        model[0].weight.add_(1.0)

    ema.update(model)

    # Shadow should be between orig_w and orig_w + 1.0
    shadow_w = ema.shadow["0.weight"]
    expected_w = 0.9 * orig_w + 0.1 * (orig_w + 1.0)
    assert torch.allclose(shadow_w, expected_w, atol=1e-5)

    # Test apply and restore
    ema.apply_shadow(model)
    assert torch.allclose(model[0].weight.data, expected_w, atol=1e-5)

    ema.restore(model)
    assert torch.allclose(model[0].weight.data, orig_w + 1.0, atol=1e-5)


def test_model_ema_warmup_and_buffers():
    # Test that BatchNorm buffers are copied directly and warmup decay progresses
    conv = nn.BatchNorm2d(4)
    conv.running_mean.fill_(0.0)
    conv.running_var.fill_(1.0)
    ema = ModelEMA(conv, decay=0.99, warmup=True)

    # Update BN stats in source model
    conv.running_mean.fill_(5.0)
    conv.running_var.fill_(2.5)
    ema.update(conv)

    # Buffer should be copied directly (not smoothed with decay)
    assert torch.allclose(ema.ema_model.running_mean, conv.running_mean)
    assert torch.allclose(ema.ema_model.running_var, conv.running_var)
    assert ema.step == 1


def test_load_pretrained_encoder_synthetic(tmp_path):
    # Create a small dummy checkpoint with encoder keys
    model = build_model()
    dummy_encoder_state = {}
    for name, param in model.named_parameters():
        if name.startswith("encoder."):
            dummy_encoder_state[name] = torch.randn_like(param)

    ckpt_path = tmp_path / "dummy_encoder.pt"
    torch.save({"model_state_dict": dummy_encoder_state}, ckpt_path)

    matched, total = load_pretrained_encoder(model, ckpt_path, verbose=False)
    assert matched == len(dummy_encoder_state)
    assert total == 704
    assert matched > 0


def test_bootstrap_paired_comparison():
    # Construct synthetic data with recipe B better than recipe A
    rng = np.random.default_rng(42)
    n_patients = 50
    rows = []
    for p in range(n_patients):
        p_id = f"P_{p:03d}"
        for img in range(2):
            ac_true = 250.0 + rng.normal(0, 20.0)
            err_a = np.abs(rng.normal(6.0, 1.0))
            err_b = np.abs(rng.normal(4.0, 1.0))
            rows.append({
                "patient": p_id,
                "image_name": f"{p_id}_img_{img}.png",
                "ac_true": ac_true,
                "ac_pred_a": ac_true + err_a,
                "ac_pred_b": ac_true + err_b,
            })
    df = pd.DataFrame(rows)

    boot = patient_level_paired_bootstrap(df, n_bootstraps=500, seed=42)
    assert boot["n_patients"] == 50
    assert boot["n_images"] == 100
    assert boot["delta_mae"] < 0.0  # B has lower error than A
    assert boot["ci95_delta_mae"][0] < boot["ci95_delta_mae"][1]


def test_train_ac_args_support_data_dir(monkeypatch):
    import importlib.util
    train_ac_path = REPO_ROOT / "scripts" / "04_train_ac.py"
    spec = importlib.util.spec_from_file_location("train_ac_script", str(train_ac_path))
    train_ac_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_ac_mod)

    test_args = ["04_train_ac.py", "--data-dir", "/kaggle/working/data", "--fold", "1", "--smoke"]
    monkeypatch.setattr(sys, "argv", test_args)
    parsed = train_ac_mod.parse_args()
    assert parsed.data_dir == "/kaggle/working/data"
    assert parsed.fold == "1"
    assert parsed.smoke is True

