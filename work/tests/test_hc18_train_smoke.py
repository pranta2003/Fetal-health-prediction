"""Smoke test for HC18 training pipeline on CPU."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fetal.models.hc18_unet import HC18Loss, build_segmentation_model
from fetal.train.train_hc18 import run_hc18_training, train_one_fold


def test_hc18_training_step_cpu():
    """Fast smoke test: 2 tiny training iterations on CPU.

    Verifies loss is finite and AdamW step modifies model weights without NaN.
    """
    device = torch.device("cpu")
    torch.manual_seed(42)

    model = build_segmentation_model(encoder_name="efficientnet-b4", encoder_weights=None).to(device)
    loss_fn = HC18Loss(mode="binary")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    p_before = [p.clone().detach() for p in model.parameters() if p.requires_grad]

    # Two mini-batches of size 2, 64x64
    x = torch.randn(2, 3, 64, 64, device=device)
    y = (torch.rand(2, 1, 64, 64, device=device) > 0.5).float()

    for _ in range(2):
        opt.zero_grad()
        preds = model(x)
        loss = loss_fn(preds, y)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        loss.backward()
        opt.step()

    p_after = [p.clone().detach() for p in model.parameters() if p.requires_grad]
    changed = [not torch.equal(pb, pa) for pb, pa in zip(p_before, p_after)]
    assert any(changed), "Optimizer step did not modify any model weights!"
    assert all(torch.isfinite(pa).all() for pa in p_after), "Weights contain NaN or Inf values!"


@pytest.mark.slow
def test_hc18_train_smoke_cpu(tmp_path, monkeypatch):
    """End-to-end 2-epoch CPU smoke run on 16 HC18 images verifying output artifacts."""
    monkeypatch.setenv("HC18_SMOKE", "1")

    out_dir = tmp_path / "smoke_models"
    summary = run_hc18_training(
        output_dir=str(out_dir),
        target_fold=1,
        smoke=True,
    )

    ckpt_file = out_dir / "hc18_seg_fold1_eff_b4_128.pt"
    metric_file = out_dir / "fold1_metrics.json"
    pred_file = out_dir / "fold1_predictions.json"
    summary_file = out_dir / "metrics.json"

    assert ckpt_file.exists(), f"Missing checkpoint: {ckpt_file}"
    assert metric_file.exists(), f"Missing fold metric file: {metric_file}"
    assert pred_file.exists(), f"Missing prediction file: {pred_file}"
    assert summary_file.exists(), f"Missing summary file: {summary_file}"

    with open(metric_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "hc_mae" in metrics
    assert "hc_rmse" in metrics
    assert "bpd_mae" in metrics
    assert "bpd_rmse" in metrics
    assert metrics["n_samples"] > 0
    assert metrics["hc_mae"] >= 0.0
