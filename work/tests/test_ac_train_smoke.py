import os
import pytest
import torch
import pandas as pd
import numpy as np

from fetal.models.ac_landmark import build_model
from fetal.models.losses import compute_loss
from fetal.train.train_ensemble import split_seed, get_ens_seeds, get_skip_512


def test_loss_and_optimizer_step():
    """
    Smoke-level test: 2 tiny batches on CPU.
    Asserts loss is finite and optimizer step changes model weights.
    """
    device = "cpu"
    torch.manual_seed(42)

    model = build_model(encoder_weights=None).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Save copy of initial weights
    p_before = [p.clone().detach() for p in model.parameters() if p.requires_grad]

    S = 128
    batch_size = 2

    for b in range(2):
        x = torch.randn(batch_size, 3, S, S, dtype=torch.float32, device=device)
        gt = torch.rand(batch_size, 4, 2, dtype=torch.float32, device=device)
        mmpp = torch.tensor([0.2, 0.25], dtype=torch.float32, device=device)

        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss, lm_err = compute_loss(logits, gt, mmpp, S=S, diam_w=1.0)

        assert torch.isfinite(loss), f"Batch {b} loss is not finite: {loss.item()}"
        assert np.isfinite(lm_err), f"Batch {b} lm_err is not finite: {lm_err}"

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

    p_after = [p.clone().detach() for p in model.parameters() if p.requires_grad]

    # Verify parameters were modified and remain finite
    changed = [not torch.equal(pb, pa) for pb, pa in zip(p_before, p_after)]
    assert any(changed), "Optimizer step did not modify any model weights!"
    assert all(torch.isfinite(pa).all() for pa in p_after), "Weights contain NaN or Inf values!"


def test_split_seed_smoke():
    """Verify split_seed patient separation and reproducibility across seeds."""
    records = []
    # 20 train images across 10 patients
    for p in range(1, 11):
        for img_id in range(2):
            records.append({
                "image_name": f"img_p{p}_{img_id}.png",
                "patient": f"patient_{p}",
                "official_split": "train"
            })
    # 4 test images across 2 patients
    for p in range(11, 13):
        for img_id in range(2):
            records.append({
                "image_name": f"img_p{p}_{img_id}.png",
                "patient": f"patient_{p}",
                "official_split": "test"
            })

    df = pd.DataFrame(records)

    tr, dv, te = split_seed(df, seed=1, dev_fraction=0.2)

    # Patient isolation
    assert set(tr.patient).isdisjoint(set(dv.patient))
    assert set(tr.patient).isdisjoint(set(te.patient))
    assert set(dv.patient).isdisjoint(set(te.patient))

    # All official test images preserved
    assert len(te) == 4
    assert len(tr) + len(dv) == 20

    # Reproducibility with same seed
    tr2, dv2, te2 = split_seed(df, seed=1, dev_fraction=0.2)
    pd.testing.assert_frame_equal(tr, tr2)
    pd.testing.assert_frame_equal(dv, dv2)


def test_ensemble_env_switches(monkeypatch):
    """Verify ENS_SEEDS and ENS_SKIP_512 environment switches."""
    monkeypatch.setenv("ENS_SEEDS", "1, 3, 5")
    assert get_ens_seeds() == [1, 3, 5]

    monkeypatch.setenv("ENS_SKIP_512", "1")
    assert get_skip_512() is True

    monkeypatch.setenv("ENS_SKIP_512", "0")
    assert get_skip_512() is False
