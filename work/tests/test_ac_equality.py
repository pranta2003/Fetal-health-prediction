import numpy as np
import pandas as pd
import pytest
import torch

import ac_v3_original as orig_ac
from fetal import config
from fetal.data import ac_dataset
from fetal.models import ac_landmark, losses
from fetal.eval import ac_evaluator


def test_build_data_frame_equal():
    """Verify build_data produces identical DataFrame to original."""
    orig_ac.BASE = config.DATA_DIR
    orig_df = orig_ac.build_data()
    new_df = ac_dataset.build_data()
    pd.testing.assert_frame_equal(orig_df, new_df)


def test_abdomen_ds_items_identical():
    """Verify AbdomenDS items are identical under deterministic seeds."""
    data = ac_dataset.build_data()
    tr, dv, te = ac_dataset.split_data(data)

    # 1. Eval mode (train=False)
    ds_orig_eval = orig_ac.AbdomenDS(te, 384, train=False, strong=False, inpaint=True)
    ds_new_eval = ac_dataset.AbdomenDS(te, 384, train=False, strong=False, inpaint=True)

    for idx in [0, 10, 50, 100]:
        x0, pts0, mm0, i0 = ds_orig_eval[idx]
        x1, pts1, mm1, i1 = ds_new_eval[idx]
        assert torch.equal(x0, x1)
        assert torch.equal(pts0, pts1)
        assert torch.equal(mm0, mm1)
        assert i0 == i1

    # 2. Train mode with fixed seed
    np.random.seed(123)
    ds_orig_tr = orig_ac.AbdomenDS(tr, 384, train=True, strong=True, inpaint=True)
    orig_sample = ds_orig_tr[5]

    np.random.seed(123)
    ds_new_tr = ac_dataset.AbdomenDS(tr, 384, train=True, strong=True, inpaint=True)
    new_sample = ds_new_tr[5]

    assert torch.allclose(orig_sample[0], new_sample[0], atol=1e-6)
    assert torch.allclose(orig_sample[1], new_sample[1], atol=1e-6)
    assert torch.equal(orig_sample[2], new_sample[2])


def test_build_model_state_dict_keys_identical():
    """Verify build_model architectures match exactly."""
    m_orig = orig_ac.build_model()
    m_new = ac_landmark.build_model()
    assert list(m_orig.state_dict().keys()) == list(m_new.state_dict().keys())


def test_decode_random_inputs():
    """Verify decode produces identical outputs on random logits."""
    torch.manual_seed(42)
    logits = torch.randn(4, 4, 96, 96, dtype=torch.float32)

    pts_orig, p_orig = orig_ac.decode(logits)
    pts_new, p_new = ac_landmark.decode(logits)

    assert torch.allclose(pts_orig, pts_new, atol=1e-7)
    assert torch.allclose(p_orig, p_new, atol=1e-7)


def test_js_loss_random_inputs():
    """Verify js_loss produces identical outputs on random inputs."""
    torch.manual_seed(42)
    B, S = 4, 96
    p = torch.softmax(torch.randn(B, 4, S * S), dim=-1)
    order = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]], dtype=torch.long)
    gt = torch.rand(B, 4, 2) * float(S)

    loss_orig = orig_ac.js_loss(p, order, gt, S, 4.0)
    loss_new = losses.js_loss(p, order, gt, S, 4.0)

    assert torch.allclose(loss_orig, loss_new, atol=1e-7)


def test_compute_loss_random_inputs():
    """Verify compute_loss produces identical loss values on random inputs."""
    torch.manual_seed(42)
    B, S = 2, 96
    logits = torch.randn(B, 4, S, S, dtype=torch.float32)
    gt = torch.rand(B, 4, 2, dtype=torch.float32) * float(S)
    mmpp = torch.tensor([0.25, 0.3], dtype=torch.float32)

    loss_orig, coord_orig = orig_ac.compute_loss(logits, gt, mmpp, S, 1.0)
    loss_new, coord_new = losses.compute_loss(logits, gt, mmpp, S, 1.0)

    assert torch.allclose(loss_orig, loss_new, atol=1e-6)
    assert abs(coord_orig - coord_new) < 1e-6


def test_summarize_identical():
    """Verify summarize produces identical dictionary metrics on sample frame."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "ac_err": np.random.exponential(4.0, n),
        "ac_signed": np.random.normal(0, 5.0, n),
        "tad_err": np.random.exponential(1.5, n),
        "apad_err": np.random.exponential(1.7, n),
        "lm_mm": np.random.exponential(2.5, n),
        "nme": np.random.uniform(0.02, 0.12, n)
    })

    s_orig = orig_ac.summarize(df)
    s_new = ac_evaluator.summarize(df)

    assert s_orig.keys() == s_new.keys()
    for k in s_orig:
        assert np.isclose(s_orig[k], s_new[k], atol=1e-7), f"Metric mismatch for {k}"

