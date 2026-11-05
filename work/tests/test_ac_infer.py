"""Tests for AC single-image inference API (fetal.infer.ac_infer)."""

import os
import pytest
from fetal import config
from fetal.data.ac_dataset import build_data, split_data
from fetal.eval.ensemble_evaluator import eval_ensemble, load_ensemble_models
from fetal.infer.ac_infer import predict_ac


@pytest.mark.slow
def test_predict_ac_official_test_equivalence():
    """Verify predict_ac reproduces eval_ensemble ac_pred on official test images within 1e-4 mm."""
    if not os.path.isdir(config.DATA_DIR):
        pytest.skip(f"Data directory missing: {config.DATA_DIR}")
    for p in config.AC_ENS_CKPTS:
        if not os.path.isfile(p):
            pytest.skip(f"Checkpoint missing: {p}")

    data = build_data(config.DATA_DIR)
    _, _, te = split_data(data)

    # Take first 3 official test images
    test_subset = te.iloc[:3].reset_index(drop=True)

    # Run eval_ensemble on the subset
    models = load_ensemble_models(config.AC_ENS_CKPTS, device=config.DEVICE)
    eval_df = eval_ensemble(models, test_subset, S=384, inpaint=True, agg="median", device=config.DEVICE)

    # Verify predict_ac matches eval_ensemble row-by-row
    for i in range(len(test_subset)):
        row = test_subset.iloc[i]
        expected_ac = float(eval_df.iloc[i]["ac_pred"])

        res = predict_ac(
            row["image_path"],
            ckpt_paths=config.AC_ENS_CKPTS,
            flip_tta=True,
            models=models,
            device=config.DEVICE,
        )
        actual_ac = res["ac_mm"]

        print(
            f"\n[Test image {i+1}] {row['image_name']}: actual_ac={actual_ac:.6f}, "
            f"expected_ac={expected_ac:.6f}, diff={abs(actual_ac - expected_ac):.2e}"
        )
        assert abs(actual_ac - expected_ac) <= 1e-4, (
            f"Image {row['image_name']}: predict_ac ac_mm={actual_ac:.6f} differs from eval_ensemble={expected_ac:.6f} by > 1e-4"
        )
