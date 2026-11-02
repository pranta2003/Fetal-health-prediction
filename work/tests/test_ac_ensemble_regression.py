import os
import pytest
import torch

from fetal import config
from fetal.data.ac_dataset import build_data, split_data
from fetal.eval.ensemble_evaluator import load_ensemble_models, eval_ensemble, summarize_ensemble


@pytest.mark.slow
def test_regression_b_ensemble_median_official_test():
    """
    REGRESSION TEST B:
    Evaluate 4-model ensemble (s1-s4, median pooling, flip-TTA) on official test set using ONLY new modules.
    Target: AC MAE 4.3426 +/- 0.002 mm (rounds to 4.34 mm), RMSE ~6.03 mm.
    """
    if not os.path.isdir(config.DATA_DIR):
        pytest.skip(f"Data directory missing: {config.DATA_DIR}")

    for ckpt in config.AC_ENS_CKPTS:
        if not os.path.isfile(ckpt):
            pytest.skip(f"Ensemble seed checkpoint missing: {ckpt}")

    data = build_data(config.DATA_DIR)
    _, _, te = split_data(data)
    assert len(te) == 147, f"Expected 147 test images, got {len(te)}"

    seed_dirs = [os.path.dirname(ckpt) for ckpt in config.AC_ENS_CKPTS]
    models = load_ensemble_models(seed_dirs, device=config.DEVICE)
    res_df = eval_ensemble(
        models, te, S=384, inpaint=True, agg="median", bs=8, num_workers=config.NUM_WORKERS, device=config.DEVICE
    )
    s = summarize_ensemble(res_df)

    actual_mae = s["ac_mae"]
    expected_mae = 4.3426
    diff = abs(actual_mae - expected_mae)

    print(f"\n[REGRESSION B RESULT]")
    print(f"  Test samples : {s['n']}")
    print(f"  AC MAE       : {actual_mae:.6f} mm (Expected: {expected_mae:.4f} +/- 0.002 mm)")
    print(f"  Difference   : {diff:.6f} mm")
    print(f"  AC RMSE      : {s['ac_rmse']:.4f} mm")
    print(f"  AC Median    : {s['ac_median']:.4f} mm")
    print(f"  AC Bias      : {s['ac_bias']:.4f} mm")
    print(f"  TAD MAE      : {s['tad_mae']:.4f} mm")
    print(f"  APAD MAE     : {s['apad_mae']:.4f} mm")
    print(f"  NME          : {s['nme']:.4f}")
    print(f"  >10mm Outlier: {100*s['over10']:.2f}%")

    assert diff <= 0.002, (
        f"Regression B Ensemble AC MAE failed tolerance +/-0.002 mm: actual={actual_mae:.6f} mm, "
        f"expected={expected_mae:.4f} mm, diff={diff:.6f} mm"
    )
