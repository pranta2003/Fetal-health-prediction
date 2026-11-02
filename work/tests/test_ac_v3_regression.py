import os
import pytest
import torch

from fetal import config
from fetal.data.ac_dataset import build_data, split_data
from fetal.models.ac_landmark import build_model
from fetal.eval.ac_evaluator import run_eval, summarize


@pytest.mark.slow
def test_regression_a_v3_full_official_test():
    """
    REGRESSION TEST A:
    Evaluate v3_full on official test set (147 images) with flip-TTA using ONLY new modules.
    Target: AC MAE 4.3951 +/- 0.002 mm (rounds to 4.40 mm), RMSE ~6.16 mm.
    """
    if not os.path.isdir(config.DATA_DIR):
        pytest.skip(f"Data directory missing: {config.DATA_DIR}")

    ckpt_path = config.AC_V3_FULL_CKPT
    if not os.path.isfile(ckpt_path):
        pytest.skip(f"Checkpoint missing: {ckpt_path}")

    data = build_data(config.DATA_DIR)
    _, _, te = split_data(data)
    assert len(te) == 147, f"Expected 147 test images, got {len(te)}"

    model = build_model(encoder_weights=None).to(config.DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))
    model.eval()

    cfg = dict(name="v3_full", img=384, diam_w=1.0, strong_aug=True, inpaint=True)
    res_df = run_eval(model, te, cfg, tta=True, bs=8, num_workers=config.NUM_WORKERS, device=config.DEVICE)
    s = summarize(res_df)

    actual_mae = s["ac_mae"]
    expected_mae = 4.3951
    diff = abs(actual_mae - expected_mae)

    print(f"\n[REGRESSION A RESULT]")
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
        f"Regression A AC MAE failed tolerance +/-0.002 mm: actual={actual_mae:.6f} mm, "
        f"expected={expected_mae:.4f} mm, diff={diff:.6f} mm"
    )
