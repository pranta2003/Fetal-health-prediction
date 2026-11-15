"""test_hc18_calibration.py: Unit tests for HC18 OOF calibration study using synthetic data."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

REPO_ROOT = Path(__file__).resolve().parent.parent
script_path = REPO_ROOT / "scripts" / "34_hc18_oof_calibration.py"
spec = importlib.util.spec_from_file_location("hc18_oof_calibration", script_path)
cal_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal_mod)

check_local_oof_probability_maps = cal_mod.check_local_oof_probability_maps
run_hc18_calibration = cal_mod.run_hc18_calibration


def _create_synthetic_oof_csv(csv_path: Path, n_samples: int = 100, n_patients: int = 20) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    pats = [f"P_{i}" for i in range(1, n_patients + 1)]
    
    rows = []
    for idx in range(n_samples):
        pid = rng.choice(pats)
        fold = (idx % 5) + 1
        true_hc = float(rng.uniform(100.0, 300.0))
        # Add negative bias so true > pred by ~5mm + noise
        pred_hc = float(true_hc - 5.0 + rng.normal(0, 1.0))
        
        rows.append({
            "image": f"img_{idx:03d}.png",
            "patient_id": pid,
            "fold": fold,
            "true_HC_mm": true_hc,
            "pred_HC_mm": pred_hc,
            "abs_err_HC": abs(true_hc - pred_hc),
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df


def test_check_local_oof_probability_maps_empty(tmp_path: Path):
    res = check_local_oof_probability_maps([tmp_path])
    assert res["local_oof_maps_exist"] is False
    assert res["count"] == 0
    assert "Zero local OOF probability maps" in res["statement"]


def test_check_local_oof_probability_maps_found(tmp_path: Path):
    fake_map = tmp_path / "oof_prob_map.npy"
    np.save(fake_map, np.zeros((10, 10)))
    res = check_local_oof_probability_maps([tmp_path])
    assert res["local_oof_maps_exist"] is True
    assert res["count"] == 1


def test_isotonic_regression_monotonicity():
    rng = np.random.RandomState(42)
    x = np.sort(rng.uniform(50, 350, 50))
    # True relationship with noise
    y = 0.95 * x + 2.0 + rng.normal(0, 5.0, 50)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x, y)

    test_x = np.linspace(40, 360, 100)
    pred_y = iso.predict(test_x)

    # Monotonicity check: pred_y[i+1] >= pred_y[i]
    diffs = np.diff(pred_y)
    assert np.all(diffs >= -1e-9), f"Monotonicity violated: min diff = {np.min(diffs)}"


def test_run_hc18_calibration_synthetic(tmp_path: Path):
    oof_csv = tmp_path / "mock_oof.csv"
    out_dir = tmp_path / "results"
    _create_synthetic_oof_csv(oof_csv, n_samples=100, n_patients=20)

    results = run_hc18_calibration(oof_csv, out_dir, n_bootstrap=200, seed=42)

    assert results["total_images"] == 100
    assert results["total_patients"] == 20
    assert "methods_metrics" in results
    assert "raw" in results["methods_metrics"]
    assert "median_add" in results["methods_metrics"]
    assert "median_mult" in results["methods_metrics"]
    assert "linear_reg" in results["methods_metrics"]
    assert "isotonic" in results["methods_metrics"]

    # Since synthetic data had ~5mm negative bias (true - pred ~ +5mm),
    # calibration should significantly reduce MAE
    raw_mae = results["methods_metrics"]["raw"]["mae"]
    add_mae = results["methods_metrics"]["median_add"]["mae"]
    assert add_mae < raw_mae, f"Expected median_add MAE ({add_mae}) < raw ({raw_mae})"

    # Check bootstrap results exist
    assert "median_add" in results["bootstrap_diff_vs_raw"]
    b_res = results["bootstrap_diff_vs_raw"]["median_add"]
    assert b_res["mean_diff_vs_raw"] < 0
    assert len(b_res["ci_95"]) == 2

    # Check output files written
    assert (out_dir / "oof_calibration_results.json").is_file()
    assert (out_dir / "oof_calibration.md").is_file()
    assert (out_dir / "calibrated_predictions.csv").is_file()
