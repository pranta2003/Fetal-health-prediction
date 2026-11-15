"""Automated regression and verification tests for Femur Round 2 baseline sensitivity and run integrity."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

work_root = Path(__file__).resolve().parent.parent


@pytest.fixture
def sensitivity_data():
    json_path = work_root / "docs" / "results" / "femur_round2" / "baseline_sensitivity.json"
    assert json_path.is_file(), f"Missing baseline_sensitivity.json at {json_path}"
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def test_baseline_sensitivity_json_exists(sensitivity_data):
    """Verify that baseline_sensitivity.json exists and contains top-level keys."""
    assert "sensitivity" in sensitivity_data
    assert "row_R1" in sensitivity_data
    assert "row_R3" in sensitivity_data
    assert "row_B" in sensitivity_data
    assert "worst_10_table" in sensitivity_data
    assert "worst_image_005_3FL_shares" in sensitivity_data


def test_sensitivity_differences_match_prediction_csvs(sensitivity_data):
    """Verify paired difference means match the underlying predictions CSVs exactly."""
    df1 = pd.read_csv(work_root / "docs" / "results" / "femur_ablation" / "femur_predictions.csv")
    df2 = pd.read_csv(work_root / "docs" / "results" / "femur_round2" / "femur_round2_predictions.csv")

    sens = sensitivity_data["sensitivity"]

    # PR #32 baseline error
    base_err = df1["fl_err"].to_numpy()

    # a) R4 vs PR #32 Row A
    expected_diff_a = float(np.mean(df2["row_R4_fl_err"].to_numpy() - base_err))
    assert pytest.approx(sens["a_R4_vs_PR32_RowA"]["mean_diff"], abs=1e-4) == expected_diff_a
    assert sens["a_R4_vs_PR32_RowA"]["ci_includes_zero"] is True

    # b) R2 vs PR #32 Row A
    expected_diff_b = float(np.mean(df2["row_R2_fl_err"].to_numpy() - base_err))
    assert pytest.approx(sens["b_R2_vs_PR32_RowA"]["mean_diff"], abs=1e-4) == expected_diff_b
    assert sens["b_R2_vs_PR32_RowA"]["ci_includes_zero"] is True

    # c) R6 vs PR #32 Row A
    expected_diff_c = float(np.mean(df2["row_R6_fl_err"].to_numpy() - base_err))
    assert pytest.approx(sens["c_R6_vs_PR32_RowA"]["mean_diff"], abs=1e-4) == expected_diff_c
    assert sens["c_R6_vs_PR32_RowA"]["ci_includes_zero"] is True

    # d) Baseline drift: Round-2 Row A vs PR #32 Row A
    expected_diff_d = float(np.mean(df2["row_A_fl_err"].to_numpy() - base_err))
    assert pytest.approx(sens["d_Round2_RowA_vs_PR32_RowA_drift"]["mean_diff"], abs=1e-4) == expected_diff_d
    assert sens["d_Round2_RowA_vs_PR32_RowA_drift"]["ci_includes_zero"] is False


def test_seed_maes_and_determinism(sensitivity_data):
    """Verify seed MAEs match results JSONs and determinism is recorded correctly."""
    with open(work_root / "docs" / "results" / "femur_ablation" / "femur_ablation_results.json", encoding="utf-8") as f:
        res1 = json.load(f)
    with open(work_root / "docs" / "results" / "femur_round2" / "femur_round2_results.json", encoding="utf-8") as f:
        res2 = json.load(f)

    sens = sensitivity_data["sensitivity"]
    for s in ["42", "43", "44"]:
        expected_pr32 = res1["row_A"]["seeds"][f"seed_{s}"]["all"]["fl_mae"]
        expected_r2 = res2["row_A"]["seeds"][f"seed_{s}"]["all"]["fl_mae"]
        assert pytest.approx(sens["pr32_row_a_seed_maes"][s], abs=1e-4) == expected_pr32
        assert pytest.approx(sens["round2_row_a_seed_maes"][s], abs=1e-4) == expected_r2

    det = sens["determinism"]
    assert det["cudnn_deterministic"] is False
    assert det["amp_autocast"] is True
    assert det["is_deterministic"] is False


def test_run_integrity_r1_r3_row_b(sensitivity_data):
    """Verify failed run classifications and Row B convergence cap status."""
    r1 = sensitivity_data["row_R1"]
    r3 = sensitivity_data["row_R3"]
    row_b = sensitivity_data["row_B"]

    assert r1["classification"] == "FAILED RUN (did not train: val MAE far above a centre baseline or flat)"
    assert r1["mark"] == "run failed, not a method comparison"

    assert r3["classification"] == "FAILED RUN (did not train: val MAE far above a centre baseline or flat)"
    assert r3["mark"] == "run failed, not a method comparison"

    assert row_b["still_improving_at_cap"] is True
    assert row_b["mark"] == "capped at 35 epochs, not converged"


def test_worst_10_table_and_shares(sensitivity_data):
    """Verify 10 worst images audit, provenance, and 005_3FL squared error shares."""
    w10 = sensitivity_data["worst_10_table"]
    assert len(w10) == 10
    assert w10[0]["image_name"] == "005_3FL.jpeg"
    assert w10[0]["source"] == "UCL"
    assert w10[0]["classification"] == "visual judgement, not computed"
    for item in w10:
        assert item["classification"] == "visual judgement, not computed"
        assert "docs/results/femur_ablation/femur_predictions.csv" in item["script_sources"]["fl_err_mm"]

    shares = sensitivity_data["worst_image_005_3FL_shares"]
    assert pytest.approx(shares["Row_A_PR32"]["share_all_pct"], abs=0.1) == 75.01
    assert pytest.approx(shares["Row_A_PR32"]["share_ucl_pct"], abs=0.1) == 99.47
    assert pytest.approx(shares["Row_R2"]["share_all_pct"], abs=0.1) == 1.03
