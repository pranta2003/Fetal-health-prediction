"""test_hadlock_step2.py: Unit tests for Hadlock demo step 2."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import pandas as pd

from fetal.explain.hadlock import calculate_efw

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def hadlock_demo_data():
    results_dir = REPO_ROOT / "docs" / "results" / "hadlock_demo"
    csv_path = results_dir / "step2_per_subject.csv"
    json_path = results_dir / "step2_summary.json"
    assert csv_path.is_file(), f"Missing {csv_path}"
    assert json_path.is_file(), f"Missing {json_path}"

    df = pd.read_csv(csv_path)
    with open(json_path, encoding="utf-8") as f:
        summary = json.load(f)
    return df, summary


def test_hadlock_published_test_vector():
    """Verify Hadlock 4-parameter formula test vector (1403.0 grams)."""
    efw = calculate_efw(hc_mm=280.0, bpd_mm=80.0, ac_mm=250.0, fl_mm=55.0)
    assert abs(efw - 1403.0) < 0.1, f"Expected 1403.0 grams, got {efw:.4f}"
    assert f"{efw:.1f}" == "1403.0"


def test_cohort_count_equals_three_plane_audit(hadlock_demo_data):
    """Verify cohort size and source counts match scripts/23_three_plane_check.py audit."""
    df, summary = hadlock_demo_data
    tp_json_path = REPO_ROOT / "docs" / "results" / "three_plane" / "three_plane_check.json"
    assert tp_json_path.is_file(), f"Missing {tp_json_path}"

    with open(tp_json_path, encoding="utf-8") as f:
        tp_data = json.load(f)

    exp_fp = tp_data["splits_and_leakage_audit"]["fp"]["three_plane_subjects_all_test"]
    exp_ucl = tp_data["splits_and_leakage_audit"]["ucl_px_to_mm"]["three_plane_subjects_all_test"]

    assert len(df) == 22, f"Expected 22 subjects, got {len(df)}"
    assert (df["source"] == "FP").sum() == exp_fp == 17
    assert (df["source"] == "UCL").sum() == exp_ucl == 5
    assert summary["metadata"]["n_total_cohort"] == 22
    assert summary["metadata"]["n_fp"] == 17
    assert summary["metadata"]["n_ucl"] == 5


def test_per_subject_math_recomputed(hadlock_demo_data):
    """Verify per-subject EFW and error math recomputes cleanly from recorded CSV columns."""
    df, _ = hadlock_demo_data

    for _, row in df.iterrows():
        recomputed_ann = calculate_efw(
            row["hc_annotated_mm"],
            row["bpd_annotated_mm"],
            row["ac_annotated_mm"],
            row["fl_annotated_mm"],
        )
        recomputed_pred = calculate_efw(
            row["hc_predicted_mm"],
            row["bpd_predicted_mm"],
            row["ac_predicted_mm"],
            row["fl_predicted_mm"],
        )
        assert abs(recomputed_ann - row["efw_annotated_g"]) < 0.05
        assert abs(recomputed_pred - row["efw_predicted_g"]) < 0.05

        expected_err_g = recomputed_pred - recomputed_ann
        assert abs(expected_err_g - row["err_g"]) < 0.05

        expected_err_pct = (expected_err_g / recomputed_ann) * 100.0
        assert abs(expected_err_pct - row["err_pct"]) < 0.02


def test_replacement_analysis_consistent(hadlock_demo_data):
    """Verify one-at-a-time replacement math and dominant error driver ranking."""
    df, summary = hadlock_demo_data
    rep = summary["replacement_analysis"]

    # Recompute replacement errors row by row
    bpd_errs, hc_errs, ac_errs, fl_errs, all_errs = [], [], [], [], []
    for _, row in df.iterrows():
        efw_ann = row["efw_annotated_g"]
        efw_bpd = calculate_efw(row["hc_annotated_mm"], row["bpd_predicted_mm"], row["ac_annotated_mm"], row["fl_annotated_mm"])
        efw_hc = calculate_efw(row["hc_predicted_mm"], row["bpd_annotated_mm"], row["ac_annotated_mm"], row["fl_annotated_mm"])
        efw_ac = calculate_efw(row["hc_annotated_mm"], row["bpd_annotated_mm"], row["ac_predicted_mm"], row["fl_annotated_mm"])
        efw_fl = calculate_efw(row["hc_annotated_mm"], row["bpd_annotated_mm"], row["ac_annotated_mm"], row["fl_predicted_mm"])
        efw_all = calculate_efw(row["hc_predicted_mm"], row["bpd_predicted_mm"], row["ac_predicted_mm"], row["fl_predicted_mm"])

        bpd_errs.append(abs(efw_bpd - efw_ann) / efw_ann * 100.0)
        hc_errs.append(abs(efw_hc - efw_ann) / efw_ann * 100.0)
        ac_errs.append(abs(efw_ac - efw_ann) / efw_ann * 100.0)
        fl_errs.append(abs(efw_fl - efw_ann) / efw_ann * 100.0)
        all_errs.append(abs(efw_all - efw_ann) / efw_ann * 100.0)

    assert pytest.approx(rep["replace_bpd_only_mean_abs_pct"], abs=0.01) == float(pd.Series(bpd_errs).mean())
    assert pytest.approx(rep["replace_hc_only_mean_abs_pct"], abs=0.01) == float(pd.Series(hc_errs).mean())
    assert pytest.approx(rep["replace_ac_only_mean_abs_pct"], abs=0.01) == float(pd.Series(ac_errs).mean())
    assert pytest.approx(rep["replace_fl_only_mean_abs_pct"], abs=0.01) == float(pd.Series(fl_errs).mean())
    assert pytest.approx(rep["replace_all_four_mean_abs_pct"], abs=0.01) == float(pd.Series(all_errs).mean())

    # Dominant driver ranking: AC > FL > HC > BPD
    assert rep["dominant_driver"] == "AC"
    assert rep["replace_ac_only_mean_abs_pct"] > rep["replace_fl_only_mean_abs_pct"]
    assert rep["replace_fl_only_mean_abs_pct"] > rep["replace_hc_only_mean_abs_pct"]
    assert rep["replace_hc_only_mean_abs_pct"] > rep["replace_bpd_only_mean_abs_pct"]


def test_outlier_ucl5_and_sensitivity(hadlock_demo_data):
    """Verify UCL_5 outlier analysis and PR #32 Row A sensitivity."""
    df, summary = hadlock_demo_data

    # UCL_5 check
    ucl5 = df[df["subject_id"] == "UCL_5"].iloc[0]
    assert ucl5["n_femur_images"] == 16
    assert abs(ucl5["err_g"] - 18.3) < 0.2
    assert abs(ucl5["err_pct"] - 1.7) < 0.2

    out_ucl5 = summary["outlier_subject_ucl5"]
    assert out_ucl5["subject_id"] == "UCL_5"
    assert out_ucl5["contains_005_3FL_jpeg"] is True

    # Excluding UCL_5
    sum_no_ucl5 = summary["primary_evaluation_r4"]["without_outlier_ucl5_21"]
    assert sum_no_ucl5["n_subjects"] == 21
    assert pytest.approx(sum_no_ucl5["mape_pct"], abs=0.05) == 2.97

    # PR #32 Row A sensitivity
    sens = summary["sensitivity_pr32_row_a"]
    assert sens["all_cohort_22"]["n_subjects"] == 22
    assert pytest.approx(sens["all_cohort_22"]["mape_pct"], abs=0.05) == 3.10
