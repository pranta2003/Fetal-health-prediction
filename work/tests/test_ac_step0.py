"""tests/test_ac_step0.py

Unit tests for AC Campaign Step 0 audit functions and integrity checks.
"""

from pathlib import Path
import pytest
import os
import sys

import importlib

# Ensure repo root and scripts are accessible
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "scripts"))

audit_mod = importlib.import_module("38_ac_step0_audit")
get_repo_root = audit_mod.get_repo_root
check_hrnet_support = audit_mod.check_hrnet_support
check_fassd_checkpoint = audit_mod.check_fassd_checkpoint
audit_ac_data = audit_mod.audit_ac_data
parse_lab_pc_timing = audit_mod.parse_lab_pc_timing
parse_committed_results = audit_mod.parse_committed_results
generate_step0_report = audit_mod.generate_step0_report


def test_repo_root():
    root = get_repo_root()
    assert (root / "scripts" / "38_ac_step0_audit.py").is_file()
    assert (root / "docs" / "results").is_dir()


def test_check_hrnet_support():
    info = check_hrnet_support()
    assert isinstance(info, dict)
    assert "timm_w18" in info
    assert "smp_hrnet" in info
    assert "hrnet_w18" in info["timm_w18"]
    # smp registry does not include hrnet encoders natively
    assert info["smp_hrnet"] == []


def test_check_fassd_checkpoint():
    ckpt_path = r"E:\BUBT-CSE-4th yr\Capstone\kaggle_outputs\fassd_run1\fassd_gray_seed42\best_model.pt"
    info = check_fassd_checkpoint(ckpt_path)
    assert isinstance(info, dict)
    if os.path.isfile(ckpt_path):
        assert info["exists"] is True
        assert info["size_mb"] > 50.0
        assert info["encoder_name"] == "efficientnet-b4"
        assert info["val_macro_dice"] > 0.70


def test_audit_ac_data():
    audit = audit_ac_data()
    assert audit["total_images"] == 789
    assert audit["total_patients"] == 615
    assert audit["official_train_images"] == 642
    assert audit["official_train_patients"] == 501
    assert audit["official_test_images"] == 147
    assert audit["official_test_patients"] == 114

    # Strict leakage checks
    assert audit["cross_patient_dup_count"] == 0
    assert audit["cross_split_dup_count"] == 0
    assert audit["patient_overlap_count"] == 0

    # Duplicate images are all within-patient
    assert audit["same_patient_dup_count"] == 11
    assert audit["unique_hashes"] == 777


def test_parse_lab_pc_timing():
    timing = parse_lab_pc_timing(repo_root)
    assert "v3_full" in timing
    assert timing["v3_full"]["epochs"] == 51
    assert timing["v3_full"]["total_seconds"] == 4498
    assert round(timing["v3_full"]["mean_seconds_per_ep"], 1) == 88.2

    assert "ens_runs" in timing
    assert set(timing["ens_runs"].keys()) == {"s1", "s2", "s3", "s4", "res512"}
    assert timing["ens_runs"]["s1"]["epochs"] == 33

    assert "dsnt_5fold" in timing
    assert set(timing["dsnt_5fold"].keys()) == {1, 2, 3, 4, 5}
    assert sum(f["total_seconds"] for f in timing["dsnt_5fold"].values()) == 15033


def test_parse_committed_results():
    results = parse_committed_results(repo_root)
    assert results["oof_shape"] == [789, 6]
    assert abs(results["oof_ac_mae"] - 5.1801) < 0.001
    assert results["test_shape"] == [147, 10]
    assert abs(results["test_ac_mae"] - 4.3956) < 0.001


def test_generate_step0_report_headings():
    audit = audit_ac_data()
    timing = parse_lab_pc_timing(repo_root)
    hrnet_info = check_hrnet_support()
    fassd_ckpt = check_fassd_checkpoint(r"E:\BUBT-CSE-4th yr\Capstone\kaggle_outputs\fassd_run1\fassd_gray_seed42\best_model.pt")
    results_info = parse_committed_results(repo_root)

    report = generate_step0_report(audit, timing, hrnet_info, fassd_ckpt, results_info, repo_root)
    assert "# AC Campaign Step 0: Dataset Audit, Inventory & Experimental Plan" in report
    assert "## 1. Inventory & Dataset Architecture" in report
    assert "## 2. Leakage & Data Integrity Audit" in report
    assert "## 3. Timing & Resource Profiling" in report
    assert "## 4. Proposed Campaign Protocol & Execution Plan" in report
    assert "ZERO cross-patient duplicates detected" in report
    assert "147 images / 114 patients" in report
