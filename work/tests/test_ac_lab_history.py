"""test_ac_lab_history.py: Unit tests verifying AC lab-PC history proof data and facts."""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure scripts dir is on sys.path
scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

import importlib.util
spec = importlib.util.spec_from_file_location("ac_history_table", scripts_dir / "28_ac_history_table.py")
ac_hist = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac_hist)


@pytest.fixture(scope="module")
def inputs_dir():
    work_dir = Path(__file__).resolve().parent.parent
    p = work_dir / "notebooks" / "history" / "ac_lab_pc" / "inputs"
    assert p.is_dir(), f"Missing inputs directory: {p}"
    return p


@pytest.fixture(scope="module")
def history_data(inputs_dir):
    return ac_hist.generate_history_data(inputs_dir)


def test_committed_inputs_exist(inputs_dir):
    expected_files = [
        "dsnt_full_oof_results.csv",
        "ens_log.txt",
        "ens_results.json",
        "ens_test_predictions.csv",
        "full_log.txt",
        "smoke_log.txt",
        "v3_base_results.json",
        "v3_base_test_predictions.csv",
        "v3_full_results.json",
        "v3_full_test_predictions.csv",
        "v3_log.txt",
        "scripts_original/ac_dsnt.py",
        "scripts_original/ac_v3.py",
        "scripts_original/ac_v3_ens.py",
    ]
    for ef in expected_files:
        fp = inputs_dir / ef
        assert fp.is_file(), f"Missing expected input: {ef}"
        assert fp.stat().st_size > 0, f"File is empty: {ef}"


def test_dsnt_oof_metrics_match_full_log(inputs_dir):
    df_oof = pd.read_csv(inputs_dir / "dsnt_full_oof_results.csv")
    full_log_text = (inputs_dir / "full_log.txt").read_text(encoding="utf-16")
    parsed = ac_hist.parse_full_log_oof(full_log_text)

    oof_err = (df_oof["ac_pred"] - df_oof["ac_true"]).abs()
    assert len(df_oof) == parsed["n"] == 789
    assert round(float(oof_err.mean()), 2) == parsed["ac_mae"] == 5.18
    assert round(float(np.sqrt(((df_oof["ac_pred"] - df_oof["ac_true"]) ** 2).mean())), 2) == parsed["ac_rmse"] == 9.07
    assert round(float((100.0 * oof_err / df_oof["ac_true"]).mean()), 2) == parsed["mean_abs_pct_error"] == 2.60
    assert round(float(oof_err[df_oof["source"] == "FP"].mean()), 2) == parsed["fp_mae"] == 5.08
    assert round(float(oof_err[df_oof["source"] == "UCL"].mean()), 2) == parsed["ucl_mae"] == 5.91


def test_v3_results_match_v3_log(inputs_dir):
    v3_log_text = (inputs_dir / "v3_log.txt").read_text(encoding="utf-16")
    parsed = ac_hist.parse_v3_log_table(v3_log_text)

    b = json.loads((inputs_dir / "v3_base_results.json").read_text())["test_tta"]
    f = json.loads((inputs_dir / "v3_full_results.json").read_text())["test_tta"]

    assert round(b["ac_mae"], 2) == parsed["base"]["ac_mae"] == 5.81
    assert round(b["ac_rmse"], 2) == parsed["base"]["ac_rmse"] == 8.87
    assert round(b["ac_median"], 2) == parsed["base"]["ac_median"] == 3.65
    assert round(b["ac_bias"], 2) == parsed["base"]["ac_bias"] == -3.26
    assert round(b["nme"], 4) == parsed["base"]["nme"] == 0.0735
    assert round(b["over10"] * 100, 1) == round(parsed["base"]["over10"] * 100, 1) == 16.3

    assert round(f["ac_mae"], 2) == parsed["v3_full"]["ac_mae"] == 4.40
    assert round(f["ac_rmse"], 2) == parsed["v3_full"]["ac_rmse"] == 6.16
    assert round(f["ac_median"], 2) == parsed["v3_full"]["ac_median"] == 2.91
    assert round(f["ac_bias"], 2) == parsed["v3_full"]["ac_bias"] == -1.55
    assert round(f["nme"], 4) == parsed["v3_full"]["nme"] == 0.0731
    assert round(f["over10"] * 100, 1) == round(parsed["v3_full"]["over10"] * 100, 1) == 8.2


def test_ens_results_match_ens_log(inputs_dir):
    ens_log_text = (inputs_dir / "ens_log.txt").read_text(encoding="utf-16")
    parsed = ac_hist.parse_ens_log_table(ens_log_text)
    ens_json = json.loads((inputs_dir / "ens_results.json").read_text())

    for k in ["s1", "s2", "s3", "s4", "res512", "ENS_mean", "ENS_median"]:
        assert round(ens_json[k]["ac_mae"], 2) == parsed[k]["ac_mae"]
        assert round(ens_json[k]["ac_rmse"], 2) == parsed[k]["ac_rmse"]
        assert round(ens_json[k]["ac_median"], 2) == parsed[k]["ac_median"]
        assert round(ens_json[k]["ac_bias"], 2) == parsed[k]["ac_bias"]
        assert round(ens_json[k]["ac_pct"], 2) == parsed[k]["ac_pct"]


def test_recomputed_ensemble_mae(inputs_dir):
    df_ens = pd.read_csv(inputs_dir / "ens_test_predictions.csv")
    ens_json = json.loads((inputs_dir / "ens_results.json").read_text())
    recomputed_mae = float((df_ens["ac_pred"] - df_ens["ac_true"]).abs().mean())
    assert np.isclose(recomputed_mae, ens_json["ENS_mean"]["ac_mae"])


def test_seeds_mean_std_min_max(history_data):
    ss = history_data["seeds_summary"]
    assert ss["mean"] == 4.58
    assert ss["std"] == 0.27
    assert ss["min"] == 4.32
    assert ss["max"] == 4.96


def test_seed5_assert_and_log_facts(inputs_dir):
    facts = ac_hist.extract_log_facts(inputs_dir)
    assert facts["v3_log.txt"]["invocations_count"] == 1
    assert facts["ens_log.txt"]["invocations_count"] == 4
    assert facts["full_log.txt"]["invocations_count"] == 5
    assert facts["smoke_log.txt"]["invocations_count"] == 1

    # Seed 5 assertion: exactly 0 [s5] lines in any log
    assert sum(f["s5_lines_count"] for f in facts.values()) == 0


def test_table_json_and_md_integrity():
    work_dir = Path(__file__).resolve().parent.parent
    json_path = work_dir / "docs" / "results" / "ac_lab_pc" / "history_table.json"
    md_path = work_dir / "docs" / "results" / "ac_lab_pc" / "history_table.md"
    assert json_path.is_file()
    assert md_path.is_file()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(data["rows"]) == 13
    assert data["assertions"]["seed5_never_trained"] is True

    # Check v1 row
    v1_row = [r for r in data["rows"] if "v1 heatmap" in r["run"]][0]
    assert v1_row["resolution"] == 384
    assert v1_row["n"] == 789
    assert v1_row["ac_mae"] == 182.60
    assert v1_row["ac_rmse"] == 195.13
    assert "FAILED/COLLAPSED run" in v1_row["status"]

    # Check v2 colab row
    v2_row = [r for r in data["rows"] if "v2 DSNT smoke" in r["run"]][0]
    assert v2_row["status"] == "DSNT smoke on Colab CPU, 1 fold, 5 epochs: code check only; the real v2 run is the lab-PC full_log row"

    # Check Colab history dict
    colab = data["colab_history"]
    assert colab["v1"]["dataset_line"] == "Pooled dataset: 789 images from 615 unique patients (586 FP + 29 UCL)"
    assert colab["v1"]["folds"]["fold_4"]["early_stopping_epoch"] == 97
    assert colab["v1"]["folds"]["fold_5"]["early_stopping_epoch"] == 89
    assert colab["v1"]["folds"]["fold_4"]["best_val_loss"] == 0.000344
    assert colab["v1"]["folds"]["fold_5"]["best_val_loss"] == 0.000351

    # Check smoke_log resolution is None (displays as '-')
    smoke_row = [r for r in data["rows"] if "smoke_log" in r["run"]][0]
    assert smoke_row["resolution"] is None

    # Check full_log crash note: contains error list and wording, no 'handled'
    md_content = md_path.read_text(encoding="utf-8")
    assert "handled" not in md_content.lower()
    full_facts = data["facts"]["full_log.txt"]
    assert len(full_facts["traceback_errors"]) == 5
    assert "refuses to resume with a different CONFIG" in md_content
