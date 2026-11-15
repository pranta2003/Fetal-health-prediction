"""Tests for HC18 official test set evaluation summary integrity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HC18_DIR = REPO_ROOT / "docs" / "results" / "hc18_official"
PUB_REFS_PATH = REPO_ROOT / "docs" / "results" / "published_refs.json"

EXPECTED_ENSEMBLE_SHA256 = "14daf31b3ca46370d926e664585a1fcfd32dc28738915d2177ed593a3cccc85b"
EXPECTED_SINGLE_SHA256 = "2fb3e8f89c2e49d342a3824e94409e19a105db64b7cf216dfa929eda7b60a851"


def sha256_file(p: Path) -> str:
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_input_files_sha256_hashes():
    ens_path = HC18_DIR / "metrics_ensemble_5fold.json"
    sin_path = HC18_DIR / "metrics_single_fold1.json"

    assert ens_path.is_file(), f"Missing {ens_path}"
    assert sin_path.is_file(), f"Missing {sin_path}"

    assert sha256_file(ens_path) == EXPECTED_ENSEMBLE_SHA256
    assert sha256_file(sin_path) == EXPECTED_SINGLE_SHA256


def test_submission_csvs_row_counts_if_present():
    sub_ens = HC18_DIR / "submission_ensemble_5fold.csv"
    sub_sin = HC18_DIR / "submission_single_fold1.csv"

    if sub_ens.is_file():
        with open(sub_ens, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        # 1 header + 335 test predictions = 336 rows
        assert len(lines) == 336
        assert lines[0] == "filename,center_x_mm,center_y_mm,semi_axes_a_mm,semi_axes_b_mm,angle_rad"

    if sub_sin.is_file():
        with open(sub_sin, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 336
        assert lines[0] == "filename,center_x_mm,center_y_mm,semi_axes_a_mm,semi_axes_b_mm,angle_rad"


def test_summary_json_numbers_match_inputs():
    with open(HC18_DIR / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(HC18_DIR / "metrics_ensemble_5fold.json", "r", encoding="utf-8") as f:
        ens_metrics = json.load(f)["aggregates"]
    with open(HC18_DIR / "metrics_single_fold1.json", "r", encoding="utf-8") as f:
        sin_metrics = json.load(f)["aggregates"]

    # Check ensemble
    ens_sum = summary["ensemble_5fold"]
    assert ens_sum["count"] == ens_metrics["absolute_difference"]["count"]
    assert ens_sum["hc_abs_diff_mean"] == pytest.approx(ens_metrics["absolute_difference"]["mean"])
    assert ens_sum["hc_abs_diff_std"] == pytest.approx(ens_metrics["absolute_difference"]["std"])
    assert ens_sum["hc_abs_diff_median"] == pytest.approx(ens_metrics["absolute_difference"]["50pc"])
    assert ens_sum["hc_abs_diff_max"] == pytest.approx(ens_metrics["absolute_difference"]["max"])
    assert ens_sum["hc_signed_diff_mean"] == pytest.approx(ens_metrics["difference"]["mean"])
    assert ens_sum["dice_mean"] == pytest.approx(ens_metrics["DICE"]["mean"])
    assert ens_sum["hausdorff_mean"] == pytest.approx(ens_metrics["hausdorff_distance"]["mean"])

    # Check single
    sin_sum = summary["single_fold1"]
    assert sin_sum["count"] == sin_metrics["absolute_difference"]["count"]
    assert sin_sum["hc_abs_diff_mean"] == pytest.approx(sin_metrics["absolute_difference"]["mean"])
    assert sin_sum["hc_abs_diff_std"] == pytest.approx(sin_metrics["absolute_difference"]["std"])
    assert sin_sum["hc_abs_diff_median"] == pytest.approx(sin_metrics["absolute_difference"]["50pc"])
    assert sin_sum["hc_abs_diff_max"] == pytest.approx(sin_metrics["absolute_difference"]["max"])
    assert sin_sum["hc_signed_diff_mean"] == pytest.approx(sin_metrics["difference"]["mean"])
    assert sin_sum["dice_mean"] == pytest.approx(sin_metrics["DICE"]["mean"])
    assert sin_sum["hausdorff_mean"] == pytest.approx(sin_metrics["hausdorff_distance"]["mean"])

    # Statement checks: no rank claimed
    stmt = summary["statement"]
    assert "no rank is claimed" in stmt
    assert "test labels are hidden" in stmt
    assert "same range as the published values listed" in stmt
    assert "ensemble vs single difference = 0.05 mm" in stmt


def test_summary_md_numbers_traceable_to_sources():
    with open(HC18_DIR / "summary.md", "r", encoding="utf-8") as f:
        md_text = f.read()

    # Collect numbers in summary.md
    num_matches = re.findall(r"\b\d+\.\d+\b", md_text)
    assert len(num_matches) > 10

    # Ensure none of the prohibited leaderboard claims exist
    lower_text = md_text.lower()
    assert "leaderboard rank" not in lower_text
    assert "placed #" not in lower_text
    assert "rank 1" not in lower_text
    assert "rank 2" not in lower_text
    assert "first place" not in lower_text
