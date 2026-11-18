"""tests/test_lit_benchmark.py

Tests for:
- Damped multiplicative calibration logic and generated candidate CSVs
- Structure and metrics of docs/results/literature/BEST_PUBLISHED.md
- SHA256 integrity of exploratory calibration submission candidates
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
src_dir = REPO_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import importlib
gen_mod = importlib.import_module("scripts.44_generate_best_published")
load_committed_metrics = gen_mod.load_committed_metrics
build_markdown = gen_mod.build_markdown


def test_committed_metrics_loading():
    metrics = load_committed_metrics()
    assert "hc18_official_ens" in metrics
    assert "hc18_official_ens_corrected" in metrics
    assert "hc18_candidates" in metrics
    assert "ac_v3_full" in metrics
    assert "femur" in metrics
    assert "stream3" in metrics

    # Verify key values are valid numbers
    assert metrics["hc18_official_ens"]["mad_mean"] == pytest.approx(1.95, rel=1e-2)
    assert metrics["hc18_official_ens_corrected"]["mad_mean"] == pytest.approx(1.9472, rel=1e-3)
    assert metrics["hc18_official_ens_corrected"]["dice_mean"] == pytest.approx(97.977, rel=1e-3)
    assert metrics["hc18_official_ens_corrected"]["hd_mean"] == pytest.approx(1.2144, rel=1e-3)
    assert metrics["hc18_official_ens_corrected"]["signed_diff_mean"] == pytest.approx(0.7844, rel=1e-3)
    assert metrics["ac_v3_full"]["ac_mae"] == pytest.approx(4.3956, rel=1e-3)
    assert metrics["femur"]["row_R4"]["ensemble_all"]["fl_mae"] == pytest.approx(1.12, rel=1e-2)
    assert metrics["stream3"]["Our XGBoost"] == pytest.approx(0.7381, rel=1e-3)


def test_best_published_markdown_generation(tmp_path):
    metrics = load_committed_metrics()
    md = build_markdown(metrics)
    assert "# Best Published Literature Benchmarks & Modern Methods Scan" in md
    assert "karanchand2525" in md
    assert "arXiv:2512.16710" in md
    assert "metrics_ensemble_5fold_corrected.json" in md
    assert "submission 1, angle bug" in md
    assert "arXiv:2609.19230" in md
    assert "BiometryNet" in md
    assert "SonoBase" in md
    assert "UltraSam" in md
    assert "van den Heuvel" in md
    assert "1.9846" in md  # Full calibrated nested OOF MAE


def test_calibration_candidate_csvs_integrity():
    cand_dir = REPO_ROOT / "docs" / "results" / "hc18_calibration" / "candidates"
    orig_sub = REPO_ROOT / "docs" / "results" / "hc18_official" / "submission_ensemble_5fold_corrected.csv"

    assert orig_sub.exists(), f"Missing base submission CSV: {orig_sub}"
    df_orig = pd.read_csv(orig_sub)
    assert len(df_orig) == 335

    summary_json = cand_dir / "candidates_summary.json"
    assert summary_json.exists()
    with open(summary_json, "r") as f:
        summary = json.load(f)

    # Check candidates for alpha=0.50 and alpha=1.00
    for alpha, pct in [(0.50, 50), (1.00, 100)]:
        fname = f"submission_ensemble_5fold_corrected_mult{pct:03d}_EXPLORATORY.csv"
        cand_csv = cand_dir / fname
        assert cand_csv.exists(), f"Missing candidate CSV: {cand_csv}"

        df_cand = pd.read_csv(cand_csv)
        assert len(df_cand) == 335
        assert list(df_cand.columns) == [
            "filename",
            "center_x_mm",
            "center_y_mm",
            "semi_axes_a_mm",
            "semi_axes_b_mm",
            "angle_rad",
        ]

        # Verify filenames, centers, and angle are strictly identical
        assert (df_cand["filename"] == df_orig["filename"]).all()
        assert np.allclose(df_cand["center_x_mm"], df_orig["center_x_mm"])
        assert np.allclose(df_cand["center_y_mm"], df_orig["center_y_mm"])
        assert np.allclose(df_cand["angle_rad"], df_orig["angle_rad"])

        # Verify scaling factor
        scale_expected = summary["generated_csvs"][fname]["scale_factor_R"]
        assert np.allclose(df_cand["semi_axes_a_mm"] / df_orig["semi_axes_a_mm"], scale_expected, rtol=1e-5)
        assert np.allclose(df_cand["semi_axes_b_mm"] / df_orig["semi_axes_b_mm"], scale_expected, rtol=1e-5)

        # Verify SHA-256 matches summary
        computed_sha = hashlib.sha256(cand_csv.read_bytes()).hexdigest()
        assert computed_sha == summary["generated_csvs"][fname]["sha256"]


def test_sha256sums_in_candidates_dir():
    cand_dir = REPO_ROOT / "docs" / "results" / "hc18_calibration" / "candidates"
    sha_file = cand_dir / "SHA256SUMS"
    assert sha_file.exists()

    lines = [line.strip() for line in sha_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2

    for line in lines:
        expected_sha, fname = line.split(maxsplit=1)
        target_file = cand_dir / fname
        assert target_file.exists()
        actual_sha = hashlib.sha256(target_file.read_bytes()).hexdigest()
        assert actual_sha == expected_sha
