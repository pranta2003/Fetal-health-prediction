"""Unit tests for multi-modal benchmark summary (scripts/20_benchmark_summary.py)."""

from __future__ import annotations

import json
from pathlib import Path
import re
import pytest

work_root = Path(__file__).resolve().parent.parent
scripts_dir = work_root / "scripts"
import sys

if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from importlib import import_module

summary_mod = import_module("20_benchmark_summary")
generate_benchmark_summary = summary_mod.generate_benchmark_summary


@pytest.fixture
def summary_md_text():
    results_dir = work_root / "docs" / "results"
    return generate_benchmark_summary(results_dir)


def test_ac_ours_and_claim_values_match(summary_md_text):
    """Verify that AC Landmark NME and AC mm rows use ONE model per row,

    and that the values in the 'Ours' cell and 'What we may claim' cell match identically.
    """
    table_lines = [l for l in summary_md_text.splitlines() if l.startswith("|")]
    
    # 1. AC Landmark NME row
    ac_nme_row = next(l for l in table_lines if "Abdomen (AC)" in l and "Landmark NME" in l)
    cols = [c.strip() for c in ac_nme_row.split("|")[1:-1]]
    ours_cell = cols[2]
    claim_cell = cols[5]

    tad_ours = re.search(r"TAD NME:\s*([0-9.]+\s*\+-\s*[0-9.]+)", ours_cell).group(1)
    apad_ours = re.search(r"APAD NME:\s*([0-9.]+\s*\+-\s*[0-9.]+)", ours_cell).group(1)

    tad_claim = re.search(r"TAD\s*([0-9.]+\s*\+-\s*[0-9.]+)", claim_cell).group(1)
    apad_claim = re.search(r"APAD\s*([0-9.]+\s*\+-\s*[0-9.]+)", claim_cell).group(1)

    assert tad_ours == tad_claim, f"TAD mismatch: Ours '{tad_ours}' vs Claim '{tad_claim}'"
    assert apad_ours == apad_claim, f"APAD mismatch: Ours '{apad_ours}' vs Claim '{apad_claim}'"
    assert "4-model median ensemble" in ours_cell
    assert "4-model median ensemble" in claim_cell

    # 2. AC Circumference mm row
    ac_mm_row = next(l for l in table_lines if "AC MAE (mm)" in l)
    cols_mm = [c.strip() for c in ac_mm_row.split("|")[1:-1]]
    ours_mm = cols_mm[2]
    claim_mm = cols_mm[5]

    mm_ours = re.search(r"([0-9.]+\s*mm)", ours_mm).group(1)
    mm_claim = re.search(r"([0-9.]+\s*mm)", claim_mm).group(1)

    assert mm_ours == mm_claim, f"AC mm mismatch: Ours '{mm_ours}' vs Claim '{mm_claim}'"
    assert "4-model median ensemble" in ours_mm
    assert "4-model median ensemble" in claim_mm


def test_femur_claim(summary_md_text):
    """Verify femur claim cell reflects that model is not yet trained."""
    table_lines = [l for l in summary_md_text.splitlines() if l.startswith("|")]
    femur_row = next(l for l in table_lines if "Femur (FL)" in l)
    cols = [c.strip() for c in femur_row.split("|")[1:-1]]
    claim_cell = cols[5]

    assert claim_cell == "model not trained; no comparison yet"
    assert "Reference baseline established" not in claim_cell


def test_stream3_published_refs_and_claims(summary_md_text):
    """Verify Stream 3 published baseline, comparable column, and claims."""
    # Check published_refs.json
    pub_refs_path = work_root / "docs" / "results" / "published_refs.json"
    with open(pub_refs_path, "r", encoding="utf-8") as f:
        pub_refs = json.load(f)

    s3_pub = pub_refs["stream3"]
    assert s3_pub["doi"] == "10.1007/s13755-020-00105-9"
    assert "10.1016" not in json.dumps(s3_pub)
    assert s3_pub["data"] == "CDC NVSS 2013-2016"
    assert "RANDOM" in s3_pub["protocol"]
    assert s3_pub["auc_cdc_early_stillbirth"]["LightGBM"] == 0.75
    assert s3_pub["auc_cdc_late_stillbirth"]["LightGBM"] == 0.60

    # Check Markdown rows
    table_lines = [l for l in summary_md_text.splitlines() if l.startswith("|")]
    e1_row = next(l for l in table_lines if "E1 Test AUC" in l)
    e4_row = next(l for l in table_lines if "E4 Test AUC" in l)

    e1_cols = [c.strip() for c in e1_row.split("|")[1:-1]]
    e4_cols = [c.strip() for c in e4_row.split("|")[1:-1]]

    # E1 checks
    assert e1_cols[4].startswith("Partly")
    assert "same order of magnitude; not directly comparable" in e1_cols[5]
    assert "term-band AUC 0.748 must not be compared with their late-stillbirth 0.60; possible source-file artifact." in e1_cols[5]

    # E4 checks
    assert e4_cols[4].startswith("No")
    assert "same order of magnitude; not directly comparable" in e4_cols[5]
    assert "0.60–0.76" not in e4_cols[3]
    assert "0.60-0.76" not in e4_cols[3]


def test_ladder_fold1_note_matches_files(summary_md_text):
    """Verify that fold-1 note under ladder table matches the dynamic JSON file values."""
    results_dir = work_root / "docs" / "results"
    with open(results_dir / "hc18_ladder" / "row_D_metrics.json", "r", encoding="utf-8") as f:
        d1 = json.load(f)
    with open(results_dir / "hc18_ladder" / "row_D_fold1_metrics.json", "r", encoding="utf-8") as f:
        d2 = json.load(f)

    expected_note = (
        f"Fold-1 numbers are single runs. The D recipe gave {d1['hc_mae']:.2f} in run 1 "
        f"and {d2['hc_mae']:.2f} in run 2 on fold 1 (GPU training is not bit-reproducible); "
        f"differences under about 0.3 mm are noise."
    )
    assert expected_note in summary_md_text


def test_head_landmark_summary_row(summary_md_text):
    """Verify that Head Landmark row in benchmark summary matches results json and claims at most 'on par'."""
    table_lines = [l for l in summary_md_text.splitlines() if l.startswith("|")]
    head_row = next((l for l in table_lines if "Head (FP+UCL)" in l), None)
    assert head_row is not None, "Head (FP+UCL) row missing from summary table"
    cols = [c.strip() for c in head_row.split("|")[1:-1]]
    ours_cell = cols[2]
    claim_cell = cols[5]

    assert "3-seed median ensemble" in ours_cell
    assert "at most 'on par'" in claim_cell
    assert "BPD NME:" in ours_cell
    assert "OFD NME:" in ours_cell

