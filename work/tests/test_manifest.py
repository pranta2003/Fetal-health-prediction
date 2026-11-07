"""Tests for weights manifest generation and verification tool (scripts/08_make_manifest.py)."""

import sys
from pathlib import Path

# Add scripts directory to sys.path so 08_make_manifest can be imported
scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

manifest_module = __import__("08_make_manifest")
generate_manifest = manifest_module.generate_manifest
verify_manifest = manifest_module.verify_manifest

hash_results_module = __import__("14_hash_results")
generate_sums = hash_results_module.generate_sums
verify_sums = hash_results_module.verify_sums



def test_manifest_generate_and_detect_modification(tmp_path):
    # Setup dummy model directory structure
    sub1 = tmp_path / "model_a"
    sub1.mkdir()
    f1 = sub1 / "best.pt"
    f1.write_bytes(b"dummy weights for model a")

    sub2 = tmp_path / "model_b"
    sub2.mkdir()
    f2 = sub2 / "best.pt"
    f2.write_bytes(b"dummy weights for model b")

    # 1. Generate manifest
    manifest_path = generate_manifest(tmp_path)
    assert manifest_path.is_file()

    # 2. Verify clean state
    all_ok, results = verify_manifest(tmp_path)
    assert all_ok is True
    assert len(results) == 2
    assert all(r["status"] == "OK" for r in results)

    # 3. Modify a file and verify mismatch detection
    f1.write_bytes(b"tampered weights for model a")
    all_ok, results = verify_manifest(tmp_path)
    assert all_ok is False
    res_map = {r["rel_path"]: r["status"] for r in results}
    assert res_map["model_a/best.pt"] == "MISMATCH"
    assert res_map["model_b/best.pt"] == "OK"

    # 4. Delete a file and verify missing detection
    f2.unlink()
    all_ok, results = verify_manifest(tmp_path)
    assert all_ok is False
    res_map = {r["rel_path"]: r["status"] for r in results}
    assert res_map["model_b/best.pt"] == "MISSING"


def test_hash_results_generate_and_verify(tmp_path):
    f1 = tmp_path / "metrics.json"
    f1.write_text('{"hc_mae": 2.30}', encoding="utf-8")
    f2 = tmp_path / "oof_predictions.csv"
    f2.write_text("image,patient_id,pred_HC_mm\n001.png,1,140.0\n", encoding="utf-8")

    sums_path = generate_sums(tmp_path)
    assert sums_path.is_file()
    assert verify_sums(tmp_path) is True

    # Tamper
    f1.write_text('{"hc_mae": 9.99}', encoding="utf-8")
    assert verify_sums(tmp_path) is False

    # Remove
    f1.unlink()
    assert verify_sums(tmp_path) is False

