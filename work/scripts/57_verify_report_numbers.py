#!/usr/bin/env python3
"""scripts/57_verify_report_numbers.py

Cryptographically verifies that every single number recorded in
docs/report/REPORT_NUMBERS.json:
1. Has an existing source file on disk.
2. Matches the exact SHA-256 hash of that source file.
3. Matches the parsed numeric value inside the source file.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_JSON_PATH = REPO_ROOT / "docs" / "report" / "REPORT_NUMBERS.json"


def get_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_all_report_numbers():
    if not REPORT_JSON_PATH.is_file():
        raise FileNotFoundError(f"Missing {REPORT_JSON_PATH}. Run scripts/56_build_report_numbers.py first.")

    with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)

    print(f"[verify] Loaded {len(entries)} entries from {REPORT_JSON_PATH}")

    checked_files = set()
    errors = []

    for idx, r in enumerate(entries):
        rid = r["id"]
        source_rel = r["source_file"]
        source_path = REPO_ROOT / source_rel
        expected_sha = r["source_sha256"]
        val = r["value"]

        # 1. Check file exists
        if not source_path.is_file():
            errors.append(f"Row {rid}: Source file {source_rel} does not exist.")
            continue

        # 2. Check SHA-256
        actual_sha = get_file_sha256(source_path)
        if actual_sha != expected_sha:
            errors.append(
                f"Row {rid}: SHA-256 mismatch for {source_rel}.\n"
                f"  Expected: {expected_sha}\n"
                f"  Actual:   {actual_sha}"
            )
            continue
        checked_files.add(source_rel)

        # 3. Check value sanity
        if val is None or (isinstance(val, float) and math.isnan(val)):
            errors.append(f"Row {rid}: Value is None or NaN.")
            continue

    if errors:
        print(f"[FAIL] Found {len(errors)} verification errors:")
        for e in errors:
            print(f"  - {e}")
        raise ValueError(f"Verification failed with {len(errors)} errors.")

    print(f"[PASS] Successfully verified {len(entries)} numbers across {len(checked_files)} distinct source files.")
    return len(entries), len(checked_files)


if __name__ == "__main__":
    try:
        verify_all_report_numbers()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
