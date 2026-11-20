"""tests/test_report_numbers.py

Pytest suite ensuring all numbers in docs/report/REPORT_NUMBERS.json
are strictly backed by committed files on disk with exact SHA256 integrity.
"""

import importlib.util
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "57_verify_report_numbers.py"


def _load_verify_func():
    spec = importlib.util.spec_from_file_location("verify_module", str(SCRIPT_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.verify_all_report_numbers


def test_all_report_numbers_verified():
    """Ensure all dissertation report numbers match disk sources and SHA-256 hashes."""
    verify_func = _load_verify_func()
    n_entries, n_sources = verify_func()
    assert n_entries >= 100, f"Expected at least 100 verified entries, got {n_entries}"
    assert n_sources >= 10, f"Expected at least 10 source files, got {n_sources}"
