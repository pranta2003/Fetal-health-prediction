"""test_code_map.py: Verify that all code files are mapped in docs/CODE_MAP.md."""

from __future__ import annotations

import fnmatch
from pathlib import Path
import re


def test_code_map_mentions_all_files():
    work_dir = Path(__file__).resolve().parent.parent
    code_map_path = work_dir / "docs" / "CODE_MAP.md"
    assert code_map_path.is_file(), f"Missing {code_map_path}"

    content = code_map_path.read_text(encoding="utf-8")
    patterns = re.findall(r"[\w\-*]+\.py", content)

    # Collect all python files under src/fetal, scripts, tests
    # Excluding __init__.py and tests/reference
    search_dirs = [
        work_dir / "src" / "fetal",
        work_dir / "scripts",
        work_dir / "tests",
    ]

    all_py_files: list[Path] = []
    for sdir in search_dirs:
        for p in sorted(sdir.rglob("*.py")):
            if p.name == "__init__.py":
                continue
            if "reference" in p.parts:
                continue
            all_py_files.append(p)

    missing: list[str] = []
    for f in all_py_files:
        fname = f.name
        # Check direct mention in CODE_MAP.md
        if fname in content:
            continue
        # Check wildcard pattern from CODE_MAP.md (e.g. test_hc18_*.py, test_ac_*.py, test_clinical_*.py)
        if any(fnmatch.fnmatch(fname, pat) for pat in patterns if "*" in pat):
            continue
        # conftest.py is pytest framework runner configuration
        # test_code_map.py tests CODE_MAP.md itself
        if fname in {"conftest.py", "test_code_map.py"}:
            continue
        # Check if base component/stem is explicitly mentioned (e.g. test_manifest.py covers 08_make_manifest.py)
        stem = f.stem.removeprefix("test_")
        if stem in content.lower():
            continue

        missing.append(str(f.relative_to(work_dir).as_posix()))

    assert not missing, f"Files missing from docs/CODE_MAP.md: {missing}"
