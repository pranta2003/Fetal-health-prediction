# FIELD: General
# MODEL: None
# STEP: utility
# USED BY: work/docs/CODE_MAP.md

"""check_secrets.py: Scan workspace for leaked tokens, keys, and private credentials."""

from __future__ import annotations

from pathlib import Path
import re
import sys

PATTERNS = [
    (re.compile(r"ghp_[A-Za-z0-9_]{36}"), "GitHub Personal Access Token"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{82}"), "Fine-grained GitHub Token"),
    (re.compile(r"hf_[A-Za-z0-9]{34}"), "HuggingFace Token"),
    (re.compile(r"ey[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}"), "JWT / Bearer Token"),
]

IGNORE_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache"}


def scan_file(path: Path) -> list[str]:
    findings = []
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return findings

    for pat, desc in PATTERNS:
        matches = pat.findall(content)
        if matches:
            findings.append(f"{path}: Found {desc} ({len(matches)} match(es))")
    return findings


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    all_findings = []
    for p in root.rglob("*"):
        if p.is_file() and not any(part in IGNORE_DIRS for part in p.parts):
            # Skip binary / image files
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".pt", ".npy", ".npz", ".zip"}:
                continue
            findings = scan_file(p)
            all_findings.extend(findings)

    if all_findings:
        print("SECRET SCAN FAILED: Detected sensitive tokens:")
        for f in all_findings:
            print("  ", f)
        return 1

    print("SECRET SCAN PASSED: Zero leaked tokens or credentials detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
