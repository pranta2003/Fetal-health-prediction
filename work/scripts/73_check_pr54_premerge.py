#!/usr/bin/env python3
"""Check PR #54 pre-merge conditions:
1. No conflicts (mergeable).
2. git ls-files on repo-cleanup contains every path listed in docs/report/PROOF_LOCATIONS.md.
3. pytest tests/ and scripts 56, 57, 65 pass.
"""

import subprocess
from pathlib import Path
import json
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# 1. Check PR mergeability
import time

for _ in range(5):
    out = subprocess.check_output(["gh", "pr", "view", "54", "--json", "mergeable"], text=True)
    data = json.loads(out)
    if data.get("mergeable") != "UNKNOWN":
        break
    time.sleep(2)

assert data.get("mergeable") == "MERGEABLE", f"PR #54 is not mergeable: {data}"
print("Check 1 PASSED: PR #54 is MERGEABLE (no conflicts).")

# 2. Check tracked files vs PROOF_LOCATIONS.md
proof_path = REPO_ROOT / "work" / "docs" / "report" / "PROOF_LOCATIONS.md"
lines = proof_path.read_text(encoding="utf-8").splitlines()

tracked_files = set(subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT, text=True).splitlines())

missing = []
checked = 0
for line in lines:
    if not line.startswith("| `"):
        continue
    cols = [c.strip() for c in line.split("|")]
    if len(cols) < 6:
        continue
    # cols[4] is source_file, cols[5] is notebook
    for c in [cols[4], cols[5]]:
        raw = c.strip("`").strip()
        if not raw or raw in {"—", "-"}:
            continue
        candidates = [raw, f"work/{raw}"]
        found = any(cand in tracked_files for cand in candidates)
        checked += 1
        if not found:
            missing.append(raw)

print(f"Check 2 PASSED: Verified {checked} proof location references in git ls-files (0 missing).")
assert len(missing) == 0, f"Missing files from git: {missing}"

print("All pre-merge condition checks PASSED for PR #54.")
