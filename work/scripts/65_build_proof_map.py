#!/usr/bin/env python3
"""Build docs/report/PROOF_LOCATIONS.md mapping every report number to file, script/notebook, and commit."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
numbers_path = REPO_ROOT / "docs" / "report" / "REPORT_NUMBERS.json"
out_path = REPO_ROOT / "docs" / "report" / "PROOF_LOCATIONS.md"

with open(numbers_path, "r", encoding="utf-8") as f:
    numbers = json.load(f)

print(f"Loaded {len(numbers)} numbers from {numbers_path}")

# Verify file existence on disk
missing_count = 0
rows = []
for entry in numbers:
    src_rel = entry.get("source_file")
    src_full = REPO_ROOT / src_rel if src_rel else None
    exists = src_full.is_file() if src_full else False
    if not exists:
        missing_count += 1
        status = "MISSING"
    else:
        status = "EXISTS"

    val_str = f"{entry['value']}"
    if entry.get("ci"):
        val_str += f" [{entry['ci'][0]}, {entry['ci'][1]}]"

    producer = entry.get("notebook") or entry.get("script") or "N/A"
    claim = f"{entry.get('approach', '')} - {entry.get('metric', '')}"

    rows.append({
        "id": entry["id"],
        "field": entry.get("field", "other"),
        "claim": claim,
        "value": val_str,
        "source_file": src_rel if exists else f"MISSING ({src_rel})",
        "producer": producer,
        "commit": entry.get("pr_commit", "N/A"),
        "status": status,
    })

# Group rows by field
fields_order = ["HC18", "AC", "femur", "tabular", "fassd", "head", "fusion", "dataset_audit", "other"]
grouped = {}
for r in rows:
    f = r["field"]
    grouped.setdefault(f, []).append(r)

md_lines = [
    "# Proof Locations Map\n",
    "This document maps every empirical claim and metric across the thesis/report to its exact on-disk artifact, producing script or notebook, and git commit provenance.\n",
    f"**Total Numbers Mapped**: {len(rows)} | **Verified Files on Disk**: {len(rows) - missing_count} | **Missing Files**: {missing_count}\n",
    "---\n",
]

for f in sorted(grouped.keys(), key=lambda x: fields_order.index(x) if x in fields_order else 99):
    group_rows = grouped[f]
    md_lines.append(f"## {f.upper()} Domain Claims ({len(group_rows)} metrics)\n")
    md_lines.append("| ID | Report Claim | Value | Result File Path | Producing Script / Notebook | Git Commit | File Status |")
    md_lines.append("|---|---|:---:|---|---|:---:|:---:|")
    for r in group_rows:
        md_lines.append(
            f"| `{r['id']}` | {r['claim']} | **{r['value']}** | `{r['source_file']}` | `{r['producer']}` | `{r['commit']}` | {r['status']} |"
        )
    md_lines.append("\n---\n")

content = "\n".join(md_lines)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Generated {out_path} with {len(rows)} claims. Missing files: {missing_count}.")
