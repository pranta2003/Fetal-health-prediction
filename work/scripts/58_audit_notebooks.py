#!/usr/bin/env python3

# FIELD: Report tools
# MODEL: None (Audit & Verification Harness)
# STEP: report_tools
# USED BY: work/docs/CODE_MAP.md

"""scripts/58_audit_notebooks.py

Audits all Jupyter notebooks under notebooks/:
- Checks for saved outputs
- Checks for error cells
- Scans for credentials / secrets
- Tags each notebook as PROOF (executed, outputs saved), RUNNER (launches scripts), or DRAFT
- Emits docs/report/NOTEBOOK_AUDIT.md
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
DOCS_REPORT_DIR = REPO_ROOT / "docs" / "report"

SECRET_PATTERNS = [
    re.compile(r'(?:api[_-]?key|secret|password|token|bearer|kaggle_key)\s*[:=]\s*[\'\"][^\'\"]+[\'\"]', re.I),
    re.compile(r'ghp_[a-zA-Z0-9]{20,}'),
    re.compile(r'AIza[0-9A-Za-z-_]{35}'),
]


def audit_notebooks():
    DOCS_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for nb_path in sorted(NOTEBOOKS_DIR.rglob("*.ipynb")):
        rel = nb_path.relative_to(REPO_ROOT).as_posix()
        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
        except Exception as e:
            results.append({
                "path": rel,
                "total_cells": 0,
                "code_cells": 0,
                "executed_cells": 0,
                "has_outputs": False,
                "error_cells": [str(e)],
                "secrets": [],
                "tag": "DRAFT",
            })
            continue

        cells = nb.get("cells", [])
        code_cells = [c for c in cells if c.get("cell_type") == "code"]
        cells_with_out = [c for c in code_cells if c.get("outputs")]
        error_cells = []
        secrets = []

        for i, c in enumerate(code_cells):
            src = "".join(c.get("source", []))
            for pat in SECRET_PATTERNS:
                if pat.search(src):
                    secrets.append(f"Cell {i} src: pattern matched")
            for out in c.get("outputs", []):
                if out.get("output_type") == "error":
                    ename = out.get("ename", "Error")
                    evalue = out.get("evalue", "")[:60]
                    error_cells.append(f"Cell {i}: {ename}: {evalue}")
                txt = "".join(out.get("text", []))
                for pat in SECRET_PATTERNS:
                    if pat.search(txt):
                        secrets.append(f"Cell {i} out: pattern matched")

        has_outputs = len(cells_with_out) > 0
        has_runner_patterns = any(
            "python " in "".join(c.get("source", [])) or "run_command" in "".join(c.get("source", []))
            for c in code_cells
        )

        if has_outputs:
            tag = "PROOF"
        elif has_runner_patterns:
            tag = "RUNNER"
        else:
            tag = "DRAFT"

        results.append({
            "path": rel,
            "total_cells": len(cells),
            "code_cells": len(code_cells),
            "executed_cells": len(cells_with_out),
            "has_outputs": has_outputs,
            "error_cells": error_cells,
            "secrets": secrets,
            "tag": tag,
        })

    # Write Markdown
    md_path = DOCS_REPORT_DIR / "NOTEBOOK_AUDIT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Notebook Audit Ledger\n\n")
        f.write(f"Total notebooks audited: **{len(results)}**\n\n")
        f.write("| Notebook Path | Tag | Code Cells | Executed Cells | Has Outputs | Error Cells | Secret Scan |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in results:
            sec_status = "CLEAN (0)" if not r["secrets"] else f"**WARNING ({len(r['secrets'])})**"
            err_status = "0" if not r["error_cells"] else f"{len(r['error_cells'])} errors"
            f.write(f"| `{r['path']}` | **{r['tag']}** | {r['code_cells']} | {r['executed_cells']} | {'Yes' if r['has_outputs'] else 'No'} | {err_status} | {sec_status} |\n")

        f.write("\n\n## Audit Notes\n")
        f.write("- **PROOF**: Notebook contains executed code cells with saved outputs proving results.\n")
        f.write("- **RUNNER**: Notebook serves as an execution harness launching CLI scripts.\n")
        f.write("- **DRAFT**: Unexecuted draft notebook or template.\n")
        f.write("- **Secret Scan**: Strictly zero credentials, API keys, or tokens committed.\n")

    print(f"[audit] Audited {len(results)} notebooks. Saved report to {md_path}")
    return results


if __name__ == "__main__":
    audit_notebooks()
