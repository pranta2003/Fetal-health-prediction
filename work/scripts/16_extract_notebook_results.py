"""16_extract_notebook_results.py: Extract historical metrics from notebook.

Reads historical Capstone notebook outputs and extracts lines containing
MAE, RMSE, AUC, NME verbatim, recording cell index and nearest markdown heading.
Writes docs/results/history/notebook_results.json and notebook_results.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

# Ensure src/ is on sys.path if needed
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

DEFAULT_NOTEBOOK = "notebooks/history/Capstone_fetalDataset_history.ipynb"
DEFAULT_OUT_DIR = "docs/results/history"
METRIC_REGEX = re.compile(r"\b(MAE|RMSE|AUC|NME)\b", re.IGNORECASE)


def extract_notebook_results(
    notebook_path: str | Path,
    out_dir: str | Path,
) -> list[dict]:
    nb_path = Path(notebook_path).resolve()
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")

    target_dir = Path(out_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    results = []
    latest_heading = ""

    for c_idx, cell in enumerate(nb.get("cells", [])):
        cell_type = cell.get("cell_type", "")
        if cell_type == "markdown":
            src = "".join(cell.get("source", []))
            for line in src.splitlines():
                line_str = line.strip()
                if line_str.startswith("#"):
                    latest_heading = line_str
        elif cell_type == "code":
            matching_lines: list[str] = []
            for out in cell.get("outputs", []):
                # Inspect text stream outputs (stdout/stderr)
                if out.get("output_type") == "stream":
                    text_data = out.get("text", [])
                    raw_lines = (
                        text_data
                        if isinstance(text_data, list)
                        else text_data.splitlines(keepends=True)
                    )
                    for raw in raw_lines:
                        for sub in raw.splitlines():
                            sub_strip = sub.rstrip("\r\n")
                            if METRIC_REGEX.search(sub_strip):
                                matching_lines.append(sub_strip)

            if matching_lines:
                results.append(
                    {
                        "cell_index": c_idx,
                        "heading": latest_heading,
                        "lines": matching_lines,
                    }
                )

    # Write JSON output
    json_path = target_dir / "notebook_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Write Markdown output
    md_path = target_dir / "notebook_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Capstone Historical Notebook Results\n\n")
        f.write(f"Source notebook: `{nb_path.name}`\n\n")
        f.write(
            f"Extracted {len(results)} cells containing metric lines "
            f"(MAE, RMSE, AUC, NME) verbatim from execution outputs.\n\n"
        )
        for item in results:
            f.write(f"## Cell {item['cell_index']}\n\n")
            f.write(f"- **Nearest Heading**: `{item['heading']}`\n")
            f.write("- **Extracted Lines**:\n\n")
            f.write("```text\n")
            for l in item["lines"]:
                f.write(f"{l}\n")
            f.write("```\n\n")

    print(f"Extracted {len(results)} cells to:")
    print(f"  {json_path}")
    print(f"  {md_path}")
    print("\nSummary of extracted cells:")
    for item in results:
        print(f"\n--- Cell {item['cell_index']} [{item['heading']}] ---")
        for l in item["lines"]:
            print(f"  {l}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract metric results from historical notebook."
    )
    parser.add_argument(
        "--notebook",
        type=str,
        default=DEFAULT_NOTEBOOK,
        help=f"Path to notebook (default: {DEFAULT_NOTEBOOK})",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    extract_notebook_results(args.notebook, args.out_dir)


if __name__ == "__main__":
    main()
