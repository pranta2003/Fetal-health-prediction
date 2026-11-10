"""Script to construct benchmark notebook notebooks/benchmarks/head_landmark.ipynb."""

from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def build_head_landmark_notebook(dest_path: Path):
    nb = new_notebook()
    nb.metadata["language_info"] = {"name": "python"}

    md_intro = """# Head Landmark Model Benchmark (Hadlock Demo + Dataset Paper Benchmark)

## 1. Background & Protocol
This notebook benchmarks our multi-center Head landmark regression model trained on the combined **FP + UCL Head** ultrasound datasets (calibrated `px_to_mm` images):
- **Model**: U-Net architecture with an `efficientnet-b4` encoder, 384x384 letterbox input, spatial soft-argmax keypoint decoding, and horizontal-flip test-time augmentation (TTA).
- **Landmarks**: 4 keypoints predicting the two OFD caliper endpoints and two BPD caliper endpoints (pairs, mirroring the AC landmark pipeline).
- **Training Setup**:
  - Seeds: 42, 43, 44 with early stopping monitored solely on a 10% patient-grouped validation slice of the official Training cohort.
  - Exclusions from Training: 4 images with collapsed/corrupted calipers (BPD/OFD < 10 mm, HC < 20 mm, or the Patient01738 annotation anomaly).
- **Testing Setup**: Official Test split (880 FP + 35 UCL images across 465 patients). Nothing excluded from Test.
- **Reporting**:
  - Per-seed metrics and 3-seed median ensemble with 1000-iteration patient-level bootstrap 95% confidence intervals.
  - Comparison against HC18 5-fold ensemble segmentation baseline and the dataset paper published reference (arXiv 2512.16710, Table 2).
"""

    code_run = """import sys
import json
from pathlib import Path
import pandas as pd
from IPython.display import display, Markdown

# Locate work root
work_dir = Path.cwd()
while work_dir.name != "work" and work_dir.parent != work_dir:
    work_dir = work_dir.parent
if str(work_dir / "src") not in sys.path:
    sys.path.insert(0, str(work_dir / "src"))

results_dir = work_dir / "docs" / "results" / "head_landmark"
print(f"Results dir: {results_dir}")
"""

    code_table = """# Display primary benchmark table
table_file = results_dir / "head_landmark_table.md"
if table_file.exists():
    display(Markdown(table_file.read_text(encoding="utf-8")))
else:
    print(f"Table file not found: {table_file}")
"""

    code_metrics = """# Load detailed JSON results
json_file = results_dir / "head_landmark_results.json"
if json_file.exists():
    with open(json_file, "r", encoding="utf-8") as f:
        res = json.load(f)
    print("Ensemble Test Summary:")
    print("  FP Test HC MAE:  ", f"{res['ensemble_test_fp']['hc_mae']:.3f} mm", res['ensemble_test_fp']['hc_mae_ci'])
    print("  FP Test BPD MAE: ", f"{res['ensemble_test_fp']['bpd_mae']:.3f} mm", res['ensemble_test_fp']['bpd_mae_ci'])
    print("  FP Test OFD MAE: ", f"{res['ensemble_test_fp']['ofd_mae']:.3f} mm", res['ensemble_test_fp']['ofd_mae_ci'])
    print("  FP Test BPD NME: ", f"{res['ensemble_test_fp']['bpd_nme_mean']:.4f} +- {res['ensemble_test_fp']['bpd_nme_sd']:.4f}")
    print("  FP Test OFD NME: ", f"{res['ensemble_test_fp']['ofd_nme_mean']:.4f} +- {res['ensemble_test_fp']['ofd_nme_sd']:.4f}")
    print("  UCL Test HC MAE: ", f"{res['ensemble_test_ucl']['hc_mae']:.3f} mm", res['ensemble_test_ucl']['hc_mae_ci'])
    print("  UCL Test BPD MAE:", f"{res['ensemble_test_ucl']['bpd_mae']:.3f} mm", res['ensemble_test_ucl']['bpd_mae_ci'])
"""

    code_outliers = """# Outliers excluded from training
outliers_dir = results_dir / "outliers"
outlier_files = sorted(list(outliers_dir.glob("*.png")))
print(f"Found {len(outlier_files)} outlier overlay images:")
for of in outlier_files:
    print(f"  - {of.name}")
"""

    nb.cells = [
        new_markdown_cell(md_intro),
        new_code_cell(code_run),
        new_code_cell(code_table),
        new_code_cell(code_metrics),
        new_code_cell(code_outliers),
    ]

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Created notebook at {dest_path}")


if __name__ == "__main__":
    dest = Path("notebooks/benchmarks/head_landmark.ipynb")
    build_head_landmark_notebook(dest)
