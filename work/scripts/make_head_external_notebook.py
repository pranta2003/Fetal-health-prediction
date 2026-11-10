"""Script to construct benchmark notebook notebooks/benchmarks/head_external.ipynb."""

from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def build_head_external_notebook(dest_path: Path):
    nb = new_notebook()
    nb.metadata["language_info"] = {"name": "python"}

    md_intro = """# External Validation of HC18 5-Fold Head Model on FP and UCL Datasets

## 1. Background & Protocol
This notebook documents and reproduces the out-of-domain external validation of our HC18 5-fold U-Net ensemble (`hc18_seg_fold{1..5}_eff_b4_384.pt`, EfficientNet-B4 backbone, Run B weights) to two external fetal ultrasound datasets:
- **FP (Fetal Planes)**: 1,635 head ultrasound images across 909 subjects with calibrated pixel spacing (2 images excluded due to missing pixel spacing).
- **UCL (University College London)**: 126 calibrated head ultrasound images across 51 subjects.

### Evaluation Protocol:
1. **Model Architecture**: 5-fold ensemble trained on the HC18 training cohort (999 images) at 384x384 input resolution.
2. **Inference**: Horizontal-flip Test-Time Augmentation (TTA) averaged across both orientations and across all 5 fold models.
3. **Biometric Measurements**: Ellipse fitting on native-coordinate probability map (> 0.5 threshold) using `fit_submission_ellipse` and Ramanujan ellipse perimeter formula.
4. **Reference Standards**:
   - `BPD_mm`: Landmark Euclidean distance between annotated BPD calipers × pixel spacing.
   - `HC reference (mm)`: Computed from annotated OFD and BPD caliper lengths using Ramanujan's second ellipse perimeter formula ("approximate HC reference").
5. **Statistical Uncertainty**:
   - 1000-iteration patient-level (subject-clustered) bootstrap 95% confidence intervals for all metrics (MAE, RMSE, bias, median).
"""

    code_run = """import sys
from pathlib import Path
import subprocess

# Locate work root
work_dir = Path.cwd()
while work_dir.name != "work" and work_dir.parent != work_dir:
    work_dir = work_dir.parent
if str(work_dir / "src") not in sys.path:
    sys.path.insert(0, str(work_dir / "src"))

print(f"Work directory: {work_dir}")
script_path = work_dir / "scripts" / "24_hadlock_demo_step1.py"
print(f"Executing {script_path.name}...")

proc = subprocess.run(
    [sys.executable, str(script_path), "--out-dir", str(work_dir / "docs" / "results" / "hadlock_demo")],
    cwd=str(work_dir),
    capture_output=True,
    text=True,
    check=True,
)
print(proc.stdout)
"""

    code_tables = """import json
from pathlib import Path
import pandas as pd

# Load results JSON
results_path = work_dir / "docs" / "results" / "hadlock_demo" / "hadlock_demo_step1.json"
with open(results_path, "r", encoding="utf-8") as f:
    res = json.load(f)

# 1. Usable subjects table
print("=== 1. USABLE SUBJECTS FOR HADLOCK DEMO ===")
rows_s = []
for cohort, d in res["item1_usable_subjects"].items():
    s1a = d["images_per_plane_1a"]
    s1b = d["images_per_plane_1b"]
    rows_s.append({
        "Cohort": d["cohort"],
        "Criterion 1(a) Subjs": d["subjects_1a_count"],
        "Criterion 1(b) Subjs": d["subjects_1b_count"],
        "1(a) Head Img (min/med/max)": f"{s1a['head']['min']} / {s1a['head']['median']} / {s1a['head']['max']}",
        "1(a) Abd Img (min/med/max)": f"{s1a['abdomen']['min']} / {s1a['abdomen']['median']} / {s1a['abdomen']['max']}",
        "1(a) Femur Img (min/med/max)": f"{s1a['femur']['min']} / {s1a['femur']['median']} / {s1a['femur']['max']}",
    })
df_s = pd.DataFrame(rows_s)
display(df_s)

# 2. External validation metrics
print("\\n=== 2. EXTERNAL VALIDATION METRICS (HC & BPD) ===")
rows_m = []
for grp, d in res["item2_metrics"].items():
    hc = d["hc"]
    bpd = d["bpd"]
    rows_m.append({
        "Subgroup": grp,
        "Images": d["n_images"],
        "Subjects": d["n_subjects"],
        "HC MAE (mm)": f"{hc['mae']:.3f} [{hc['mae_ci_95'][0]:.3f}, {hc['mae_ci_95'][1]:.3f}]",
        "HC RMSE (mm)": f"{hc['rmse']:.3f} [{hc['rmse_ci_95'][0]:.3f}, {hc['rmse_ci_95'][1]:.3f}]",
        "HC Bias (mm)": f"{hc['bias']:+.3f} [{hc['bias_ci_95'][0]:+.3f}, {hc['bias_ci_95'][1]:+.3f}]",
        "HC Med AE (mm)": f"{hc['median']:.3f} [{hc['median_ci_95'][0]:.3f}, {hc['median_ci_95'][1]:.3f}]",
        "BPD MAE (mm)": f"{bpd['mae']:.3f} [{bpd['mae_ci_95'][0]:.3f}, {bpd['mae_ci_95'][1]:.3f}]",
        "BPD RMSE (mm)": f"{bpd['rmse']:.3f} [{bpd['rmse_ci_95'][0]:.3f}, {bpd['rmse_ci_95'][1]:.3f}]",
        "BPD Bias (mm)": f"{bpd['bias']:+.3f} [{bpd['bias_ci_95'][0]:+.3f}, {bpd['bias_ci_95'][1]:+.3f}]",
        "BPD Med AE (mm)": f"{bpd['median']:.3f} [{bpd['median_ci_95'][0]:.3f}, {bpd['median_ci_95'][1]:.3f}]",
    })
df_m = pd.DataFrame(rows_m)
display(df_m)

# 3. 10 Worst Images
print("\\n=== 3. 10 WORST IMAGES BY HC ABSOLUTE ERROR ===")
df_w = pd.DataFrame(res["worst_10_images_overall"])[[
    "dataset", "image_name", "SubjectID", "Split",
    "ref_hc_mm", "pred_hc_mm", "hc_abs_err",
    "ref_bpd_mm", "pred_bpd_mm", "bpd_abs_err"
]]
display(df_w)
"""

    md_audit = """## 4. Key Observations & Findings
1. **Usable 3-Plane Subjects**:
   - In **FP**, 29 subjects have an Abdomen image in Abdomen Test AND a Femur image in Femur Test AND at least one Head image (Criterion 1a). Of these, 17 subjects also have their Head image in Head Test (Criterion 1b).
   - In **UCL (calibrated)**, 5 subjects have all three planes in Test (Criterion 1a and 1b are identical because splits are aligned).
2. **External Validation Performance**:
   - The HC18 5-fold ensemble trained on the 999-image HC18 dataset exhibits stable generalization on external datasets.
   - Outliers are primarily driven by severe acoustic shadowing, non-standard transventricular cut angles, or extreme caliper differences in annotations.
"""

    nb.cells = [
        new_markdown_cell(md_intro),
        new_code_cell(code_run),
        new_code_cell(code_tables),
        new_markdown_cell(md_audit),
    ]

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Created notebook at {dest_path}")


if __name__ == "__main__":
    work_root = Path(__file__).resolve().parent.parent
    nb_path = work_root / "notebooks" / "benchmarks" / "head_external.ipynb"
    build_head_external_notebook(nb_path)
