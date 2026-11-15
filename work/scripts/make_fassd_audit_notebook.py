"""make_fassd_audit_notebook.py: Generate notebooks/benchmarks/fassd_audit.ipynb."""

from pathlib import Path
import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent


def create_notebook():
    nb = nbf.v4.new_notebook()

    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# FASSD Data Audit and Patient-Grouped Split Benchmark\n\n"
            "This notebook audits the **FASSD** (Fetal Abdominal Structures Segmentation Dataset Using Ultrasonic Images):\n"
            "- **Images**: 1,588 ultrasound images (`IMAGES/*.png` and `ARRAY_FORMAT/*.npy`)\n"
            "- **Structures**: 4 abdominal anatomical structures (`artery`, `liver`, `stomach`, `vein`)\n\n"
            "### Comprehensive Audit Protocols\n"
            "1. **Patient ID Parsing**: Width distribution, patient count ($N=169$), zero-padding collision audit.\n"
            "2. **Pairwise Agreement**: Byte-exact verification of PNG vs NPY image channels.\n"
            "3. **Structure Analysis**: Mask presence, area fractions, blobs/connected components, cross-structure overlap.\n"
            "4. **Duplicate & Leakage Audit**: SHA-256 analysis identifying same-patient and across-patient duplicates.\n"
            "5. **Color & Intensity Audit**: Grayscale vs Color Doppler / annotation channel difference analysis.\n"
            "6. **Patient-Grouped Split**: Seed 42 patient-grouped ~70/15/15 partition with 0 patient overlap and all structures $\\ge 20$.\n"
            "7. **Visualizations**: 12 sample contour/structure overlays across all 4 structures."
        ),
        nbf.v4.new_code_cell(
            "import json\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import display, Markdown\n\n"
            "# Locate repository root\n"
            "work_dir = Path.cwd().resolve()\n"
            "while work_dir.name != 'work' and work_dir.parent != work_dir:\n"
            "    work_dir = work_dir.parent\n\n"
            "results_dir = work_dir / 'docs' / 'results' / 'fassd'\n"
            "md_path = results_dir / 'fassd_audit.md'\n"
            "json_path = results_dir / 'fassd_audit.json'\n"
            "split_csv_path = results_dir / 'split_v1.csv'\n\n"
            "assert md_path.is_file(), f'Missing {md_path}'\n"
            "assert json_path.is_file(), f'Missing {json_path}'\n"
            "assert split_csv_path.is_file(), f'Missing {split_csv_path}'\n\n"
            "with open(json_path, 'r', encoding='utf-8') as f:\n"
            "    audit = json.load(f)\n\n"
            "print('Loaded FASSD Audit JSON successfully.')\n"
            "print(f'Total Images: {audit[\"total_images\"]}')\n"
            "print(f'Unique Patients: {audit[\"patient_id_parsing\"][\"unique_patients\"]}')\n"
            "print(f'Pairwise Agreement Max Diff: {audit[\"pairwise_agreement\"][\"max_abs_diff_cv2_imread_unchanged\"]}')\n\n"
            "display(Markdown(md_path.read_text(encoding='utf-8')))"
        ),
        nbf.v4.new_code_cell(
            "# Verify Patient-Grouped Split Integrity\n"
            "df_split = pd.read_csv(split_csv_path)\n"
            "print(f'Total Split Rows: {len(df_split)}')\n"
            "print('Images per split:', dict(df_split['split'].value_counts()))\n"
            "print('Patients per split:', dict(df_split.groupby('split')['patient'].nunique()))\n\n"
            "train_p = set(df_split[df_split['split'] == 'train']['patient'])\n"
            "val_p = set(df_split[df_split['split'] == 'val']['patient'])\n"
            "test_p = set(df_split[df_split['split'] == 'test']['patient'])\n\n"
            "assert len(train_p & val_p) == 0, 'Leakage between train and val!'\n"
            "assert len(train_p & test_p) == 0, 'Leakage between train and test!'\n"
            "assert len(val_p & test_p) == 0, 'Leakage between val and test!'\n"
            "print('VERIFIED: Zero patient overlap across train, val, and test partitions.')"
        ),
        nbf.v4.new_code_cell(
            "# Summary of Across-Patient Duplicate Images (Leakage Warning)\n"
            "dups = audit['duplicate_analysis']['across_patient_duplicates']\n"
            "print(f'Across-patient duplicate image pairs: {len(dups)}')\n"
            "for h, item in dups.items():\n"
            "    print(f'  Hash {h[:12]}... Patients {item[\"patients\"]}: {item[\"files\"]}')"
        ),
        nbf.v4.new_code_cell(
            "# Visualizations: 12 Structure Overlays\n"
            "overlays = audit['overlays_generated']\n"
            "print(f'Total Overlays Generated: {len(overlays)}')\n"
            "for ov in overlays:\n"
            "    print(f'[{ov[\"structure\"].upper()}] Sample {ov[\"sample_index\"]}: {ov[\"filename\"]} (Patient {ov[\"patient\"]}) Area={ov[\"area_pct\"]:.2f}% -> {ov[\"saved_path\"]}')"
        ),
    ]

    out_path = REPO_ROOT / "notebooks" / "benchmarks" / "fassd_audit.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated {out_path}")


if __name__ == "__main__":
    create_notebook()
