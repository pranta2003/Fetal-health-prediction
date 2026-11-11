"""make_femur_ablation_notebook.py: Generate notebooks/benchmarks/femur_ablation.ipynb."""

from pathlib import Path
import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()

    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Femur Model Ablation Study: Heatmap vs Coordinate Regression\n\n"
            "Evaluates 3 architectural approaches on the official FP+UCL Femur test set (N=348 images / 266 patients):\n"
            "- **Row A**: Heatmap landmarks (U-Net EfficientNet-B4, 384px letterbox, 1-channel dual-peak, flip TTA)\n"
            "- **Row B**: Direct coordinate regression (ConvNeXt-Tiny, 384px letterbox, order-invariant loss)\n"
            "- **Row C**: Heatmap landmarks (U-Net EfficientNet-B4, 512px letterbox, 1-channel dual-peak, flip TTA)\n\n"
            "Each row trained with 3 random seeds (42, 43, 44). Early stopping on 65 patient-grouped validation images.\n"
            "Patient-level bootstrap 95% confidence intervals computed with 1,000 resamples."
        ),
        nbf.v4.new_code_cell(
            "import json\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Markdown, display\n\n"
            "results_dir = Path('../../docs/results/femur_ablation')\n"
            "if not results_dir.is_dir():\n"
            "    results_dir = Path('docs/results/femur_ablation')\n"
            "if not results_dir.is_dir():\n"
            "    results_dir = Path('/kaggle/working/out')\n\n"
            "table_path = results_dir / 'femur_ablation_table.md'\n"
            "if table_path.is_file():\n"
            "    display(Markdown(table_path.read_text(encoding='utf-8')))\n"
            "else:\n"
            "    print('Table not found at', table_path)"
        ),
        nbf.v4.new_code_cell(
            "json_path = results_dir / 'femur_ablation_results.json'\n"
            "if json_path.is_file():\n"
            "    with open(json_path, 'r', encoding='utf-8') as f:\n"
            "        results = json.load(f)\n"
            "    print('Loaded ablation results for keys:', list(results.keys()))\n"
            "    for rk in ['row_A', 'row_B', 'row_C']:\n"
            "        if rk in results:\n"
            "            ens = results[rk]['ensemble_all']\n"
            "            ms = results[rk]['seeds_mean_sd']\n"
            "            print(f'=== {results[rk][\"name\"]} ===')\n"
            "            print(f'  Seeds Mean: MAE={ms[\"fl_mae_mean\"]:.3f} +- {ms[\"fl_mae_sd\"]:.3f} mm, NME={ms[\"fl_nme_mean\"]:.3f} +- {ms[\"fl_nme_sd\"]:.3f}')\n"
            "            print(f'  Ensemble:   MAE={ens[\"fl_mae\"]:.3f} [{ens[\"fl_mae_ci\"][0]:.3f}, {ens[\"fl_mae_ci\"][1]:.3f}] mm, NME={ens[\"fl_nme_mean\"]:.3f} +- {ens[\"fl_nme_sd\"]:.3f}')\n"
            "            print(f'  Top-10 SqErr Share: {ens[\"top10_sq_err_share\"]*100:.1f}%')\n"
            "else:\n"
            "    print('Results JSON not found at', json_path)"
        ),
        nbf.v4.new_code_cell(
            "csv_path = results_dir / 'femur_predictions.csv'\n"
            "if csv_path.is_file():\n"
            "    df = pd.read_csv(csv_path)\n"
            "    print(f'Total predictions: {len(df)} images across {df[\"patient\"].nunique()} patients')\n"
            "    print(f'Source counts: {dict(df[\"source\"].value_counts())}')\n"
            "    assert len(df) == 348, f'Expected 348 test images, got {len(df)}'\n"
            "    print('Verification check passed: 348 test predictions confirmed.')\n"
            "else:\n"
            "    print('Predictions CSV not found at', csv_path)"
        ),
    ]

    out_file = Path("notebooks/benchmarks/femur_ablation.ipynb")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created {out_file}")

if __name__ == "__main__":
    create_notebook()
