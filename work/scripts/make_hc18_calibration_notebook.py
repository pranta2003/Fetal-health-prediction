# FIELD: HC18
# MODEL: EfficientNet-B4 U-Net / Post-hoc Transform
# STEP: calibration
# USED BY: work/docs/CODE_MAP.md

"""make_hc18_calibration_notebook.py: Generate notebooks/benchmarks/hc18_oof_calibration.ipynb."""

from pathlib import Path
import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent


def create_notebook():
    nb = nbf.v4.new_notebook()

    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# HC18 Out-Of-Fold Post-Hoc Calibration Benchmark\n\n"
            "This benchmark evaluates whether post-hoc calibration functions improve Head Circumference (HC) MAE "
            "on existing single-fold out-of-fold predictions (`docs/results/hc18/oof_predictions.csv`, $N=999$ images / 806 patients).\n\n"
            "### Calibration Protocol\n"
            "- **Nested 5-Fold Cross-Validation**: Calibration models are fit strictly on 4 training folds and evaluated on held-out fold $k$.\n"
            "- **Candidate Methods**:\n"
            "  1. `raw`: Uncorrected model predictions ($y_{pred}$)\n"
            "  2. `median_add`: Additive shift $y_{pred} + \\text{median}(y_{tr} - \\hat{y}_{tr})$\n"
            "  3. `median_mult`: Multiplicative scaling $y_{pred} \\times \\text{median}(y_{tr} / \\hat{y}_{tr})$\n"
            "  4. `linear_reg`: Linear regression (OLS) $a \\cdot y_{pred} + b$\n"
            "  5. `isotonic`: Non-parametric monotonic regression (`IsotonicRegression(out_of_bounds='clip')`)\n"
            "- **Statistical Inference**: Patient-level paired bootstrap (5,000 resamples, seed 42) computing 95% CIs of $\\Delta \\text{MAE}$ vs raw baseline."
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
            "results_dir = work_dir / 'docs' / 'results' / 'hc18_calibration'\n"
            "md_path = results_dir / 'oof_calibration.md'\n"
            "json_path = results_dir / 'oof_calibration_results.json'\n"
            "csv_path = results_dir / 'calibrated_predictions.csv'\n\n"
            "assert md_path.is_file(), f'Missing {md_path}'\n"
            "assert json_path.is_file(), f'Missing {json_path}'\n"
            "assert csv_path.is_file(), f'Missing {csv_path}'\n\n"
            "with open(json_path, 'r', encoding='utf-8') as f:\n"
            "    results = json.load(f)\n\n"
            "print('Loaded HC18 OOF Calibration Results JSON.')\n"
            "print('Probability Map Audit:', results['local_probability_maps_audit']['statement'])\n"
            "print(f'Total Samples: {results[\"total_images\"]} across {results[\"total_patients\"]} patients')\n\n"
            "display(Markdown(md_path.read_text(encoding='utf-8')))"
        ),
        nbf.v4.new_code_cell(
            "# Summary Table of Statistical Differences vs Baseline\n"
            "rows = []\n"
            "for m, met in results['methods_metrics'].items():\n"
            "    b_res = results['bootstrap_diff_vs_raw'][m]\n"
            "    rows.append({\n"
            "        'Method': m,\n"
            "        'MAE (mm)': round(met['mae'], 4),\n"
            "        'RMSE (mm)': round(met['rmse'], 4),\n"
            "        'Median AE (mm)': round(met['median_ae'], 4),\n"
            "        'Signed Bias Mean (mm)': round(met['signed_bias_true_minus_pred_mean'], 4),\n"
            "        'Mean Diff vs Raw (mm)': round(b_res['mean_diff_vs_raw'], 4),\n"
            "        '95% CI Lower (mm)': round(b_res['ci_95'][0], 4),\n"
            "        '95% CI Upper (mm)': round(b_res['ci_95'][1], 4),\n"
            "        'Significant (CI < 0)': b_res['significant_improvement'],\n"
            "    })\n"
            "df_summary = pd.DataFrame(rows)\n"
            "display(df_summary)"
        ),
        nbf.v4.new_code_cell(
            "# Calibrated Predictions Sample\n"
            "df_cal = pd.read_csv(csv_path)\n"
            "print(f'Calibrated predictions table: {len(df_cal)} rows')\n"
            "display(df_cal[['image', 'fold', 'tercile', 'true_HC_mm', 'pred_HC_raw', 'pred_HC_linear_reg', 'err_HC_raw', 'err_HC_linear_reg']].head(10))"
        ),
        nbf.v4.new_code_cell(
            "# Verify Explicit Caveats\n"
            "print('=== EXPLICIT CAVEATS & INTEGRITY CHECKS ===')\n"
            "for k, v in results['caveats'].items():\n"
            "    print(f'\\n[{k.upper()}]\\n{v}')"
        ),
    ]

    out_path = REPO_ROOT / "notebooks" / "benchmarks" / "hc18_oof_calibration.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated {out_path}")


if __name__ == "__main__":
    create_notebook()
