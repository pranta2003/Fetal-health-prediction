"""22_femur_ablation.py: Femur landmark ablation study runner and report generator.

Evaluates 3 architectural approaches on FP+UCL Femur dataset (348 test images):
  Row A: Heatmap U-Net EfficientNet-B4 (384px, 1 channel / 2 peaks, flip TTA)
  Row B: Direct Coordinate Regression ConvNeXt-Tiny (384px, order-invariant loss)
  Row C: Heatmap U-Net EfficientNet-B4 (512px, 1 channel / 2 peaks, flip TTA)

Metrics reported:
  - FL MAE mm [95% CI]
  - FL RMSE mm [95% CI]
  - FL Median mm [95% CI]
  - % Error [95% CI]
  - FL NME (mean +- SD)
  - Per-source breakdown (FP vs UCL tiny: 25 images / 5 patients)
  - Top-10 share of squared error (%)
  - Training time (s) & Epochs trained
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

# Ensure work/src is in sys.path
work_root = Path(__file__).resolve().parent.parent
if str(work_root / "src") not in sys.path:
    sys.path.insert(0, str(work_root / "src"))

from fetal import config
from fetal.data.femur_dataset import build_femur_data, split_femur_data
from fetal.models.femur_landmark import (
    build_femur_heatmap_model,
    build_femur_regression_model,
)
from fetal.eval.femur_evaluator import (
    evaluate_femur_dataframe,
    ensemble_3seed_predictions,
    summarize_femur_metrics,
)


def format_ablation_markdown_table(results: dict) -> str:
    """Format comprehensive markdown table comparing rows A, B, and C."""
    lines = [
        "# Femur Model Ablation Study (FP + UCL Femur Test Set, N=348 images / 266 patients)",
        "",
        "> Reference FL: Euclidean distance between annotated endpoints * px_to_mm_rate.",
        "> Protocol: Train on 451 images, early stop on 65 validation images (0 patient overlap).",
        "> Evaluation: 348 official test images (323 FP / 261 patients, 25 UCL / 5 patients [tiny]).",
        "> 95% CIs: Patient-level bootstrap with 1,000 resamples.",
        "> NME: Mean Euclidean landmark error normalized by ground-truth femur length.",
        "",
        "| Approach | Row / Model | Resolution | Epochs | Train Time (s) | FL MAE [95% CI] (mm) | FL RMSE [95% CI] (mm) | FL Median [95% CI] (mm) | % Error [95% CI] | FL NME (mean+-SD) | Top-10 SqErr Share (%) | FP MAE (mm) | UCL MAE (mm, tiny) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    row_names = [
        ("Row A (U-Net EffNet-B4 1-ch Heatmap, flip TTA)", "row_A"),
        ("Row B (ConvNeXt-Tiny Direct Reg, Order-Inv)", "row_B"),
        ("Row C (U-Net EffNet-B4 512px Heatmap, flip TTA)", "row_C"),
    ]

    for display_name, row_key in row_names:
        if row_key not in results:
            continue
        row_data = results[row_key]
        res_str = f"{row_data['resolution']} px"
        m_sd = row_data["seeds_mean_sd"]
        ens = row_data["ensemble_all"]
        ens_fp = row_data["ensemble_fp"]
        ens_ucl = row_data["ensemble_ucl"]

        # Seeds Mean +- SD row
        ep_str = f"{m_sd['epochs_mean']:.1f} +- {m_sd['epochs_sd']:.1f}"
        time_str = f"{m_sd['time_mean']:.1f} +- {m_sd['time_sd']:.1f}"
        mae_ms = f"{m_sd['fl_mae_mean']:.3f} +- {m_sd['fl_mae_sd']:.3f}"
        rmse_ms = f"{m_sd['fl_rmse_mean']:.3f} +- {m_sd['fl_rmse_sd']:.3f}"
        med_ms = f"{m_sd['fl_med_mean']:.3f} +- {m_sd['fl_med_sd']:.3f}"
        pct_ms = f"{m_sd['fl_pct_mean']:.2f}% +- {m_sd['fl_pct_sd']:.2f}%"
        nme_ms = f"{m_sd['fl_nme_mean']:.3f} +- {m_sd['fl_nme_sd']:.3f}"
        top10_ms = f"{m_sd['top10_mean']*100:.1f}% +- {m_sd['top10_sd']*100:.1f}%"
        fp_ms = f"{m_sd['fp_mae_mean']:.3f}"
        ucl_ms = f"{m_sd['ucl_mae_mean']:.3f}"

        lines.append(
            f"| {display_name} | *Seeds Mean +- SD* | {res_str} | {ep_str} | {time_str} | "
            f"{mae_ms} | {rmse_ms} | {med_ms} | {pct_ms} | {nme_ms} | {top10_ms} | {fp_ms} | {ucl_ms} |"
        )

        # 3-Seed Median Ensemble row
        e_mae = f"**{ens['fl_mae']:.3f}** [{ens['fl_mae_ci'][0]:.3f}, {ens['fl_mae_ci'][1]:.3f}]"
        e_rmse = f"**{ens['fl_rmse']:.3f}** [{ens['fl_rmse_ci'][0]:.3f}, {ens['fl_rmse_ci'][1]:.3f}]"
        e_med = f"**{ens['fl_median']:.3f}** [{ens['fl_median_ci'][0]:.3f}, {ens['fl_median_ci'][1]:.3f}]"
        e_pct = f"**{ens['fl_pct_error']:.2f}%** [{ens['fl_pct_error_ci'][0]:.2f}%, {ens['fl_pct_error_ci'][1]:.2f}%]"
        e_nme = f"**{ens['fl_nme_mean']:.3f} +- {ens['fl_nme_sd']:.3f}**"
        e_top10 = f"**{ens['top10_sq_err_share']*100:.1f}%**"
        e_fp = f"**{ens_fp['fl_mae']:.3f}**"
        e_ucl = f"**{ens_ucl['fl_mae']:.3f}**"

        lines.append(
            f"| {display_name} | **3-Seed Median Ens** | {res_str} | - | - | "
            f"{e_mae} | {e_rmse} | {e_med} | {e_pct} | {e_nme} | {e_top10} | {e_fp} | {e_ucl} |"
        )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Femur ablation study evaluator")
    parser.add_argument("--data-dir", type=Path, default=Path("C:/fetal-data"), help="Root data directory")
    parser.add_argument("--results-dir", type=Path, default=None, help="Directory containing precomputed results from Kaggle")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/results/femur_ablation"), help="Output directory")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.results_dir is not None and args.results_dir.is_dir():
        print(f"Loading precomputed results from {args.results_dir}...")
        res_json = args.results_dir / "femur_ablation_results.json"
        if res_json.is_file():
            shutil.copy2(res_json, args.out_dir / "femur_ablation_results.json")
        pred_csv = args.results_dir / "femur_predictions.csv"
        if pred_csv.is_file():
            shutil.copy2(pred_csv, args.out_dir / "femur_predictions.csv")
        table_md = args.results_dir / "femur_ablation_table.md"
        if table_md.is_file():
            shutil.copy2(table_md, args.out_dir / "femur_ablation_table.md")
        log_file = args.results_dir / "femur_ablation.log"
        if log_file.is_file():
            shutil.copy2(log_file, args.out_dir / "femur_ablation.log")
        print(f"Copied results to {args.out_dir}")
        return

    print("Running local evaluation / smoke test...")


if __name__ == "__main__":
    main()
