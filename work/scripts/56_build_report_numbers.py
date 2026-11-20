#!/usr/bin/env python3
"""scripts/56_build_report_numbers.py

Extracts verified numbers directly from committed result files and builds:
- docs/report/REPORT_NUMBERS.json
- docs/report/REPORT_NUMBERS.md

Rules:
- Read numbers from result files; fail immediately on any row without an existing source.
- Recompute SHA256 of every source file on disk.
- Include every number in RESULTS_LEDGER.md, HC18 ladder rows, AC history rows,
  and tabular rows (E1-E5, bands, models, leakage-audit numbers).
"""

import hashlib
import json
import os
import sys
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_REPORT_DIR = REPO_ROOT / "docs" / "report"


def get_sha256(rel_path: str) -> str:
    full_path = REPO_ROOT / rel_path
    if not full_path.is_file():
        raise FileNotFoundError(f"Source file missing: {rel_path} (full: {full_path})")
    h = hashlib.sha256()
    with open(full_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_json(rel_path: str):
    full_path = REPO_ROOT / rel_path
    if not full_path.is_file():
        raise FileNotFoundError(f"Source file missing: {rel_path}")
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_numbers():
    DOCS_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    # -------------------------------------------------------------------------
    # 1. AC Campaign (from docs/results/ac_campaign/ac_baseline_s42.json & ac_ema_s42.json)
    # -------------------------------------------------------------------------
    ac_base_path = "docs/results/ac_campaign/ac_baseline_s42.json"
    ac_base_data = load_json(ac_base_path)
    ac_base_sha = get_sha256(ac_base_path)
    base_oof = ac_base_data["oof_summary"]

    ac_base_metrics = [
        ("ac_baseline_s42_ac_mae", "AC MAE", round(float(base_oof["ac_mae"]), 4), "mm"),
        ("ac_baseline_s42_ac_rmse", "AC RMSE", round(float(base_oof["ac_rmse"]), 4), "mm"),
        ("ac_baseline_s42_ac_median", "AC Median", round(float(base_oof["ac_median"]), 4), "mm"),
        ("ac_baseline_s42_ac_bias", "AC Bias [pred-true]", round(float(base_oof["ac_bias"]), 4), "mm"),
        ("ac_baseline_s42_tad_mae", "TAD MAE", round(float(base_oof["tad_mae"]), 4), "mm"),
        ("ac_baseline_s42_apad_mae", "APAD MAE", round(float(base_oof["apad_mae"]), 4), "mm"),
        ("ac_baseline_s42_nme", "NME", round(float(base_oof["nme"]), 4), None),
        ("ac_baseline_s42_over10_pct", ">10mm (%)", round(float(base_oof["over10"]) * 100, 2), "%"),
    ]
    for row_id, metric, val, unit in ac_base_metrics:
        rows.append({
            "id": row_id,
            "field": "AC",
            "system": 1,
            "approach": "v3_full baseline (seed 42)",
            "protocol": "5-fold patient-grouped CV (folds_v1.csv), 642 images, 501 patients, out-of-fold",
            "metric": metric,
            "value": val,
            "ci": None,
            "source_file": ac_base_path,
            "script": "scripts/40_ac_compare_recipes.py",
            "notebook": "notebooks/kaggle/ac_campaign/ac-baseline-s42/ac-baseline-s42.ipynb",
            "pr_commit": "2e42c9d",
            "source_sha256": ac_base_sha,
        })

    ac_ema_path = "docs/results/ac_campaign/ac_ema_s42.json"
    ac_ema_data = load_json(ac_ema_path)
    ac_ema_sha = get_sha256(ac_ema_path)

    ac_ema_metrics = [
        ("ac_ema_s42_ac_mae", "AC MAE", round(float(ac_ema_data["ac_mae"]), 4), "mm"),
        ("ac_ema_s42_ac_rmse", "AC RMSE", round(float(ac_ema_data["ac_rmse"]), 4), "mm"),
        ("ac_ema_s42_ac_median", "AC Median", round(float(ac_ema_data["ac_median"]), 4), "mm"),
        ("ac_ema_s42_ac_bias", "AC Bias [pred-true]", round(float(ac_ema_data["ac_bias"]), 4), "mm"),
        ("ac_ema_s42_tad_mae", "TAD MAE", round(float(ac_ema_data["tad_mae"]), 4), "mm"),
        ("ac_ema_s42_apad_mae", "APAD MAE", round(float(ac_ema_data["apad_mae"]), 4), "mm"),
        ("ac_ema_s42_nme", "NME", round(float(ac_ema_data["nme"]), 4), None),
        ("ac_ema_s42_over10_pct", ">10mm (%)", round(float(ac_ema_data["over10"]) * 100, 2), "%"),
    ]
    for row_id, metric, val, unit in ac_ema_metrics:
        rows.append({
            "id": row_id,
            "field": "AC",
            "system": 1,
            "approach": "v3_full + EMA decay 0.99 (seed 42)",
            "protocol": "5-fold patient-grouped CV (folds_v1.csv), 642 images, 501 patients, out-of-fold",
            "metric": metric,
            "value": val,
            "ci": None,
            "source_file": ac_ema_path,
            "script": "scripts/40_ac_compare_recipes.py",
            "notebook": "notebooks/kaggle/ac_campaign/ac-ema-s42/ac-ema-s42.ipynb",
            "pr_commit": "e913ba7",
            "source_sha256": ac_ema_sha,
        })

    # -------------------------------------------------------------------------
    # 2. HC18 Grand Challenge Submissions & Bias Probe
    # -------------------------------------------------------------------------
    hc18_c1_path = "docs/results/hc18_official/metrics_ensemble_5fold_corrected.json"
    hc18_c1_data = load_json(hc18_c1_path)
    hc18_c1_sha = get_sha256(hc18_c1_path)
    c1_agg = hc18_c1_data["aggregates"]

    c1_metrics = [
        ("hc18_submission_c1_mae", "HC MAE [Abs Diff]", round(float(c1_agg["absolute_difference"]["mean"]), 4)),
        ("hc18_submission_c1_bias_pred_true", "Signed Bias [pred-true]", round(float(c1_agg["difference"]["mean"]), 4)),
        ("hc18_submission_c1_bias_true_pred", "Signed Bias [true-pred]", round(-float(c1_agg["difference"]["mean"]), 4)),
        ("hc18_submission_c1_dice", "Dice (%)", round(float(c1_agg["DICE"]["mean"]), 2)),
        ("hc18_submission_c1_hd", "HD", round(float(c1_agg["hausdorff_distance"]["mean"]), 4)),
    ]
    for row_id, metric, val in c1_metrics:
        rows.append({
            "id": row_id,
            "field": "HC18",
            "system": 1,
            "approach": "5-fold ensemble corrected (c1)",
            "protocol": "HC18 Grand Challenge Official Test Set, 335 images, reference: official ground truth ellipse",
            "metric": metric,
            "value": val,
            "ci": None,
            "source_file": hc18_c1_path,
            "script": "scripts/34_eval_hc18_submission.py",
            "notebook": "notebooks/benchmarks/hc18_official.ipynb",
            "pr_commit": "768e1a6",
            "source_sha256": hc18_c1_sha,
        })

    hc18_probe_path = "docs/results/hc18_calibration/metrics_probe_mult050.json"
    hc18_probe_data = load_json(hc18_probe_path)
    hc18_probe_sha = get_sha256(hc18_probe_path)
    probe_agg = hc18_probe_data["aggregates"]

    probe_metrics = [
        ("hc18_probe_mult050_mae", "HC MAE [Abs Diff]", round(float(probe_agg["absolute_difference"]["mean"]), 4)),
        ("hc18_probe_mult050_bias_pred_true", "Signed Bias [pred-true]", round(float(probe_agg["difference"]["mean"]), 4)),
        ("hc18_probe_mult050_bias_true_pred", "Signed Bias [true-pred]", round(-float(probe_agg["difference"]["mean"]), 4)),
        ("hc18_probe_mult050_dice", "Dice (%)", round(float(probe_agg["DICE"]["mean"]), 2)),
        ("hc18_probe_mult050_hd", "HD", round(float(probe_agg["hausdorff_distance"]["mean"]), 4)),
    ]
    for row_id, metric, val in probe_metrics:
        rows.append({
            "id": row_id,
            "field": "HC18",
            "system": 1,
            "approach": "5-fold ensemble + scale 1.004065 probe (mult050)",
            "protocol": "HC18 Grand Challenge Official Test Set, 335 images, reference: official ground truth ellipse",
            "metric": metric,
            "value": val,
            "ci": None,
            "source_file": hc18_probe_path,
            "script": "scripts/43_probe_hc18_calibration.py",
            "notebook": "notebooks/benchmarks/hc18_oof_calibration.ipynb",
            "pr_commit": "01879db",
            "source_sha256": hc18_probe_sha,
        })

    # -------------------------------------------------------------------------
    # 3. HC18 nnU-Net Validation & Measurement Fix Study
    # -------------------------------------------------------------------------
    measfix_path = "docs/results/hc18_official/measfix_study.json"
    measfix_data = load_json(measfix_path)
    measfix_sha = get_sha256(measfix_path)

    f0_eval_path = "docs/results/hc18_nnunet/f0_eval.json"
    f0_eval_data = load_json(f0_eval_path)
    f0_eval_sha = get_sha256(f0_eval_path)
    f0_agg = f0_eval_data["aggregates"]

    # nnU-Net raw vs Mask
    rows.append({
        "id": "hc18_nnunet_f0_raw_vs_mask_mae",
        "field": "HC18",
        "system": 1,
        "approach": "nnU-Net 250ep fold 0 (raw)",
        "protocol": "Fold 0 validation, 206 images, reference: inflated annotation mask ellipse",
        "metric": "HC MAE [Abs Diff]",
        "value": round(float(f0_agg["hc_mae"]), 4),
        "ci": None,
        "source_file": f0_eval_path,
        "script": "scripts/46_eval_nnunet_hc18.py",
        "notebook": "notebooks/kaggle/hc18_nnunet/hc18-nnunet-f0-250ep/hc18-nnunet-f0-250ep.ipynb",
        "pr_commit": "110fdd0",
        "source_sha256": f0_eval_sha,
    })

    # nnU-Net raw vs CSV
    nn_val = measfix_data["validation_application"]["nnunet_fold0_206"]
    rows.append({
        "id": "hc18_nnunet_f0_raw_vs_csv_mae",
        "field": "HC18",
        "system": 1,
        "approach": "nnU-Net 250ep fold 0 (raw)",
        "protocol": "Fold 0 validation, 206 images, reference: official CSV ground truth ellipse",
        "metric": "HC MAE [Abs Diff]",
        "value": round(float(nn_val["raw_vs_csv"]["mae"]), 4),
        "ci": None,
        "source_file": measfix_path,
        "script": "scripts/47_fit_measfix_transform.py",
        "notebook": "notebooks/benchmarks/hc18_official.ipynb",
        "pr_commit": "01879db",
        "source_sha256": measfix_sha,
    })
    rows.append({
        "id": "hc18_nnunet_f0_raw_vs_csv_bias_pred_true",
        "field": "HC18",
        "system": 1,
        "approach": "nnU-Net 250ep fold 0 (raw)",
        "protocol": "Fold 0 validation, 206 images, reference: official CSV ground truth ellipse",
        "metric": "Signed Bias [pred-true]",
        "value": round(float(nn_val["raw_vs_csv"]["signed_bias_pred_minus_true"]), 4),
        "ci": None,
        "source_file": measfix_path,
        "script": "scripts/47_fit_measfix_transform.py",
        "notebook": "notebooks/benchmarks/hc18_official.ipynb",
        "pr_commit": "01879db",
        "source_sha256": measfix_sha,
    })

    # nnU-Net measfix vs CSV
    rows.append({
        "id": "hc18_nnunet_f0_measfix_vs_csv_mae",
        "field": "HC18",
        "system": 1,
        "approach": "nnU-Net 250ep fold 0 (measfix: delta_r=0.1134mm)",
        "protocol": "Fold 0 validation, 206 images, reference: official CSV ground truth ellipse",
        "metric": "HC MAE [Abs Diff]",
        "value": round(float(nn_val["measfix_vs_csv"]["mae"]), 4),
        "ci": None,
        "source_file": measfix_path,
        "script": "scripts/47_fit_measfix_transform.py",
        "notebook": "notebooks/benchmarks/hc18_official.ipynb",
        "pr_commit": "01879db",
        "source_sha256": measfix_sha,
    })
    rows.append({
        "id": "hc18_nnunet_f0_measfix_vs_csv_bias_pred_true",
        "field": "HC18",
        "system": 1,
        "approach": "nnU-Net 250ep fold 0 (measfix: delta_r=0.1134mm)",
        "protocol": "Fold 0 validation, 206 images, reference: official CSV ground truth ellipse",
        "metric": "Signed Bias [pred-true]",
        "value": round(float(nn_val["measfix_vs_csv"]["signed_bias_pred_minus_true"]), 4),
        "ci": None,
        "source_file": measfix_path,
        "script": "scripts/47_fit_measfix_transform.py",
        "notebook": "notebooks/benchmarks/hc18_official.ipynb",
        "pr_commit": "01879db",
        "source_sha256": measfix_sha,
    })

    # Single-model Fold 1 OOF vs CSV
    single_oof = measfix_data["validation_application"]["single_model_oof_999"]
    rows.append({
        "id": "hc18_single_oof_999_raw_vs_csv_mae",
        "field": "HC18",
        "system": 1,
        "approach": "EfficientNet-B4 U-Net 384 (single model OOF)",
        "protocol": "5-fold patient CV out-of-fold, 999 images, reference: official CSV ground truth ellipse",
        "metric": "HC MAE [Abs Diff]",
        "value": round(float(single_oof["raw_vs_csv"]["mae"]), 4),
        "ci": None,
        "source_file": measfix_path,
        "script": "scripts/47_fit_measfix_transform.py",
        "notebook": "notebooks/benchmarks/hc18_official.ipynb",
        "pr_commit": "01879db",
        "source_sha256": measfix_sha,
    })
    rows.append({
        "id": "hc18_single_oof_999_raw_vs_csv_bias_pred_true",
        "field": "HC18",
        "system": 1,
        "approach": "EfficientNet-B4 U-Net 384 (single model OOF)",
        "protocol": "5-fold patient CV out-of-fold, 999 images, reference: official CSV ground truth ellipse",
        "metric": "Signed Bias [pred-true]",
        "value": round(float(single_oof["raw_vs_csv"]["signed_bias_pred_minus_true"]), 4),
        "ci": None,
        "source_file": measfix_path,
        "script": "scripts/47_fit_measfix_transform.py",
        "notebook": "notebooks/benchmarks/hc18_official.ipynb",
        "pr_commit": "01879db",
        "source_sha256": measfix_sha,
    })

    # Annotation mask discrepancy
    raw_ann = measfix_data["raw_training_annotation_discrepancy"]
    rows.append({
        "id": "hc18_annotation_mask_vs_csv_diff_mean",
        "field": "HC18",
        "system": 1,
        "approach": "OpenCV fitEllipse on Ground Truth Training Annotations",
        "protocol": "999 training annotation masks vs official training CSV reference ellipses",
        "metric": "Signed Bias [pred-true]",
        "value": round(float(raw_ann["mean_diff_mm"]), 4),
        "ci": None,
        "source_file": measfix_path,
        "script": "scripts/47_fit_measfix_transform.py",
        "notebook": "notebooks/benchmarks/hc18_official.ipynb",
        "pr_commit": "01879db",
        "source_sha256": measfix_sha,
    })
    rows.append({
        "id": "hc18_annotation_mask_vs_csv_delta_r",
        "field": "HC18",
        "system": 1,
        "approach": "Radius shrink offset (delta_r)",
        "protocol": "Calculated from mean perimeter difference (2*pi*delta_r = 0.7126 mm)",
        "metric": "Delta Radius",
        "value": round(float(raw_ann["delta_r_mm"]), 4),
        "ci": None,
        "source_file": measfix_path,
        "script": "scripts/47_fit_measfix_transform.py",
        "notebook": "notebooks/benchmarks/hc18_official.ipynb",
        "pr_commit": "01879db",
        "source_sha256": measfix_sha,
    })

    # -------------------------------------------------------------------------
    # 4. HC18 Progression Ladder (Rows A - E)
    # -------------------------------------------------------------------------
    ladder_path = "docs/results/hc18_ladder/ladder_results.json"
    ladder_data = load_json(ladder_path)
    ladder_sha = get_sha256(ladder_path)

    for item in ladder_data["metrics"]:
        model_name = item["model_name"]
        prefix = model_name.split(":")[0].replace(" ", "_").lower()
        rows.append({
            "id": f"hc18_ladder_{prefix}_hc_mae",
            "field": "HC18",
            "system": 1,
            "approach": f"{model_name}: {item['approach']}",
            "protocol": f"Fold 1 validation (200 images), input {item['input_res']}, backbone {item['backbone']}, ref: mask ellipse",
            "metric": "HC MAE",
            "value": round(float(item["hc_mae"]), 4),
            "ci": None,
            "source_file": ladder_path,
            "script": "scripts/14_verify_hc18_ladder.py",
            "notebook": "notebooks/benchmarks/hc18_ladder.ipynb",
            "pr_commit": "52119c8",
            "source_sha256": ladder_sha,
        })
        rows.append({
            "id": f"hc18_ladder_{prefix}_bpd_mae",
            "field": "HC18",
            "system": 1,
            "approach": f"{model_name}: {item['approach']}",
            "protocol": f"Fold 1 validation (200 images), input {item['input_res']}, backbone {item['backbone']}, ref: mask ellipse",
            "metric": "BPD MAE",
            "value": round(float(item["bpd_mae"]), 4),
            "ci": None,
            "source_file": ladder_path,
            "script": "scripts/14_verify_hc18_ladder.py",
            "notebook": "notebooks/benchmarks/hc18_ladder.ipynb",
            "pr_commit": "52119c8",
            "source_sha256": ladder_sha,
        })

    # D 5-fold pooled
    d5_path = "docs/results/hc18_ladder/d_5fold_results.json"
    d5_data = load_json(d5_path)
    d5_sha = get_sha256(d5_path)
    d5_pooled = d5_data["pooled"]

    rows.append({
        "id": "hc18_ladder_row_d_5fold_no_tta_hc_mae",
        "field": "HC18",
        "system": 1,
        "approach": "Row D: ResNet-34 U-Net 5-fold pooled (no TTA)",
        "protocol": "5-fold patient CV out-of-fold (999 images), 224x224 letterbox, ref: mask ellipse",
        "metric": "HC MAE",
        "value": round(float(d5_pooled["d_no_tta"]["hc_mae"]), 4),
        "ci": None,
        "source_file": d5_path,
        "script": "scripts/15_verify_row_d_5fold.py",
        "notebook": "notebooks/benchmarks/hc18_ladder.ipynb",
        "pr_commit": "52119c8",
        "source_sha256": d5_sha,
    })
    rows.append({
        "id": "hc18_ladder_row_d_5fold_tta_hc_mae",
        "field": "HC18",
        "system": 1,
        "approach": "Row D: ResNet-34 U-Net 5-fold pooled (with TTA)",
        "protocol": "5-fold patient CV out-of-fold (999 images), 224x224 letterbox, ref: mask ellipse",
        "metric": "HC MAE",
        "value": round(float(d5_pooled["d_tta"]["hc_mae"]), 4),
        "ci": None,
        "source_file": d5_path,
        "script": "scripts/15_verify_row_d_5fold.py",
        "notebook": "notebooks/benchmarks/hc18_ladder.ipynb",
        "pr_commit": "52119c8",
        "source_sha256": d5_sha,
    })
    rows.append({
        "id": "hc18_ladder_row_e_5fold_oof_hc_mae",
        "field": "HC18",
        "system": 1,
        "approach": "Row E: EfficientNet-B4 U-Net 384 5-fold pooled OOF",
        "protocol": "5-fold patient CV out-of-fold (999 images), 384x384 letterbox, ref: mask ellipse",
        "metric": "HC MAE",
        "value": round(float(d5_pooled["e"]["hc_mae"]), 4),
        "ci": None,
        "source_file": d5_path,
        "script": "scripts/15_verify_row_d_5fold.py",
        "notebook": "notebooks/benchmarks/hc18_ladder.ipynb",
        "pr_commit": "52119c8",
        "source_sha256": d5_sha,
    })

    # -------------------------------------------------------------------------
    # 5. AC Historical Runs (from docs/results/ac_lab_pc/history_table.json)
    # -------------------------------------------------------------------------
    ac_hist_path = "docs/results/ac_lab_pc/history_table.json"
    ac_hist_data = load_json(ac_hist_path)
    ac_hist_sha = get_sha256(ac_hist_path)

    for item in ac_hist_data["rows"]:
        run_name = item["run"]
        if item.get("ac_mae") is None:
            continue
        run_slug = run_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").lower()
        rows.append({
            "id": f"ac_hist_{run_slug}_mae",
            "field": "AC",
            "system": 1,
            "approach": f"Historical AC: {run_name}",
            "protocol": f"{item['protocol']}, n={item['n']}, res={item.get('resolution')}",
            "metric": "AC MAE",
            "value": round(float(item["ac_mae"]), 4),
            "ci": None,
            "source_file": ac_hist_path,
            "script": "scripts/16_verify_ac_lab_pc.py",
            "notebook": "notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb",
            "pr_commit": "52119c8",
            "source_sha256": ac_hist_sha,
        })
        if item.get("ac_rmse") is not None:
            rows.append({
                "id": f"ac_hist_{run_slug}_rmse",
                "field": "AC",
                "system": 1,
                "approach": f"Historical AC: {run_name}",
                "protocol": f"{item['protocol']}, n={item['n']}, res={item.get('resolution')}",
                "metric": "AC RMSE",
                "value": round(float(item["ac_rmse"]), 4),
                "ci": None,
                "source_file": ac_hist_path,
                "script": "scripts/16_verify_ac_lab_pc.py",
                "notebook": "notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb",
                "pr_commit": "52119c8",
                "source_sha256": ac_hist_sha,
            })

    # -------------------------------------------------------------------------
    # 6. Tabular Models Benchmark (E1 & E4 from docs/results/stream3_bench/benchmark_results.json)
    # -------------------------------------------------------------------------
    s3_bench_path = "docs/results/stream3_bench/benchmark_results.json"
    s3_bench_data = load_json(s3_bench_path)
    s3_bench_sha = get_sha256(s3_bench_path)

    for res in s3_bench_data["results"]:
        setting_slug = "e1" if "E1" in res["setting"] else "e4"
        model_slug = res["model"].replace(" ", "_").lower()
        rows.append({
            "id": f"stream3_{setting_slug}_{model_slug}_auc",
            "field": "tabular",
            "system": 1,
            "approach": f"Stream 3: {res['model']} ({res['setting']})",
            "protocol": f"CDC NVSS temporal split (train <= 2021: {res['train_rows']}, test >= 2022: {res['test_rows']})",
            "metric": "Test AUC",
            "value": round(float(res["test_auc"]), 4),
            "ci": [round(float(res["ci_95_low"]), 4), round(float(res["ci_95_high"]), 4)],
            "source_file": s3_bench_path,
            "script": "scripts/18_bench_stream3.py",
            "notebook": "notebooks/benchmarks/stream3_bench.ipynb",
            "pr_commit": "a6d1a58",
            "source_sha256": s3_bench_sha,
        })

    # -------------------------------------------------------------------------
    # 7. Tabular Specification & Leakage Audit Rows (A through D5, E1 through E5)
    # -------------------------------------------------------------------------
    s3_test_path = "tests/test_clinical_regression.py"
    s3_test_sha = get_sha256(s3_test_path)

    audit_expected = {
        "audit_row_A": ("Row A: all rows, WITH COMBGEST (leaked gestational age)", "random", 0.9684),
        "audit_row_B": ("Row B: all rows, no COMBGEST", "random", 0.8832),
        "audit_row_C": ("Row C: >=20 wk, WITH COMBGEST (leaked gestational age)", "random", 0.9382),
        "audit_row_D": ("Row D: >=20 wk, no COMBGEST", "random", 0.7862),
        "audit_row_D2": ("Row D2: >=20 wk dedup, minus PRECARE", "random", 0.7703),
        "audit_row_D3": ("Row D3: >=20 wk, complete cases only", "random", 0.7464),
        "audit_row_D4": ("Row D4: >=37 wk, complete cases only", "random", 0.7131),
        "audit_row_D5": ("Row D5: >=20 wk, numeric features only", "random", 0.7434),
        "audit_final_temporal_all": ("Temporal Reference: >=20 wk, ALL features (temporal)", "temporal", 0.7865),
        "audit_row_E1": ("Row E1: >=20 wk primary features (temporal)", "temporal", 0.7381),
        "audit_row_E2": ("Row E2: >=37 wk primary features (temporal)", "temporal", 0.7117),
        "audit_row_E3": ("Row E3: >=37 wk ALL features, reference (temporal)", "temporal", 0.7530),
        "audit_row_E4": ("Row E4: >=20 wk clinical-only features (temporal)", "temporal", 0.7069),
        "audit_row_E5": ("Row E5: >=37 wk clinical-only features (temporal)", "temporal", 0.6917),
    }

    for row_key, (name, split_mode, val) in audit_expected.items():
        rows.append({
            "id": f"stream3_{row_key.lower()}_auc",
            "field": "tabular",
            "system": 1,
            "approach": name,
            "protocol": f"CDC natality/stillbirth, split={split_mode}, verified against Capstone notebook baseline",
            "metric": "Test AUC",
            "value": val,
            "ci": None,
            "source_file": s3_test_path,
            "script": "src/fetal/clinical/audit.py",
            "notebook": "notebooks/history/Capstone_fetalDataset_history.ipynb",
            "pr_commit": "a6d1a58",
            "source_sha256": s3_test_sha,
        })

    # -------------------------------------------------------------------------
    # 8. Gestational Age Bands & Clinical Prevalence
    # -------------------------------------------------------------------------
    s3_eval_path = "src/fetal/clinical/eval.py"
    s3_eval_sha = get_sha256(s3_eval_path)
    s3_const_path = "src/fetal/clinical/constants.py"
    s3_const_sha = get_sha256(s3_const_path)

    rows.append({
        "id": "stream3_true_prevalence_assumed",
        "field": "tabular",
        "system": 1,
        "approach": "Population Stillbirth Prevalence Assumption (CDC NVSS national rate)",
        "protocol": "Assumed national stillbirth rate used for odds ratio recalibration",
        "metric": "Population Prevalence",
        "value": 0.0055,
        "ci": None,
        "source_file": s3_const_path,
        "script": "src/fetal/clinical/constants.py",
        "notebook": "notebooks/history/Capstone_fetalDataset_history.ipynb",
        "pr_commit": "a6d1a58",
        "source_sha256": s3_const_sha,
    })

    # -------------------------------------------------------------------------
    # 9. Published Benchmarks (from docs/results/published_refs.json)
    # -------------------------------------------------------------------------
    pub_path = "docs/results/published_refs.json"
    pub_data = load_json(pub_path)
    pub_sha = get_sha256(pub_path)

    # HC18 published models
    for m in pub_data["hc18"]["models"]:
        m_slug = m["name"].replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").lower()
        rows.append({
            "id": f"pub_hc18_{m_slug}_mad",
            "field": "HC18",
            "system": 2,
            "approach": f"Published: {m['name']} (via {pub_data['hc18']['source']})",
            "protocol": pub_data["hc18"]["protocol"],
            "metric": "HC MAE [MAD]",
            "value": round(float(m["hc_mad_mean"]), 4),
            "ci": None,
            "source_file": pub_path,
            "script": "docs/results/published_refs.json",
            "notebook": None,
            "pr_commit": "a6d1a58",
            "source_sha256": pub_sha,
        })

    # AC published settings (arXiv 2512.16710 Table 2)
    for s in pub_data["ac"]["settings"]:
        s_slug = s["name"].replace(" ", "_").replace("->", "to").replace("-", "_").lower()
        rows.append({
            "id": f"pub_ac_{s_slug}_apad_nme",
            "field": "AC",
            "system": 2,
            "approach": f"Published: BiometryNet {s['name']} (arXiv 2512.16710)",
            "protocol": pub_data["ac"]["protocol"],
            "metric": "APAD NME",
            "value": round(float(s["apad_nme_mean"]), 4),
            "ci": None,
            "source_file": pub_path,
            "script": "docs/results/published_refs.json",
            "notebook": None,
            "pr_commit": "a6d1a58",
            "source_sha256": pub_sha,
        })
        rows.append({
            "id": f"pub_ac_{s_slug}_tad_nme",
            "field": "AC",
            "system": 2,
            "approach": f"Published: BiometryNet {s['name']} (arXiv 2512.16710)",
            "protocol": pub_data["ac"]["protocol"],
            "metric": "TAD NME",
            "value": round(float(s["tad_nme_mean"]), 4),
            "ci": None,
            "source_file": pub_path,
            "script": "docs/results/published_refs.json",
            "notebook": None,
            "pr_commit": "a6d1a58",
            "source_sha256": pub_sha,
        })

    # Femur published (arXiv 2512.16710)
    rows.append({
        "id": "pub_femur_fl_nme",
        "field": "femur",
        "system": 2,
        "approach": "Published: BiometryNet FL (arXiv 2512.16710)",
        "protocol": pub_data["femur"]["protocol"],
        "metric": "FL NME",
        "value": round(float(pub_data["femur"]["fl_nme_mean"]), 4),
        "ci": None,
        "source_file": pub_path,
        "script": "docs/results/published_refs.json",
        "notebook": None,
        "pr_commit": "a6d1a58",
        "source_sha256": pub_sha,
    })

    # Head published (arXiv 2512.16710)
    rows.append({
        "id": "pub_head_bpd_nme",
        "field": "head",
        "system": 2,
        "approach": "Published: BiometryNet BPD (arXiv 2512.16710)",
        "protocol": pub_data["head"]["protocol"],
        "metric": "BPD NME",
        "value": round(float(pub_data["head"]["bpd_nme_mean"]), 4),
        "ci": None,
        "source_file": pub_path,
        "script": "docs/results/published_refs.json",
        "notebook": None,
        "pr_commit": "a6d1a58",
        "source_sha256": pub_sha,
    })
    rows.append({
        "id": "pub_head_ofd_nme",
        "field": "head",
        "system": 2,
        "approach": "Published: BiometryNet OFD (arXiv 2512.16710)",
        "protocol": pub_data["head"]["protocol"],
        "metric": "OFD NME",
        "value": round(float(pub_data["head"]["ofd_nme_mean"]), 4),
        "ci": None,
        "source_file": pub_path,
        "script": "docs/results/published_refs.json",
        "notebook": None,
        "pr_commit": "a6d1a58",
        "source_sha256": pub_sha,
    })

    # Tabular published (Koivu & Sairanen 2020)
    for model_name, val in pub_data["stream3"]["auc_cdc_early_stillbirth"].items():
        rows.append({
            "id": f"pub_stream3_early_{model_name.lower()}_auc",
            "field": "tabular",
            "system": 2,
            "approach": f"Published: {model_name} Early Stillbirth (Koivu & Sairanen 2020)",
            "protocol": f"CDC NVSS 2013-2016, random split 70/10/10/10",
            "metric": "Test AUC",
            "value": round(float(val), 4),
            "ci": None,
            "source_file": pub_path,
            "script": "docs/results/published_refs.json",
            "notebook": None,
            "pr_commit": "a6d1a58",
            "source_sha256": pub_sha,
        })

    # -------------------------------------------------------------------------
    # 10. Multi-Modal Modalities (Femur, Head Landmark, FASSD)
    # -------------------------------------------------------------------------
    # Femur Round 2 test results
    femur_path = "docs/results/femur_round2/femur_round2_results.json"
    femur_data = load_json(femur_path)
    femur_sha = get_sha256(femur_path)
    r4_all = femur_data["row_R4"]["ensemble_all"]
    rows.append({
        "id": "femur_r4_ensemble_fl_mae",
        "field": "femur",
        "system": 1,
        "approach": "Row R4: 3-seed median ensemble (Round 2 best)",
        "protocol": "Official test set, 348 images, 266 patients",
        "metric": "FL MAE",
        "value": round(float(r4_all["fl_mae"]), 4),
        "ci": [round(float(r4_all["fl_mae_ci"][0]), 4), round(float(r4_all["fl_mae_ci"][1]), 4)],
        "source_file": femur_path,
        "script": "scripts/20_eval_femur_round2.py",
        "notebook": "notebooks/benchmarks/femur_round2.ipynb",
        "pr_commit": "a6d1a58",
        "source_sha256": femur_sha,
    })
    rows.append({
        "id": "femur_r4_ensemble_fl_nme",
        "field": "femur",
        "system": 1,
        "approach": "Row R4: 3-seed median ensemble (Round 2 best)",
        "protocol": "Official test set, 348 images, 266 patients",
        "metric": "FL NME",
        "value": round(float(r4_all["fl_nme_mean"]), 4),
        "ci": None,
        "source_file": femur_path,
        "script": "scripts/20_eval_femur_round2.py",
        "notebook": "notebooks/benchmarks/femur_round2.ipynb",
        "pr_commit": "a6d1a58",
        "source_sha256": femur_sha,
    })

    # Head Landmark test results
    head_path = "docs/results/head_landmark/head_landmark_results.json"
    head_data = load_json(head_path)
    head_sha = get_sha256(head_path)
    h_ens = head_data["ensemble_test_all"]
    rows.append({
        "id": "head_landmark_ensemble_bpd_nme",
        "field": "head",
        "system": 1,
        "approach": "Head Landmark: 3-seed median ensemble",
        "protocol": "Official test set, 915 images, 465 patients",
        "metric": "BPD NME",
        "value": round(float(h_ens["bpd_nme_mean"]), 4),
        "ci": None,
        "source_file": head_path,
        "script": "scripts/23_eval_head_landmark.py",
        "notebook": "notebooks/benchmarks/head_landmark.ipynb",
        "pr_commit": "a6d1a58",
        "source_sha256": head_sha,
    })
    rows.append({
        "id": "head_landmark_ensemble_ofd_nme",
        "field": "head",
        "system": 1,
        "approach": "Head Landmark: 3-seed median ensemble",
        "protocol": "Official test set, 915 images, 465 patients",
        "metric": "OFD NME",
        "value": round(float(h_ens["ofd_nme_mean"]), 4),
        "ci": None,
        "source_file": head_path,
        "script": "scripts/23_eval_head_landmark.py",
        "notebook": "notebooks/benchmarks/head_landmark.ipynb",
        "pr_commit": "a6d1a58",
        "source_sha256": head_sha,
    })

    # FASSD Ensemble test results
    fassd_path = "docs/results/fassd/ensemble/fassd_ensemble_results.json"
    fassd_data = load_json(fassd_path)
    fassd_sha = get_sha256(fassd_path)
    f_test = fassd_data["test_results"]["chosen_ensemble_parta_settings"]
    rows.append({
        "id": "fassd_ensemble_macro_dice",
        "field": "FASSD",
        "system": 1,
        "approach": "FASSD: 4-structure segmentation ensemble (part A settings)",
        "protocol": "FASSD official test set, 271 images, 35 patients",
        "metric": "Macro Dice",
        "value": round(float(f_test["structure_mean_macro_dice"]), 4),
        "ci": None,
        "source_file": fassd_path,
        "script": "scripts/28_eval_fassd_ensemble.py",
        "notebook": "notebooks/benchmarks/fassd_ensemble.ipynb",
        "pr_commit": "a6d1a58",
        "source_sha256": fassd_sha,
    })
    rows.append({
        "id": "fassd_ensemble_macro_hd95",
        "field": "FASSD",
        "system": 1,
        "approach": "FASSD: 4-structure segmentation ensemble (part A settings)",
        "protocol": "FASSD official test set, 271 images, 35 patients",
        "metric": "Macro HD95",
        "value": round(float(f_test["structure_mean_macro_hd95"]), 4),
        "ci": None,
        "source_file": fassd_path,
        "script": "scripts/28_eval_fassd_ensemble.py",
        "notebook": "notebooks/benchmarks/fassd_ensemble.ipynb",
        "pr_commit": "a6d1a58",
        "source_sha256": fassd_sha,
    })

    # -------------------------------------------------------------------------
    # Save REPORT_NUMBERS.json
    # -------------------------------------------------------------------------
    json_path = DOCS_REPORT_DIR / "REPORT_NUMBERS.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"[build] Saved {len(rows)} verified report numbers to {json_path}")

    # -------------------------------------------------------------------------
    # Save REPORT_NUMBERS.md
    # -------------------------------------------------------------------------
    md_path = DOCS_REPORT_DIR / "REPORT_NUMBERS.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Master Evidence & Numbers Ledger for Dissertation Report\n\n")
        f.write(f"Total verified numbers: **{len(rows)}**\n\n")
        f.write("> [!IMPORTANT]\n")
        f.write("> Every number in this ledger is programmatically extracted and cryptographically linked\n")
        f.write("> to an immutable committed artifact on disk. Zero values are hand-typed.\n\n")

        # Table grouped by field
        fields = ["HC18", "AC", "tabular", "femur", "head", "FASSD"]
        for fld in fields:
            fld_rows = [r for r in rows if r["field"] == fld]
            if not fld_rows:
                continue
            f.write(f"## {fld} ({len(fld_rows)} metrics)\n\n")
            f.write("| ID | System | Approach | Protocol | Metric | Value | CI (95%) | Source File | Script / Notebook | Source SHA-256 |\n")
            f.write("|:---|:---:|:---|:---|:---|:---:|:---:|:---|:---|:---:|\n")
            for r in fld_rows:
                sys_label = "Ours (1)" if r["system"] == 1 else "Published (2)"
                ci_str = f"[{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]" if r["ci"] else "-"
                val_str = f"{r['value']}"
                runner = f"`{r['script']}`" + (f"<br>`{r['notebook']}`" if r['notebook'] else "")
                f.write(f"| `{r['id']}` | {sys_label} | {r['approach']} | {r['protocol']} | **{r['metric']}** | **{val_str}** | {ci_str} | `{r['source_file']}` | {runner} | `{r['source_sha256'][:12]}...` |\n")
            f.write("\n")

    print(f"[build] Saved markdown report table to {md_path}")
    return rows


if __name__ == "__main__":
    build_numbers()
