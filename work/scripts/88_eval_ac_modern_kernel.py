#!/usr/bin/env python3

# FIELD: AC
# MODEL: Modern Encoder Landmark U-Net
# STEP: evaluation
# USED BY: work/docs/AC_MODERN_QUEUE.md, work/docs/MODEL_SELECTION_TABLES.md

"""scripts/88_eval_ac_modern_kernel.py

Evaluates a modern-encoder AC Kaggle kernel output on folds_v1.csv (642 images,
501 patients) with paired patient-level cluster bootstrap (5,000 resamples, seed 42)
vs ac-baseline-s42 (4.7744 mm) and vs the 4-seed EfficientNet-B4 average (4.4028 mm),
and parses per-fold best epochs and dev AC MAE > 50 mm spikes after epoch 5.
"""

import argparse
import json
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fetal.eval.ac_evaluator import patient_level_paired_bootstrap

OUT_DIR = REPO_ROOT / "docs" / "results" / "ac_campaign"
FOUR_SEED_CSV = OUT_DIR / "oof_predictions_4seed_avg.csv"


def run_boot(patient: pd.Series, true: pd.Series, pred_ref: pd.Series, pred_cand: pd.Series) -> dict:
    df = pd.DataFrame({
        "patient": patient,
        "ac_true": true,
        "ac_pred_a": pred_ref,
        "ac_pred_b": pred_cand,
    })
    res = patient_level_paired_bootstrap(df, n_bootstraps=5000, seed=42)
    ci_lo, ci_hi = res["ci95_delta_mae"]
    excludes_zero = bool((ci_hi < 0.0) or (ci_lo > 0.0))
    return {
        "delta_mae": float(res["delta_mae"]),
        "ci95_delta_mae": [float(ci_lo), float(ci_hi)],
        "p_val_mae": float(res["p_val_mae"]),
        "delta_rmse": float(res["delta_rmse"]),
        "ci95_delta_rmse": [float(res["ci95_delta_rmse"][0]), float(res["ci95_delta_rmse"][1])],
        "p_val_rmse": float(res["p_val_rmse"]),
        "ci_excludes_zero": excludes_zero,
    }


def parse_log(log_path: Path) -> dict:
    if not log_path.is_file():
        return {"per_fold_best_epochs": {}, "jumps_gt50_after_ep5": []}
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    pat = re.compile(r"\[fold_(\d+)\]\s+ep\s+(\d+)/(\d+).*?dev AC MAE\s+([0-9.]+)mm.*?score\s+[0-9.]+\s*(\*?)\s*\|")
    folds: dict[int, list[tuple[int, float, bool]]] = {}
    jumps = []
    for m in pat.finditer(txt):
        fk, ep, mae, star = int(m.group(1)), int(m.group(2)), float(m.group(4)), m.group(5) == "*"
        folds.setdefault(fk, []).append((ep, mae, star))
        if ep > 5 and mae > 50.0:
            jumps.append({"fold": fk, "epoch": ep, "dev_ac_mae": mae})
    best_epochs = {}
    for fk in sorted(folds):
        eps = folds[fk]
        best_ep = [e for e, _, s in eps if s][-1] if any(s for _, _, s in eps) else None
        best_mae = [m for _, m, s in eps if s][-1] if any(s for _, _, s in eps) else None
        best_epochs[f"fold_{fk}"] = {
            "total_epochs": eps[-1][0],
            "best_epoch": best_ep,
            "best_dev_ac_mae": best_mae,
        }
    return {"per_fold_best_epochs": best_epochs, "jumps_gt50_after_ep5": jumps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True, help="Kernel slug, e.g. ac-convnext-lr1e4-s42")
    parser.add_argument("--download-dir", required=True, help="Path to downloaded Kaggle output folder")
    parser.add_argument("--out-json", required=True, help="Path to save evaluation JSON")
    args = parser.parse_args()

    dl_dir = Path(args.download_dir)
    sub_dir = dl_dir / args.slug
    df_4seed = pd.read_csv(FOUR_SEED_CSV)
    df_cand = pd.read_csv(sub_dir / "oof_predictions_pooled.csv")
    with open(sub_dir / "oof_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    df_cand = df_4seed[["image_name"]].merge(df_cand, on="image_name", how="left")
    assert len(df_4seed) == 642
    assert not df_cand["ac_pred"].isna().any()
    assert np.allclose(df_cand["ac_true"], df_4seed["ac_true"])

    patient = df_4seed["patient"]
    ac_true = df_4seed["ac_true"]
    boot_vs_s42 = run_boot(patient, ac_true, df_4seed["ac_pred_s42"], df_cand["ac_pred"])
    boot_vs_4seed = run_boot(patient, ac_true, df_4seed["ac_pred_4seed"], df_cand["ac_pred"])
    log_info = parse_log(dl_dir / f"{args.slug}.log")

    payload = dict(summary)
    payload["oof_summary"] = dict(summary)
    payload["paired_bootstrap_vs_baseline_s42"] = boot_vs_s42
    payload["paired_bootstrap_vs_4seed_avg"] = boot_vs_4seed
    payload.update(log_info)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)

    print(f"=== {args.slug} (N=642, 501 patients) ===")
    print(f"MAE: {summary['ac_mae']:.4f} mm | RMSE: {summary['ac_rmse']:.4f} mm | Med: {summary['ac_median']:.4f} mm | Bias: {summary['ac_bias']:+.4f} mm | NME: {summary['nme']:.4f} | >10mm: {summary['over10']*100:.2f}%")
    print(f"vs s42   : delta {boot_vs_s42['delta_mae']:+.4f} mm | 95% CI [{boot_vs_s42['ci95_delta_mae'][0]:+.4f}, {boot_vs_s42['ci95_delta_mae'][1]:+.4f}] | p={boot_vs_s42['p_val_mae']:.4f} | excl0={boot_vs_s42['ci_excludes_zero']}")
    print(f"vs 4-seed: delta {boot_vs_4seed['delta_mae']:+.4f} mm | 95% CI [{boot_vs_4seed['ci95_delta_mae'][0]:+.4f}, {boot_vs_4seed['ci95_delta_mae'][1]:+.4f}] | p={boot_vs_4seed['p_val_mae']:.4f} | excl0={boot_vs_4seed['ci_excludes_zero']}")
    print(f"Per-fold best epochs: {log_info['per_fold_best_epochs']}")
    print(f"Jumps >50mm after ep 5: {log_info['jumps_gt50_after_ep5']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
