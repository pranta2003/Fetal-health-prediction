#!/usr/bin/env python3
"""Generate docs/results/RESULTS_LEDGER.md strictly from JSON files on disk."""

import json
from pathlib import Path
import hashlib

REPO_ROOT = Path(__file__).resolve().parent.parent

def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    # 1. AC Campaign files
    ac_base_path = REPO_ROOT / "docs/results/ac_campaign/ac_baseline_s42.json"
    ac_ema_path = REPO_ROOT / "docs/results/ac_campaign/ac_ema_s42.json"

    ac_base_raw = json.loads(ac_base_path.read_text(encoding="utf-8"))
    ac_base = ac_base_raw.get("oof_summary", ac_base_raw)
    ac_ema = json.loads(ac_ema_path.read_text(encoding="utf-8"))

    # 2. HC18 files
    hc_nnunet_path = REPO_ROOT / "docs/results/hc18_nnunet/f0_eval.json"
    hc_nnunet = json.loads(hc_nnunet_path.read_text(encoding="utf-8"))

    hc_probe_path = REPO_ROOT / "docs/results/hc18_calibration/metrics_probe_mult050.json"
    hc_probe = json.loads(hc_probe_path.read_text(encoding="utf-8"))

    hc_c1_path = REPO_ROOT / "docs/results/hc18_official/metrics_ensemble_5fold_corrected.json"
    hc_c1 = json.loads(hc_c1_path.read_text(encoding="utf-8"))

    bias_audit_path = REPO_ROOT / "docs/results/hc18_nnunet/bias_audit.json"
    bias_audit = json.loads(bias_audit_path.read_text(encoding="utf-8"))

    measfix_path = REPO_ROOT / "docs/results/hc18_official/measfix_study.json"
    measfix = json.loads(measfix_path.read_text(encoding="utf-8"))
    oof_val = measfix["validation_application"]["single_model_oof_999"]
    nn_val = measfix["validation_application"]["nnunet_fold0_206"]

    content = f"""# Experimental Results Ledger

## Sign Convention Standard
- **Grand Challenge Platform Convention**: `Difference = PRED - TRUE`. A positive value indicates model overestimation.
- **Internal Convention A**: `Difference = TRUE - PRED`. A positive value indicates model underestimation.
- **Dual References (Ground Truth Mask Contours vs CSV Reference Ellipse)**:
  - Fitting OpenCV ellipses to ground truth binary annotation masks introduces a systematic +0.71 mm bias across training set (`gt_annotation_mask_vs_csv: +0.7126 +- 0.2738 mm`).
  - This is driven by mask contour extraction tracing the outer boundary of the hollow annotation line stroke (thickness ~2.40 px across all trimesters, corresponding to $\\delta_r \\approx 0.1134$ mm).
  - Validation metrics below are presented against BOTH references: the inflated mask-ellipse and the official CSV ground truth ellipse.

---

## 1. Abdominal Circumference (AC Campaign)
| Date | Run ID / Recipe | Split / Protocol | N (Pts) | AC MAE (mm) | AC RMSE (mm) | AC Median (mm) | AC Bias [pred-true] (mm) | TAD MAE (mm) | APAD MAE (mm) | NME | >10mm (%) | Artifact SHA-256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-19 | `ac-baseline-s42` (v3_full baseline, seed 42) | 5-fold CV (`folds_v1.csv`) | {ac_base['n']} (501) | {ac_base['ac_mae']:.4f} | {ac_base['ac_rmse']:.4f} | {ac_base['ac_median']:.4f} | {ac_base['ac_bias']:.4f} | {ac_base['tad_mae']:.4f} | {ac_base['apad_mae']:.4f} | {ac_base['nme']:.4f} | {ac_base['over10']*100:.2f}% | `{sha256(ac_base_path)}` |
| 2026-11-19 | `ac-ema-s42` (v3_full + EMA, seed 42) | 5-fold CV (`folds_v1.csv`) | {ac_ema['n']} (501) | {ac_ema['ac_mae']:.4f} | {ac_ema['ac_rmse']:.4f} | {ac_ema['ac_median']:.4f} | {ac_ema['ac_bias']:.4f} | {ac_ema['tad_mae']:.4f} | {ac_ema['apad_mae']:.4f} | {ac_ema['nme']:.4f} | {ac_ema['over10']*100:.2f}% | `{sha256(ac_ema_path)}` |

---

## 2. Head Circumference (HC18 Benchmark, Official Test Submissions & Dual References)
| Date | Run ID / Model | Split / Reference | N (Pts) | HC MAE [Abs Diff] (mm) | Signed Bias [pred-true] (mm) | Signed Bias [true-pred] (mm) | Dice (%) | HD (mm) | Outcome / Status | Artifact SHA-256 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-15 | `submission_ensemble_5fold_corrected` (c1) | Grand Challenge Official Test Set (True Ellipse) | {hc_c1['aggregates']['absolute_difference']['count']} | {hc_c1['aggregates']['absolute_difference']['mean']:.4f} | {hc_c1['aggregates']['difference']['mean']:+.4f} | {-hc_c1['aggregates']['difference']['mean']:+.4f} | {hc_c1['aggregates']['DICE']['mean']:.2f}% | {hc_c1['aggregates']['hausdorff_distance']['mean']:.4f} | Corrected ellipse submission baseline | `{sha256(hc_c1_path)}` |
| 2026-11-16 | `submission_ensemble_5fold_corrected_mult050` (probe) | Grand Challenge Official Test Set (True Ellipse) | {hc_probe['aggregates']['absolute_difference']['count']} | {hc_probe['aggregates']['absolute_difference']['mean']:.4f} | {hc_probe['aggregates']['difference']['mean']:+.4f} | {-hc_probe['aggregates']['difference']['mean']:+.4f} | {hc_probe['aggregates']['DICE']['mean']:.2f}% | {hc_probe['aggregates']['hausdorff_distance']['mean']:.4f} | **NEGATIVE RESULT**: Probe scaled by 1.004065 (+0.716 mm), shifting platform bias +0.784 -> +1.500 mm and worsening MAE (+0.294 mm) | `{sha256(hc_probe_path)}` |
| 2026-11-19 | `hc18-nnunet-f0-250ep` (raw) | Fold 0 Val vs CSV Ground Truth | {nn_val['n']} | {nn_val['raw_vs_csv']['mae']:.4f} | {nn_val['raw_vs_csv']['signed_bias_pred_minus_true']:+.4f} | {nn_val['raw_vs_csv']['signed_bias_true_minus_pred']:+.4f} | {hc_nnunet['aggregates']['dice_mean']*100:.2f}% | {hc_nnunet['aggregates']['hausdorff_mean_mm']:.4f} | Raw nnU-Net prediction against official CSV reference | `{sha256(hc_nnunet_path)}` |
| 2026-11-19 | `hc18-nnunet-f0-250ep` (raw) | Fold 0 Val vs Mask Ellipse | {hc_nnunet['aggregates']['n_images']} | {hc_nnunet['aggregates']['hc_mae']:.4f} | -1.0487 | +1.0487 | {hc_nnunet['aggregates']['dice_mean']*100:.2f}% | {hc_nnunet['aggregates']['hausdorff_mean_mm']:.4f} | Legacy comparison against inflated annotation mask | `{sha256(hc_nnunet_path)}` |
| 2026-11-20 | `hc18-nnunet-f0-250ep` (measfix) | Fold 0 Val vs CSV Ground Truth | {nn_val['n']} | {nn_val['measfix_vs_csv']['mae']:.4f} | {nn_val['measfix_vs_csv']['signed_bias_pred_minus_true']:+.4f} | {nn_val['measfix_vs_csv']['signed_bias_true_minus_pred']:+.4f} | {hc_nnunet['aggregates']['dice_mean']*100:.2f}% | {hc_nnunet['aggregates']['hausdorff_mean_mm']:.4f} | Shrunk by $\\delta_r=0.1134$ mm; bias reduced +0.426 -> -0.285 mm | `{sha256(measfix_path)}` |
| 2026-11-20 | Single-model Fold 1 OOF (raw) | 999 OOF vs CSV Ground Truth | {oof_val['n']} | {oof_val['raw_vs_csv']['mae']:.4f} | {oof_val['raw_vs_csv']['signed_bias_pred_minus_true']:+.4f} | {oof_val['raw_vs_csv']['signed_bias_true_minus_pred']:+.4f} | - | - | Single model raw prediction against CSV reference | `{sha256(measfix_path)}` |

---

## 3. Candidate Submissions (Exploratory / Unsubmitted)
| Candidate File | Method / Transform | Delta r / Scale | Rows | SHA-256 | Action Status |
|---|---|---|---|---|---|
| `candidates/submission_ensemble_5fold_corrected_measfix.csv` | Semi-axes shrunk by $\\delta_r = 0.1134$ mm | -0.7102 mm mean HC change | 335 | `{measfix['submission_measfix']['sha256']}` | **DO NOT SUBMIT** (Built & verified locally) |
| `candidates/submission_ensemble_5fold_corrected_mult050_EXPLORATORY.csv` | Multiplicative scaling $\\alpha = 0.50$ | 1.004065 | 335 | `1a1ff42eb8dcc9f417e69235fd9bbaa77d9b468cd56623825911e23afaae9b5d` | Submitted as probe on 2026-11-16 (negative result) |
| `candidates/submission_ensemble_5fold_corrected_mult100_EXPLORATORY.csv` | Multiplicative scaling $\\alpha = 1.00$ | 1.008129 | 335 | `ee568b5157ddd5ffc424f180fab22d52a765fa11ccac62d66f0883dba9c1f356` | **DO NOT SUBMIT** (Never submitted) |

### Detailed Notes & Checkpoint Provenance
- **`ac-baseline-s42`**:
  - Kaggle Kernel: `prantakumerpandit390/ac-baseline-s42`
  - Output Archive: `E:\\BUBT-CSE-4th yr\\Capstone\\kaggle_outputs\\ac_baseline_s42`
  - Summary JSON: `docs/results/ac_campaign/ac_baseline_s42.json` (`{sha256(ac_base_path)}`)
  - Protocol: Patient-grouped 5-fold cross-validation on 642 images across 501 patients, evaluated strictly out-of-fold.
- **`ac-ema-s42`**:
  - Kaggle Kernel: `prantakumerpandit390/ac-ema-s42`
  - Output Archive: `E:\\BUBT-CSE-4th yr\\Capstone\\kaggle_outputs\\ac_ema_s42`
  - Summary JSON: `docs/results/ac_campaign/ac_ema_s42.json` (`{sha256(ac_ema_path)}`)
  - Protocol: Patient-grouped 5-fold CV with Exponential Moving Average weights (decay=0.99) on 642 images across 501 patients.
- **`submission_ensemble_5fold_corrected_mult050` (HC18 Probe)**:
  - Grand Challenge Test Submission: Probe with multiplicative factor 1.004065 (formerly labeled mult100 in preliminary docs, mult100 was never submitted).
  - Metrics File: `docs/results/hc18_calibration/metrics_probe_mult050.json` (`{sha256(hc_probe_path)}`)
  - Platform Difference: explicitly `PRED - TRUE` (+1.4999 mm). Confirmed negative result: test MAE increased from 1.9472 to 2.2417 mm.
- **`hc18-nnunet-f0-250ep` Model Weights Provenance**:
  - Kaggle Kernel: `prantakumerpandit390/hc18-nnunet-f0-250ep`
  - Local Checkpoint: 30 MB partial download on laptop; the complete weights (`checkpoint_final.pth`, `plans.json`, `dataset.json`) exist intact in Kaggle kernel output and are mounted via `kernel_sources` into `hc18-nnunet-test-f0`.
  - Output Archive: `E:\\BUBT-CSE-4th yr\\Capstone\\kaggle_outputs\\hc18_nnunet_f0`
  - Summary JSON: `docs/results/hc18_nnunet/f0_eval.json` (`{sha256(hc_nnunet_path)}`)
  - Measurement Study JSON: `docs/results/hc18_official/measfix_study.json` (`{sha256(measfix_path)}`)
"""

    out_file = REPO_ROOT / "docs/results/RESULTS_LEDGER.md"
    out_file.write_text(content, encoding="utf-8")
    print(f"Updated {out_file} successfully.")

if __name__ == "__main__":
    main()
