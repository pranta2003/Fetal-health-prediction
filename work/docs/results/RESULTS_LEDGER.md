# Experimental Results Ledger

## Sign Convention Standard
- **Grand Challenge Platform Convention**: `Difference = PRED - TRUE`. A positive value indicates model overestimation.
- **Internal Convention A**: `Difference = TRUE - PRED`. A positive value indicates model underestimation.
- **Dual References (Ground Truth Mask Contours vs CSV Reference Ellipse)**:
  - Fitting OpenCV ellipses to ground truth binary annotation masks introduces a systematic +0.71 mm bias across training set (`gt_annotation_mask_vs_csv: +0.7126 +- 0.2738 mm`).
  - This is driven by mask contour extraction tracing the outer boundary of the hollow annotation line stroke (thickness ~2.40 px across all trimesters, corresponding to $\delta_r \approx 0.1134$ mm).
  - Validation metrics below are presented against BOTH references: the inflated mask-ellipse and the official CSV ground truth ellipse.

---

## 1. Abdominal Circumference (AC Campaign)
| Date | Run ID / Recipe | Split / Protocol | N (Pts) | AC MAE (mm) | AC RMSE (mm) | AC Median (mm) | AC Bias [pred-true] (mm) | TAD MAE (mm) | APAD MAE (mm) | NME | >10mm (%) | Artifact SHA-256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-19 | `ac-baseline-s42` (v3_full baseline, seed 42) | 5-fold CV (`folds_v1.csv`) | 642 (501) | 4.7744 | 7.6074 | 2.8474 | -0.0826 | 1.7331 | 2.0793 | 0.0808 | 12.15% | `8ca2aed7810fcb0b4a84ca5ba57e53f101d50f1349ac6484a351a070ec779f52` |
| 2026-11-19 | `ac-ema-s42` (v3_full + EMA, seed 42) | 5-fold CV (`folds_v1.csv`) | 642 (501) | 4.6430 | 7.2886 | 2.8375 | -0.7910 | 1.6868 | 2.0389 | 0.0793 | 11.84% | `ff8ca0558d9cbebbc9024296586377764f27d4168de3be6c33fb94572a771637` |

---

## 2. Head Circumference (HC18 Benchmark, Official Test Submissions & Dual References)
| Date | Run ID / Model | Split / Reference | N (Pts) | HC MAE [Abs Diff] (mm) | Signed Bias [pred-true] (mm) | Signed Bias [true-pred] (mm) | Dice (%) | HD (mm) | Outcome / Status | Artifact SHA-256 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-15 | `submission_ensemble_5fold_corrected` (c1) | Grand Challenge Official Test Set (True Ellipse) | 335 | 1.9472 | +0.7844 | -0.7844 | 97.98% | 1.2144 | Corrected ellipse submission baseline | `5bf81467e417b84343cb6f313845ee813a6c9f31cd41f8a9800f50b9c177f6cb` |
| 2026-11-16 | `submission_ensemble_5fold_corrected_mult050` (probe) | Grand Challenge Official Test Set (True Ellipse) | 335 | 2.2417 | +1.4999 | -1.4999 | 97.88% | 1.2580 | **NEGATIVE RESULT**: Probe scaled by 1.004065 (+0.716 mm), shifting platform bias +0.784 -> +1.500 mm and worsening MAE (+0.294 mm) | `682672a906d6317fdaca1e7c40b2b03c465cb98d4e4472d71b8d666f210141a1` |
| 2026-11-20 | `submission_ensemble_5fold_corrected_measfix` | Grand Challenge Official Test Set (True Ellipse) | 335 | 1.8350 | +0.0743 | -0.0743 | 98.02% | 1.1990 | **OFFICIAL BEST**: Semi-axes shrunk by $\delta_r = 0.1134$ mm; platform bias reduced +0.7844 -> +0.0743 mm; MAE improved 1.9472 -> 1.8350 mm | `e051fda4415c6b488e29665a086104fac639b74bee6f0cbc7a819a97cb6904f3` |
| 2026-11-19 | `hc18-nnunet-f0-250ep` (raw) | Fold 0 Val vs CSV Ground Truth | 206 | 1.8879 | +0.4256 | -0.4256 | 97.96% | 1.2711 | Raw nnU-Net prediction against official CSV reference | `b62968b65f911fd8777b1ab4d705c8075a2c907a0f52d85486ece5e9838287c7` |
| 2026-11-19 | `hc18-nnunet-f0-250ep` (raw) | Fold 0 Val vs Mask Ellipse | 206 | 1.9267 | -1.0487 | +1.0487 | 97.96% | 1.2711 | Legacy comparison against inflated annotation mask | `b62968b65f911fd8777b1ab4d705c8075a2c907a0f52d85486ece5e9838287c7` |
| 2026-11-20 | `hc18-nnunet-f0-250ep` (measfix) | Fold 0 Val vs CSV Ground Truth | 206 | 1.8686 | -0.2846 | +0.2846 | 97.96% | 1.2711 | Shrunk by $\delta_r=0.1134$ mm; bias reduced +0.426 -> -0.285 mm | `9b5a3b53d47a376a3af0040debec6f68d07dc4d8396854f950e3634676cb1086` |
| 2026-11-20 | Single-model Fold 1 OOF (raw) | 999 OOF vs CSV Ground Truth | 999 | 2.3019 | -1.2576 | +1.2576 | - | - | Single model raw prediction against CSV reference | `9b5a3b53d47a376a3af0040debec6f68d07dc4d8396854f950e3634676cb1086` |

### HC18 Ladder Benchmark Reference Dual-Scoring (Rows A – E)
Audit of ground-truth reference definitions across ladder rows from `docs/results/hc18_ladder/hc18_ladder_ref_audit.json`:

| Row ID | Approach / Model | Split | N | Reference in Original Run | Original Ref MAE (mm) | **Official CSV Ref MAE (mm)** | Orig Signed Bias [pred-true] (mm) | **CSV Signed Bias [pred-true] (mm)** | Status / Action |
|---|---|---|:---:|---|:---:|:---:|:---:|:---:|---|
| **Row A** | Direct Regression (Cell 22) | Fold 1 | 206 | Official CSV HC | 10.6728 | **10.6728** | +5.3052 | **+5.3052** | Recomputed from `row_A_predictions.csv` |
| **Row B** | Geometry Approach v3 (Cell 27) | Fold 1 | 206 | Official CSV HC | 11.5964 | **11.5964** | +5.6149 | **+5.6149** | Recomputed from `row_B_predictions.csv` |
| **Row C** | Multi-Task U-Net (Cell 25) | Fold 1 | 206 | Official CSV HC | 10.5104 | **10.5104** | +3.5311 | **+3.5311** | Recomputed from `row_C_predictions.csv` |
| **Row D** | ResNet-34 U-Net (Cell 31) | Fold 1 | 206 | Official CSV HC | 1.9246 | **1.9246** | -0.2607 | **-0.2607** | Recomputed from `row_D_predictions.csv` |
| **Row D 5-fold** | ResNet-34 U-Net (5-fold pooled) | 5-Fold CV | 999 | Official CSV HC | 2.2688 | **2.2688** | +0.5435 | **+0.5435** | Recomputed from `row_D_5fold_predictions.csv` |
| **Row D TTA 5-fold** | ResNet-34 U-Net + TTA | 5-Fold CV | 999 | Official CSV HC | 2.2323 | **2.2323** | +0.5395 | **+0.5395** | Recomputed from `row_D_tta_5fold_predictions.csv` |
| **Row E** | EfficientNet-B4 U-Net 384 | 999 OOF | 999 | Mask-Ellipse HC | 2.3019 | **1.9933** | -1.2576 | **+0.2171** | Recomputed from `docs/results/hc18/oof_predictions.csv` |
| *Row D Run 1* | Historical single-run (Cell 31) | Fold 1 | 206 | — | — | — | — | — | *cannot recompute against CSV HC (prediction CSV not preserved)* |

---

## 3. Candidate Submissions (Exploratory / Unsubmitted)
| Candidate File | Method / Transform | Delta r / Scale | Rows | SHA-256 | Action Status |
|---|---|---|---|---|---|
| `candidates/submission_ensemble_5fold_corrected_measfix.csv` | Semi-axes shrunk by $\delta_r = 0.1134$ mm | -0.7102 mm mean HC change | 335 | `93cbf78d98404545dbe0cc78dd58399f73bedc7065b2cfd01e9d0627dd25ab81` | **DO NOT SUBMIT** (Built & verified locally) |
| `candidates/submission_ensemble_5fold_corrected_mult050_EXPLORATORY.csv` | Multiplicative scaling $\alpha = 0.50$ | 1.004065 | 335 | `1a1ff42eb8dcc9f417e69235fd9bbaa77d9b468cd56623825911e23afaae9b5d` | Submitted as probe on 2026-11-16 (negative result) |
| `candidates/submission_ensemble_5fold_corrected_mult100_EXPLORATORY.csv` | Multiplicative scaling $\alpha = 1.00$ | 1.008129 | 335 | `ee568b5157ddd5ffc424f180fab22d52a765fa11ccac62d66f0883dba9c1f356` | **DO NOT SUBMIT** (Never submitted) |

### Detailed Notes & Checkpoint Provenance
- **`ac-baseline-s42`**:
  - Kaggle Kernel: `prantakumerpandit390/ac-baseline-s42`
  - Output Archive: `E:\BUBT-CSE-4th yr\Capstone\kaggle_outputs\ac_baseline_s42`
  - Summary JSON: `docs/results/ac_campaign/ac_baseline_s42.json` (`8ca2aed7810fcb0b4a84ca5ba57e53f101d50f1349ac6484a351a070ec779f52`)
  - Protocol: Patient-grouped 5-fold cross-validation on 642 images across 501 patients, evaluated strictly out-of-fold.
- **`ac-ema-s42`**:
  - Kaggle Kernel: `prantakumerpandit390/ac-ema-s42`
  - Output Archive: `E:\BUBT-CSE-4th yr\Capstone\kaggle_outputs\ac_ema_s42`
  - Summary JSON: `docs/results/ac_campaign/ac_ema_s42.json` (`ff8ca0558d9cbebbc9024296586377764f27d4168de3be6c33fb94572a771637`)
  - Protocol: Patient-grouped 5-fold CV with Exponential Moving Average weights (decay=0.99) on 642 images across 501 patients.
- **`submission_ensemble_5fold_corrected_mult050` (HC18 Probe)**:
  - Grand Challenge Test Submission: Probe with multiplicative factor 1.004065 (formerly labeled mult100 in preliminary docs, mult100 was never submitted).
  - Metrics File: `docs/results/hc18_calibration/metrics_probe_mult050.json` (`682672a906d6317fdaca1e7c40b2b03c465cb98d4e4472d71b8d666f210141a1`)
  - Platform Difference: explicitly `PRED - TRUE` (+1.4999 mm). Confirmed negative result: test MAE increased from 1.9472 to 2.2417 mm.
- **`submission_ensemble_5fold_corrected_measfix` (HC18 Official Best)**:
  - Grand Challenge Test Submission: 5-fold ensemble with measurement-fix shrink ($\delta_r = 0.1134$ mm).
  - Metrics File: `docs/results/hc18_official/metrics_ensemble_5fold_measfix.json` (`e051fda4415c6b488e29665a086104fac639b74bee6f0cbc7a819a97cb6904f3`)
  - Platform Difference: `PRED - TRUE` (+0.0743 mm). Official MAE 1.8350 mm, Dice 98.02%, HD 1.1990 mm.
- **`hc18-nnunet-f0-250ep` Model Weights Provenance**:
  - Kaggle Kernel: `prantakumerpandit390/hc18-nnunet-f0-250ep`
  - Local Checkpoint: 30 MB partial download on laptop; the complete weights (`checkpoint_final.pth`, `plans.json`, `dataset.json`) exist intact in Kaggle kernel output and are mounted via `kernel_sources` into `hc18-nnunet-test-f0`.
  - Output Archive: `E:\BUBT-CSE-4th yr\Capstone\kaggle_outputs\hc18_nnunet_f0`
  - Summary JSON: `docs/results/hc18_nnunet/f0_eval.json` (`b62968b65f911fd8777b1ab4d705c8075a2c907a0f52d85486ece5e9838287c7`)
  - Measurement Study JSON: `docs/results/hc18_official/measfix_study.json` (`9b5a3b53d47a376a3af0040debec6f68d07dc4d8396854f950e3634676cb1086`)
