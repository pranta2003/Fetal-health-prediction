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
| 2026-11-24 | `ac-baseline-s43` (v3_full baseline, seed 43) | 5-fold CV (`folds_v1.csv`) | 642 (501) | 4.5406 | 7.1260 | 2.9084 | -0.1976 | 1.6685 | 2.0281 | 0.0789 | 9.81% | `9c92e54993b5e8d2cea156ded48e24d067219a3a604becc0de8bba4b64756e13` |
| 2026-11-19 | `ac-ema-s42` (v3_full + EMA, seed 42) | 5-fold CV (`folds_v1.csv`) | 642 (501) | 4.6430 | 7.2886 | 2.8375 | -0.7910 | 1.6868 | 2.0389 | 0.0793 | 11.84% | `ff8ca0558d9cbebbc9024296586377764f27d4168de3be6c33fb94572a771637` |
| 2026-11-24 | `ac-wing-s42` (v3_full + Wing loss, seed 42) | 5-fold CV (`folds_v1.csv`) | 642 (501) | 4.8796 | 7.4940 | 3.1506 | -0.1662 | 1.6705 | 2.1797 | 0.0829 | 11.99% | `af79383c71c4754d011897d5969768ef10d8616b928434fea885f77475515909` |
| 2026-11-21 | `ac-fassd-s42` (v3_full + FASSD pretraining, seed 42) | 5-fold CV (`folds_v1.csv`) | 642 (501) | 4.9459 | 8.3335 | 2.9702 | -0.4653 | 1.7911 | 2.1315 | 0.0820 | 12.62% | `6153da5481a8d6654020d0ad9c2ed821a6a51b1fd87238db374d02704dca9eb9` |
| 2026-11-24 | `ac-2seed-avg` (mean of s42 and s43) | 5-fold CV (`folds_v1.csv`) | 642 (501) | 4.4780 | 6.9725 | 2.8152 | -0.1401 | - | - | 0.0228 | 10.75% | `40c6f7ca72498b9b5916db05d7aacc4300de8859d5a536920fff8d25d41b6831` |

---

## 2. Head Circumference (HC18 Benchmark, Official Test Submissions & Dual References)
| Date | Run ID / Model | Split / Reference | N (Pts) | HC MAE [Abs Diff] (mm) | Signed Bias [pred-true] (mm) | Signed Bias [true-pred] (mm) | Dice (%) | HD (mm) | Outcome / Status | Artifact SHA-256 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-15 | `submission_ensemble_5fold_corrected` (c1) | Grand Challenge Official Test Set (True Ellipse) | 335 | 1.9472 | +0.7844 | -0.7844 | 97.98% | 1.2144 | Corrected ellipse submission baseline | `5bf81467e417b84343cb6f313845ee813a6c9f31cd41f8a9800f50b9c177f6cb` |
| 2026-11-16 | `submission_ensemble_5fold_corrected_mult050` (probe) | Grand Challenge Official Test Set (True Ellipse) | 335 | 2.2417 | +1.4999 | -1.4999 | 97.88% | 1.2580 | **NEGATIVE RESULT**: Probe scaled by 1.004065 (+0.716 mm), shifting platform bias +0.784 -> +1.500 mm and worsening MAE (+0.294 mm) | `682672a906d6317fdaca1e7c40b2b03c465cb98d4e4472d71b8d666f210141a1` |
| 2026-11-20 | `submission_ensemble_5fold_corrected_measfix` | Grand Challenge Official Test Set (True Ellipse) | 335 | 1.8350 | +0.0743 | -0.0743 | 98.02% | 1.1990 | **OFFICIAL BEST**: Semi-axes shrunk by $\delta_r = 0.1134$ mm; platform bias reduced +0.7844 -> +0.0743 mm; MAE improved 1.9472 -> 1.8350 mm | `e051fda4415c6b488e29665a086104fac639b74bee6f0cbc7a819a97cb6904f3` |
| 2026-11-21 | `submission_nnunet_f0_measfix` | Grand Challenge Official Test Set (True Ellipse) | 335 | 1.8275 | +0.0382 | -0.0382 | 97.99% | 1.2039 | **OFFICIAL (Rank 107)**: nnU-Net fold 0 + measfix (user-reported file name, comment on the site is wrong) | `5ad02ab58039e40ff6fd52628f969ad86f61b3a5f50cc53dd28755f830f3f4c5` |
| 2026-11-23 | `submission_nnunet_f01234_measfix` | Grand Challenge Official Test Set (True Ellipse) | 335 | 1.7832 | +0.0684 | -0.0684 | 98.08% | 1.1623 | **NEW OFFICIAL BEST (Rank 40)**: nnU-Net full 5-fold ensemble + measfix ($\delta_r=0.1134$ mm); breaks below 1.80 mm barrier | `ac24296c6575ad0c6a685115ae734ecf3ea5b105a6c9b1f8bdf1c44f5d25b31b` |
| 2026-11-19 | `hc18-nnunet-f0-250ep` (raw) | Fold 0 Val vs CSV Ground Truth | 206 | 1.8879 | +0.4256 | -0.4256 | 97.96% | 1.2711 | Raw nnU-Net prediction against official CSV reference | `b62968b65f911fd8777b1ab4d705c8075a2c907a0f52d85486ece5e9838287c7` |
| 2026-11-19 | `hc18-nnunet-f0-250ep` (raw) | Fold 0 Val vs Mask Ellipse | 206 | 1.9267 | -1.0487 | +1.0487 | 97.96% | 1.2711 | Legacy comparison against inflated annotation mask | `b62968b65f911fd8777b1ab4d705c8075a2c907a0f52d85486ece5e9838287c7` |
| 2026-11-20 | `hc18-nnunet-f0-250ep` (measfix) | Fold 0 Val vs CSV Ground Truth | 206 | 1.8686 | -0.2846 | +0.2846 | 97.96% | 1.2711 | Shrunk by $\delta_r=0.1134$ mm; bias reduced +0.426 -> -0.285 mm | `9b5a3b53d47a376a3af0040debec6f68d07dc4d8396854f950e3634676cb1086` |
| 2026-11-21 | `hc18-nnunet-f1-100ep` (raw) | Fold 1 Val vs CSV Ground Truth | 199 (161) | 1.9582 | +0.7910 | -0.7910 | 97.36% | 1.5262 | Raw nnU-Net prediction against official CSV reference | `87cfc34fdcb90d78c407bc715a1c615d413aec2702866ccc33ab37cca8a4996a` |
| 2026-11-21 | `hc18-nnunet-f1-100ep` (measfix) | Fold 1 Val vs CSV Ground Truth | 199 (161) | 1.8767 | +0.0806 | -0.0806 | 97.36% | 1.5262 | Shrunk by $\delta_r=0.1134$ mm; bias reduced +0.791 -> +0.081 mm | `87cfc34fdcb90d78c407bc715a1c615d413aec2702866ccc33ab37cca8a4996a` |
| 2026-11-21 | `hc18-nnunet-f2-100ep` (raw) | Fold 2 Val vs CSV Ground Truth | 208 (161) | 2.1076 | +0.4470 | -0.4470 | 97.39% | 1.3962 | Raw nnU-Net prediction against official CSV reference | `a4edf10eb05f93f690659f17111d7b91069dddd90272de42632f07875137fc32` |
| 2026-11-21 | `hc18-nnunet-f2-100ep` (measfix) | Fold 2 Val vs CSV Ground Truth | 208 (161) | 2.0993 | -0.2633 | +0.2633 | 97.39% | 1.3962 | Shrunk by $\delta_r=0.1134$ mm; bias reduced +0.447 -> -0.263 mm | `a4edf10eb05f93f690659f17111d7b91069dddd90272de42632f07875137fc32` |
| 2026-11-21 | `hc18-nnunet-pooled-f012` (raw) | Pooled Folds 0, 1, 2 Val vs CSV Ground Truth | 613 (484) | 1.9853 | +0.5515 | -0.5515 | 97.57% | 1.3964 | Raw pooled nnU-Net prediction across folds 0, 1, 2 | `0268ad1e72644a35e91b2526b364329aad7f20a1cb8c98372429958cc090aa26` |
| 2026-11-21 | `hc18-nnunet-pooled-f012` (measfix) | Pooled Folds 0, 1, 2 Val vs CSV Ground Truth | 613 (484) | 1.9495 | -0.1588 | +0.1588 | 97.57% | 1.3964 | Shrunk by $\delta_r=0.1134$ mm; bias reduced +0.5515 -> -0.1588 mm | `0268ad1e72644a35e91b2526b364329aad7f20a1cb8c98372429958cc090aa26` |
| 2026-11-21 | `heterogeneous-blend-nnunet-eff` (measfix+raw) | Pooled Folds 0, 1, 2 Val vs CSV Ground Truth | 613 (484) | 1.8540 | -0.0016 | +0.0016 | - | - | Heterogeneous blend: 0.5 * nnU-Net (measfix) + 0.5 * EffNet (raw); bootstrap vs nnU-Net $\Delta = -0.0955$ mm, $p=0.0888$ | `5045dc810f2ec4a6a95cde82a3659d1606a21066cc82631b073260ac0ca08ed9` |
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
| `candidates/submission_blend_nnunetf0_ens5_measfix.csv` | Heterogeneous ellipse blend: nnU-Net f0 measfix + 5-fold ensemble measfix | Mean HC: 175.2869 mm (+0.0173 mm vs nnU-Net, -0.0187 mm vs ens5) | 335 | `aabfb91e54cd6b36b57aa52eab5d2752057203bb319f32d45c75106a7d5afd51` | **DO NOT SUBMIT** (Candidate for tomorrow's slot) |
| `candidates/submission_nnunet_f012_raw.csv` | nnU-Net 3-fold ensemble (folds 0, 1, 2) raw | Mean HC: 175.9986 mm | 335 | `c8ed4c15a43e99adf4b9614d7b25aee32c404f31b9c97c6a59755549dd8d1ddb` | **DO NOT SUBMIT** (Built & verified locally) |
| `candidates/submission_nnunet_f012_measfix.csv` | nnU-Net 3-fold ensemble (folds 0, 1, 2) measfix | $\delta_r = 0.1134$ mm, Mean HC: 175.2884 mm | 335 | `f5c6daaa51ff2f5e192303c8e1a769fc2dcf8ff2f4d437f150000d9c5f57e700` | **DO NOT SUBMIT** (Built & verified locally) |
| `candidates/submission_blend_nnunetf012_ens5_measfix.csv` | Heterogeneous blend: nnU-Net f012 measfix + 5-fold ensemble measfix | Mean HC: 175.2964 mm | 335 | `1519186ebdffd941c77b6e6636b6137b56ac9d59676b8af3e236cd79b984eebe` | **DO NOT SUBMIT** (Built & verified locally) |
| `candidates/submission_nnunet_f01234_raw.csv` | nnU-Net 5-fold ensemble (folds 0, 1, 2, 3, 4) raw | Mean HC: 176.0100 mm | 335 | `270136b25ce86fb3bd5e3ed053569540fba6e745a61f71d101b094c4f84e4fdc` | **DO NOT SUBMIT** (Built & verified locally) |
| `candidates/submission_nnunet_f01234_measfix.csv` | nnU-Net 5-fold ensemble (folds 0, 1, 2, 3, 4) measfix | $\delta_r = 0.1134$ mm, Mean HC: 175.2998 mm | 335 | `b62f2a15d5b4d5d028aa5b4bd1b817594a537d8184891af63184a35b7d375ca5` | Submitted by user on 2026-11-22: **Official Rank 40 (MAE 1.7832 mm, Dice 98.08%, HD 1.1623 mm)** |
| `candidates/submission_blend_nnunetf01234_ens5_measfix.csv` | Heterogeneous blend: nnU-Net f01234 measfix + 5-fold ensemble measfix | Mean HC: 175.3021 mm | 335 | `ca9acd5bc0caf79d03b08909a6ef140d974dbdb5500240af557a68833ac1349a` | **DO NOT SUBMIT** (Built & verified locally) |
| `candidates/submission_ensemble_5fold_corrected_mult050_EXPLORATORY.csv` | Multiplicative scaling $\alpha = 0.50$ | 1.004065 | 335 | `1a1ff42eb8dcc9f417e69235fd9bbaa77d9b468cd56623825911e23afaae9b5d` | Submitted as probe on 2026-11-16 (negative result) |
| `candidates/submission_ensemble_5fold_corrected_mult100_EXPLORATORY.csv` | Multiplicative scaling $\alpha = 1.00$ | 1.008129 | 335 | `ee568b5157ddd5ffc424f180fab22d52a765fa11ccac62d66f0883dba9c1f356` | **DO NOT SUBMIT** (Never submitted) |

---

## 4. Tabular Stillbirth Prediction (Stream 3 Modern GBDT Campaign)
- **Protocol**: CDC NVSS Temporal split (train <= 2021 [N=402,282], held-out test >= 2022 [N=130,110], total cohort 532,392 rows, 40,221 test stillbirths).
- **Features**: 15 clinical & pre-pregnancy variables (5 numeric, 10 categorical, zero post-delivery / outcome-time leakage).
- **Hyperparameter Optimization**: Optuna (40 trials).
- **Results File**: `docs/results/stream3_modern/stream3_modern_results.json` (`b06e046d95925f02ec5fdc9cb77e87a6f440fbe85957ab8cb49eb8c54315ecfa`)
- **Table File**: `docs/results/stream3_modern/stream3_modern_table.md` (`e4616211ef38f049ea22ea58f1d39aae88554afa5d3ec357b5cf4e99243f7b12`)

| Model | Test ROC-AUC | 95% Bootstrap CI | Brier Score (Raw -> Calibrated) | Key Architecture / Hyperparameters |
|---|:---:|:---:|:---:|---|
| **Tuned XGBoost** | **0.7382** | [0.7349, 0.7412] | — | max_depth=5, lr=0.0656, n_estimators=237, subsample=0.6171, colsample=0.7687, scale_pos_weight=3.8731 |
| **Tuned LightGBM** | **0.7383** | [0.7351, 0.7412] | — | max_depth=6, num_leaves=41, lr=0.0560, n_estimators=288, subsample=0.6088, colsample=0.6602, balanced |
| **Tuned CatBoost** | **0.7383** | [0.7350, 0.7414] | — | Optuna tuned with Balanced auto class weights |
| **Tuned Logistic Regression** | **0.6659** | [0.6624, 0.6691] | — | Standardized features, C=0.01, balanced |
| **Calibrated GBM+LR Blend** | **0.7382** | [0.7350, 0.7412] | 0.19641 -> **0.17095** | `0.90 * LightGBM + 0.10 * LR` with Isotonic Calibration |

### Detailed Notes & Checkpoint Provenance
- **`stream3-modern-s1`**:
  - Kaggle Kernel: `prantakumerpandit390/stream3-modern-s1` (Status: `COMPLETE`)
  - Summary JSON: `docs/results/stream3_modern/stream3_modern_results.json` (`b06e046d95925f02ec5fdc9cb77e87a6f440fbe85957ab8cb49eb8c54315ecfa`)
  - Summary Table: `docs/results/stream3_modern/stream3_modern_table.md` (`e4616211ef38f049ea22ea58f1d39aae88554afa5d3ec357b5cf4e99243f7b12`)
  - Execution Log: `docs/results/stream3_modern/stream3-modern-s1.log` (`82677763684ad266b1a37da933d4684583c1fa5d3284cb29964f59e271e24b74`)
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
- **`ac-fassd-s42`**:
  - Kaggle Kernel: `prantakumerpandit390/ac-fassd-s42`
  - Output Archive: `E:\BUBT-CSE-4th yr\Capstone\kaggle_outputs\ac_fassd_s42`
  - Summary JSON: `docs/results/ac_campaign/ac_fassd_s42.json` (`6153da5481a8d6654020d0ad9c2ed821a6a51b1fd87238db374d02704dca9eb9`)
  - Protocol: Patient-grouped 5-fold CV with FASSD self-supervised pretrained backbone weights on 642 images across 501 patients. Paired bootstrap vs baseline s42: $\Delta\text{MAE} = +0.1715$ mm, 95% CI `[-0.2306, +0.5989]`, $p = 0.4108$. Clearly did not help.
- **`ac-baseline-s43` & `ac-wing-s42` (Kaggle Runs)**:
  - Kaggle Kernels: `prantakumerpandit390/ac-baseline-s43` and `prantakumerpandit390/ac-wing-s42` (both `COMPLETE`).
  - Output Archives: `C:\fetal-data\kaggle_outputs\ac_baseline_s43\` and `C:\fetal-data\kaggle_outputs\ac_wing_s42\`.
  - Summary JSONs: `docs/results/ac_campaign/ac_baseline_s43.json` (`9c92e54993b5e8d2cea156ded48e24d067219a3a604becc0de8bba4b64756e13`), `docs/results/ac_campaign/ac_wing_s42.json` (`af79383c71c4754d011897d5969768ef10d8616b928434fea885f77475515909`), and comparison `docs/results/ac_campaign/ac_s43_wing_comparison.json` (`40c6f7ca72498b9b5916db05d7aacc4300de8859d5a536920fff8d25d41b6831`).
  - Protocol: 5-fold patient-grouped CV on 642 images across 501 patients (`folds_v1.csv`).
  - Paired Patient-Level Bootstrap (5000 resamples, seed 42):
    - **Seed Noise [s43 vs s42]**: $\Delta\text{MAE} = -0.2337$ mm, 95% CI `[-0.5773, +0.1279]`, $p = 0.1856$. Random seed fluctuation is ~0.23 mm and not statistically significant.
    - **Wing Loss [wing s42 vs baseline s42]**: $\Delta\text{MAE} = +0.1052$ mm, 95% CI `[-0.2537, +0.4676]`, $p = 0.5500$. Wing loss does not outperform default loss.
    - **Wing Loss vs Seed 43 [wing s42 vs baseline s43]**: $\Delta\text{MAE} = +0.3389$ mm, 95% CI `[-0.0233, +0.7047]`, $p = 0.0664$.
    - **2-Seed Ensemble [mean(s42, s43) vs baseline s42]**: $\Delta\text{MAE} = -0.2963$ mm, 95% CI `[-0.4750, -0.1099]`, $p = 0.0012$. Significant gain achieved simply by ensembling two random initializations.
- **`ac-baseline-s44` & `ac-baseline-s45` (Kaggle Runs)**:
  - Kaggle Kernels: `prantakumerpandit390/ac-baseline-s44` (Status: `COMPLETE`) and `prantakumerpandit390/ac-baseline-s45` (Status: `COMPLETE`).
  - Recipes: v3_full baseline, patient-grouped 5-fold cross-validation on 642 images across 501 patients, seeds 44 and 45.
- **`Dataset502_AC` (nnU-Net AC Dataset Preparation)**:
  - Dataset Directory: `C:\fetal-data\nnunet_ac\Dataset502_AC\`
  - Kaggle Dataset: `prantakumerpandit390/dataset502-ac` (status: `ready`, private dataset with Dataset502_AC and offline wheels)
  - Summary JSON: `docs/results/ac_campaign/ac_nnunet_prep_summary.json` (`0c6dd39d20f7dbc98fd565fecebf8023c66504e0f7f6e6830b14f2e2f06c5729`)
  - Generating Script: `scripts/83_ac_to_nnunet.py`
  - Cohort: 642 training images in `imagesTr/` with solid filled ellipse masks in `labelsTr/`, plus 147 test images in `imagesTs/`.
  - Ellipse Fitting & Quality: 0 bad ellipse fits out of 642 cases (rejection rule: semi-axis $\le 0$, aspect ratio $> 4.0$ or $< 0.25$, midpoints distance $> \max(r_{\text{tad}}, r_{\text{apad}})$, or contour $< 5$ points).
  - Mask Circumference Agreement vs Reference: Mean signed difference $-0.0951$ mm, mean absolute difference $0.1953$ mm, max absolute difference $2.9493$ mm vs sonographer caliper Ramanujan AC.
  - Splits: `splits_final.json` maps 5 folds (Fold 0: 142 val / 500 train; Fold 1: 121 val / 521 train; Fold 2: 124 val / 518 train; Fold 3: 121 val / 521 train; Fold 4: 134 val / 508 train) with 0 patient overlap across all folds.
- **`ac-nnunet-f0` (AC nnU-Net Fold 0 Training Kernel Pushed)**:
  - Kernel Directory: `notebooks/kaggle/ac_nnunet/ac-nnunet-f0/`
  - Generating Script: `scripts/84_make_ac_nnunet_kaggle_kernel.py`
  - Configuration: 2D (`nnUNetPlans__2d`), `nnUNetTrainer_100epochs`, Fold 0 (500 train, 142 val cases), machine shape `NvidiaTeslaT4` (T4 x2), private dataset `prantakumerpandit390/dataset502-ac`, `enable_internet: false`.
  - Local Smoke Test: `scripts/smoke_test_ac_nnunet_f0.py` verified dataset loading, 2D PlainConvUNet forward pass, and Ramanujan AC measurement on 3 validation images (`020_2AC`, `020_3AC`, `020_4AC`).
  - Pushed to Kaggle: `2026-11-24 18:37:49 +06:00` (Kaggle Kernel ID: `prantakumerpandit390/ac-nnunet-f0`).
  - Status: `KernelWorkerStatus.QUEUED` (single status check, no polling loop).
  - Prepared Evaluation Script: `scripts/85_eval_ac_nnunet_f0.py` (loads 142 val cases, largest component extraction, `cv2.fitEllipseAMS`, Ramanujan AC in mm, MAE/RMSE/median/>10mm/bias metrics, paired patient-level cluster bootstrap with 5,000 resamples at seed 42 vs `s42`, `s43`, and 2-seed average).
- **`submission_ensemble_5fold_corrected_mult050` (HC18 Probe)**:
  - Grand Challenge Test Submission: Probe with multiplicative factor 1.004065 (formerly labeled mult100 in preliminary docs, mult100 was never submitted).
  - Metrics File: `docs/results/hc18_calibration/metrics_probe_mult050.json` (`682672a906d6317fdaca1e7c40b2b03c465cb98d4e4472d71b8d666f210141a1`)
  - Platform Difference: explicitly `PRED - TRUE` (+1.4999 mm). Confirmed negative result: test MAE increased from 1.9472 to 2.2417 mm.
- **`submission_ensemble_5fold_corrected_measfix` (HC18 Official Best)**:
  - Grand Challenge Test Submission: 5-fold ensemble with measurement-fix shrink ($\delta_r = 0.1134$ mm).
  - Metrics File: `docs/results/hc18_official/metrics_ensemble_5fold_measfix.json` (`e051fda4415c6b488e29665a086104fac639b74bee6f0cbc7a819a97cb6904f3`)
  - Platform Difference: `PRED - TRUE` (+0.0743 mm). Official MAE 1.8350 mm, Dice 98.02%, HD 1.1990 mm.
- **`submission_nnunet_f0_measfix` (HC18 Official Best, Rank 107)**:
  - Grand Challenge Test Submission: 2026-11-21 23:37 Dhaka.
  - Metrics File: `docs/results/hc18_official/metrics_nnunet_f0_measfix.json` (`5ad02ab58039e40ff6fd52628f969ad86f61b3a5f50cc53dd28755f830f3f4c5`)
  - Platform Metrics: Abs Diff MAE 1.8275 +- 1.7280 mm, Dice 97.988%, Difference [PRED - TRUE] +0.0382 mm, HD 1.2039 mm.
  - Provenance: nnU-Net fold 0 + measfix (user-reported file name, comment on the site is wrong: website comment says '5-fold ensemble, patient-grouped CV'). Single nnU-Net fold equals/surpasses our 5-model ensemble (1.8275 vs 1.8350 mm).
- **`hc18-nnunet-f0-250ep` Model Weights Provenance**:
  - Kaggle Kernel: `prantakumerpandit390/hc18-nnunet-f0-250ep`
  - Local Checkpoint: 30 MB partial download on laptop; the complete weights (`checkpoint_final.pth`, `plans.json`, `dataset.json`) exist intact in Kaggle kernel output and are mounted via `kernel_sources` into `hc18-nnunet-test-f0`.
  - Output Archive: `E:\BUBT-CSE-4th yr\Capstone\kaggle_outputs\hc18_nnunet_f0`
  - Summary JSON: `docs/results/hc18_nnunet/f0_eval.json` (`b62968b65f911fd8777b1ab4d705c8075a2c907a0f52d85486ece5e9838287c7`)
  - Measurement Study JSON: `docs/results/hc18_official/measfix_study.json` (`9b5a3b53d47a376a3af0040debec6f68d07dc4d8396854f950e3634676cb1086`)
- **`hc18-nnunet-f1-100ep` & `hc18-nnunet-f2-100ep`**:
  - Kaggle Kernel: `prantakumerpandit390/hc18-nnunet-f1f2-100ep` (Status: `COMPLETE`)
  - Output Archive: `E:\BUBT-CSE-4th yr\Capstone\kaggle_outputs\hc18_nnunet_f1f2`
  - Fold 1 Evaluation JSON: `docs/results/hc18_nnunet/f1_eval.json` (`87cfc34fdcb90d78c407bc715a1c615d413aec2702866ccc33ab37cca8a4996a`)
    - N=199 (161 pts). Raw MAE: 1.9582 mm, RMSE: 2.5205 mm, Bias [pred-true]: +0.7910 mm. Measfix MAE: 1.8767 mm, RMSE: 2.3946 mm, Bias [pred-true]: +0.0806 mm. Dice: 0.9736, HD: 1.5262 mm.
    - Paired bootstrap measfix vs raw: $\Delta\text{MAE} = -0.0815$ mm, 95% CI `[-0.1721, +0.0135]`, $p = 0.0924$. Measfix vs OOF: $\Delta\text{MAE} = -0.4172$ mm, 95% CI `[-0.6921, -0.1440]`, $p = 0.0032$.
  - Fold 2 Evaluation JSON: `docs/results/hc18_nnunet/f2_eval.json` (`a4edf10eb05f93f690659f17111d7b91069dddd90272de42632f07875137fc32`)
    - N=208 (161 pts). Raw MAE: 2.1076 mm, RMSE: 3.0350 mm, Bias [pred-true]: +0.4470 mm. Measfix MAE: 2.0993 mm, RMSE: 3.0133 mm, Bias [pred-true]: -0.2633 mm. Dice: 0.9739, HD: 1.3962 mm.
    - Paired bootstrap measfix vs raw: $\Delta\text{MAE} = -0.0083$ mm, 95% CI `[-0.1030, +0.0834]`, $p = 0.8748$. Measfix vs OOF: $\Delta\text{MAE} = -0.0589$ mm, 95% CI `[-0.3045, +0.1737]`, $p = 0.6348$.
- **`hc18-nnunet-pooled-f012` (Pooled Folds 0, 1, 2)**:
  - Evaluation JSON: `docs/results/hc18_nnunet/pooled_eval.json` (`0268ad1e72644a35e91b2526b364329aad7f20a1cb8c98372429958cc090aa26`)
  - Cohort: N=613 images, 484 unique patients across folds 0, 1, 2.
  - Raw: MAE 1.9853 mm, RMSE 2.7024 mm, Bias [pred-true] +0.5515 mm. Dice: 0.9757, HD: 1.3964 mm.
  - Measfix ($\delta_r = 0.1134$ mm): MAE 1.9495 mm, RMSE 2.6503 mm, Bias [pred-true] -0.1588 mm.
  - Per size band (measfix): under 110 mm (N=133) MAE 1.6827 mm; 110 to 260 mm (N=407) MAE 1.7655 mm; over 260 mm (N=73) MAE 3.4615 mm.
  - Significance Note: Fold 0 ($p=0.0028$) and Fold 1 ($p=0.0032$) improvements were statistically significant, whereas Fold 2 was not significant ($\Delta = -0.0589$ mm, $p=0.6348$).
- **Heterogeneous Blend Study**:
  - Evaluation JSON: `docs/results/hc18_nnunet/blend_study.json` (`5045dc810f2ec4a6a95cde82a3659d1606a21066cc82631b073260ac0ca08ed9`)
  - Summary Markdown: `docs/results/hc18_nnunet/BLEND_STUDY.md` (`c01ea80aecb4829739654ef7b6b73139f05111d5fe431c7053e2c15e42676fbd`)
  - Blending nnU-Net (measfix) with our EfficientNet-B4 U-Net (raw) gives pooled MAE 1.8540 mm (paired bootstrap vs nnU-Net alone: $\Delta = -0.0955$ mm, 95% CI `[-0.1869, +0.0178]`, $p = 0.0888$, not statistically significant).
  - Part (iv) ellipse parameter blend stopped: per-image ellipse parameters (`center_x`, `center_y`, `semi_axes_a`, `semi_axes_b`, `angle`) were not preserved in OOF prediction files.
- **`hc18-nnunet-test-f012` & `hc18-nnunet-test-f012-cpu` (Official Test Set 3-Fold Ensemble)**:
  - Kaggle Kernels: `prantakumerpandit390/hc18-nnunet-test-f012` (GPU, `COMPLETE`) and `prantakumerpandit390/hc18-nnunet-test-f012-cpu` (CPU, `COMPLETE`).
  - Output Directories: `C:\fetal-data\kaggle_outputs\hc18_nnunet_test_f012\` and `C:\fetal-data\kaggle_outputs\hc18_nnunet_test_f012_cpu\`.
  - Summary JSON: `docs/results/hc18_nnunet/test_f012_candidates_summary.json`
  - GPU vs CPU Invariance: 335 images evaluated; max absolute HC difference between GPU and CPU is 0.004840 mm (mean absolute diff 0.000636 mm), confirming near-exact inference equivalence across hardware backends.
  - Candidate Raw: `candidates/submission_nnunet_f012_raw.csv` (SHA256: `c8ed4c15a43e99adf4b9614d7b25aee32c404f31b9c97c6a59755549dd8d1ddb`). Mean HC: 175.9986 mm. Mean diff vs Fold 0 measfix: +0.7290 mm (abs: 0.8158 mm); mean diff vs 5-fold ensemble measfix: +0.6930 mm (abs: 1.2745 mm).
  - Candidate Measfix: `candidates/submission_nnunet_f012_measfix.csv` (SHA256: `f5c6daaa51ff2f5e192303c8e1a769fc2dcf8ff2f4d437f150000d9c5f57e700`). Mean HC: 175.2884 mm. Mean diff vs Fold 0 measfix: +0.0189 mm (abs: 0.4165 mm); mean diff vs 5-fold ensemble measfix: -0.0172 mm (abs: 1.0406 mm).
- **`hc18-nnunet-f3-100ep` & `hc18-nnunet-f4-100ep`**:
  - Kaggle Kernel: `prantakumerpandit390/hc18-nnunet-f3f4-100ep` (Status: `COMPLETE`)
  - Output Archive: `C:\fetal-data\kaggle_outputs\hc18_nnunet_f3f4\`
  - Fold 3 Evaluation JSON: `docs/results/hc18_nnunet/f3_eval.json`
    - N=194 images, 161 unique patients. Raw MAE: 2.2217 mm, Bias [pred-true]: +0.7280 mm. Measfix MAE: 2.1408 mm, Bias [pred-true]: +0.0178 mm. Dice: 0.9691, HD: 1.5130 mm.
    - Paired bootstrap measfix vs raw: $\Delta\text{MAE} = -0.0809$ mm, 95% CI `[-0.1772, +0.0097]`, $p = 0.0852$. Measfix vs OOF: $\Delta\text{MAE} = -0.6698$ mm, 95% CI `[-1.0125, -0.3444]`, $p = 0.0004$.
  - Fold 4 Evaluation JSON: `docs/results/hc18_nnunet/f4_eval.json`
    - N=192 images, 161 unique patients. Raw MAE: 1.9760 mm, Bias [pred-true]: +0.7405 mm. Measfix MAE: 1.8798 mm, Bias [pred-true]: +0.0303 mm. Dice: 0.9781, HD: 1.4032 mm.
    - Paired bootstrap measfix vs raw: $\Delta\text{MAE} = -0.0962$ mm, 95% CI `[-0.1979, +0.0041]`, $p = 0.0636$. Measfix vs OOF: $\Delta\text{MAE} = -0.2241$ mm, 95% CI `[-0.4398, -0.0055]`, $p = 0.0456$.
- **`hc18-nnunet-pooled-5fold` (Full 5-Fold Benchmark)**:
  - Evaluation JSON: `docs/results/hc18_nnunet/pooled_eval_5fold.json`
  - Cohort: N=999 images, 806 unique patients across all 5 folds (complete HC18 training set).
  - Raw: MAE 2.0294 mm, RMSE 2.6844 mm, Signed Bias [pred-true] +0.6221 mm. Dice: 0.9749, HD: 1.4203 mm.
  - Measfix ($\delta_r = 0.1134$ mm): MAE 1.9733 mm, RMSE 2.6289 mm, Signed Bias [pred-true] -0.0882 mm.
  - Per size band (measfix): under 110 mm (N=208) MAE 1.7483 mm; 110 to 250 mm (N=682) MAE 1.7643 mm; over 250 mm (N=109) MAE 3.7088 mm.
  - Paired bootstrap (5000 resamples): Measfix vs Raw $\Delta\text{MAE} = -0.0561$ mm, 95% CI `[-0.0975, -0.0134]`, $p = 0.0080$. Measfix vs OOF $\Delta\text{MAE} = -0.3287$ mm, 95% CI `[-0.5032, -0.1836]`, $p = 0.0000$.
- **Advanced Heterogeneous Blend Candidate**:
  - Candidate: `candidates/submission_blend_nnunetf012_ens5_measfix.csv` (SHA256: `1519186ebdffd941c77b6e6636b6137b56ac9d59676b8af3e236cd79b984eebe`).
  - Averaged ellipse parameters between 3-fold nnU-Net measfix (175.2884 mm) and 5-fold ensemble measfix (175.3056 mm).
  - Mean HC: 175.2964 mm. Max angular difference modulo $\pi$: 0.3126 rad ($\le 0.5$ rad).
- **5-Fold Test Kernels (`hc18-nnunet-test-f01234` & `hc18-nnunet-test-f01234-cpu`)**:
  - Kaggle Kernels: `prantakumerpandit390/hc18-nnunet-test-f01234` (GPU T4, `COMPLETE`) and `prantakumerpandit390/hc18-nnunet-test-f01234-cpu` (CPU, `RUNNING`).
  - Input Sources: `prantakumerpandit390/hc18-ultrasound`, `hc18-nnunet-f0-250ep`, `hc18-nnunet-f1f2-100ep`, `hc18-nnunet-f3f4-100ep`.
  - Pushed at: 2026-11-22 13:51:20 Dhaka.
  - Output Archive: `C:\fetal-data\kaggle_outputs\hc18_nnunet_test_f01234\`
  - Summary JSON: `docs/results/hc18_nnunet/test_f01234_candidates_summary.json`
  - Candidate Raw: `candidates/submission_nnunet_f01234_raw.csv` (SHA256: `270136b25ce86fb3bd5e3ed053569540fba6e745a61f71d101b094c4f84e4fdc`). Mean HC: 176.0100 mm. Mean diff vs f012 measfix: +0.7215 mm (abs: 0.7262 mm); vs 5-fold ensemble measfix: +0.7043 mm (abs: 1.2677 mm).
  - Candidate Measfix: `candidates/submission_nnunet_f01234_measfix.csv` (SHA256: `b62f2a15d5b4d5d028aa5b4bd1b817594a537d8184891af63184a35b7d375ca5`). Mean HC: 175.2998 mm. Mean diff vs f012 measfix: +0.0113 mm (mean abs diff 0.1728 mm $\le 0.3$ mm, verified). Mean diff vs 5-fold ensemble measfix: -0.0059 mm (abs: 1.0087 mm).
  - Blend Candidate: `candidates/submission_blend_nnunetf01234_ens5_measfix.csv` (SHA256: `ca9acd5bc0caf79d03b08909a6ef140d974dbdb5500240af557a68833ac1349a`). Mean HC: 175.3021 mm. Max angular difference modulo $\pi$: 0.2827 rad ($\le 0.5$ rad). Mean diff vs nnunet_f01234_measfix: +0.0024 mm (abs: 0.5043 mm); vs 5-fold ensemble measfix: -0.0035 mm (abs: 0.5044 mm).

---

## 5. Multi-Modal Fusion Biometric Reference (CDC NVSS Gestational Age Percentiles)
- **Protocol**: CDC NVSS live births only (`OUTCOME == 0`, stillbirths excluded), training years $\le 2021$ (2016–2021), completed gestational age $20 \le \text{COMBGEST} \le 42$ weeks, plausible birth weight $227 \le \text{DBWT} \le 8165$ g (missing `9999` dropped).
- **Cohort Size**: $N = 262,542$ verified live births across 23 completed weeks.
- **Reference CSV**: `docs/results/fusion/gestational_percentiles.csv` (`95aa3fb988ca87fcfc3a9f427bb3a756ecffb23fb0f4f2d72fba9558df65873b`)
- **Summary JSON**: `docs/results/fusion/gestational_percentiles_summary.json` (`ded130be563e7aad919815fbdcd9a725edbba82c0be7a2cfd4f2631fa82e236c`)
- **Generating Script**: `scripts/81_build_gestational_percentiles.py`

| GA Week | Sample N | 10th Pct $p_{10}$ (g) | Median $p_{50}$ (g) | 90th Pct $p_{90}$ (g) | Mean (g) | Std (g) | Clinical Milestone |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **20** | 60 | 254.7 | 336.0 | 687.4 | 429.7 | 257.9 | Anomaly scan baseline |
| **24** | 217 | 496.0 | 690.0 | 1300.0 | 847.5 | 576.9 | Viability threshold |
| **28** | 539 | 740.0 | 1191.0 | 3095.2 | 1493.6 | 835.8 | Early 3rd trimester |
| **32** | 1,675 | 1442.0 | 1984.0 | 3345.0 | 2192.8 | 731.3 | Late preterm growth monitoring |
| **34** | 4,366 | 1800.0 | 2410.0 | 3370.0 | 2505.7 | 609.8 | Late preterm threshold |
| **37** | 26,228 | 2466.0 | 3033.0 | 3665.6 | 3054.8 | 484.4 | Early term delivery reference |
| **40** | 50,335 | 2945.0 | 3459.0 | 4030.0 | 3473.4 | 440.5 | Full term expected birth weight |
| **42** | 6,563 | 2870.0 | 3435.0 | 4053.0 | 3448.8 | 477.3 | Post-term surveillance |






