# Multi-Modal Fetal Health Benchmark Summary

Comprehensive benchmark report comparing multimodal stream models (Head HC18, Abdomen AC, Femur FL, and Clinical Tabular Risk) against published literature under strictly verified protocols.

## 1. Supervisor Benchmark Summary

| Task | Metric | Ours (value, CI, protocol, source file) | Published (value, source, protocol) | Comparable? (Yes / Partly / No + reason) | What we may claim |
|:---|:---|:---|:---|:---|:---|
| **Head (HC18)** | HC MAE | **2.30 mm** (95% CI: [2.13, 2.48])<br>• Protocol: 5-fold grouped patient CV (999 images)<br>• Source: `docs/results/hc18/metrics.json` | **1.90 +- 1.77 mm** (Fiorentino2020 (with post-proc))<br>• Mask-R2CNN: 1.95 +- 1.92 mm<br>• Range: 1.90–2.80 mm across 6 models<br>• Source: [Mask-R2CNN (PMC8580944)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8580944/)<br>• Protocol: Official HC18 challenge 335-image test set, HC mean absolute difference (MAD) in mm (mean +- SD) | No (5-fold patient CV on 999 training images vs official 335 challenge test set) | Claim only 'same range' (Ours: 2.30 mm vs published 1.90–2.80 mm); cannot claim superior/inferior due to different evaluation sets. |
| **Abdomen (AC)** | Landmark NME (TAD, APAD) | **TAD NME: 0.076 +- 0.060**<br>**APAD NME: 0.075 +- 0.061** (4-model median ensemble)<br>• Protocol: 147 official test images (114 patients)<br>• Source: `docs/results/ac/ac_v3_ens/results.json` | Combined->Combined:<br>• APAD: **0.07 +- 0.08**<br>• TAD: **0.08 +- 0.08**<br>FP->FP: APAD 0.08 +- 0.06, TAD 0.08 +- 0.06<br>• Source: [arXiv 2512.16710, Table 2](https://arxiv.org/abs/2512.16710)<br>• Protocol: NME +- SD, model trained on combined data, tested on combined; FP->FP | Partly (same NME formula; test set split sizes not stated in publication) | Claim 'on par', not 'better' (Ours [4-model median ensemble]: TAD 0.076 +- 0.060, APAD 0.075 +- 0.061 vs Published: TAD 0.08 +- 0.08, APAD 0.07 +- 0.08). |
| **Abdomen (AC)** | AC MAE (mm) | **4.34 mm** (4-model median ensemble)<br>• Protocol: 147 official test images (114 patients)<br>• Source: `docs/results/ac/ac_v3_ens/results.json` | **Not reported in mm**<br>• Dataset paper reports NME only<br>• Source: [arXiv 2512.16710, Table 2](https://arxiv.org/abs/2512.16710)<br>• Note: reports NME, not AC in mm; test set sizes not stated. | Partly (different metric: mm vs NME; paper does not report mm) | Direct mm comparison unavailable in published baseline; established absolute geometric benchmark (Ours: 4.34 mm [4-model median ensemble]). |
| **Femur (FL)** | FL NME | **Data verification complete**<br>• Model training pending<br>• Protocol: Multi-center FP+UCL cohort<br>• Source: `scripts/15_femur_data_report.py` | **FL NME: 0.03 +- 0.09** (m-c->m-c)<br>• Source: [arXiv 2512.16710, Table 2](https://arxiv.org/abs/2512.16710)<br>• Protocol: FL NME m-c->m-c (multi-center to multi-center) | No (model training pending; data check complete) | model not trained; no comparison yet |
| **Maternal Risk (Stream 3)** | E1 Test AUC (Primary, >=20 wk) | **0.7381** (95% CI: [0.7358, 0.7413])<br>• Protocol: Temporal split: Train <=2021 (402,282 rows), Test >=2022 (130,110 rows), COMBGEST >=20<br>• Source: `docs/results/stream3_bench/benchmark_results.json` | CDC early stillbirth: LR 0.73, SELU 0.75, LightGBM 0.75<br>CDC late stillbirth: LR 0.58, SELU 0.59, LightGBM 0.60<br>• Source: [Koivu and Sairanen 2020 (Springer Health Inf Sci Syst)](https://link.springer.com/article/10.1007/s13755-020-00105-9) (DOI: 10.1007/s13755-020-00105-9)<br>• Protocol: RANDOM split 70/10/10/10 + external NYC set; not temporal (CDC NVSS 2013-2016) | Partly (random vs temporal split, different years [2013-2016 vs <=2021/>=2022], different outcome definitions) | same order of magnitude; not directly comparable (Ours: 0.7381 [95% CI: 0.7358, 0.7413] vs Published early stillbirth: 0.73–0.75).<br>Warning: term-band AUC 0.748 must not be compared with their late-stillbirth 0.60; possible source-file artifact. |
| **Maternal Risk (Stream 3)** | E4 Test AUC (Clinical-Only, >=20 wk) | **0.7070** (95% CI: [0.7040, 0.7101])<br>• Protocol: Strictly pre-delivery clinical features (no GA/sex)<br>• Source: `docs/results/stream3_bench/benchmark_results.json` | No clinical-only (GA-free) subset reported in Koivu et al. 2020 (all published models include GA/timing)<br>• Source: [Koivu and Sairanen 2020 (Springer Health Inf Sci Syst)](https://link.springer.com/article/10.1007/s13755-020-00105-9) (DOI: 10.1007/s13755-020-00105-9) | No (random vs temporal split, different years, different feature subsets and outcome definitions) | same order of magnitude; not directly comparable (Ours clinical-only: 0.7070 [95% CI: 0.7040, 0.7101]). |

---

## 2. HC18 Historical Approach Ladder

Progression of architectural families re-evaluated under strict, leak-free protocols.

### A. Held-Out Fold 1 Comparison (206 Held-Out Images)

| Approach Row | Model Family / Cell | Input Res | HC MAE (mm) | HC RMSE (mm) | BPD MAE (mm) | BPD RMSE (mm) | Epochs Trained | Fit Time (s) |
|:---|:---|:---:|---:|---:|---:|---:|:---:|---:|
| Row A: Direct Regression (Cell 22) | Direct continuous scalar regression | 224x224 (direct) | **10.67** | 15.97 | 3.26 | 4.73 | 25 | 695.4 |
| Row B: Geometry Approach v3 (Cell 27) | Ellipse parameter regression (Huber loss) | 224x224 (letterbox) | **11.60** | 16.18 | 3.22 | 4.54 | 20 | 307.4 |
| Row C: Multi-Task U-Net (Cell 25) | Segmentation + shared-encoder ellipse regression | 224x224 (direct) | **10.51** | 15.51 | 2.95 | 4.32 | 52 | 548.1 |
| Row D: ResNet-34 U-Net (Cell 31 recipe) | Segmentation mask + robust native ellipse fit | 224x224 (letterbox) | **1.92** | 2.44 | 0.60 | 0.83 | 48 | 495.6 |
| Row E: EfficientNet-B4 U-Net 384 (Stream 1 Final) | High-res segmentation + TTA + fitEllipseAMS | 384x384 (letterbox) | **2.48** | 5.10 | 0.70 | 0.92 | — | — |

> **Note on Fold-1 Re-runs**: Fold-1 numbers are single runs. The D recipe gave 1.92 in run 1 and 2.17 in run 2 on fold 1 (GPU training is not bit-reproducible); differences under about 0.3 mm are noise.

### B. 5-Fold Cross-Validation: Row D vs Row E (All 999 Images)

| Approach | Prediction TTA | Pooled HC MAE (mm) | Pooled HC RMSE (mm) | Pooled BPD MAE (mm) | Pooled BPD RMSE (mm) |
|:---|:---:|---:|---:|---:|---:|
| Row D (Segmentation Recipe) | None | **2.2688** | 3.9175 | 0.7560 | 1.1999 |
| Row D (Segmentation Recipe) | Flip-TTA | **2.2323** | 3.9700 | 0.7560 | 1.4066 |
| Row E (Final Production Model) | Flip-TTA | **2.3019** | 3.5797 | 0.7535 | 1.0938 |

> **Statistical Equivalence Test (Row D+TTA vs Row E)**:
> - All 999 images: Paired difference in HC MAE = **-0.0696 mm** (Patient-level 95% Bootstrap CI: [-0.2299, 0.1045]).
> - Excluding outlier 065_HC.png (998 images): Paired difference = **-0.0735 mm** (95% CI: [-0.2374, 0.0890]).
> 
> **Conclusion**: Because both 95% confidence intervals cross zero, the historical segmentation recipe (Row D) and our final production model (Row E) perform equivalently within statistical noise.


---

## 3. Stream 3 Clinical Risk Model Family Comparison

Comparative benchmark across four major machine learning families evaluated under identical temporal splits (Train <= 2021: 402,282 rows; Test >= 2022: 130,110 rows; Gestation >= 20 weeks):

| Setting | Model Family | Test AUC | 95% Bootstrap CI | Fit Time (s) | Historical Match |
|:---|:---|:---:|:---:|---:|:---:|
| E1 (Primary, >=20 wk) | Logistic Regression | **0.6659** | [0.6629, 0.6691] | 5.7 | — |
| E1 (Primary, >=20 wk) | Random Forest | **0.7322** | [0.7297, 0.7354] | 50.5 | — |
| E1 (Primary, >=20 wk) | LightGBM | **0.7380** | [0.7357, 0.7413] | 10.8 | — |
| E1 (Primary, >=20 wk) | Our XGBoost | **0.7381** | [0.7358, 0.7413] | 22.6 | Target 0.738 (Exact) |
| E4 (Clinical-Only, >=20 wk) | Logistic Regression | **0.6585** | [0.6557, 0.6617] | 4.8 | — |
| E4 (Clinical-Only, >=20 wk) | Random Forest | **0.7043** | [0.7013, 0.7075] | 41.8 | — |
| E4 (Clinical-Only, >=20 wk) | LightGBM | **0.7070** | [0.7041, 0.7101] | 12.0 | — |
| E4 (Clinical-Only, >=20 wk) | Our XGBoost | **0.7070** | [0.7040, 0.7101] | 19.5 | Target 0.707 (Exact) |

> **Key Takeaway for Supervisor**: Gradient boosted trees (XGBoost / LightGBM) demonstrate clear superiority over linear models (+0.072 AUC) and tree ensembles (+0.006 AUC over Random Forest), while maintaining strict freedom from post-delivery data leakage.

