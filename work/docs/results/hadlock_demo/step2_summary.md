# Hadlock Demo Step 2: Multi-Plane EFW Benchmark Report

Evaluation of Estimated Fetal Weight (EFW) using Hadlock's 4-parameter formula ($EFW = f(BPD, HC, AC, FL)$) on the strictly held-out 3-plane test cohort ($N = 22$ subjects: 17 FP + 5 UCL).

## 1. Overall Cohort Performance (Primary: Row R4)

| Cohort / Split | N | MAE (g, 95% CI) | MAPE (%, 95% CI) | Bias (%, 95% CI) | Median Abs Error (g) | Max Abs Error (g) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **All Subjects (Primary R4)** | 22 | **46.43** [25.16, 70.01] | **2.92%** [2.03, 3.86] | -1.36% [-2.76, 0.08] | 18.66 | 140.43 |
| • FP Cohort | 17 | 50.68 [25.97, 77.68] | 2.92% [1.87, 3.97] | -1.10% [-2.80, 0.50] | 19.04 | 140.43 |
| • UCL Cohort | 5 | 31.97 [7.54, 67.35] | 2.91% [1.27, 4.70] | -2.24% [-4.69, 0.08] | 18.27 | 102.10 |
| **Excluding subject with the femur outlier (005_3FL)** | 21 | **47.77** [26.23, 72.35] | **2.97%** [2.08, 3.95] | -1.50% [-2.98, -0.06] | 19.04 | 140.43 |

> **Note on Sample Size**: The cohort size is small ($N=22$ overall; FP=17, UCL=5) because it strictly enforces zero cross-plane training leakage. The 95% bootstrap confidence intervals reflect this limited sample size.

## 2. One-at-a-Time Measurement Replacement Analysis

Starting from annotated ground-truth EFW, replacing each individual measurement by its model prediction reveals which plane drives EFW error:

| Replacement Condition | Mean Abs EFW Error (%) | single replacement as % of full replacement error |
|:---|:---:|:---:|
| Replace **BPD** only | 0.12% | 4.1% |
| Replace **HC** only | 0.35% | 11.8% |
| Replace **AC** only | **2.42%** | **83.1%** |
| Replace **FL** only | 1.69% | 57.9% |
| Replace **All Four** (Full Model) | **2.92%** | 100.0% |

> **Note on Replacement Shares**: The individual single replacement error shares do not add up to 100% because measurement errors across planes partly cancel each other out.

> **Error Driver Finding**: From the numbers, **AC** drives the error (2.42% MAPE), followed by FL (1.69%), HC (0.35%), and BPD (0.12%).

## 3. Outlier Audit: UCL_5 & Image 005_3FL.jpeg

- **Subject**: `UCL_5` (UCL)
- **Image Counts**: Head = 15, Abdomen = 13, Femur = 16 (contains `005_3FL.jpeg`)
- **Annotated EFW**: 1080.3 g vs **Predicted EFW**: 1098.6 g
- **Subject EFW Error**: **+18.3 g** (+1.7%)
- **Impact Assessment**: The subject's own EFW error is small (+18.3 g, +1.7%). Because UCL_5 has 16 femur images, subject-level averaging dampens the single-image dual-peak flip of `005_3FL.jpeg`. Excluding this subject with the femur outlier changes cohort MAPE marginally from 2.92% to 2.97% (+0.05%).

## 4. Sensitivity Analysis: PR #32 Row A Baseline

| Femur Model Run | All Cohort MAPE (%) | All Cohort MAE (g) | Excl. outlier subject MAPE (%) | Excl. outlier subject MAE (g) |
|:---|:---:|:---:|:---:|:---:|
| **Row R4 (Modern Augmentation)** | **2.92%** | **46.43 g** | **2.97%** | **47.77 g** |
| **Row A (PR #32 Baseline)** | 3.10% | 47.39 g | 3.16% | 48.67 g |

> **Note**: Row R4 is not shown to be statistically superior to Row A (difference is 0.18% MAPE, 0.96 g MAE), which is within run-to-run baseline drift.

## 5. Context with Hadlock Published Reference Standard

- **Source Reference**: `src/fetal/explain/hadlock.py:23` (`SD as a uniform 12.7% coefficient of variation, per the paper."""`)
- **Hadlock Formula Residual SD**: **12.7%** coefficient of variation (Hadlock et al. 1991).
- **Model Measurement Propagation Error**: **2.92%** MAPE (22 subjects).
- **Assessment**: The measurement propagation error is substantially smaller than Hadlock's intrinsic formula modeling error (2.92% vs 12.7% CV). (Note: not a benchmark comparison; reporting relative magnitude only).
