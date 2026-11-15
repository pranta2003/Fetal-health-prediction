# HC18 Official Challenge Test Set Benchmark Summary

Official platform evaluation metrics returned by the HC18 Grand Challenge evaluation system for the 335 hidden test images.

## 1. Overall Test Set Performance (335 Images)

| Submission / Model | Images | HC Abs Diff (mm) | Median (mm) | Max (mm) | HC Signed Diff (mm) | Dice (%) | Hausdorff (mm) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ensemble (5-Fold)** | 335 | **1.95 +- 1.91** | 1.38 | 13.69 | +0.78 +- 2.62 | 85.59 +- 3.11 | 7.20 +- 3.21 |
| **Single Model (Fold 1)** | 335 | **1.99 +- 1.91** | 1.39 | 13.93 | +0.70 +- 2.67 | 85.66 +- 3.07 | 7.18 +- 3.20 |

## 2. Trimester Breakdown

| Submission | Trimester | Images | HC Abs Diff (mm) | Dice (%) | Hausdorff (mm) |
|:---|:---|:---:|:---:|:---:|:---:|
| Ensemble (5-Fold) | Trimester 1 | 55 | 1.60 +- 1.61 | 87.91 +- 3.28 | 2.74 +- 0.83 |
| Ensemble (5-Fold) | Trimester 2 | 233 | 1.81 +- 1.67 | 85.17 +- 2.82 | 7.20 +- 1.82 |
| Ensemble (5-Fold) | Trimester 3 | 47 | 3.02 +- 2.84 | 84.98 +- 3.11 | 12.43 +- 2.38 |
| Single Model (Fold 1) | Trimester 1 | 55 | 1.64 +- 1.68 | 87.88 +- 3.28 | 2.75 +- 0.83 |
| Single Model (Fold 1) | Trimester 2 | 233 | 1.83 +- 1.61 | 85.25 +- 2.79 | 7.18 +- 1.80 |
| Single Model (Fold 1) | Trimester 3 | 47 | 3.20 +- 2.90 | 85.09 +- 3.08 | 12.42 +- 2.38 |

## 3. Comparison with Published Baselines

| Model / Architecture | Evaluation Set | HC Absolute Difference / MAD (mm) | Source |
|:---|:---:|:---:|:---|
| **Ours: Ensemble (5-Fold)** | 335 Test Images | **1.95 +- 1.91** | Platform Evaluation (`metrics_ensemble_5fold.json`) |
| **Ours: Single (Fold 1)** | 335 Test Images | **1.99 +- 1.91** | Platform Evaluation (`metrics_single_fold1.json`) |
| Fiorentino2020 (with post-proc) | 335 Test Images | 1.90 +- 1.77 | [Mask-R2CNN (PMC8580944)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8580944/) |
| Mask-R2CNN | 335 Test Images | 1.95 +- 1.92 | [Mask-R2CNN (PMC8580944)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8580944/) |
| Sobhaninia2019 | 335 Test Images | 2.12 +- 1.87 | [Mask-R2CNN (PMC8580944)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8580944/) |

## 4. Formal Claim Statement

> **Statement**: same range as the published values listed; no rank is claimed; test labels are hidden so no paired significance test is possible; ensemble vs single difference = 0.05 mm (no test)

