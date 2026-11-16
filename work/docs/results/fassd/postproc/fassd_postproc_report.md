# FASSD Multi-Label Post-Processing Benchmark & Validation Report

## 1. Study Overview

- **Models Evaluated**:
  - `fassd_gray_seed42`: U-Net (`efficientnet-b4`) trained with single-channel grayscale input.
  - `fassd_rgb_seed42`: U-Net (`efficientnet-b4`) trained with standard 3-channel RGB input.
- **Data Splits** (Leakage-Free `split_v2.csv`):
  - **Validation Set**: 254 images across 26 patient groups.
  - **Test Set**: 271 images across 26 patient groups.
- **Resolution**: Inference performed at native $(768 \times 1024)$ resolution.
- **Empty Ground Truth Protocol**:
  - For each anatomical structure, images with 0 annotated pixels (`gt_empty == True`) are evaluated separately.
  - Detection status is scored as `empty_detected = (pred_pixels == 0)`. The non-empty prediction rate (`1.0 - detection_rate`) measures false positive blobs on empty ground truth.
  - Empty images are excluded from standard Dice, IoU, and HD95 averages to prevent skewing and ensure clinical validity.
- **Macro Dice Metric Clarification**:
  - **Per-Image Macro Dice**: Average Dice across non-empty structures present on each individual image.
  - **Image-Level Mean Macro Dice**: Mean of per-image macro Dice values across all images in the split.
  - **Structure-Level Macro Dice**: Unweighted arithmetic mean of the 4 individual structure Dice averages.
  - **Patient-Level Mean Macro Dice**: Grouped mean of per-image macro Dice values by patient, used for bootstrap resampling.
- **Post-Processing Protocol**:
  - Hyperparameters tuned **strictly on VAL**: threshold $T \in [0.30..0.70]$ and morphology (none, largest component, min area filter).
  - Selected per-structure settings applied **once** to TEST.
  - Statistical significance assessed via **patient-level paired bootstrap** (5,000 resamples, seed 42) for $\Delta = \text{postproc} - \text{baseline}$.

## 2. Model: `fassd_gray_seed42`

### 2.1 Selected Validation Configurations (Tuned on VAL Only)

| Structure | Selected Threshold | Morphology Filter | Min Area (px) | VAL Dice (Tuned) |
|---|---|---|---|---|
| **Artery** | 0.60 | `largest` | 0 | 0.7091 |
| **Liver** | 0.30 | `none` | 0 | 0.8563 |
| **Stomach** | 0.35 | `min_area_200` | 200 | 0.7644 |
| **Vein** | 0.50 | `largest` | 0 | 0.7996 |

### 2.2 Validation Set (254 images) Results

#### Structure-by-Structure Performance

| Structure | Baseline Dice | PostProc Dice | Paired $\Delta$ Dice (95% CI) | Baseline HD95 (px) | PostProc HD95 (px) | Paired $\Delta$ HD95 (95% CI) |
|---|---|---|---|---|---|---|
| **Artery** | 0.7081 ± 0.2660 | 0.7091 ± 0.2699 | +0.0018 [-0.0012, +0.0047] | 25.14 ± 114.61 | 24.51 ± 114.63 | -0.72 [-1.49, -0.07] |
| **Liver** | 0.8549 ± 0.1529 | 0.8563 ± 0.1533 | +0.0016 [-0.0001, +0.0033] | 31.89 ± 41.65 | 31.92 ± 42.04 | -0.01 [-0.39, +0.39] |
| **Stomach** | 0.7636 ± 0.2358 | 0.7644 ± 0.2352 | +0.0002 [-0.0018, +0.0020] | 26.56 ± 50.50 | 26.56 ± 50.40 | +0.01 [-0.14, +0.17] |
| **Vein** | 0.7958 ± 0.1454 | 0.7996 ± 0.1513 | +0.0037 [+0.0015, +0.0063] | 24.49 ± 83.26 | 19.25 ± 80.86 | -4.95 [-9.16, -1.68] |

#### Empty Ground Truth & False Positive Rates

| Structure | Total Empty GT Images | Baseline Detected Empty | Baseline FP Rate | PostProc Detected Empty | PostProc FP Rate |
|---|---|---|---|---|---|
| **Artery** | 0 | 0 / 0 (100.0%) | 0.0% | 0 / 0 (100.0%) | 0.0% |
| **Liver** | 0 | 0 / 0 (100.0%) | 0.0% | 0 / 0 (100.0%) | 0.0% |
| **Stomach** | 25 | 0 / 25 (0.0%) | 100.0% | 0 / 25 (0.0%) | 100.0% |
| **Vein** | 1 | 0 / 1 (0.0%) | 100.0% | 0 / 1 (0.0%) | 100.0% |

#### Overall Summary & Patient-Level Bootstrap

- **Structure-Mean Macro Dice**: Baseline = 0.7806 -> PostProc = 0.7824
- **Per-Image Macro Dice Mean**: Baseline = 0.7818 -> PostProc = 0.7835
- **Patient-Level Macro Dice Mean**: Baseline = 0.7761 -> PostProc = 0.7780
- **Patient-Level Paired $\Delta$ Dice (5000 resamples, 95% CI)**: `+0.0019 [+0.0010, +0.0028]`
- **Patient-Level Macro HD95 Mean**: Baseline = 27.17 px -> PostProc = 25.76 px
- **Patient-Level Paired $\Delta$ HD95 (5000 resamples, 95% CI)**: `-1.41 px [-2.50, -0.54]`

### 2.2 Test Set (271 images - Evaluated Once) Results

#### Structure-by-Structure Performance

| Structure | Baseline Dice | PostProc Dice | Paired $\Delta$ Dice (95% CI) | Baseline HD95 (px) | PostProc HD95 (px) | Paired $\Delta$ HD95 (95% CI) |
|---|---|---|---|---|---|---|
| **Artery** | 0.7592 ± 0.1807 | 0.7589 ± 0.1845 | +0.0004 [-0.0025, +0.0035] | 10.41 ± 15.57 | 9.35 ± 8.27 | -0.93 [-2.40, -0.02] |
| **Liver** | 0.8777 ± 0.0702 | 0.8780 ± 0.0707 | -0.0000 [-0.0020, +0.0018] | 24.59 ± 14.45 | 24.82 ± 14.66 | +0.25 [-0.13, +0.62] |
| **Stomach** | 0.7400 ± 0.2511 | 0.7413 ± 0.2507 | +0.0009 [-0.0008, +0.0027] | 21.54 ± 24.33 | 21.70 ± 24.53 | +0.16 [-0.04, +0.42] |
| **Vein** | 0.7729 ± 0.1397 | 0.7714 ± 0.1483 | -0.0005 [-0.0041, +0.0031] | 24.29 ± 79.85 | 23.77 ± 79.84 | -0.54 [-1.97, +0.94] |

#### Empty Ground Truth & False Positive Rates

| Structure | Total Empty GT Images | Baseline Detected Empty | Baseline FP Rate | PostProc Detected Empty | PostProc FP Rate |
|---|---|---|---|---|---|
| **Artery** | 1 | 0 / 1 (0.0%) | 100.0% | 0 / 1 (0.0%) | 100.0% |
| **Liver** | 0 | 0 / 0 (100.0%) | 0.0% | 0 / 0 (100.0%) | 0.0% |
| **Stomach** | 8 | 1 / 8 (12.5%) | 87.5% | 1 / 8 (12.5%) | 87.5% |
| **Vein** | 1 | 0 / 1 (0.0%) | 100.0% | 0 / 1 (0.0%) | 100.0% |

#### Overall Summary & Patient-Level Bootstrap

- **Structure-Mean Macro Dice**: Baseline = 0.7874 -> PostProc = 0.7874
- **Per-Image Macro Dice Mean**: Baseline = 0.7876 -> PostProc = 0.7876
- **Patient-Level Macro Dice Mean**: Baseline = 0.7933 -> PostProc = 0.7934
- **Patient-Level Paired $\Delta$ Dice (5000 resamples, 95% CI)**: `+0.0002 [-0.0010, +0.0013]`
- **Patient-Level Macro HD95 Mean**: Baseline = 19.37 px -> PostProc = 19.13 px
- **Patient-Level Paired $\Delta$ HD95 (5000 resamples, 95% CI)**: `-0.25 px [-0.86, +0.26]`

## 2. Model: `fassd_rgb_seed42`

### 2.1 Selected Validation Configurations (Tuned on VAL Only)

| Structure | Selected Threshold | Morphology Filter | Min Area (px) | VAL Dice (Tuned) |
|---|---|---|---|---|
| **Artery** | 0.30 | `largest` | 0 | 0.7080 |
| **Liver** | 0.35 | `largest` | 0 | 0.8570 |
| **Stomach** | 0.30 | `min_area_200` | 200 | 0.7634 |
| **Vein** | 0.40 | `largest` | 0 | 0.8011 |

### 2.2 Validation Set (254 images) Results

#### Structure-by-Structure Performance

| Structure | Baseline Dice | PostProc Dice | Paired $\Delta$ Dice (95% CI) | Baseline HD95 (px) | PostProc HD95 (px) | Paired $\Delta$ HD95 (95% CI) |
|---|---|---|---|---|---|---|
| **Artery** | 0.7055 ± 0.2700 | 0.7080 ± 0.2684 | +0.0014 [-0.0043, +0.0077] | 24.49 ± 114.66 | 24.23 ± 114.66 | -0.13 [-1.13, +0.60] |
| **Liver** | 0.8562 ± 0.1529 | 0.8570 ± 0.1529 | +0.0008 [-0.0005, +0.0021] | 32.23 ± 42.39 | 31.68 ± 41.38 | -0.43 [-1.47, +0.22] |
| **Stomach** | 0.7611 ± 0.2347 | 0.7634 ± 0.2339 | +0.0011 [-0.0017, +0.0036] | 27.63 ± 52.97 | 27.63 ± 52.97 | +0.05 [-0.12, +0.23] |
| **Vein** | 0.7979 ± 0.1455 | 0.8011 ± 0.1435 | +0.0028 [-0.0011, +0.0085] | 32.46 ± 138.54 | 28.81 ± 137.77 | -3.43 [-6.79, -0.84] |

#### Empty Ground Truth & False Positive Rates

| Structure | Total Empty GT Images | Baseline Detected Empty | Baseline FP Rate | PostProc Detected Empty | PostProc FP Rate |
|---|---|---|---|---|---|
| **Artery** | 0 | 0 / 0 (100.0%) | 0.0% | 0 / 0 (100.0%) | 0.0% |
| **Liver** | 0 | 0 / 0 (100.0%) | 0.0% | 0 / 0 (100.0%) | 0.0% |
| **Stomach** | 25 | 0 / 25 (0.0%) | 100.0% | 0 / 25 (0.0%) | 100.0% |
| **Vein** | 1 | 0 / 1 (0.0%) | 100.0% | 0 / 1 (0.0%) | 100.0% |

#### Overall Summary & Patient-Level Bootstrap

- **Structure-Mean Macro Dice**: Baseline = 0.7802 -> PostProc = 0.7824
- **Per-Image Macro Dice Mean**: Baseline = 0.7817 -> PostProc = 0.7839
- **Patient-Level Macro Dice Mean**: Baseline = 0.7764 -> PostProc = 0.7781
- **Patient-Level Paired $\Delta$ Dice (5000 resamples, 95% CI)**: `+0.0017 [-0.0004, +0.0042]`
- **Patient-Level Macro HD95 Mean**: Baseline = 29.88 px -> PostProc = 28.89 px
- **Patient-Level Paired $\Delta$ HD95 (5000 resamples, 95% CI)**: `-0.99 px [-1.95, -0.19]`

### 2.2 Test Set (271 images - Evaluated Once) Results

#### Structure-by-Structure Performance

| Structure | Baseline Dice | PostProc Dice | Paired $\Delta$ Dice (95% CI) | Baseline HD95 (px) | PostProc HD95 (px) | Paired $\Delta$ HD95 (95% CI) |
|---|---|---|---|---|---|---|
| **Artery** | 0.7645 ± 0.1754 | 0.7659 ± 0.1763 | +0.0008 [-0.0047, +0.0062] | 13.95 ± 77.63 | 13.69 ± 77.57 | -0.26 [-0.70, +0.07] |
| **Liver** | 0.8741 ± 0.0802 | 0.8748 ± 0.0883 | +0.0005 [-0.0028, +0.0039] | 27.47 ± 25.36 | 25.56 ± 19.19 | -1.59 [-4.09, +0.17] |
| **Stomach** | 0.7359 ± 0.2486 | 0.7386 ± 0.2489 | +0.0021 [-0.0004, +0.0048] | 26.59 ± 81.51 | 26.53 ± 81.50 | -0.03 [-0.21, +0.18] |
| **Vein** | 0.7717 ± 0.1417 | 0.7694 ± 0.1501 | -0.0021 [-0.0076, +0.0035] | 20.04 ± 24.13 | 19.70 ± 23.44 | -0.32 [-2.85, +2.01] |

#### Empty Ground Truth & False Positive Rates

| Structure | Total Empty GT Images | Baseline Detected Empty | Baseline FP Rate | PostProc Detected Empty | PostProc FP Rate |
|---|---|---|---|---|---|
| **Artery** | 1 | 0 / 1 (0.0%) | 100.0% | 0 / 1 (0.0%) | 100.0% |
| **Liver** | 0 | 0 / 0 (100.0%) | 0.0% | 0 / 0 (100.0%) | 0.0% |
| **Stomach** | 8 | 0 / 8 (0.0%) | 100.0% | 0 / 8 (0.0%) | 100.0% |
| **Vein** | 1 | 0 / 1 (0.0%) | 100.0% | 0 / 1 (0.0%) | 100.0% |

#### Overall Summary & Patient-Level Bootstrap

- **Structure-Mean Macro Dice**: Baseline = 0.7866 -> PostProc = 0.7872
- **Per-Image Macro Dice Mean**: Baseline = 0.7866 -> PostProc = 0.7871
- **Patient-Level Macro Dice Mean**: Baseline = 0.7925 -> PostProc = 0.7927
- **Patient-Level Paired $\Delta$ Dice (5000 resamples, 95% CI)**: `+0.0002 [-0.0020, +0.0027]`
- **Patient-Level Macro HD95 Mean**: Baseline = 21.32 px -> PostProc = 20.79 px
- **Patient-Level Paired $\Delta$ HD95 (5000 resamples, 95% CI)**: `-0.54 px [-1.70, +0.32]`
