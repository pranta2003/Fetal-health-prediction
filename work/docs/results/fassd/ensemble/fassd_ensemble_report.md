# FASSD Multi-Label Segmentation Ensemble Benchmark Report

## 1. Study Configuration

- **Models Evaluated**: 5 Single-Channel Grayscale Models
  1. `fassd_gray_seed42`: Run 1 U-Net (`efficientnet-b4`, seed 42)
  2. `run2_unet_effb4`: Run 2 U-Net (`efficientnet-b4`, seed 42)
  3. `run2_unetplusplus_effb4`: Run 2 U-Net++ (`efficientnet-b4`, seed 42)
  4. `run2_unet_resnet34`: Run 2 U-Net (`resnet34`, seed 42)
  5. `run2_fpn_effb4`: Run 2 FPN (`efficientnet-b4`, seed 42)
- **Modality Uniformity**: Run 1 RGB is excluded to maintain strictly uniform single-channel grayscale preprocessing.
- **Dataset Split**: `split_v2.csv` (VAL = 254 images, TEST = 271 images).
- **Ensemble Selection Protocol**: Evaluated on VAL with macro Dice only; TEST evaluated once on the chosen ensemble.
- **Bootstrap Evaluation**: Patient-level paired bootstrap (5,000 resamples, seed 42).

## 2. Validation Candidate Comparison (Selection by VAL Macro Dice)

| Candidate Configuration | Models Included | VAL Macro Dice | Per-Image Mean Dice | Status |
|:---|:---|:---:|:---:|:---:|
| **All 5 Gray Models** | fassd_gray_seed42, run2_unet_effb4, run2_unetplusplus_effb4, run2_unet_resnet34, run2_fpn_effb4 | 0.7841 | 0.7854 | Candidate |
| **Top-3 by VAL Dice** | fassd_gray_seed42, run2_unet_resnet34, run2_unet_effb4 | 0.7845 | 0.7857 | Candidate |
| **Top-2 by VAL Dice** | fassd_gray_seed42, run2_unet_resnet34 | 0.7848 | 0.7862 | **Chosen Winner** |

**Chosen Ensemble**: `Top-2 by VAL Dice` (top2_val) selected strictly by highest validation macro Dice (`0.7848`).
**Best Single Model on VAL**: `Run 1 Gray Seed 42 (UNet EffB4)` (fassd_gray_seed42) with VAL macro Dice = `0.7806`.

## 3. Test Set Performance (271 images, Evaluated Once)

### Macro Performance Summary

| Evaluation Target | Macro Dice | Macro HD95 (px) | Paired $\Delta$ Dice vs Best Single (95% CI) | Paired $\Delta$ HD95 vs Best Single (95% CI) |
|:---|:---:|:---:|:---:|:---:|
| **Best Single Model** (`Run 1 Gray Seed 42 (UNet EffB4)`) | 0.7874 | 20.21 | Baseline | Baseline |
| **Chosen Ensemble** (Raw, threshold 0.5) | 0.7927 | 21.33 | +0.0057 [+0.0022, +0.0094] | +0.87 [-1.32, +3.79] |
| **Chosen Ensemble + Largest CC** (threshold 0.5) | 0.7905 | 21.69 | +0.0038 [+0.0001, +0.0077] | +1.22 [-1.07, +4.15] |
| **Chosen Ensemble + Part A Settings** | 0.7914 | 22.31 | - | - |

### Structure-by-Structure Breakdown (Test Set)

| Structure | Best Single Dice | Ensemble Dice | Paired $\Delta$ Dice (95% CI) | Best Single HD95 (px) | Ensemble HD95 (px) | Paired $\Delta$ HD95 (95% CI) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Artery** | 0.7592 | 0.7698 | +0.0109 [+0.0037, +0.0188] | 10.41 | 9.05 | -1.23 [-2.62, -0.27] |
| **Liver** | 0.8777 | 0.8814 | +0.0038 [+0.0018, +0.0059] | 24.59 | 23.82 | -0.76 [-1.37, -0.18] |
| **Stomach** | 0.7400 | 0.7445 | +0.0044 [-0.0005, +0.0097] | 21.54 | 20.94 | -0.66 [-1.90, +0.35] |
| **Vein** | 0.7729 | 0.7751 | +0.0037 [-0.0033, +0.0094] | 24.29 | 31.50 | +6.11 [-2.28, +18.02] |

## 4. Key Findings

1. **Ensemble Gain**: Ensembling models by probability averaging achieved a test macro Dice of **0.7927** vs **0.7874** for the best single validation model (`Run 1 Gray Seed 42 (UNet EffB4)`).
2. **Boundary Precision**: The test macro 95% Hausdorff Distance was **21.33 px** for the ensemble vs **20.21 px** for the single model.
3. **Post-Processing Synergy**: Applying fixed largest-component filtering yielded macro Dice of **0.7905** and HD95 of **21.69 px**.
