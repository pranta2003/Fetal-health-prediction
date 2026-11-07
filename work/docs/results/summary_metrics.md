# HC18 5-Fold Segmentation Out-Of-Fold Summary

## Evaluation Setup
- **Architecture**: EfficientNet-B4 U-Net (384x384 input resolution)
- **Checkpoints**: Run B with `--inner-holdout 0.1` (`fm_B/hc18_seg_fold{1..5}_eff_b4_384.pt`)
- **Validation**: Patient-grouped 5-fold cross-validation (`make_grouped_folds(names, 5, 42)`)
- **Inference**: Test-Time Augmentation (TTA: original + horizontal flip), robust ellipse fitting (`fit_ellipse_robust`)

## Per-Fold & Pooled Biometric Performance

| Fold | Images | Evaluated Patients | HC MAE (mm) | BPD MAE (mm) |
|:---|:---:|:---:|:---:|:---:|
| Fold 1 | 206 | 146 | 2.48 | 0.70 |
| Fold 2 | 199 | 146 | 2.29 | 0.81 |
| Fold 3 | 208 | 146 | 2.16 | 0.66 |
| Fold 4 | 194 | 146 | 2.24 | 0.77 |
| Fold 5 | 192 | 146 | 2.33 | 0.84 |
| **Pooled OOF** | **999** | **730** | **2.30** | **0.75** |

- **Pooled HC RMSE**: 3.58 mm
- **Pooled BPD RMSE**: 1.09 mm
- **Patient-Level Bootstrap 95% CI (1000 resamples)**: [2.18, 2.42] mm (Pooled HC MAE)
- **Tolerance Verification**: All 4 metrics match target within tolerance $\le 0.02$ mm.
