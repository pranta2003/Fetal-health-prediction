# HC18 Row D (ResNet-34 U-Net 224) 5-Fold Benchmark vs Row E

Evaluated on all 999 images across all 5 grouped folds (`make_grouped_folds`, seed 42) with inner 10% patient-isolation validation.

## Per-Fold Performance

| Fold | N Images | D HC MAE (mm) | D+TTA HC MAE (mm) | E HC MAE (mm) | D BPD MAE (mm) | D+TTA BPD MAE (mm) | E BPD MAE (mm) |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 206 | 2.1662 | 2.1578 | 2.4817 | 0.6874 | 0.6821 | 0.6982 |
| 2 | 199 | 2.2055 | 2.3016 | 2.2939 | 0.7359 | 0.8030 | 0.8052 |
| 3 | 208 | 2.3151 | 2.2983 | 2.1582 | 0.7246 | 0.7174 | 0.6586 |
| 4 | 194 | 2.3044 | 2.3258 | 2.2446 | 0.8484 | 0.8319 | 0.7726 |
| 5 | 192 | 2.3582 | 2.0746 | 2.3311 | 0.7913 | 0.7517 | 0.8430 |
| **Pooled** | **999** | **2.2688** | **2.2323** | **2.3019** | **0.7560** | **0.7560** | **0.7535** |

## Pooled Metrics Summary

| Model / Approach | Input Res | TTA | Pooled HC MAE (mm) | Pooled HC RMSE (mm) | Pooled BPD MAE (mm) | Pooled BPD RMSE (mm) |
|:---|:---:|:---:|---:|---:|---:|---:|
| Row D: ResNet-34 U-Net | 224x224 | No | **2.2688** | 3.9175 | **0.7560** | 1.1999 |
| Row D: ResNet-34 U-Net (+TTA) | 224x224 | Yes | **2.2323** | 3.9700 | **0.7560** | 1.4066 |
| Row E: EfficientNet-B4 U-Net 384 | 384x384 | Yes | **2.3019** | 3.5797 | **0.7535** | 1.0938 |

## Paired Error Differences (D minus E) with Patient-Level Bootstrap 95% CI

| Comparison | Sample | $\Delta$ HC MAE (mm) | 95% Bootstrap CI (mm) |
|:---|:---|---:|:---:|
| Row D (no TTA) vs Row E | All 999 images | -0.0332 | [-0.1922, 0.1272] |
| Row D (no TTA) vs Row E | Excl. 065_HC.png (998 images) | -0.0347 | [-0.1920, 0.1306] |
| Row D (+TTA) vs Row E | All 999 images | -0.0696 | [-0.2299, 0.1045] |
| Row D (+TTA) vs Row E | Excl. 065_HC.png (998 images) | -0.0735 | [-0.2374, 0.0890] |
