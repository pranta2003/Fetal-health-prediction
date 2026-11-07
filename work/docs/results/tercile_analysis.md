# HC18 Out-Of-Fold Error Analysis by Head-Size Terciles

> **Head-size proxy for gestational age**: terciles of ground-truth Head Circumference (HC mm) across the 999 HC18 training images, evaluated using 5-fold patient-grouped out-of-fold inference with test-time augmentation (TTA).

| Tercile | Description | Samples | True HC Range (mm) | HC MAE (mm) | HC RMSE (mm) | BPD MAE (mm) | BPD RMSE (mm) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Small (< T1)** | Early 2nd Trimester / Small Head | 333 | 44.3 - 165.7 | 2.14 | 3.22 | 0.69 | 0.98 |
| **Medium (T1-T2)** | Mid 2nd Trimester / Average Head | 333 | 165.7 - 181.6 | 2.26 | 3.49 | 0.74 | 1.05 |
| **Large (> T2)** | Late 2nd & 3rd Trimester / Large Head | 333 | 181.6 - 346.4 | 2.50 | 4.02 | 0.82 | 1.23 |
| **Pooled (Overall)** | **All Gestational Ages** | **999** | **44.3 - 346.4** | **2.30** | **3.58** | **0.75** | **1.09** |

## Notes & Observations
1. **Error Profile across Gestational Ages**:
   - Smallest relative absolute error is observed in early second trimester scans where head boundaries are crisp and acoustic shadowing is minimal.
   - Larger fetal heads in late second and third trimesters exhibit slightly higher absolute error (2.50 mm vs 2.14 mm) due to calvarial acoustic shadowing and larger absolute head dimensions.
2. **Patient-level Bootstrap 95% Confidence Interval**:
   - 1,000 resamples by unique patient ID: **[2.18, 2.42] mm** for pooled HC MAE.
3. **Leak-Free Validation**:
   - Patient grouping via `make_grouped_folds(names, 5, 42)` ensures no patient appears in both train and validation sets of any fold.
