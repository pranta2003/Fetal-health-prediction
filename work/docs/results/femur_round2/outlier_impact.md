# Femur Landmark FL MAE: Impact of Outlier Image (005_3FL.jpeg)

Comparison of Femur Length (FL) Mean Absolute Error (MAE) across all 348 test images versus excluding 
the single visual outlier `005_3FL.jpeg` (UCL test set), with patient-level 95% bootstrap confidence intervals.

| Model Architecture / Run | All Images (N) | All Images FL MAE (mm, 95% CI) | Excl. Outlier (N) | Excl. Outlier FL MAE (mm, 95% CI) | MAE Impact (mm, %) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Row A (PR #32)** | 348 | **1.1466** [0.8895, 1.4848] | 347 | **0.9851** [0.8505, 1.1504] | **-0.1615** (-14.1%) |
| **Row A (round 2)** | 348 | **1.3496** [1.0654, 1.6924] | 347 | **1.1881** [1.0121, 1.3911] | **-0.1616** (-12.0%) |
| **Row R4 (round 2)** | 348 | **1.1179** [0.8560, 1.4588] | 347 | **0.9561** [0.8111, 1.1158] | **-0.1617** (-14.5%) |
| **Row R2 (round 2)** | 348 | **1.0711** [0.9110, 1.2616] | 347 | **1.0632** [0.9022, 1.2577] | **-0.0079** (-0.7%) |

## Key Observations
- For dual-peak heatmap models (**Row A PR #32**, **Row A round 2**, **Row R4**), removing `005_3FL.jpeg` reduces FL MAE by exactly **-0.162 mm** (~12-14% reduction).
- In contrast, for the soft-argmax coordinate model (**Row R2**), removing `005_3FL.jpeg` changes FL MAE by only **-0.008 mm** (-0.7% change), confirming soft-argmax resilience against single-image peak ambiguity.
