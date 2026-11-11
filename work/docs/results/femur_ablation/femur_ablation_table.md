# Femur Model Ablation Study (FP + UCL Femur Test Set, N=348 images / 266 patients)

> Reference FL: Euclidean distance between annotated endpoints * px_to_mm_rate.
> Protocol: Train on 451 images, early stop on 65 validation images (0 patient overlap).
> Evaluation: 348 official test images (323 FP / 261 patients, 25 UCL / 5 patients [tiny]).
> 95% CIs: Patient-level bootstrap with 1,000 resamples.
> NME: Mean Euclidean landmark error normalized by ground-truth femur length.

| Approach | Row / Model | Resolution | Epochs | Train Time (s) | FL MAE [95% CI] (mm) | FL RMSE [95% CI] (mm) | FL Median [95% CI] (mm) | % Error [95% CI] | FL NME (mean+-SD) | Top-10 SqErr Share (%) | FP MAE (mm) | UCL MAE (mm, tiny) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Row A (U-Net EffNet-B4 1-ch Heatmap, flip TTA) | *Seeds Mean +- SD* | 384 px | 17.7 +- 2.9 | 330.4 +- 26.9 | 1.210 +- 0.051 | 3.708 +- 0.078 | 0.597 +- 0.049 | 3.11% +- 0.14% | 0.036 +- 0.001 | 90.8% +- 0.5% | 1.074 | 2.966 |
| Row A (U-Net EffNet-B4 1-ch Heatmap, flip TTA) | **3-Seed Median Ens** | 384 px | - | - | **1.147** [0.890, 1.485] | **3.540** [1.457, 5.552] | **0.610** [0.519, 0.704] | **2.94%** [2.03%, 4.29%] | **0.037 +- 0.151** | **90.5%** | **1.011** | **2.893** |
| Row B (ConvNeXt-Tiny Direct Reg, Order-Inv) | *Seeds Mean +- SD* | 384 px | 35.0 +- 0.0 | 397.1 +- 3.0 | 4.287 +- 0.415 | 7.318 +- 0.512 | 2.350 +- 0.354 | 10.81% +- 1.12% | 0.135 +- 0.006 | 51.8% +- 2.9% | 3.841 | 10.050 |
| Row B (ConvNeXt-Tiny Direct Reg, Order-Inv) | **3-Seed Median Ens** | 384 px | - | - | **4.149** [3.469, 4.981] | **7.088** [5.299, 9.116] | **2.197** [1.749, 2.658] | **10.40%** [8.45%, 13.02%] | **0.132 +- 0.183** | **51.5%** | **3.712** | **9.788** |
| Row C (U-Net EffNet-B4 512px Heatmap, flip TTA) | *Seeds Mean +- SD* | 512 px | 17.3 +- 5.4 | 507.8 +- 159.4 | 1.526 +- 0.287 | 6.097 +- 3.264 | 0.622 +- 0.010 | 3.97% +- 1.00% | 0.044 +- 0.004 | 90.6% +- 7.8% | 1.330 | 4.063 |
| Row C (U-Net EffNet-B4 512px Heatmap, flip TTA) | **3-Seed Median Ens** | 512 px | - | - | **1.210** [0.969, 1.472] | **2.957** [1.734, 3.972] | **0.632** [0.532, 0.706] | **2.97%** [2.23%, 3.84%] | **0.038 +- 0.131** | **83.3%** | **1.148** | **2.012** |
