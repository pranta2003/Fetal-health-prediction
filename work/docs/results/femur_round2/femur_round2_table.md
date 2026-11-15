# Femur Round 2: Modern Techniques (FP + UCL Femur Test Set, N=348 images / 266 patients)

> Reference FL: Euclidean distance between annotated endpoints * px_to_mm_rate.
> Protocol: Train on 451 images, early stop on 65 validation images (0 patient overlap).
> Evaluation: 348 official test images (323 FP / 261 patients, 25 UCL / 5 patients [tiny]).
> Paired Diff: Patient-level bootstrap with 1,000 resamples of difference in FL MAE vs Row A.
> Selection: Best recipe chosen strictly by validation MAE only.

| Approach | Row / Model | Resolution | Val MAE (mm) | Epochs | Train Time (s) | FL MAE [95% CI] (mm) | Paired Diff vs Row A [95% CI] (mm) | FL RMSE [95% CI] (mm) | FL Median [95% CI] (mm) | % Error [95% CI] | FL NME (mean+-SD) | Top-10 SqErr Share (%) | FP MAE (mm) | UCL MAE (mm, tiny) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Row A (Baseline EffNet-B4 1-ch Heatmap, flip TTA) | *Seeds Mean +- SD* | 384 px | 1.150 | 14.3 +- 2.5 | 252.4 +- 34.9 | 1.314 +- 0.162 | - | 3.256 +- 0.777 | 0.704 +- 0.137 | 3.29% +- 0.51% | 0.039 +- 0.003 | 81.8% +- 7.9% | 1.230 | 2.397 |
| Row A (Baseline EffNet-B4 1-ch Heatmap, flip TTA) | **3-Seed Median Ens** | 384 px | 1.150 | - | - | **1.350** [1.065, 1.692] | Reference | **3.786** [1.799, 5.723] | **0.667** [0.597, 0.786] | **3.46%** [2.48%, 4.85%] | **0.038 +- 0.151** | **88.7%** | **1.227** | **2.937** |
| Row R1 (HRNet-W32 Heatmap, DARK Sub-pixel) | *Seeds Mean +- SD* | 384 px | 16.863 | 33.7 +- 1.9 | 338.8 +- 26.5 | 18.213 +- 0.898 | - | 25.911 +- 0.706 | 12.946 +- 1.875 | 40.31% +- 2.55% | 0.424 +- 0.009 | 18.4% +- 0.8% | 18.453 | 15.117 |
| Row R1 (HRNet-W32 Heatmap, DARK Sub-pixel) | **3-Seed Median Ens** | 384 px | 16.863 | - | - | **18.333** [16.119, 20.535] | **16.983** [14.770, 19.191] | **26.143** [23.712, 28.328] | **13.618** [9.349, 16.827] | **40.20%** [36.07%, 44.04%] | **0.415 +- 0.389** | **17.8%** | **18.863** | **11.489** |
| Row R2 (Row-A Net, Integral Soft-Argmax Loss) | *Seeds Mean +- SD* | 384 px | 1.220 | 30.0 +- 3.7 | 512.7 +- 64.0 | 1.223 +- 0.079 | - | 2.351 +- 0.347 | 0.645 +- 0.049 | 2.91% +- 0.26% | 0.043 +- 0.001 | 63.3% +- 6.5% | 1.179 | 1.794 |
| Row R2 (Row-A Net, Integral Soft-Argmax Loss) | **3-Seed Median Ens** | 384 px | 1.220 | - | - | **1.071** [0.911, 1.262] | **-0.279** [-0.640, 0.007] *(no measurable difference)* | **2.017** [1.522, 2.540] | **0.550** [0.493, 0.685] | **2.49%** [2.10%, 2.96%] | **0.040 +- 0.167** | **63.6%** | **1.097** | **0.742** |
| Row R3 (Row-A Net, Bone Line + Length Loss, MinDist=50px) | *Seeds Mean +- SD* | 384 px | 28.293 | 10.7 +- 1.7 | 183.4 +- 29.4 | 43.204 +- 15.958 | - | 50.121 +- 17.762 | 41.482 +- 16.670 | 104.88% +- 39.20% | 1.159 +- 0.304 | 15.9% +- 0.1% | 43.073 | 44.894 |
| Row R3 (Row-A Net, Bone Line + Length Loss, MinDist=50px) | **3-Seed Median Ens** | 384 px | 28.293 | - | - | **47.505** [44.820, 50.539] | **46.155** [43.247, 49.214] | **54.760** [51.881, 57.986] | **46.231** [40.249, 49.928] | **118.92%** [110.00%, 128.88%] | **1.062 +- 0.505** | **15.6%** | **47.713** | **44.809** |
| ★ Row R4 (Row-A Net + Strong Ultrasound Aug) | *Seeds Mean +- SD* | 384 px | 1.070 | 21.0 +- 5.7 | 367.5 +- 100.1 | 1.129 +- 0.090 | - | 3.577 +- 0.197 | 0.553 +- 0.038 | 2.93% +- 0.25% | 0.037 +- 0.003 | 91.0% +- 0.4% | 0.995 | 2.869 |
| ★ Row R4 (Row-A Net + Strong Ultrasound Aug) | **3-Seed Median Ens** | 384 px | 1.070 | - | - | **1.118** [0.856, 1.459] | **-0.232** [-0.398, -0.086] | **3.546** [1.471, 5.507] | **0.538** [0.459, 0.604] | **2.90%** [1.96%, 4.23%] | **0.033 +- 0.141** | **91.1%** | **0.980** | **2.901** |
| Row R6 (Ensemble of Row A + Best 2 Val Rows) | **9-Model Multi-Row Ens** | 384 px | 1.147 | - | - | **1.196** [0.919, 1.537] | **-0.154** [-0.272, -0.038] | **3.590** [1.600, 5.522] | **0.578** [0.500, 0.654] | **3.08%** [2.12%, 4.42%] | **0.036 +- 0.147** | **89.8%** | **1.065** | **2.883** |

### Pretrained Encoder Check (Row R5):
- **DINOv2 ViT-S/14**: Apache 2.0 license, publicly downloadable weights from Facebook Research hub. Incompatible with standard 384px resolution because patch size is 14 (`384 % 14 != 0`), violating the standardized letterbox resolution protocol. Skipped.
- **UltraSam (arXiv 2411.16222)**: Derivative of SAM requiring external heavy checkpoint hosting and 1024px architecture. Skipped due to lack of standard PyTorch distribution.
