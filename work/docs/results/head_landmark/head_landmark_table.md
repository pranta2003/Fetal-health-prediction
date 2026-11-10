# Head Landmark Model Evaluation (FP + UCL Head Test Set)

## 1. Head Test Landmark & Biometry Performance

> Reference BPD & OFD are from annotated calipers Ã— px_to_mm_rate.
> Reference HC is computed from annotated OFD & BPD using Ramanujan's formula (labelled 'approximate HC reference').
> Model: U-Net EfficientNet-B4 (384 letterbox, heatmap JS + length/HC loss, flip TTA).
> Evaluation: Patient-level bootstrap 95% CIs (1000 iterations).

| Source | Model / Evaluation | N (Img / Subj) | BPD MAE [95% CI] (mm) | BPD Bias [95% CI] (mm) | OFD MAE [95% CI] (mm) | OFD Bias [95% CI] (mm) | HC MAE [95% CI] (mm) | HC Bias [95% CI] (mm) | BPD NME (meanÂ±SD) | OFD NME (meanÂ±SD) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP Test** | Seed 42 | 880 / 460 | 0.497 | -0.010 | 1.071 | -0.167 | 1.832 | -0.291 | 0.030Â±0.046 | 0.028Â±0.040 |
| **FP Test** | Seed 43 | 880 / 460 | 0.516 | +0.252 | 1.090 | -0.204 | 1.750 | +0.046 | 0.029Â±0.043 | 0.028Â±0.040 |
| **FP Test** | Seed 44 | 880 / 460 | 0.504 | +0.038 | 1.077 | -0.024 | 1.819 | +0.016 | 0.030Â±0.046 | 0.029Â±0.042 |
| **FP Test** | Seeds Mean Â± SD | 880 / 460 | 0.506Â±0.008 | - | 1.079Â±0.008 | - | 1.800Â±0.036 | - | 0.030Â±0.001 | 0.028Â±0.000 |
| **FP Test** | **3-Seed Median Ens** | 880 / 460 | **0.474** [0.414, 0.545] | +0.091 [+0.023, +0.167] | **1.056** [0.955, 1.168] | -0.108 [-0.232, +0.010] | **1.743** [1.577, 1.920] | -0.045 [-0.242, +0.134] | 0.029Â±0.044 | 0.027Â±0.039 |
| *FP Test (Comparison)* | *HC18 5-Fold Ens* | 880 / 460 | 1.664 | +1.644 | - | - | 7.255 | +7.240 | - | - |
| **UCL Test (tiny)** | Seeds Mean Â± SD | 35 / 5 | 1.070Â±0.455 | - | 2.315Â±0.717 | - | 4.806Â±1.727 | - | 0.033Â±0.002 | 0.029Â±0.002 |
| **UCL Test (tiny)** | **3-Seed Median Ens** | 35 / 5 | **0.833** [0.284, 1.220] | -0.393 [-0.929, +0.380] | **2.616** [0.433, 3.499] | -1.547 [-2.613, +0.329] | **4.916** [0.985, 6.766] | -3.026 [-5.457, +1.102] | 0.033Â±0.066 | 0.028Â±0.053 |
| *UCL Test (Comparison)* | *HC18 5-Fold Ens* | 35 / 5 | 0.985 | +0.594 | - | - | 3.664 | +1.316 | - | - |
| **All Test (Combined)** | **3-Seed Median Ens (All)** | 915 / 465 | **0.488** [0.420, 0.558] | +0.072 [-0.002, +0.155] | **1.115** [0.997, 1.264] | -0.163 [-0.326, -0.026] | **1.864** [1.650, 2.131] | -0.159 [-0.454, +0.085] | 0.029Â±0.045 | 0.027Â±0.040 |
| **All Test (Combined)** | **3-Seed Median Ens (No Flagged)** | 915 / 465 | **0.488** [0.420, 0.558] | +0.072 [-0.002, +0.155] | **1.115** [0.997, 1.264] | -0.163 [-0.326, -0.026] | **1.864** [1.650, 2.131] | -0.159 [-0.454, +0.085] | 0.029Â±0.045 | 0.027Â±0.040 |

## 2. Excluded Training Outliers

| Image Name | Subject | Source | Split | Annotated BPD (mm) | Annotated OFD (mm) | Annotated HC (mm) | Reason |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Patient00957_Plane3_2_of_3.png` | FP_957 | FP | train | 0.21 | 0.64 | 1.42 | collapsed calipers (<10mm BPD/OFD or <20mm HC) |
| `Patient00955_Plane3_2_of_3.png` | FP_955 | FP | train | 0.24 | 0.49 | 1.18 | collapsed calipers (<10mm BPD/OFD or <20mm HC) |
| `Patient01070_Plane3_2_of_6.png` | FP_1070 | FP | train | 0.32 | 0.87 | 1.97 | collapsed calipers (<10mm BPD/OFD or <20mm HC) |
| `Patient01738_Plane3_2_of_2.png` | FP_1738 | FP | train | 18.55 | 34.01 | 84.37 | 84mm vs 152mm annotated outlier |