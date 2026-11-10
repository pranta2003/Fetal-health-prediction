# External Validation of HC18 5-Fold Ensemble on FP and UCL Head Ultrasound

## 1. Usable Subjects for Hadlock Demo

| Cohort | Criterion 1(a): Abd Test & Femur Test & Head (any) | Criterion 1(b): Abd Test & Femur Test & Head Test | Images/Subj 1(a) [Head, Abd, Femur] (min, med, max) | Images/Subj 1(b) [Head, Abd, Femur] (min, med, max) |
| :--- | :---: | :---: | :---: | :---: |
| **FP** | **29** | **17** | H: (1,2.0,5) A: (1,1.0,2) F: (1,1.0,2) | H: (1,2.0,5) A: (1,1.0,2) F: (1,1.0,2) |
| **UCL (calibrated)** | **5** | **5** | H: (1,4.0,15) A: (1,3.0,13) F: (1,2.0,16) | H: (1,4.0,15) A: (1,3.0,13) F: (1,2.0,16) |

## 2. External Validation Metrics (HC & BPD against Reference Annotations)

> Note: Reference BPD is derived from annotated BPD landmarks × px_to_mm_rate. Approximate HC reference is computed from annotated OFD and BPD axes using the Ramanujan ellipse perimeter formula.

| Subgroup | N (Images / Subj) | Metric | MAE [95% CI] (mm) | RMSE [95% CI] (mm) | Bias (Signed Mean) [95% CI] (mm) | Median Abs Error [95% CI] (mm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP All** | 1635 / 908 | **HC** | 6.991 [6.647, 7.387] | 10.344 [7.831, 12.747] | +6.928 [+6.573, +7.331] | 5.770 [5.587, 5.973] |
| | | **BPD** | 1.738 [1.645, 1.847] | 2.758 [2.088, 3.430] | +1.633 [+1.528, +1.745] | 1.444 [1.401, 1.500] |
| **FP Train** | 755 / 448 | **HC** | 6.684 [6.021, 7.489] | 12.374 [7.362, 16.748] | +6.564 [+5.862, +7.375] | 5.273 [4.996, 5.571] |
| | | **BPD** | 1.826 [1.651, 2.049] | 3.394 [2.151, 4.549] | +1.621 [+1.426, +1.851] | 1.444 [1.372, 1.517] |
| **FP Test** | 880 / 460 | **HC** | 7.255 [6.959, 7.604] | 8.212 [7.815, 8.636] | +7.240 [+6.936, +7.588] | 6.228 [5.874, 6.575] |
| | | **BPD** | 1.664 [1.575, 1.762] | 2.062 [1.876, 2.292] | +1.644 [+1.555, +1.741] | 1.452 [1.382, 1.516] |
| **UCL All** | 126 / 31 | **HC** | 3.588 [2.417, 4.669] | 4.646 [3.058, 5.846] | +1.662 [-0.426, +3.508] | 2.848 [1.922, 3.955] |
| | | **BPD** | 1.097 [0.715, 1.503] | 1.437 [0.939, 1.844] | +0.786 [+0.302, +1.303] | 0.916 [0.542, 1.548] |
| **UCL Train** | 91 / 26 | **HC** | 3.559 [2.147, 4.971] | 4.729 [2.788, 6.191] | +1.795 [-0.648, +4.047] | 2.816 [1.653, 4.031] |
| | | **BPD** | 1.140 [0.617, 1.644] | 1.490 [0.821, 1.974] | +0.859 [+0.191, +1.510] | 0.947 [0.372, 1.769] |
| **UCL Test** | 35 / 5 | **HC** | 3.664 [2.942, 4.458] | 4.421 [3.561, 5.243] | +1.316 [-1.016, +4.236] | 2.982 [1.962, 4.007] |
| | | **BPD** | 0.985 [0.782, 1.224] | 1.289 [1.026, 1.503] | +0.594 [+0.258, +1.043] | 0.739 [0.717, 1.209] |

## 3. 10 Worst Images by HC Absolute Error

| # | Dataset | Image Name | Subject ID | Split | Ref HC (mm) | Pred HC (mm) | HC Abs Error (mm) | Ref BPD (mm) | Pred BPD (mm) | BPD Abs Error (mm) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | FP | `Patient01070_Plane3_2_of_6.png` | 1070 | Train | 1.97 | 175.32 | **173.35** | 0.32 | 50.65 | 50.32 |
| 2 | FP | `Patient00955_Plane3_2_of_3.png` | 955 | Train | 1.18 | 161.47 | **160.29** | 0.24 | 35.11 | 34.87 |
| 3 | FP | `Patient00957_Plane3_2_of_3.png` | 957 | Train | 1.42 | 138.55 | **137.13** | 0.21 | 39.01 | 38.79 |
| 4 | FP | `Patient01738_Plane3_2_of_2.png` | 1738 | Train | 84.37 | 152.05 | **67.68** | 18.55 | 38.54 | 19.98 |
| 5 | FP | `Patient00840_Plane3_2_of_3.png` | 840 | Test | 310.94 | 343.03 | **32.09** | 93.74 | 97.04 | 3.30 |
| 6 | FP | `Patient01490_Plane3_2_of_2.png` | 1490 | Train | 264.95 | 296.01 | **31.07** | 74.18 | 93.50 | 19.31 |
| 7 | FP | `Patient00646_Plane3_1_of_1.png` | 646 | Test | 268.12 | 297.21 | **29.09** | 71.37 | 86.62 | 15.25 |
| 8 | FP | `Patient00765_Plane3_5_of_6.png` | 765 | Test | 105.78 | 132.15 | **26.37** | 28.01 | 41.74 | 13.73 |
| 9 | FP | `Patient01729_Plane3_2_of_3.png` | 1729 | Train | 134.11 | 109.92 | **24.19** | 36.67 | 26.65 | 10.02 |
| 10 | FP | `Patient00984_Plane3_8_of_8.png` | 984 | Test | 231.39 | 255.40 | **24.01** | 71.79 | 73.29 | 1.50 |