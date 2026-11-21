# Dataset Image Count Audit (Abdomen, Femur, Head)

Comparison of image counts in arXiv 2512.16710 vs local repository annotations.

## 1. Summary Comparison Table

| Anatomical Plane | Paper Published Total | FP Raw | UCL Raw | UCL Calibrated (px_to_mm) | Excluded Corrupt | Final Used in Repo | Missing vs Paper |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Abdomen** | **823** (FP 693 + UCL 130) | 693 | 130 | 99 | 3 (pat 125) | **789** (642 train + 147 test) | **34** (31 no px_to_mm + 3 corrupt) |
| **Femur** | **895** (FP 760 + UCL 135) | 760 | 135 | 107 | 0 | **867** (FP 760 + UCL 107) | **28** (28 no px_to_mm) |
| **Head** | **1796** (FP 1637 + UCL 159) | 1637 | 159 | 126 | 0 | **1763** (FP 1637 + UCL 126) | **33** (33 no px_to_mm) |

## 2. Abdomen 34 Missing Images Breakdown

### A. Missing `px_to_mm` Calibration (31 images)
Present in raw UCL annotation CSV but excluded because physical mm scaling is unknown:
- `001_2AC.jpg`
- `001_AC.jpg`
- `005_5AC.jpeg`
- `011_AC.png`
- `031_2AC.png`
- `031_3AC.png`
- `031_4AC.png`
- `031_5AC.png`
- `031_AC.png`
- `032_AC.png`
- `034_2AC.png`
- `034_3AC.png`
- `034_AC.png`
- `035_2AC.png`
- `035_3AC.png`
- `035_4AC.png`
- `035_AC.png`
- `047_AC.png`
- `051_AC.png`
- `052_AC.png`
- `053_AC.png`
- `054_AC.png`
- `056_2AC.png`
- `056_AC.png`
- `058_2AC.png`
- `058_AC.png`
- `059_2AC.png`
- `059_AC.png`
- `120_AC.jpeg`
- `125_2AC.jpeg`
- `125_AC.jpeg`

### B. Corrupted Acquisition Exclusions (3 images)
Excluded by `df[~df['image_name'].str.startswith('125_')]` in `src/fetal/data/ac_dataset.py`:
- `125_3AC.jpeg`
- `125_4AC.jpeg`
- `125_5AC.jpeg`
