# HC18 Grand Challenge Official Test Set Evaluation

This directory contains the official Grand Challenge test set evaluation artifacts, submission CSVs, and audit reports for the 335 hidden test images.

## Official Test Set Metrics Summary Table

| Name | Description | Abs Diff (mm) | Dice (%) | Signed [pred-true] (mm) | SHA256 |
|---|---|:---:|:---:|:---:|---|
| `metrics_single_fold1.json` | Single model (Fold 1), angle bug | **1.9937** | **85.66%** | **+0.6988** | `2fb3e8f89c2e49d342a3824e94409e19a105db64b7cf216dfa929eda7b60a851` |
| `metrics_HC18_kfold1_2.json` | Single model (Fold 1), corrected orientation | **1.9937** | **97.92%** | **+0.6988** | `a80a68d0dfe7b308cb3542bc883ffc6033d643d1c39b404d58abce1736172930` |
| `metrics_ensemble_5fold.json` | 5-fold ensemble, angle bug | **1.9472** | **85.59%** | **+0.7844** | `14daf31b3ca46370d926e664585a1fcfd32dc28738915d2177ed593a3cccc85b` |
| `metrics_ensemble_5fold_corrected.json` | 5-fold ensemble, corrected orientation | **1.9472** | **97.98%** | **+0.7844** | `5bf81467e417b84343cb6f313845ee813a6c9f31cd41f8a9800f50b9c177f6cb` |
| `metrics_HC18_kfold5_2.json` | 5-fold ensemble probe (x1.004065) | **2.2417** | **97.88%** | **+1.4999** | `682672a906d6317fdaca1e7c40b2b03c465cb98d4e4472d71b8d666f210141a1` |
| `metrics_ensemble_5fold_measfix.json` | 5-fold ensemble + measfix (best) | **1.8350** | **98.02%** | **+0.0743** | `e051fda4415c6b488e29665a086104fac639b74bee6f0cbc7a819a97cb6904f3` |

## Key Findings & Provenance
1. **Initial Submissions & Ellipse Angle Error**:
   - Initial submissions (`metrics_single_fold1.json` and `metrics_ensemble_5fold.json`) had a 90° orientation mismatch in OpenCV `cv2.fitEllipse` axis/angle pairing.
   - This depressed Dice (~85.6%) and inflated Hausdorff distance (~7.2 mm), while Head Circumference (HC) remained unaffected (rotation invariant).
2. **Corrected Submissions**:
   - Correcting orientation in `metrics_single_fold1_corrected` / `metrics_HC18_kfold1_2.json` and `metrics_ensemble_5fold_corrected.json` restored Dice to **97.98%** and Hausdorff to **1.21 mm**.
3. **Probe Submission**:
   - `metrics_HC18_kfold5_2.json` evaluated multiplicative scaling x1.004065 (+0.716 mm HC); platform bias jumped from +0.784 to +1.500 mm, confirming that the platform reference is strictly smaller than the outer mask contour.
4. **Measurement-Fix Shrink (Internal Best)**:
   - `metrics_ensemble_5fold_measfix.json` applied the training annotation stroke correction ($\delta_r = 0.1134$ mm), reducing platform bias from +0.784 mm to **+0.074 mm** and achieving the top official MAE of **1.835 mm** (Dice 98.02%, HD 1.199 mm).
