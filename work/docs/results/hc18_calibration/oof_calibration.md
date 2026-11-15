# HC18 Out-Of-Fold Post-Hoc Calibration Study

## Executive Summary
- **Dataset**: HC18 5-fold cross-validation (999 images / 806 patients)
- **Local Probability Maps**: NO: Zero local OOF probability maps (.npy/.pt) exist in docs/ or data/. Only scalar tabular predictions (oof_predictions.csv) are stored.
- **Evaluation Protocol**: Strictly nested 5-fold cross-validation (calibration parameters fit on 4 folds, evaluated on held-out fold).
- **Key Finding**: Raw single-fold models exhibit systematic underestimation (signed bias +1.258 mm under `y_true - y_pred`). Post-hoc calibration reduces HC MAE from **2.302 mm** down to **1.985 mm** (a significant improvement of -0.317 mm, 95% CI [-0.393, -0.240]).

## Overall Calibration Comparison

| Method | MAE (mm) | RMSE (mm) | Median AE (mm) | Signed Bias (True - Pred) Mean / Median | MAE Diff vs Raw [95% CI] (mm) |
|---|---|---|---|---|---|
| Raw (No Correction) | **2.302** | 3.580 | 1.760 | +1.258 / +1.349 | Baseline |
| Median Additive Bias | **2.003** | 3.358 | 1.444 | -0.091 / +0.018 | **-0.299** [-0.374, -0.218] |
| Median Multiplicative Ratio | **1.985** | 3.342 | 1.426 | -0.178 / -0.015 | **-0.317** [-0.404, -0.227] |
| Linear Regression (OLS) | **1.985** | 3.335 | 1.427 | -0.002 / +0.154 | **-0.317** [-0.393, -0.240] |
| Isotonic Regression | **2.184** | 3.566 | 1.523 | -0.028 / +0.031 | **-0.118** [-0.227, -0.005] |

## Per-Tercile MAE Analysis (Head-Size Proxy for Gestational Age)

- **Low Tercile**: 45.1 – 167.2 mm ($N=333$)
- **Mid Tercile**: 167.5 – 183.0 mm ($N=333$)
- **High Tercile**: 183.0 – 349.3 mm ($N=333$)

| Method | Low Tercile MAE (mm) | Mid Tercile MAE (mm) | High Tercile MAE (mm) | Overall MAE (mm) |
|---|---|---|---|---|
| Raw (No Correction) | 1.917 | 2.023 | 2.966 | 2.302 |
| Median Additive Bias | 1.798 | 1.618 | 2.592 | 2.003 |
| Median Multiplicative Ratio | 1.758 | 1.628 | 2.568 | 1.985 |
| Linear Regression (OLS) | 1.753 | 1.638 | 2.562 | 1.985 |
| Isotonic Regression | 2.080 | 1.673 | 2.798 | 2.184 |

## Explicit Caveats & Integrity Guards

1. **Single-Fold vs Ensemble**: This study evaluates single-fold out-of-fold predictions. The official HC18 challenge test submission used a 5-fold ensemble average of predicted ellipse parameters. Averaging across 5 independent fold models already reduces prediction variance and naturally shifts predictions toward the ensemble centroid.
2. **Signed Bias Conventions**:
   - **Convention A (`y_true - y_pred`)**: Raw mean = +1.2576 mm, median = +1.3490 mm (positive indicates underprediction).
   - **Convention B (`y_pred - y_true`)**: Raw mean = -1.2576 mm, median = -1.3490 mm (negative indicates underprediction).
3. **Submission Immutability**: Zero changes are made to the official challenge submission CSVs (submission_ensemble_5fold.csv and submission_single_fold1.csv) or submission generation scripts. This audit is strictly observational on the existing held-out cross-validation records.