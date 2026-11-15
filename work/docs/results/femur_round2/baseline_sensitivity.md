# Femur Round 2 Checks: Sensitivity of Baseline & Run Integrity

## 1. Sensitivity of the Baseline & Paired Bootstrap Differences
> Protocol: 348 official test images, 266 patients. Clustered patient-level paired bootstrap (1,000 resamples, 95% CI).

| Comparison | Mean Diff (mm) | 95% CI (mm) | Includes Zero? |
| :--- | :---: | :---: | :---: |
| a) Round-2 Row R4 vs PR #32 Row A | **-0.03** | [-0.14, 0.09] | True |
| b) Round-2 Row R2 vs PR #32 Row A | **-0.08** | [-0.43, 0.21] | True |
| c) Round-2 Row R6 vs PR #32 Row A | **0.05** | [-0.02, 0.14] | True |
| d) Baseline Drift: Round-2 Row A vs PR #32 Row A | **0.20** | [0.11, 0.33] | False |
| Reference: Row R4 Gain over Round-2 Row A | **-0.23** | [-0.40, -0.09] | False |

### Row A Per-Seed Test MAE Comparison:
- **PR #32 Row A**: Seed 42: 1.2512 mm | Seed 43: 1.1380 mm | Seed 44: 1.2410 mm
- **Round 2 Row A**: Seed 42: 1.4546 mm | Seed 43: 1.0869 mm | Seed 44: 1.4009 mm
- **Seeds**: [42, 43, 44] in both runs.
- **Deterministic Training**: False (`torch.backends.cudnn.deterministic` not set, AMP GradScaler enabled; cuDNN non-deterministic convolution algorithms and atomic additions active).

**Statement**: Row R4 shows a gain of 0.23 mm (95% CI: [-0.40, -0.09]) over the round-2 Row A re-run, which is comparable in magnitude to the run-to-run baseline drift of 0.20 mm (95% CI: [0.11, 0.33]) between round-2 Row A and PR #32 Row A; evaluated directly against the PR #32 Row A baseline, the difference is -0.03 mm (95% CI: [-0.14, 0.09]), whose 95% CI includes zero.

## 2. Rows R1 and R3 Run Integrity & Failure Audit
- **Row R1**: FAILED RUN (did not train: val MAE far above a centre baseline or flat). Mark: `run failed, not a method comparison`.
  - Pretrained weights: TIMM HRNet-W32 loaded with unexpected batchnorm tracking keys
  - Decode settings: DARK sub-pixel Taylor-expansion decoding on 384px heatmap
  - Test MAE: 18.33 mm
- **Row R3**: FAILED RUN (did not train: val MAE far above a centre baseline or flat). Mark: `run failed, not a method comparison`.
  - Pretrained weights: EfficientNet-B4 loaded with auxiliary bone-line heatmap head
  - Decode settings: Dual peak detection with bone-line auxiliary loss and minimum distance constraint (min_dist=50px)
  - Test MAE: 47.50 mm

## 3. Row B Epoch Convergence Check
- Epochs 25 to 35: Val MAE decreased from 9.23 mm to 5.15 mm with loss dropping from 5.99 to 3.71.
- Model was still improving at the epoch 35 cap.
- Mark: `capped at 35 epochs, not converged`.

## 4. 10 Worst Row-A Test Images (Audit & Provenance)
| Rank | Image Name | Dataset | True FL (mm) | Pred FL (mm) | FL Error (mm) | Squared Error (mm²) | Share of Total Sq Err (%) | Classification | Emitting Function |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| 1 | `005_3FL.jpeg` | UCL | 23.66 | 80.85 | 57.19 | 3271.16 | 75.01% | visual judgement, not computed | `fetal.eval.femur_evaluator:eval_femur_ensemble` |
| 2 | `Patient00695_Plane5_1_of_1.png` | FP | 56.22 | 43.17 | 13.05 | 170.23 | 3.90% | visual judgement, not computed | `fetal.eval.femur_evaluator:eval_femur_ensemble` |
| 3 | `Patient01427_Plane5_1_of_1.png` | FP | 36.13 | 49.02 | 12.90 | 166.32 | 3.81% | visual judgement, not computed | `fetal.eval.femur_evaluator:eval_femur_ensemble` |
| 4 | `Patient01177_Plane5_1_of_2.png` | FP | 42.96 | 32.93 | 10.03 | 100.56 | 2.31% | visual judgement, not computed | `fetal.eval.femur_evaluator:eval_femur_ensemble` |
| 5 | `Patient01648_Plane5_1_of_5.png` | FP | 30.73 | 40.67 | 9.94 | 98.89 | 2.27% | visual judgement, not computed | `fetal.eval.femur_evaluator:eval_femur_ensemble` |
| 6 | `Patient00705_Plane5_1_of_2.png` | FP | 64.34 | 58.06 | 6.29 | 39.54 | 0.91% | visual judgement, not computed | `fetal.eval.femur_evaluator:eval_femur_ensemble` |
| 7 | `Patient00662_Plane5_1_of_1.png` | FP | 59.54 | 65.17 | 5.63 | 31.68 | 0.73% | visual judgement, not computed | `fetal.eval.femur_evaluator:eval_femur_ensemble` |
| 8 | `Patient00958_Plane5_1_of_1.png` | FP | 72.94 | 67.40 | 5.54 | 30.68 | 0.70% | visual judgement, not computed | `fetal.eval.femur_evaluator:eval_femur_ensemble` |
| 9 | `Patient00679_Plane5_1_of_1.png` | FP | 65.59 | 61.09 | 4.50 | 20.22 | 0.46% | visual judgement, not computed | `fetal.eval.femur_evaluator:eval_femur_ensemble` |
| 10 | `Patient01682_Plane5_2_of_2.png` | FP | 58.21 | 62.62 | 4.41 | 19.48 | 0.45% | visual judgement, not computed | `fetal.eval.femur_evaluator:eval_femur_ensemble` |

## 5. Single Worst Image (`005_3FL.jpeg`) Squared Error Share Audit
| Model | FL Error (mm) | Squared Error | Share of Total Sq Err (348) | Share of UCL Sq Err (25) |
| :--- | :---: | :---: | :---: | :---: |
| Row A (PR #32) | 57.19 | 3271.16 | **75.01%** | **99.47%** |
| Row A (Round 2) | 57.42 | 3296.88 | **66.10%** | **99.31%** |
| Row R4 (Dual-peak) | 57.24 | 3276.59 | **74.90%** | **99.45%** |
| Row R2 (Soft-Argmax) | 3.81 | 14.55 | **1.03%** | **48.47%** |

> Classification column audit: marked 'visual judgement, not computed' for post-hoc qualitative assessments.
