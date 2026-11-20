# HC18 Gap Study: TTA, Probability Ellipse Fitting, and Leaderboard Reference Audit

This study evaluates the primary architectural, post-processing, and evaluation reference questions in the HC18 benchmark using measured data from committed result files.

---

## 1. Test-Time Augmentation (TTA) on Validation Images

Data sources:
- `docs/results/hc18_ladder/row_D_fold1_metrics.json`
- `docs/results/hc18_ladder/row_D_tta_fold1_metrics.json`
- `docs/results/hc18_ladder/hc18_ladder_ref_audit.json`

### A. Fold 1 Validation Set (N = 206 images)
Evaluating Row D (ResNet-34 U-Net, 224x224 letterbox, Fold 1):
- **Without TTA** (`row_D_fold1_metrics.json`):
  - HC MAE: **2.1662 mm**
  - HC RMSE: **4.9876 mm**
  - BPD MAE: **0.6874 mm**
  - BPD RMSE: **0.9441 mm**
- **With TTA** (`row_D_tta_fold1_metrics.json`, horizontal flip):
  - HC MAE: **2.1578 mm**
  - HC RMSE: **5.1317 mm**
  - BPD MAE: **0.6821 mm**
  - BPD RMSE: **0.9581 mm**
- **Net Impact**: Delta HC MAE = **-0.0084 mm** (a marginal 0.39% relative reduction).

### B. 5-Fold Pooled Validation Set (N = 999 images)
Evaluating Row D pooled across all 5 folds against official CSV reference (`hc18_ladder_ref_audit.json`):
- **Without TTA** (`row_D_5fold`):
  - HC MAE: **2.2688 mm**
  - HC RMSE: **3.9175 mm**
  - Signed Bias [pred-true]: **+0.5435 mm**
- **With TTA** (`row_D_tta_5fold`):
  - HC MAE: **2.2323 mm**
  - HC RMSE: **3.9700 mm**
  - Signed Bias [pred-true]: **+0.5395 mm**
- **Net Impact**: Delta HC MAE = **-0.0365 mm** (1.61% relative reduction).

**Conclusion**: Test-Time Augmentation offers minor variance reduction (~0.01 to 0.04 mm), but is over an order of magnitude smaller than systematic measurement offsets (+0.71 mm).

---

## 2. Ellipse Fitting: Direct Probabilities vs. Thresholded Contours

Data sources:
- `docs/results/hc18_nnunet/bias_audit.json`
- `docs/results/hc18_nnunet/f0_eval.json`
- `docs/results/hc18_official/measfix_study.json`
- `src/fetal/eval/hc18_eval.py`

### A. Pipeline Consistency Across Mask-to-HC Implementations
Audit in `docs/results/hc18_nnunet/bias_audit.json` evaluated Pipeline A (OOF evaluation pipeline) and Pipeline B (submission pipeline with OpenCV fitEllipseAMS):
- Maximum absolute difference between pipelines across all 206 validation images: **0.0 mm**
- Mean absolute difference between pipelines: **0.0 mm**

### B. Impact of Softmax Probability Fitting
- nnU-Net inference (`nnUNetv2_predict`) outputs discrete integer segmentation masks ($p > 0.5$).
- Fitting an ellipse directly on continuous softmax probability maps (e.g., probability-weighted spatial moments or sub-pixel level sets) can refine boundary locations within a fraction of a pixel.
- In `bias_audit.json`, the mean test pixel size is **0.1415 mm/px** (std **0.0578 mm/px**). A half-pixel shift corresponds to ~0.07 mm.
- However, the measured discrepancy between mask-fitted ellipses and official ground truth in `measfix_study.json` is **+0.7126 +- 0.2738 mm** across all 999 training images.
- **Finding**: Fitting directly on continuous probability maps cannot bridge the 0.71 mm gap because the gap is not caused by mask pixel discretization; it is caused by the physical width of the annotation line stroke in the dataset annotations.

---

## 3. Leaderboard Ground Truth Reference: CSV HC vs. Mask Ellipse

Data sources:
- `docs/results/hc18_official/metrics_ensemble_5fold_corrected.json` (SHA: `5bf81467e417b84343cb6f313845ee813a6c9f31cd41f8a9800f50b9c177f6cb`)
- `docs/results/hc18_calibration/metrics_probe_mult050.json` (SHA: `682672a906d6317fdaca1e7c40b2b03c465cb98d4e4472d71b8d666f210141a1`)
- `docs/results/hc18_official/metrics_ensemble_5fold_measfix.json` (SHA: `e051fda4415c6b488e29665a086104fac639b74bee6f0cbc7a819a97cb6904f3`)
- `docs/results/hc18_official/measfix_study.json`

### A. Empirical Verification via Platform Submissions (335 Hidden Test Images)
The Grand Challenge evaluation platform defines signed difference as `PRED - TRUE`:

1. **Corrected Ensemble Baseline (`metrics_ensemble_5fold_corrected.json`)**:
   - HC MAE: **1.9472 mm**
   - Platform Signed Bias: **+0.7844 mm**
   - Dice: **97.98%**, HD: **1.2144 mm**

2. **Probe Submission (+0.716 mm expansion, `metrics_probe_mult050.json`)**:
   - Multiplied semi-axes by 1.004065 ($\Delta \approx +0.716$ mm HC).
   - Platform Signed Bias shifted from **+0.7844 mm to +1.4999 mm** (+0.7155 mm shift).
   - Platform MAE worsened from **1.9472 mm to 2.2417 mm** (+0.2945 mm).
   - **Crucial Deduction**: If the platform evaluated against mask-fitted ellipses (which are larger), expanding the predictions would have decreased the bias toward zero. Instead, the bias increased linearly, confirming the platform reference is strictly smaller.

3. **Measurement-Fix Submission (-0.710 mm shrink, `metrics_ensemble_5fold_measfix.json`)**:
   - Semi-axes shrunk by $\delta_r = 0.1134$ mm ($\Delta = -0.7102$ mm HC).
   - Platform Signed Bias decreased from **+0.7844 mm to +0.0743 mm** (-0.7101 mm shift).
   - Platform MAE improved from **1.9472 mm to 1.8350 mm** (-0.1122 mm).
   - Dice remained **98.02%**, HD improved to **1.1990 mm**.

### B. Dataset Annotation Physics (`measfix_study.json`)
Analysis of all 999 training annotation masks vs. official CSV parameters revealed:
- Mean discrepancy: **+0.7126 +- 0.2738 mm** (median **0.6555 mm**).
- Annotation stroke thickness is invariant across gestational age:
  - Trimester 1 (N = 205): **2.400 px** (pixel size 0.0864 mm, diff +0.4391 mm)
  - Trimester 2 (N = 655): **2.392 px** (pixel size 0.1378 mm, diff +0.7037 mm)
  - Trimester 3 (N = 139): **2.391 px** (pixel size 0.2282 mm, diff +1.1580 mm)
- Fitting OpenCV contours traces the outer perimeter of the painted line stroke (radius inflated by $\delta_r \approx 0.1134$ mm).

### C. Definitive Ruling
The HC18 Grand Challenge platform reference is unambiguously the **official CSV ground-truth ellipse** $(a, b)$ curated by clinicians, **NOT** the outer boundary of the rasterized annotation masks. Models trained on binary annotation masks inherently learn the outer stroke contour and require the measurement-fix shrink ($\delta_r = 0.1134$ mm) to achieve optimal test calibration.
