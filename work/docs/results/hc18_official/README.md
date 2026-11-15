# HC18 Grand Challenge Official Test Set Evaluation

This directory contains the official Grand Challenge test set evaluation artifacts, submission CSVs, and audit reports for the 335 hidden test images.

## 1. Initial Submissions & Official Platform Metrics
- **Ensemble (5-Fold)** (`metrics_ensemble_5fold.json`):
  - HC Absolute Difference: **1.95 +- 1.91 mm** (Median: 1.38 mm)
  - Dice Similarity Coefficient: **85.59 +- 3.11 %**
  - Hausdorff Distance: **7.20 +- 3.21 mm**
- **Single Model (Fold 1)** (`metrics_single_fold1.json`):
  - HC Absolute Difference: **1.99 +- 1.91 mm** (Median: 1.39 mm)
  - Dice Similarity Coefficient: **85.66 +- 3.07 %**
  - Hausdorff Distance: **7.18 +- 3.20 mm**

## 2. Ellipse Orientation Audit & Root Cause
- **Root Cause**: In OpenCV (`cv2.fitEllipse`), the returned angle in `[0, 180)` belongs to the minor axis (`d1 <= d2`), whereas the submission CSV specifies `a` as the major semi-axis. The first submissions had the axis/angle pairing 90 degrees off.
- **Mask Reconstruction Analysis**: On all 999 training masks (`roundtrip_dice.json`), the legacy buggy convention achieved mean Dice **85.42%**, whereas the corrected convention (`angle_rad = (angle_cv2 - pi/2) wrapped to [-pi/2, pi/2)`) achieves mean Dice **99.41%** (+13.99% gain).
- **Impact on Platform Metrics**:
  - Official Dice and Hausdorff distance of those first submissions are affected by the 90° orientation error.
  - Head Circumference (HC) absolute difference is **NOT** affected: Ramanujan's formula depends strictly on semi-axes `(a, b)` and is invariant to rotation. Max absolute difference row-by-row between original and corrected CSVs is **0.0 mm**.

## 3. Corrected Submission Files
- `submission_ensemble_5fold_corrected.csv`: 335 rows, unique filenames, angle wrapped to `[-pi/2, pi/2)`.
- `submission_single_fold1_corrected.csv`: 335 rows, unique filenames, angle wrapped to `[-pi/2, pi/2)`.

## 4. Official Corrected Status
> **Official corrected result: NOT YET (to be added after the user resubmits).**
