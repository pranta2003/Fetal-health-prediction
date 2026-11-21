#!/usr/bin/env python3
"""Compute linear fit comparison on nnU-Net 206 and OOF 999 against CSV HC."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import cv2

REPO_ROOT = Path("C:/fetal-github-clone/work")
HC18_DIR = Path("C:/fetal-data/HC18")
TRAIN_CSV = HC18_DIR / "training_set_pixel_size_and_HC.csv"
TRAIN_DIR = HC18_DIR / "training_set"

import sys
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from fetal.eval.ellipse_fitting import fit_ellipse_robust, hc_bpd_from_ellipse_params
from fetal.eval.hc18_eval import fit_submission_ellipse

# 1. Fit linear rule strictly on 999 training annotations:
df_train = pd.read_csv(TRAIN_CSV)
records = []
for idx, row in df_train.iterrows():
    fn = row["filename"]
    ps = float(row["pixel size(mm)"])
    csv_hc = float(row["head circumference (mm)"])
    ann_path = TRAIN_DIR / fn.replace(".png", "_Annotation.png")
    ann_img = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)
    contours_ext, _ = cv2.findContours(ann_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest = max(contours_ext, key=cv2.contourArea)
    (_, _), (ax1, ax2), _ = fit_ellipse_robust(largest)
    semi_major_mm = (max(ax1, ax2) / 2.0) * ps
    semi_minor_mm = (min(ax1, ax2) / 2.0) * ps
    fit_hc, _ = hc_bpd_from_ellipse_params(semi_major_mm, semi_minor_mm)
    records.append({"fit_hc": fit_hc, "csv_hc": csv_hc})

df_ann = pd.DataFrame(records)
reg = LinearRegression().fit(df_ann[["fit_hc"]], df_ann["csv_hc"])
slope = float(reg.coef_[0])
intercept = float(reg.intercept_)
print(f"Linear Fit on 999 Annotations: CSV_HC = {slope:.6f} * FIT_HC + ({intercept:.4f})")

def bootstrap_paired(a, b, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = a - b
    mean_diff = float(np.mean(diffs))
    indices = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    boot_means = np.mean(diffs[indices], axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    p_val = float(2 * min(np.mean(boot_means <= 0), np.mean(boot_means >= 0)))
    return mean_diff, [float(ci_low), float(ci_high)], p_val

# 2. Evaluate on OOF 999
oof_csv = REPO_ROOT / "docs/results/hc18/oof_predictions.csv"
df_oof = pd.read_csv(oof_csv)
oof_true = df_oof["true_HC_mm"].to_numpy()
oof_pred = df_oof["pred_HC_mm"].to_numpy()

raw_oof_ae = np.abs(oof_pred - oof_true)
raw_oof_bias = float(np.mean(oof_pred - oof_true))

lin_oof_pred = slope * oof_pred + intercept
lin_oof_ae = np.abs(lin_oof_pred - oof_true)
lin_oof_bias = float(np.mean(lin_oof_pred - oof_true))
delta_lin_oof, ci_lin_oof, p_lin_oof = bootstrap_paired(lin_oof_ae, raw_oof_ae)

# 3. Evaluate on nnU-Net fold 0 (206)
val_files = sorted(list(Path("E:/BUBT-CSE-4th yr/Capstone/kaggle_outputs/hc18_nnunet_f0/out/validation_fold0").glob("*.png")))
train_map = {r["filename"]: r for _, r in df_train.iterrows()}

nn_raw_ae = []
nn_raw_signed = []
nn_lin_ae = []
nn_lin_signed = []

for mf in val_files:
    fn = mf.name
    row = train_map[fn]
    ps = float(row["pixel size(mm)"])
    csv_hc = float(row["head circumference (mm)"])
    pred_mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
    prob_map = (pred_mask > 0).astype(np.float32)
    sub = fit_submission_ellipse(prob_map, ps, fn)
    hc_raw, _ = hc_bpd_from_ellipse_params(sub["semi_axes_a_mm"], sub["semi_axes_b_mm"])
    hc_lin = slope * hc_raw + intercept
    nn_raw_ae.append(abs(hc_raw - csv_hc))
    nn_raw_signed.append(hc_raw - csv_hc)
    nn_lin_ae.append(abs(hc_lin - csv_hc))
    nn_lin_signed.append(hc_lin - csv_hc)

nn_raw_ae = np.array(nn_raw_ae)
nn_lin_ae = np.array(nn_lin_ae)
delta_lin_nn, ci_lin_nn, p_lin_nn = bootstrap_paired(nn_lin_ae, nn_raw_ae)

print("\n--- OOF 999 ---")
print(f"Raw MAE: {np.mean(raw_oof_ae):.4f}, Signed Bias: {raw_oof_bias:+.4f}")
print(f"Linear MAE: {np.mean(lin_oof_ae):.4f}, Signed Bias: {lin_oof_bias:+.4f}")
print(f"Delta MAE: {delta_lin_oof:+.4f}, 95% CI: [{ci_lin_oof[0]:.4f}, {ci_lin_oof[1]:.4f}], p={p_lin_oof:.4f}")

print("\n--- nnU-Net fold 0 (206) ---")
print(f"Raw MAE: {np.mean(nn_raw_ae):.4f}, Signed Bias: {np.mean(nn_raw_signed):+.4f}")
print(f"Linear MAE: {np.mean(nn_lin_ae):.4f}, Signed Bias: {np.mean(nn_lin_signed):+.4f}")
print(f"Delta MAE: {delta_lin_nn:+.4f}, 95% CI: [{ci_lin_nn[0]:.4f}, {ci_lin_nn[1]:.4f}], p={p_lin_nn:.4f}")

# 4. What would the linear rule shift the test HC by on average?
test_sub_csv = REPO_ROOT / "docs/results/hc18_official/submission_ensemble_5fold_corrected.csv"
df_sub = pd.read_csv(test_sub_csv)
hc_test_raw = []
for _, r in df_sub.iterrows():
    hc, _ = hc_bpd_from_ellipse_params(r["semi_axes_a_mm"], r["semi_axes_b_mm"])
    hc_test_raw.append(hc)
hc_test_raw = np.array(hc_test_raw)
hc_test_lin = slope * hc_test_raw + intercept
linear_test_shifts = hc_test_lin - hc_test_raw
print(f"\nLinear Rule Shift on Test HC: Mean = {np.mean(linear_test_shifts):.4f} mm, Range: [{np.min(linear_test_shifts):.4f}, {np.max(linear_test_shifts):.4f}] mm")
