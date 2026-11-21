#!/usr/bin/env python3
"""Test the submission ellipse extraction pipeline on 3 validation masks locally."""

from pathlib import Path
import sys
import numpy as np
import cv2
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from fetal.eval.hc18_eval import fit_submission_ellipse
from fetal.eval.ellipse_fitting import hc_bpd_from_ellipse_params

val_dir = Path("E:/BUBT-CSE-4th yr/Capstone/kaggle_outputs/hc18_nnunet_f0/out/validation_fold0")
val_files = sorted(list(val_dir.glob("*.png")))[:3]

assert len(val_files) == 3, f"Expected 3 files, got {len(val_files)}"

rows = []
for f in val_files:
    mask = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    prob = (mask > 0).astype(np.float32)
    # Mock pixel size
    ps = 0.1234
    res = fit_submission_ellipse(prob, ps, f.name)
    hc, bpd = hc_bpd_from_ellipse_params(res["semi_axes_a_mm"], res["semi_axes_b_mm"])
    res["hc_mm"] = hc
    rows.append(res)
    print(f"Processed {f.name}: a={res['semi_axes_a_mm']:.3f} mm, b={res['semi_axes_b_mm']:.3f} mm, HC={hc:.2f} mm, angle={res['angle_rad']:.4f}")

df = pd.DataFrame(rows)
print("\nDataFrame verification:")
print(df[["filename", "center_x_mm", "center_y_mm", "semi_axes_a_mm", "semi_axes_b_mm", "angle_rad"]])
assert -np.pi / 2 <= df["angle_rad"].min() <= df["angle_rad"].max() <= np.pi / 2
print("\nAll angle checks passed within [-pi/2, pi/2]!")
