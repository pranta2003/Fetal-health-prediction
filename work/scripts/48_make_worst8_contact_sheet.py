#!/usr/bin/env python3
"""Create a 2x4 contact sheet PNG of the worst 8 AC OOF images."""

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
WORST8_DIR = REPO_ROOT / "docs/results/ac_campaign/worst8"
OUT_PNG = REPO_ROOT / "docs/results/ac_campaign/worst8_contact_sheet.png"

# Read the pooled OOF predictions to get exact error values (match by stem to handle .png vs .jpeg/.jpg)
POOLED_CSV = Path("E:/BUBT-CSE-4th yr/Capstone/kaggle_outputs/ac_baseline_s42/ac-baseline-s42/oof_predictions_pooled.csv")
df_oof = pd.read_csv(POOLED_CSV)
err_map = {Path(r["image_name"]).stem: float(r["ac_err"]) for _, r in df_oof.iterrows()}

worst8_files = [
    "002_4AC.png",
    "114_2AC.png",
    "Patient00168_Plane2_2_of_2.png",
    "Patient00731_Plane2_1_of_1.png",
    "Patient01390_Plane2_1_of_1.png",
    "Patient01604_Plane2_1_of_1.png",
    "Patient01650_Plane2_1_of_1.png",
    "Patient01780_Plane2_2_of_2.png",
]

# Sort by descending error
worst8_files.sort(key=lambda fn: err_map.get(Path(fn).stem, 0.0), reverse=True)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, fn in enumerate(worst8_files):
    img_path = WORST8_DIR / fn
    im = Image.open(img_path)
    err = err_map.get(Path(fn).stem, 0.0)
    
    ax = axes[idx]
    ax.imshow(im)
    ax.axis("off")
    ax.set_title(f"{fn}\nError: {err:.2f} mm", fontsize=12, fontweight="bold", pad=6)

plt.tight_layout()
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved contact sheet to {OUT_PNG}")
