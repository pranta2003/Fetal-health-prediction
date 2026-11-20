#!/usr/bin/env python3
"""Generate Kaggle kernel for HC18 nnU-Net fold 0 test set inference.

Outputs:
- notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f0/hc18-nnunet-test-f0.ipynb
- notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f0/kernel-metadata.json
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNEL_DIR = REPO_ROOT / "notebooks" / "kaggle" / "hc18_nnunet_test" / "hc18-nnunet-test-f0"
TEST_PIXEL_CSV = Path("C:/fetal-data/HC18/test_set_pixel_size.csv")


def make_code_cell(source: str, cell_id: str | None = None) -> dict:
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")],
    }
    if cell_id:
        cell["id"] = cell_id
    return cell


def make_markdown_cell(source: str, cell_id: str | None = None) -> dict:
    cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")],
    }
    if cell_id:
        cell["id"] = cell_id
    return cell


def main():
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)

    # Read pixel size CSV text to embed
    pixel_csv_text = TEST_PIXEL_CSV.read_text(encoding="utf-8").strip()

    metadata = {
        "id": "prantakumerpandit390/hc18-nnunet-test-f0",
        "title": "hc18-nnunet-test-f0",
        "code_file": "hc18-nnunet-test-f0.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [
            "prantakumerpandit390/hc18-ultrasound"
        ],
        "competition_sources": [],
        "kernel_sources": [
            "prantakumerpandit390/hc18-nnunet-f0-250ep"
        ],
        "model_sources": [],
    }

    cells = []

    # Cell 1: Intro Markdown
    cells.append(make_markdown_cell("""# HC18 nnU-Net v2 Test Set Inference (Fold 0)
- Predicts on official 335 HC18 test set images.
- Uses trained weights from `prantakumerpandit390/hc18-nnunet-f0-250ep`.
- Applies corrected ellipse extraction pipeline (cv2.fitEllipseAMS, angle convention [-pi/2, pi/2)).
- Backs up model weights (`checkpoint_final.pth` / `checkpoint_best.pth`, `plans.json`, `dataset.json`) to output.
- Generates `submission_nnunet_f0_raw.csv` and `submission_nnunet_f0_measfix.csv`.
""", cell_id="cell_1_intro"))

    # Cell 2: Fail-Fast Dataset Discovery & Assertion (Hard-coded path with print of input root)
    cells.append(make_code_cell("""import os
from pathlib import Path

print("=" * 60)
print("KERNEL STARTUP: Inspecting /kaggle/input and verifying test set")
print("=" * 60)

input_root = Path("/kaggle/input")
if input_root.exists():
    print("os.listdir('/kaggle/input'):", os.listdir("/kaggle/input"))
    for sub in input_root.iterdir():
        if sub.is_dir():
            try:
                print(f"os.listdir('{sub}'):", os.listdir(sub)[:10])
            except Exception as e:
                print(f"Could not list {sub}: {e}")

# Hard-coded dataset path
hardcoded_path = Path("/kaggle/input/datasets/prantakumerpandit390/hc18-ultrasound/test_set/test_set")
if not hardcoded_path.is_dir():
    fallback_path = Path("/kaggle/input/hc18-ultrasound/test_set/test_set")
    if fallback_path.is_dir():
        hardcoded_path = fallback_path

test_src_dir = hardcoded_path
print(f"Target test_src_dir: {test_src_dir}")
assert test_src_dir.is_dir(), f"Test set directory not found at {test_src_dir}"

test_pngs = sorted([p for p in test_src_dir.iterdir() if p.name.endswith(".png") and not p.name.endswith("_Annotation.png")])
print(f"Test PNG count: {len(test_pngs)}")
assert len(test_pngs) == 335, f"Expected exactly 335 test PNGs, got {len(test_pngs)}"
print("FAIL-FAST CHECK PASSED: Exactly 335 test images verified before proceeding.")
""", cell_id="cell_2_discovery_failfast"))

    # Cell 3: Install dependencies
    cells.append(make_code_cell("""!nvidia-smi
!pip install --upgrade pip
!pip install --quiet nnunetv2
""", cell_id="cell_3_install"))

    # Cell 3: Setup nnUNet environment & stage trained model
    cells.append(make_code_cell("""import os
import shutil
from pathlib import Path
import json

os.environ["nnUNet_raw"] = "/kaggle/working/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/kaggle/working/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "/kaggle/working/nnUNet_results"

raw_dir = Path("/kaggle/working/nnUNet_raw")
prep_dir = Path("/kaggle/working/nnUNet_preprocessed")
res_dir = Path("/kaggle/working/nnUNet_results")
out_dir = Path("/kaggle/working/out")
backup_dir = Path("/kaggle/working/backup_weights")

for d in [raw_dir, prep_dir, res_dir, out_dir, backup_dir]:
    d.mkdir(parents=True, exist_ok=True)

print("Locating trained nnU-Net model weights in /kaggle/input...")
model_src_fold0 = None
for p in Path("/kaggle/input").rglob("fold_0"):
    if (p / "checkpoint_final.pth").exists() or (p / "checkpoint_best.pth").exists():
        model_src_fold0 = p
        break

if model_src_fold0 is None:
    for p in Path("/kaggle/input").rglob("checkpoint_best.pth"):
        model_src_fold0 = p.parent
        break

assert model_src_fold0 is not None, "Could not find fold_0 directory in /kaggle/input"
print(f"Found trained fold_0 at: {model_src_fold0}")

trainer_src = model_src_fold0.parent
plans_src = trainer_src / "plans.json"
dataset_src = trainer_src / "dataset.json"

target_trainer_dir = res_dir / "Dataset501_HC18" / trainer_src.name
target_fold0_dir = target_trainer_dir / "fold_0"
target_fold0_dir.mkdir(parents=True, exist_ok=True)

for f in model_src_fold0.glob("*"):
    if f.is_file():
        shutil.copy2(f, target_fold0_dir / f.name)
        print(f"  Staged {f.name} ({f.stat().st_size} bytes)")

if plans_src.exists():
    shutil.copy2(plans_src, target_trainer_dir / "plans.json")
    print(f"  Staged plans.json ({plans_src.stat().st_size} bytes)")
if dataset_src.exists():
    shutil.copy2(dataset_src, target_trainer_dir / "dataset.json")
    print(f"  Staged dataset.json ({dataset_src.stat().st_size} bytes)")

# Backup essential weights to backup_dir
for ckpt_name in ["checkpoint_final.pth", "checkpoint_best.pth"]:
    src_ckpt = target_fold0_dir / ckpt_name
    if src_ckpt.exists():
        shutil.copy2(src_ckpt, backup_dir / ckpt_name)
        print(f"Backed up {ckpt_name} to {backup_dir}")

if (target_trainer_dir / "plans.json").exists():
    shutil.copy2(target_trainer_dir / "plans.json", backup_dir / "plans.json")
if (target_trainer_dir / "dataset.json").exists():
    shutil.copy2(target_trainer_dir / "dataset.json", backup_dir / "dataset.json")

print("Backup contents:", list(backup_dir.glob("*")))
""", cell_id="cell_3_env"))

    # Cell 5: Stage 335 Test Images
    cells.append(make_code_cell("""import shutil
from pathlib import Path

test_images = sorted([p for p in test_src_dir.iterdir() if p.name.endswith(".png") and not p.name.endswith("_Annotation.png")])
print(f"Staging {len(test_images)} test images from {test_src_dir}...")
assert len(test_images) == 335, f"Expected 335 test images, got {len(test_images)}"

nnunet_test_in = Path("/kaggle/working/nnunet_test_in")
nnunet_test_in.mkdir(parents=True, exist_ok=True)

for img_path in test_images:
    case_id = img_path.stem
    dest_name = f"{case_id}_0000.png"
    shutil.copy2(img_path, nnunet_test_in / dest_name)

n_staged = len(list(nnunet_test_in.glob("*.png")))
print(f"Staged {n_staged} formatted test images in {nnunet_test_in}")
assert n_staged == 335
""", cell_id="cell_5_stage_images"))

    # Cell 6: Predict
    cells.append(make_code_cell("""import subprocess
from pathlib import Path

nnunet_test_out = Path("/kaggle/working/test_predicted_masks")
nnunet_test_out.mkdir(parents=True, exist_ok=True)

fold0_dir = res_dir / "Dataset501_HC18" / trainer_src.name / "fold_0"
if (fold0_dir / "checkpoint_final.pth").exists():
    chk_flag = "-chk checkpoint_final.pth"
elif (fold0_dir / "checkpoint_best.pth").exists():
    chk_flag = "-chk checkpoint_best.pth"
else:
    raise FileNotFoundError(f"No checkpoint found in {fold0_dir}")

cmd = f"nnUNetv2_predict -i {nnunet_test_in} -o {nnunet_test_out} -d 501 -c 2d -f 0 -tr {trainer_src.name} {chk_flag}"
print("Executing nnUNetv2_predict command:\\n", cmd)
ret = subprocess.run(cmd, shell=True)
if ret.returncode != 0:
    raise RuntimeError(f"nnUNetv2_predict failed with return code {ret.returncode}")

predicted_masks = list(nnunet_test_out.glob("*.png"))
print(f"Prediction finished successfully: {len(predicted_masks)} masks generated.")
assert len(predicted_masks) == 335, f"Expected 335 masks, got {len(predicted_masks)}"
""", cell_id="cell_6_predict"))

    # Cell 6: Fit Ellipses & Generate Raw Submission CSV
    cell_6_code = f'''import io
import cv2
import numpy as np
import pandas as pd
import hashlib
from pathlib import Path

# Robust ellipse fit
def fit_ellipse_robust(contour):
    try:
        return cv2.fitEllipseAMS(contour)
    except cv2.error:
        return cv2.fitEllipse(contour)

def hc_bpd_from_ellipse_params(semi_major_mm, semi_minor_mm):
    a, b = semi_major_mm, semi_minor_mm
    hc = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
    bpd = 2 * min(a, b)
    return hc, bpd

def fit_submission_ellipse(prob_map_native, pixel_size, filename):
    binary = (prob_map_native > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {{
            "filename": filename,
            "center_x_mm": 0.0,
            "center_y_mm": 0.0,
            "semi_axes_a_mm": 0.0,
            "semi_axes_b_mm": 0.0,
            "angle_rad": 0.0,
        }}
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return {{
            "filename": filename,
            "center_x_mm": 0.0,
            "center_y_mm": 0.0,
            "semi_axes_a_mm": 0.0,
            "semi_axes_b_mm": 0.0,
            "angle_rad": 0.0,
        }}
    (cx_px, cy_px), (axis1_px, axis2_px), angle_deg = fit_ellipse_robust(largest)
    semi_major_mm = (max(axis1_px, axis2_px) / 2.0) * pixel_size
    semi_minor_mm = (min(axis1_px, axis2_px) / 2.0) * pixel_size
    center_x_mm = cx_px * pixel_size
    center_y_mm = cy_px * pixel_size
    raw_angle_rad = np.deg2rad(angle_deg) - np.pi / 2.0
    angle_rad = float((raw_angle_rad + np.pi / 2.0) % np.pi - np.pi / 2.0)

    return {{
        "filename": filename,
        "center_x_mm": float(center_x_mm),
        "center_y_mm": float(center_y_mm),
        "semi_axes_a_mm": float(semi_major_mm),
        "semi_axes_b_mm": float(semi_minor_mm),
        "angle_rad": float(angle_rad),
    }}

# Load test pixel size data
csv_embedded = """{pixel_csv_text}"""
df_ps = pd.read_csv(io.StringIO(csv_embedded))
pixel_size_map = dict(zip(df_ps["filename"], df_ps["pixel size(mm)"]))
print(f"Loaded {{len(pixel_size_map)}} test pixel sizes.")
assert len(pixel_size_map) == 335

rows = []
hc_values = []
mask_dir = Path("/kaggle/working/test_predicted_masks")
test_fns = sorted(list(pixel_size_map.keys()))

for fn in test_fns:
    mask_file = mask_dir / fn
    assert mask_file.exists(), f"Missing predicted mask: {{mask_file}}"
    mask_img = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    prob_map = (mask_img > 0).astype(np.float32)
    ps = pixel_size_map[fn]
    rec = fit_submission_ellipse(prob_map, ps, fn)
    rows.append(rec)
    hc, _ = hc_bpd_from_ellipse_params(rec["semi_axes_a_mm"], rec["semi_axes_b_mm"])
    hc_values.append(hc)

df_sub = pd.DataFrame(rows)

# Strict column order
columns = ["filename", "center_x_mm", "center_y_mm", "semi_axes_a_mm", "semi_axes_b_mm", "angle_rad"]
df_sub = df_sub[columns]

# Rounding
for col in ["center_x_mm", "center_y_mm", "semi_axes_a_mm", "semi_axes_b_mm", "angle_rad"]:
    df_sub[col] = df_sub[col].round(6)

out_csv = Path("/kaggle/working/submission_nnunet_f0_raw.csv")
csv_bytes = df_sub.to_csv(index=False, lineterminator="\\n").encode("utf-8")
out_csv.write_bytes(csv_bytes)

# Verify assertions
assert len(df_sub) == 335, f"Expected 335 rows, got {{len(df_sub)}}"
assert df_sub["filename"].nunique() == 335, "Duplicate filenames found"
assert df_sub["angle_rad"].min() >= -np.pi / 2.0 - 1e-5, "Angle below -pi/2"
assert df_sub["angle_rad"].max() <= np.pi / 2.0 + 1e-5, "Angle above pi/2"
assert b"\\r" not in csv_bytes, "CRLF detected"

sha256 = hashlib.sha256(csv_bytes).hexdigest()
print("=" * 60)
print("SUCCESSFULLY GENERATED RAW SUBMISSION CSV:")
print(f"  File  : {{out_csv}}")
print(f"  Rows  : {{len(df_sub)}}")
print(f"  SHA256: {{sha256}}")
print(f"  Mean HC on test set: {{np.mean(hc_values):.4f}} mm (std: {{np.std(hc_values):.4f}} mm)")
print(f"  Mean semi_axes_a: {{df_sub['semi_axes_a_mm'].mean():.4f}} mm")
print(f"  Mean semi_axes_b: {{df_sub['semi_axes_b_mm'].mean():.4f}} mm")
print("First 3 rows:")
print(df_sub.head(3))
print("Last 3 rows:")
print(df_sub.tail(3))
print("=" * 60)

# Build measfix candidate (semi-axes shrunk by delta_r = 0.113422 mm)
delta_r = 0.11342152722878392
df_meas = df_sub.copy()
df_meas["semi_axes_a_mm"] = (df_sub["semi_axes_a_mm"] - delta_r).clip(lower=0.1).round(6)
df_meas["semi_axes_b_mm"] = (df_sub["semi_axes_b_mm"] - delta_r).clip(lower=0.1).round(6)

out_meas_csv = Path("/kaggle/working/submission_nnunet_f0_measfix.csv")
meas_bytes = df_meas.to_csv(index=False, lineterminator="\\n").encode("utf-8")
out_meas_csv.write_bytes(meas_bytes)

sha256_meas = hashlib.sha256(meas_bytes).hexdigest()
print("SUCCESSFULLY GENERATED MEASFIX SUBMISSION CSV:")
print(f"  File  : {{out_meas_csv}}")
print(f"  Rows  : {{len(df_meas)}}")
print(f"  SHA256: {{sha256_meas}}")
print("=" * 60)
'''
    cells.append(make_code_cell(cell_6_code, cell_id="cell_6_fit_ellipses"))

    # Cell 7: Zip masks & finalize
    cells.append(make_code_cell("""import zipfile
from pathlib import Path

zip_path = Path("/kaggle/working/test_predicted_masks.zip")
print(f"Creating zip archive {zip_path}...")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for m in Path("/kaggle/working/test_predicted_masks").glob("*.png"):
        zf.write(m, arcname=m.name)
print(f"Archive created successfully: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")
print("Task B nnU-Net test inference completed successfully.")
""", cell_id="cell_7_zip"))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    nb_path = KERNEL_DIR / "hc18-nnunet-test-f0.ipynb"
    meta_path = KERNEL_DIR / "kernel-metadata.json"

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated {nb_path}")
    print(f"Generated {meta_path}")


if __name__ == "__main__":
    main()
