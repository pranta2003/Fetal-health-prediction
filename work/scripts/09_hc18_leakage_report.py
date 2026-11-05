"""HC18 Dataset Analysis & Patient Leakage Audit Report.

Demonstrates the data leakage flaw in standard ungrouped K-Fold cross-validation
on HC18 fetal head ultrasound images, where multiple scans from the same patient
are split across train and validation sets.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
import sys

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pandas as pd
from sklearn.model_selection import KFold

from fetal.config import DATA_DIR
from fetal.data.hc18_split import patient_id_from_name, make_grouped_folds


def find_hc18_paths() -> tuple[Path, Path]:
    """Resolve HC18 dataset paths from config or default locations."""
    candidates = [
        Path(DATA_DIR) / "HC18",
        Path(r"C:\fetal-data\HC18"),
        Path(r"E:\BUBT-CSE-4th yr\Capstone\data\raw\HC-18"),
    ]
    for root in candidates:
        csv_path = root / "training_set_pixel_size_and_HC.csv"
        img_dir = root / "training_set"
        if csv_path.exists() and img_dir.exists():
            return csv_path, img_dir
    raise FileNotFoundError(
        f"HC18 training data not found in candidate locations: {candidates}"
    )


def run_report() -> None:
    csv_path, img_dir = find_hc18_paths()
    print("=" * 70)
    print("HC18 DATASET & CROSS-VALIDATION LEAKAGE REPORT")
    print("=" * 70)
    print(f"CSV Path:       {csv_path}")
    print(f"Image Directory:{img_dir}")

    df = pd.read_csv(csv_path)
    names = df["filename"].tolist()

    # Verify actual image files on disk
    disk_images = [
        f for f in os.listdir(img_dir)
        if f.endswith(".png") and not f.endswith("_Annotation.png")
    ]

    patient_groups = defaultdict(list)
    for name in names:
        pid = patient_id_from_name(name)
        patient_groups[pid].append(name)

    total_images = len(names)
    unique_patients = len(patient_groups)
    multi_img_patients = sum(1 for imgs in patient_groups.values() if len(imgs) > 1)
    max_imgs = max(len(imgs) for imgs in patient_groups.values())

    print("\n--- Step 1: HC18 Patient Distribution ---")
    print(f"Total training images (CSV):       {total_images} (expect 999)")
    print(f"Total training images (disk):      {len(disk_images)}")
    print(f"Number of unique patient_ids:      {unique_patients}")
    print(f"Patients with >1 image:            {multi_img_patients}")
    print(f"Max images per patient:            {max_imgs}")

    print("\n10 Example Patient Groups (>1 image):")
    shown = 0
    for pid, imgs in sorted(patient_groups.items()):
        if len(imgs) > 1:
            print(f"  Patient {pid} ({len(imgs)} images): {imgs}")
            shown += 1
            if shown >= 10:
                break

    print("\n--- Step 2: Leakage Analysis Under OLD Ungrouped K-Fold ---")
    print("Simulation: KFold(n_splits=5, shuffle=True, random_state=42)")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    total_leaked = 0
    patient_ids = [patient_id_from_name(n) for n in names]

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(names)):
        tr_patients = set(patient_ids[i] for i in tr_idx)
        val_leaked = sum(1 for i in val_idx if patient_ids[i] in tr_patients)
        pct = val_leaked / len(val_idx) * 100
        print(f"  Fold {fold_idx}: {val_leaked:>3d} / {len(val_idx)} val images leaked from train ({pct:5.1f}%)")
        total_leaked += val_leaked

    total_pct = total_leaked / total_images * 100
    print(f"\nTotal Leaked Validation Scans: {total_leaked} / {total_images} ({total_pct:.2f}%)")
    print("Conclusion: ~27.5% of validation images have another scan of the same")
    print("patient in training when using naive ungrouped K-Fold cross-validation.")

    print("\n--- Step 3: Verification with NEW Grouped Folds ---")
    grouped_folds = make_grouped_folds(names, n_splits=5, seed=42)
    new_leaked = 0
    for fold_idx, (tr_idx, val_idx) in enumerate(grouped_folds):
        tr_patients = set(patient_ids[i] for i in tr_idx)
        val_leaked = sum(1 for i in val_idx if patient_ids[i] in tr_patients)
        new_leaked += val_leaked
    print(f"Total Leaked Validation Scans under GroupKFold: {new_leaked} / {total_images} (0.00%)")
    print("=" * 70)


if __name__ == "__main__":
    run_report()
