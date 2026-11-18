"""scripts/39_make_ac_folds.py

Builds a fixed patient-grouped 5-fold assignment on the official TRAIN set
(642 images / 501 patients), seed 42.
Saves to docs/results/ac_campaign/folds_v1.csv (filename, patient, fold).
Asserts no patient is in two folds.
Asserts that none of the 147 official test images appear in it.
Prints image and patient counts per fold.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
src_dir = REPO_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fetal import config
from fetal.data.ac_dataset import build_data


def make_ac_folds_v1(
    data_dir: str | None = None,
    out_csv: Path | str | None = None,
    seed: int = 42,
    n_splits: int = 5,
) -> pd.DataFrame:
    if data_dir is None:
        data_dir = config.DATA_DIR
    if out_csv is None:
        out_csv = REPO_ROOT / "docs" / "results" / "ac_campaign" / "folds_v1.csv"
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = build_data(data_dir)
    train_df = data[data["official_split"] == "train"].reset_index(drop=True)
    test_df = data[data["official_split"] == "test"].reset_index(drop=True)

    assert len(train_df) == 642, f"Expected 642 train images, got {len(train_df)}"
    assert train_df["patient"].nunique() == 501, f"Expected 501 train patients, got {train_df['patient'].nunique()}"
    assert len(test_df) == 147, f"Expected 147 test images, got {len(test_df)}"

    # Get sorted unique patient IDs from train pool
    unique_patients = np.array(sorted(train_df["patient"].unique()))
    assert len(unique_patients) == 501

    # Deterministic patient-grouped 5-fold partition using seed 42
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    patient_to_fold: dict[str, int] = {}
    for fold_idx, (_, val_idx) in enumerate(kf.split(unique_patients), start=1):
        for p in unique_patients[val_idx]:
            patient_to_fold[p] = fold_idx

    # Map fold to each image in train
    folds_df = pd.DataFrame({
        "filename": train_df["image_name"],
        "patient": train_df["patient"],
        "fold": train_df["patient"].map(patient_to_fold),
    })

    # Assertions
    # 1. No patient in two folds
    pat_folds = folds_df.groupby("patient")["fold"].nunique()
    assert (pat_folds == 1).all(), "Leakage detected: patient assigned to multiple folds!"

    # 2. Official test images must not appear in folds_v1
    test_filenames = set(test_df["image_name"])
    train_filenames = set(folds_df["filename"])
    overlap = train_filenames & test_filenames
    assert len(overlap) == 0, f"Leakage detected: test images present in train folds: {overlap}"

    # 3. Exactly 642 images and 501 patients
    assert len(folds_df) == 642
    assert folds_df["patient"].nunique() == 501

    # Save CSV
    folds_df.to_csv(out_path, index=False)
    print(f"Saved folds to: {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"folds_v1.csv Summary (Official Train: 642 images / 501 patients, seed {seed})")
    print("=" * 60)
    print(f"{'Fold':>5} | {'Images':>8} | {'Patients':>10}")
    print("-" * 60)
    for f in range(1, n_splits + 1):
        sub = folds_df[folds_df["fold"] == f]
        n_img = len(sub)
        n_pat = sub["patient"].nunique()
        print(f"{f:5d} | {n_img:8d} | {n_pat:10d}")
    print("-" * 60)
    print(f"{'Total':>5} | {len(folds_df):8d} | {folds_df['patient'].nunique():10d}")
    print("=" * 60 + "\n")

    return folds_df


if __name__ == "__main__":
    make_ac_folds_v1()
