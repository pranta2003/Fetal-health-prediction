"""fassd.py: Data loader and patient-grouped splitting for the FASSD dataset.

FASSD (Fetal Abdominal Structures Segmentation Dataset Using Ultrasonic Images):
- Ultrasound images: IMAGES/*.png (1,588 files)
- Array data: ARRAY_FORMAT/*.npy (1,588 files)
  Each .npy contains a 0-d object array with item() returning:
    {
      'image': ndarray (768, 1024, 3) or (768, 1024) uint8,
      'structures': {
        'artery': (768, 1024) uint8 in {0, 1},
        'liver': (768, 1024) uint8 in {0, 1},
        'stomach': (768, 1024) uint8 in {0, 1},
        'vein': (768, 1024) uint8 in {0, 1}
      }
    }
"""

from __future__ import annotations

import os
from pathlib import Path
import random
import re
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd

STRUCTURE_NAMES = ("artery", "liver", "stomach", "vein")


def parse_patient_id(filename: str | Path) -> str:
    """Extract patient ID prefix from FASSD filename (e.g. 'P01_IMG1.png' -> 'P01', 'P0100_01.npy' -> 'P0100')."""
    name = Path(filename).name
    m = re.match(r"^(P\d+)", name)
    if not m:
        raise ValueError(f"Filename does not match expected FASSD patient pattern: {name}")
    return m.group(1)


def load_fassd_sample(npy_path: str | Path) -> dict[str, Any]:
    """Load a single FASSD sample from an ARRAY_FORMAT .npy file.

    NOTE ON allow_pickle=True:
    allow_pickle=True is strictly required because FASSD .npy files store 0-dimensional
    object arrays containing a Python dict {'image': ndarray, 'structures': dict}.
    This flag is strictly isolated to this loader function for trusted local FASSD data.
    """
    path = Path(npy_path)
    if not path.is_file():
        raise FileNotFoundError(f"FASSD sample not found: {path}")

    # allow_pickle=True strictly isolated here for 0-d object array loading:
    raw_obj = np.load(path, allow_pickle=True)
    sample = raw_obj.item()
    if not isinstance(sample, dict) or "image" not in sample or "structures" not in sample:
        raise ValueError(f"Unexpected FASSD object structure in {path}: keys={list(sample.keys()) if isinstance(sample, dict) else type(sample)}")
    return sample


def load_fassd_image_rgb(path: str | Path) -> np.ndarray:
    """Load image as uint8 RGB (H, W, 3) regardless of whether source is .npy or .png."""
    p = Path(path)
    if p.suffix.lower() == ".npy":
        sample = load_fassd_sample(p)
        img = sample["image"]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            # OpenCV convention was used when saving .npy: BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    elif p.suffix.lower() == ".png":
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {p}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    else:
        raise ValueError(f"Unsupported image format: {p.suffix}")


def create_patient_grouped_split(
    filenames: Sequence[str],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> pd.DataFrame:
    """Split FASSD filenames into train, val, test grouped strictly by patient ID.

    Ensures zero patient overlap between train, val, and test sets.
    Returns a DataFrame with columns ['filename', 'patient', 'split'].
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    # Group filenames by patient
    patient_to_files: dict[str, list[str]] = {}
    for f in filenames:
        pid = parse_patient_id(f)
        patient_to_files.setdefault(pid, []).append(Path(f).name)

    unique_patients = sorted(patient_to_files.keys())
    rng = random.Random(seed)
    shuffled_patients = list(unique_patients)
    rng.shuffle(shuffled_patients)

    n_total = len(shuffled_patients)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))

    train_patients = set(shuffled_patients[:n_train])
    val_patients = set(shuffled_patients[n_train : n_train + n_val])
    test_patients = set(shuffled_patients[n_train + n_val :])

    # Assert zero patient overlap
    assert len(train_patients & val_patients) == 0, "Patient overlap between train and val"
    assert len(train_patients & test_patients) == 0, "Patient overlap between train and test"
    assert len(val_patients & test_patients) == 0, "Patient overlap between val and test"
    assert len(train_patients) + len(val_patients) + len(test_patients) == n_total, "Patient partition incomplete"

    rows: list[dict[str, str]] = []
    for pid in sorted(patient_to_files.keys()):
        if pid in train_patients:
            split_label = "train"
        elif pid in val_patients:
            split_label = "val"
        else:
            split_label = "test"

        for f in sorted(patient_to_files[pid]):
            rows.append({
                "filename": f,
                "patient": pid,
                "split": split_label,
            })

    df = pd.DataFrame(rows)
    return df
