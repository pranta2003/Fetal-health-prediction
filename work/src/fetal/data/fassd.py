# FIELD: FASSD
# MODEL: EfficientNet-B4 Multilabel U-Net
# STEP: dataset_loading
# USED BY: pipeline imports across work/scripts and work/tests

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
STRUCTURE_BITFLAGS = {
    "artery": 1,
    "liver": 2,
    "stomach": 4,
    "vein": 8,
}


def encode_multilabel_mask(masks: dict[str, np.ndarray]) -> np.ndarray:
    """Encode dictionary of binary masks into a single uint8 bit-flag mask.

    Bit flags:
      artery  = 1 (bit 0)
      liver   = 2 (bit 1)
      stomach = 4 (bit 2)
      vein    = 8 (bit 3)
    """
    first_mask = next(iter(masks.values()))
    combined = np.zeros(first_mask.shape[:2], dtype=np.uint8)
    for name, flag in STRUCTURE_BITFLAGS.items():
        if name in masks:
            combined |= (masks[name] > 0).astype(np.uint8) * flag
    return combined


def decode_multilabel_mask(combined_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Decode a single uint8 bit-flag mask into a dictionary of 4 binary masks."""
    return {
        name: ((combined_mask & flag) > 0).astype(np.uint8)
        for name, flag in STRUCTURE_BITFLAGS.items()
    }


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


def build_patient_groups_from_hashes(
    img_dir: Path | str,
    filenames: Sequence[str] | None = None,
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, str]]:
    """Build patient groups using Union-Find over identical raw image SHA-256 hashes.

    Returns:
        patient_to_group_id: dict mapping each patient ID to its group ID (e.g. 'G001').
        group_id_to_patients: dict mapping group ID to list of patient IDs.
        file_to_hash: dict mapping filename to its SHA-256 hash.
    """
    import hashlib

    img_path = Path(img_dir)
    if filenames is None:
        filenames = sorted([f for f in os.listdir(img_path) if f.lower().endswith(".png")])

    file_to_hash: dict[str, str] = {}
    hash_to_patients: dict[str, set[str]] = {}
    all_patients: set[str] = set()

    for f in filenames:
        pid = parse_patient_id(f)
        all_patients.add(pid)
        p_img = img_path / f
        with open(p_img, "rb") as fp:
            h = hashlib.sha256(fp.read()).hexdigest()
        file_to_hash[Path(f).name] = h
        hash_to_patients.setdefault(h, set()).add(pid)

    # Union-Find
    parent: dict[str, str] = {p: p for p in all_patients}

    def find(p: str) -> str:
        if parent[p] != p:
            parent[p] = find(parent[p])
        return parent[p]

    def union(p1: str, p2: str) -> None:
        r1, r2 = find(p1), find(p2)
        if r1 != r2:
            parent[r1] = r2

    for h, pats in hash_to_patients.items():
        if len(pats) > 1:
            plist = sorted(list(pats))
            for p in plist[1:]:
                union(plist[0], p)

    # Group patients by root
    root_to_patients: dict[str, list[str]] = {}
    for p in sorted(all_patients):
        r = find(p)
        root_to_patients.setdefault(r, []).append(p)

    sorted_roots = sorted(root_to_patients.keys())
    patient_to_group_id: dict[str, str] = {}
    group_id_to_patients: dict[str, list[str]] = {}

    for idx, r in enumerate(sorted_roots, 1):
        gid = f"G{idx:03d}"
        pats = root_to_patients[r]
        group_id_to_patients[gid] = pats
        for p in pats:
            patient_to_group_id[p] = gid

    return patient_to_group_id, group_id_to_patients, file_to_hash


def create_group_split_v2(
    filenames: Sequence[str],
    patient_to_group: dict[str, str],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> pd.DataFrame:
    """Split FASSD filenames into train, val, test grouped strictly by patient GROUP.

    Ensures zero patient overlap and zero image-hash overlap between splits.
    Returns DataFrame with columns ['filename', 'patient', 'group', 'split'].
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    unique_groups = sorted(list(set(patient_to_group.values())))
    rng = random.Random(seed)
    shuffled_groups = list(unique_groups)
    rng.shuffle(shuffled_groups)

    n_total = len(shuffled_groups)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))

    train_groups = set(shuffled_groups[:n_train])
    val_groups = set(shuffled_groups[n_train : n_train + n_val])
    test_groups = set(shuffled_groups[n_train + n_val :])

    assert len(train_groups & val_groups) == 0
    assert len(train_groups & test_groups) == 0
    assert len(val_groups & test_groups) == 0
    assert len(train_groups) + len(val_groups) + len(test_groups) == n_total

    group_split_map: dict[str, str] = {}
    for g in train_groups:
        group_split_map[g] = "train"
    for g in val_groups:
        group_split_map[g] = "val"
    for g in test_groups:
        group_split_map[g] = "test"

    rows: list[dict[str, str]] = []
    for f in sorted(filenames):
        fname = Path(f).name
        pid = parse_patient_id(fname)
        gid = patient_to_group[pid]
        sp = group_split_map[gid]
        rows.append({
            "filename": fname,
            "patient": pid,
            "group": gid,
            "split": sp,
        })

    df = pd.DataFrame(rows)
    return df

