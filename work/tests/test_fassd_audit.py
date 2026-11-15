"""test_fassd_audit.py: Unit tests for FASSD data loader and audit tools using synthetic data."""

from __future__ import annotations

import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytest

from fetal.data.fassd import (
    STRUCTURE_NAMES,
    create_patient_grouped_split,
    load_fassd_image_rgb,
    load_fassd_sample,
    parse_patient_id,
)
from scripts import __file__ as scripts_init_file


def _create_synthetic_fassd_dataset(root_dir: Path, n_patients: int = 10, images_per_patient: int = 3) -> tuple[Path, Path]:
    img_dir = root_dir / "IMAGES"
    npy_dir = root_dir / "ARRAY_FORMAT"
    img_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    h, w = 64, 80  # small synthetic dimensions for fast testing
    for p_idx in range(1, n_patients + 1):
        pid = f"P0{p_idx}" if p_idx < 10 else f"P0{p_idx}"
        for i_idx in range(1, images_per_patient + 1):
            stem = f"{pid}_IMG{i_idx}"
            
            # Grayscale or 3-channel image
            if (p_idx + i_idx) % 2 == 0:
                img = np.full((h, w, 3), fill_value=p_idx * 10, dtype=np.uint8)
            else:
                img = np.full((h, w), fill_value=p_idx * 10, dtype=np.uint8)

            structures = {}
            for s in STRUCTURE_NAMES:
                mask = np.zeros((h, w), dtype=np.uint8)
                # create non-empty mask for some structures
                mask[10:20, 10:20] = 1
                structures[s] = mask

            # Save PNG
            p_png = img_dir / f"{stem}.png"
            cv2.imwrite(str(p_png), img)

            # Save NPY (0-d object array with dict)
            data_dict = {"image": img, "structures": structures}
            p_npy = npy_dir / f"{stem}.npy"
            np.save(p_npy, np.array(data_dict, dtype=object))

    return img_dir, npy_dir


def test_parse_patient_id():
    assert parse_patient_id("P01_IMG1.png") == "P01"
    assert parse_patient_id("P09_02.npy") == "P09"
    assert parse_patient_id("P010_IMG5.png") == "P010"
    assert parse_patient_id("P099_01.npy") == "P099"
    assert parse_patient_id("P0100_IMG1.png") == "P0100"
    assert parse_patient_id("P0169_IMG10.png") == "P0169"
    assert parse_patient_id(Path("some/dir/P042_IMG3.png")) == "P042"

    with pytest.raises(ValueError):
        parse_patient_id("invalid_filename.png")


def test_load_fassd_sample_synthetic(tmp_path: Path):
    _, npy_dir = _create_synthetic_fassd_dataset(tmp_path, n_patients=2, images_per_patient=2)
    sample_path = npy_dir / "P01_IMG1.npy"

    sample = load_fassd_sample(sample_path)
    assert isinstance(sample, dict)
    assert "image" in sample
    assert "structures" in sample
    assert set(sample["structures"].keys()) == set(STRUCTURE_NAMES)
    for s in STRUCTURE_NAMES:
        assert sample["structures"][s].dtype == np.uint8
        assert set(np.unique(sample["structures"][s])).issubset({0, 1})


def test_load_fassd_image_rgb(tmp_path: Path):
    img_dir, npy_dir = _create_synthetic_fassd_dataset(tmp_path, n_patients=2, images_per_patient=2)
    
    rgb_from_npy = load_fassd_image_rgb(npy_dir / "P01_IMG1.npy")
    assert rgb_from_npy.ndim == 3
    assert rgb_from_npy.shape[2] == 3
    assert rgb_from_npy.dtype == np.uint8

    rgb_from_png = load_fassd_image_rgb(img_dir / "P01_IMG1.png")
    assert rgb_from_png.ndim == 3
    assert rgb_from_png.shape[2] == 3
    assert rgb_from_png.dtype == np.uint8


def test_create_patient_grouped_split():
    # 20 patients, 5 images each = 100 images
    filenames = [f"P0{p}_IMG{i}.png" if p < 10 else f"P0{p}_IMG{i}.png" for p in range(1, 21) for i in range(1, 6)]
    
    df1 = create_patient_grouped_split(filenames, seed=42, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    df2 = create_patient_grouped_split(filenames, seed=42, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    # Determinism
    pd.testing.assert_frame_equal(df1, df2)

    # Check columns
    assert list(df1.columns) == ["filename", "patient", "split"]
    assert len(df1) == len(filenames)

    # Grouped strictly by patient (0 patient overlap)
    train_patients = set(df1[df1["split"] == "train"]["patient"])
    val_patients = set(df1[df1["split"] == "val"]["patient"])
    test_patients = set(df1[df1["split"] == "test"]["patient"])

    assert len(train_patients & val_patients) == 0
    assert len(train_patients & test_patients) == 0
    assert len(val_patients & test_patients) == 0
    assert len(train_patients) + len(val_patients) + len(test_patients) == 20

    # Approximate ratios
    assert len(train_patients) == 14  # 70% of 20
    assert len(val_patients) == 3     # 15% of 20
    assert len(test_patients) == 3    # 15% of 20
