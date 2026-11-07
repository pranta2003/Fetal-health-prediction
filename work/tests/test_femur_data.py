"""Fast unit tests for femur dataset parsing and length calculation."""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from fetal.data.femur_dataset import (
    LM_COLS,
    build_femur_data,
    compute_femur_length,
    load_femur_source,
    patient_id_from_name,
    read_split_names,
)


def test_patient_id_from_name_fp():
    assert patient_id_from_name("Patient00888_Plane5_1_of_1.png", "FP") == "FP_Patient00888"
    assert patient_id_from_name("Patient00001_Plane5_2_of_2.png", "FP") == "FP_Patient00001"
    with pytest.raises(ValueError, match="Could not extract FP patient ID"):
        patient_id_from_name("InvalidFilename.png", "FP")


def test_patient_id_from_name_ucl():
    assert patient_id_from_name("042_FL.png", "UCL") == "UCL_042"
    assert patient_id_from_name("105_2FL.png", "UCL") == "UCL_105"
    with pytest.raises(ValueError, match="Could not extract UCL patient ID"):
        patient_id_from_name("invalid_fl.png", "UCL")


def test_patient_id_unknown_src():
    with pytest.raises(ValueError, match="Unknown source dataset"):
        patient_id_from_name("042_FL.png", "UNKNOWN")


def test_compute_femur_length():
    # 3-4-5 right triangle: dist = 5.0 px; rate = 0.2 mm/px -> 1.0 mm
    fl = compute_femur_length(0.0, 0.0, 3.0, 4.0, 0.2)
    assert pytest.approx(fl, rel=1e-5) == 1.0

    # Horizontal: dx = 20 px, dy = 0 px, rate = 0.05 mm/px -> 1.0 mm
    fl_horiz = compute_femur_length(10.0, 50.0, 30.0, 50.0, 0.05)
    assert pytest.approx(fl_horiz, rel=1e-5) == 1.0

    # Zero length (degenerate endpoints)
    fl_zero = compute_femur_length(15.0, 15.0, 15.0, 15.0, 0.1)
    assert fl_zero == 0.0

    # NaN px_to_mm_rate
    fl_nan = compute_femur_length(0.0, 0.0, 10.0, 10.0, float("nan"))
    assert math.isnan(fl_nan)


def test_read_split_names(tmp_path: Path):
    non_existent = tmp_path / "does_not_exist.csv"
    assert read_split_names(non_existent) is None

    split_csv = tmp_path / "split.csv"
    df = pd.DataFrame({"image_name": ["img1.png", "img2.png"]})
    df.to_csv(split_csv, index=False)
    names = read_split_names(split_csv)
    assert names == {"img1.png", "img2.png"}


def test_load_femur_source_synthetic_fp(tmp_path: Path):
    ann_dir = tmp_path / "FP" / "annotations"
    img_dir = tmp_path / "FP" / "images" / "Femur"
    ann_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)

    femur_df = pd.DataFrame(
        {
            "image_name": [
                "Patient001_Plane5_1_of_1.png",
                "Patient002_Plane5_1_of_1.png",
                "Patient003_Plane5_1_of_1.png",
            ],
            "scale": [1.0, 1.0, 1.0],
            "center_w": [100, 100, 100],
            "center_h": [100, 100, 100],
            "fl_1_x": [10.0, 20.0, 30.0],
            "fl_1_y": [10.0, 20.0, 30.0],
            "fl_2_x": [40.0, 20.0, 30.0],
            "fl_2_y": [50.0, 60.0, 30.0],
            "px_to_mm_rate": [0.1, np.nan, 0.2],
        }
    )
    femur_df.to_csv(ann_dir / "Femur.csv", index=False)

    train_df = pd.DataFrame({"image_name": ["Patient001_Plane5_1_of_1.png"]})
    train_df.to_csv(ann_dir / "Femur_Train.csv", index=False)
    test_df = pd.DataFrame(
        {"image_name": ["Patient002_Plane5_1_of_1.png", "Patient003_Plane5_1_of_1.png"]}
    )
    test_df.to_csv(ann_dir / "Femur_Test.csv", index=False)

    loaded = load_femur_source("FP", ann_dir, img_dir)
    assert len(loaded) == 3
    assert set(loaded.columns) == {
        "image_name",
        "image_path",
        "source",
        "patient",
        "official_split",
        "px_to_mm_rate",
        "fl_length_mm",
        *LM_COLS,
    }
    assert loaded.loc[0, "patient"] == "FP_Patient001"
    assert loaded.loc[0, "official_split"] == "train"
    assert pytest.approx(loaded.loc[0, "fl_length_mm"], rel=1e-5) == 5.0  # 50 px * 0.1

    assert loaded.loc[1, "patient"] == "FP_Patient002"
    assert loaded.loc[1, "official_split"] == "test"
    assert math.isnan(loaded.loc[1, "fl_length_mm"])

    assert loaded.loc[2, "patient"] == "FP_Patient003"
    assert loaded.loc[2, "fl_length_mm"] == 0.0


def test_load_femur_source_synthetic_ucl_excludes_125(tmp_path: Path):
    ann_dir = tmp_path / "UCL" / "annotations"
    img_dir = tmp_path / "UCL" / "images" / "Femur"
    ann_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)

    femur_df = pd.DataFrame(
        {
            "image_name": ["001_FL.png", "125_FL.png"],
            "fl_1_x": [10.0, 10.0],
            "fl_1_y": [10.0, 10.0],
            "fl_2_x": [40.0, 40.0],
            "fl_2_y": [50.0, 50.0],
            "px_to_mm_rate": [0.1, 0.1],
        }
    )
    femur_df.to_csv(ann_dir / "Femur.csv", index=False)

    train_df = pd.DataFrame({"image_name": ["001_FL.png"]})
    train_df.to_csv(ann_dir / "Femur_Train.csv", index=False)

    loaded = load_femur_source("UCL", ann_dir, img_dir)
    assert len(loaded) == 1
    assert loaded.iloc[0]["image_name"] == "001_FL.png"
    assert loaded.iloc[0]["patient"] == "UCL_001"


def test_build_femur_data_synthetic(tmp_path: Path):
    fp_ann = tmp_path / "FP" / "annotations"
    fp_img = tmp_path / "FP" / "images" / "Femur"
    ucl_ann = tmp_path / "UCL" / "annotations" / "px_to_mm"
    ucl_img = tmp_path / "UCL" / "images" / "Femur"

    fp_ann.mkdir(parents=True)
    fp_img.mkdir(parents=True)
    ucl_ann.mkdir(parents=True)
    ucl_img.mkdir(parents=True)

    pd.DataFrame(
        {
            "image_name": ["Patient001_Plane5_1_of_1.png"],
            "fl_1_x": [0.0],
            "fl_1_y": [0.0],
            "fl_2_x": [3.0],
            "fl_2_y": [4.0],
            "px_to_mm_rate": [0.1],
        }
    ).to_csv(fp_ann / "Femur.csv", index=False)
    pd.DataFrame({"image_name": ["Patient001_Plane5_1_of_1.png"]}).to_csv(
        fp_ann / "Femur_Train.csv", index=False
    )

    pd.DataFrame(
        {
            "image_name": ["010_FL.png"],
            "fl_1_x": [0.0],
            "fl_1_y": [0.0],
            "fl_2_x": [6.0],
            "fl_2_y": [8.0],
            "px_to_mm_rate": [0.1],
        }
    ).to_csv(ucl_ann / "Femur.csv", index=False)
    pd.DataFrame({"image_name": ["010_FL.png"]}).to_csv(ucl_ann / "Femur_Test.csv", index=False)

    combined = build_femur_data(tmp_path)
    assert len(combined) == 2
    assert set(combined["source"]) == {"FP", "UCL"}
    assert set(combined["official_split"]) == {"train", "test"}
