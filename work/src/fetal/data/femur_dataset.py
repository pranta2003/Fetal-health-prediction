"""Femur (FL) dataset parsing and biometry utilities.

Parses landmark annotations and images for fetal femur length (FL)
from the FP and UCL biometry datasets.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import numpy as np
import pandas as pd

from fetal import config

LM_COLS = ["fl_1_x", "fl_1_y", "fl_2_x", "fl_2_y"]


def patient_id_from_name(name: str, src: str) -> str:
    """Extract patient ID string from image filename based on source dataset."""
    name_str = str(name)
    if src == "FP":
        m = re.search(r"(Patient\d+)", name_str)
        if not m:
            raise ValueError(f"Could not extract FP patient ID from {name_str}")
        return f"FP_{m.group(1)}"
    elif src == "UCL":
        m = re.match(r"(\d+)_", name_str)
        if not m:
            raise ValueError(f"Could not extract UCL patient ID from {name_str}")
        return f"UCL_{m.group(1)}"
    else:
        raise ValueError(f"Unknown source dataset: {src}")


def compute_femur_length(
    fl_1_x: float, fl_1_y: float, fl_2_x: float, fl_2_y: float, px_to_mm_rate: float
) -> float:
    """Compute Euclidean femur length in millimeters from pixel endpoints and mm rate."""
    if np.isnan(px_to_mm_rate):
        return float("nan")
    dist_px = np.hypot(fl_2_x - fl_1_x, fl_2_y - fl_1_y)
    return float(dist_px * px_to_mm_rate)


def read_split_names(path: str | Path) -> set[str] | None:
    """Read set of image filenames from split CSV file if it exists."""
    p = Path(path)
    if not p.is_file():
        return None
    d = pd.read_csv(p)
    col = "image_name" if "image_name" in d.columns else d.columns[0]
    return set(d[col].astype(str))


def load_femur_source(src: str, ann_dir: str | Path, img_dir: str | Path) -> pd.DataFrame:
    """Load and parse femur annotations for a single source (FP or UCL)."""
    ann_path = Path(ann_dir) / "Femur.csv"
    if not ann_path.is_file():
        raise FileNotFoundError(f"Femur annotation file not found at {ann_path}")

    df = pd.read_csv(ann_path)
    if src == "UCL":
        # Match AC logic excluding patient 125 if present
        df = df[~df["image_name"].str.startswith("125_")].copy()
    df = df.reset_index(drop=True)

    df["source"] = src
    df["image_path"] = df["image_name"].apply(lambda f: str(Path(img_dir) / f))
    df["patient"] = df["image_name"].apply(lambda f: patient_id_from_name(f, src))

    # Official Train/Test split assignment
    te = read_split_names(Path(ann_dir) / "Femur_Test.csv")
    tr = read_split_names(Path(ann_dir) / "Femur_Train.csv")

    if te is not None:
        df["official_split"] = df["image_name"].astype(str).apply(
            lambda f: "test" if f in te else "train"
        )
    elif tr is not None:
        df["official_split"] = df["image_name"].astype(str).apply(
            lambda f: "train" if f in tr else "test"
        )
    elif "Split" in df.columns:
        df["official_split"] = df["Split"].astype(str).str.lower()
    else:
        raise RuntimeError(f"{src}: No Femur_Train.csv / Femur_Test.csv or Split column found in {ann_dir}")

    # Computed femur length in mm
    df["fl_length_mm"] = [
        compute_femur_length(r.fl_1_x, r.fl_1_y, r.fl_2_x, r.fl_2_y, r.px_to_mm_rate)
        for r in df.itertuples()
    ]

    cols = [
        "image_name",
        "image_path",
        "source",
        "patient",
        "official_split",
        "px_to_mm_rate",
        "fl_length_mm",
    ] + LM_COLS
    return df[cols]


def build_femur_data(base: str | Path | None = None) -> pd.DataFrame:
    """Load and concatenate FP and UCL femur annotations from data root."""
    if base is None:
        base = config.DATA_DIR
    base = Path(base)

    fp = load_femur_source("FP", base / "FP" / "annotations", base / "FP" / "images" / "Femur")
    ucl = load_femur_source(
        "UCL",
        base / "UCL" / "annotations" / "px_to_mm",
        base / "UCL" / "images" / "Femur",
    )
    data = pd.concat([fp, ucl], ignore_index=True)
    return data
