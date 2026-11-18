"""scripts/41_prepare_ac_kaggle_dataset.py

Packages AC data (FP + UCL images, annotations, and folds_v1.csv)
into E:\\BUBT-CSE-4th yr\\Capstone\\data\\kaggle_ac_v1 and creates/updates
private Kaggle dataset prantakumerpandit390/ac-fetal-v1.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fetal import config
from fetal.data.ac_dataset import build_data

DEFAULT_DEST = Path(r"E:\BUBT-CSE-4th yr\Capstone\data\kaggle_ac_v1")
SOURCE_DATA = Path("C:/fetal-data")


def prepare_ac_kaggle_dataset(dest_dir: Path = DEFAULT_DEST, src_data: Path = SOURCE_DATA) -> dict:
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy FP annotations and images
    fp_dest_ann = dest_dir / "FP" / "annotations"
    fp_dest_img = dest_dir / "FP" / "images" / "Abdomen"
    fp_dest_ann.mkdir(parents=True, exist_ok=True)
    fp_dest_img.mkdir(parents=True, exist_ok=True)

    fp_src_ann = src_data / "FP" / "annotations"
    fp_src_img = src_data / "FP" / "images" / "Abdomen"

    for f in fp_src_ann.glob("*.csv"):
        shutil.copy2(f, fp_dest_ann / f.name)

    fp_img_count = 0
    for f in fp_src_img.glob("*.*"):
        shutil.copy2(f, fp_dest_img / f.name)
        fp_img_count += 1

    # 2. Copy UCL annotations and images
    ucl_dest_ann = dest_dir / "UCL" / "annotations" / "px_to_mm"
    ucl_dest_img = dest_dir / "UCL" / "images" / "Abdomen"
    ucl_dest_ann.mkdir(parents=True, exist_ok=True)
    ucl_dest_img.mkdir(parents=True, exist_ok=True)

    ucl_src_ann = src_data / "UCL" / "annotations" / "px_to_mm"
    ucl_src_img = src_data / "UCL" / "images" / "Abdomen"

    for f in ucl_src_ann.glob("*.csv"):
        shutil.copy2(f, ucl_dest_ann / f.name)

    ucl_img_count = 0
    for f in ucl_src_img.glob("*.*"):
        shutil.copy2(f, ucl_dest_img / f.name)
        ucl_img_count += 1

    # 3. Copy folds_v1.csv
    folds_src = REPO_ROOT / "docs" / "results" / "ac_campaign" / "folds_v1.csv"
    shutil.copy2(folds_src, dest_dir / "folds_v1.csv")

    # 4. Write dataset-metadata.json
    meta = {
        "title": "ac-fetal-v1",
        "id": "prantakumerpandit390/ac-fetal-v1",
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True,
    }
    meta_path = dest_dir / "dataset-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 5. Verify integrity using build_data on packaged directory
    test_df = build_data(str(dest_dir))
    assert len(test_df) == 789, f"Expected 789 images, got {len(test_df)}"

    # 6. Calculate total size
    total_bytes = sum(f.stat().st_size for f in dest_dir.rglob("*") if f.is_file())
    total_mb = total_bytes / (1024 * 1024)

    stats = {
        "dest_dir": str(dest_dir),
        "fp_images": fp_img_count,
        "ucl_images": ucl_img_count,
        "total_images": fp_img_count + ucl_img_count,
        "valid_build_data_rows": len(test_df),
        "total_mb": round(total_mb, 2),
    }

    print("\n=== AC Kaggle Dataset Packaged ===")
    print(f"Destination: {dest_dir}")
    print(f"FP Images: {fp_img_count}")
    print(f"UCL Images: {ucl_img_count}")
    print(f"Total build_data rows: {len(test_df)}")
    print(f"Folds CSV: {dest_dir / 'folds_v1.csv'} (exists: {(dest_dir / 'folds_v1.csv').exists()})")
    print(f"Total size: {total_mb:.2f} MB")
    return stats


if __name__ == "__main__":
    prepare_ac_kaggle_dataset()
