"""Femur (FL) biometry dataset inspection and validation report.

Inspects FP and UCL femur annotations, checks landmark coordinates,
verifies patient-level Train/Test split isolation (leakage check),
computes femur length distribution in mm, and outputs summary metrics to JSON.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
import pandas as pd
from PIL import Image

from fetal import config
from fetal.data.femur_dataset import (
    LM_COLS,
    build_femur_data,
    compute_femur_length,
    load_femur_source,
    patient_id_from_name,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Femur (FL) Data Check and Validation Report")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(config.DATA_DIR),
        help="Root directory for fetal datasets (contains FP and UCL)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="C:/fetal-results/femur",
        help="Directory where femur_data_report.json will be saved",
    )
    return parser.parse_args()


def inspect_femur_data(data_dir: Path) -> dict:
    fp_ann_dir = data_dir / "FP" / "annotations"
    fp_img_dir = data_dir / "FP" / "images" / "Femur"
    ucl_ann_dir = data_dir / "UCL" / "annotations" / "px_to_mm"
    ucl_img_dir = data_dir / "UCL" / "images" / "Femur"
    ucl_raw_ann = data_dir / "UCL" / "annotations" / "Femur.csv"

    # Raw file path and column inspection
    fp_csv = fp_ann_dir / "Femur.csv"
    ucl_csv = ucl_ann_dir / "Femur.csv"

    fp_raw_df = pd.read_csv(fp_csv)
    ucl_cal_df = pd.read_csv(ucl_csv)
    ucl_raw_df = pd.read_csv(ucl_raw_ann) if ucl_raw_ann.is_file() else None

    # Load parsed datasets
    fp_df = load_femur_source("FP", fp_ann_dir, fp_img_dir)
    ucl_df = load_femur_source("UCL", ucl_ann_dir, ucl_img_dir)
    combined_df = pd.concat([fp_df, ucl_df], ignore_index=True)

    # 1. Image and patient counts
    total_images = len(combined_df)
    fp_images = len(fp_df)
    ucl_images = len(ucl_df)

    fp_patients = set(fp_df["patient"].unique())
    ucl_patients = set(ucl_df["patient"].unique())
    all_patients = set(combined_df["patient"].unique())

    # 2. Calibration availability (px_to_mm_rate)
    fp_missing_rate = int(fp_df["px_to_mm_rate"].isna().sum())
    fp_valid_rate = int(fp_df["px_to_mm_rate"].notna().sum())
    ucl_missing_rate = int(ucl_df["px_to_mm_rate"].isna().sum())
    ucl_valid_rate = int(ucl_df["px_to_mm_rate"].notna().sum())

    # 3. Splits
    fp_train = fp_df[fp_df["official_split"] == "train"]
    fp_test = fp_df[fp_df["official_split"] == "test"]
    ucl_train = ucl_df[ucl_df["official_split"] == "train"]
    ucl_test = ucl_df[ucl_df["official_split"] == "test"]

    comb_train = combined_df[combined_df["official_split"] == "train"]
    comb_test = combined_df[combined_df["official_split"] == "test"]

    fp_train_pts = set(fp_train["patient"].unique())
    fp_test_pts = set(fp_test["patient"].unique())
    fp_pt_overlap = sorted(list(fp_train_pts.intersection(fp_test_pts)))

    ucl_train_pts = set(ucl_train["patient"].unique())
    ucl_test_pts = set(ucl_test["patient"].unique())
    ucl_pt_overlap = sorted(list(ucl_train_pts.intersection(ucl_test_pts)))

    all_train_pts = set(comb_train["patient"].unique())
    all_test_pts = set(comb_test["patient"].unique())
    all_pt_overlap = sorted(list(all_train_pts.intersection(all_test_pts)))

    # 4. Image dimensions and landmark coordinate validation
    img_dims = []
    oob_count = 0
    neg_coords = 0
    nan_coords = 0
    degenerate_landmarks = 0

    for r in combined_df.itertuples():
        coords = [r.fl_1_x, r.fl_1_y, r.fl_2_x, r.fl_2_y]
        if any(np.isnan(c) for c in coords):
            nan_coords += 1
            continue
        if any(c < 0 for c in coords):
            neg_coords += 1
        if r.fl_1_x == r.fl_2_x and r.fl_1_y == r.fl_2_y:
            degenerate_landmarks += 1

        img_p = Path(r.image_path)
        if img_p.is_file():
            with Image.open(img_p) as im:
                w, h = im.size
                img_dims.append({"width": w, "height": h, "source": r.source})
                if r.fl_1_x >= w or r.fl_2_x >= w or r.fl_1_y >= h or r.fl_2_y >= h:
                    oob_count += 1

    dims_df = pd.DataFrame(img_dims)
    fp_dims = dims_df[dims_df["source"] == "FP"]
    ucl_dims = dims_df[dims_df["source"] == "UCL"]

    # 5. Femur Length distribution (mm)
    def stats_dict(series: pd.Series) -> dict:
        s = series.dropna()
        return {
            "count": int(len(s)),
            "min": float(np.round(s.min(), 2)),
            "q25": float(np.round(s.quantile(0.25), 2)),
            "median": float(np.round(s.median(), 2)),
            "q75": float(np.round(s.quantile(0.75), 2)),
            "max": float(np.round(s.max(), 2)),
            "mean": float(np.round(s.mean(), 2)),
            "std": float(np.round(s.std(), 2)),
        }

    fl_fp_stats = stats_dict(fp_df["fl_length_mm"])
    fl_ucl_stats = stats_dict(ucl_df["fl_length_mm"])
    fl_comb_stats = stats_dict(combined_df["fl_length_mm"])

    # 6. Comparison with AC splits
    ac_comparison = {}
    ac_fp_csv = fp_ann_dir / "Abdomen.csv"
    if ac_fp_csv.is_file():
        ac_fp = pd.read_csv(ac_fp_csv)
        ac_fp["patient"] = ac_fp["image_name"].apply(lambda f: patient_id_from_name(f, "FP"))
        ac_pts = set(ac_fp["patient"].unique())
        shared_pts = fp_patients.intersection(ac_pts)
        ac_comparison["fp_patients_shared_with_ac"] = len(shared_pts)
        ac_comparison["fp_patients_total_femur"] = len(fp_patients)
        ac_comparison["fp_patients_total_ac"] = len(ac_pts)

    report = {
        "file_paths": {
            "fp_annotations": str(fp_csv),
            "fp_train_split": str(fp_ann_dir / "Femur_Train.csv"),
            "fp_test_split": str(fp_ann_dir / "Femur_Test.csv"),
            "fp_images_dir": str(fp_img_dir),
            "ucl_calibrated_annotations": str(ucl_csv),
            "ucl_train_split": str(ucl_ann_dir / "Femur_Train.csv"),
            "ucl_test_split": str(ucl_ann_dir / "Femur_Test.csv"),
            "ucl_images_dir": str(ucl_img_dir),
            "ucl_raw_annotations": str(ucl_raw_ann),
        },
        "raw_column_names": {
            "fp": list(fp_raw_df.columns),
            "ucl_calibrated": list(ucl_cal_df.columns),
            "ucl_raw": list(ucl_raw_df.columns) if ucl_raw_df is not None else [],
        },
        "landmark_encoding": {
            "format": "Two endpoints in pixel coordinates: (fl_1_x, fl_1_y) and (fl_2_x, fl_2_y)",
            "columns": LM_COLS,
            "metric_formula": "Euclidean distance * px_to_mm_rate -> fl_length_mm",
        },
        "image_counts": {
            "fp_total": fp_images,
            "ucl_calibrated": ucl_images,
            "ucl_raw_total": len(ucl_raw_df) if ucl_raw_df is not None else ucl_images,
            "total_calibrated": total_images,
        },
        "px_to_mm_rate_availability": {
            "fp_valid": fp_valid_rate,
            "fp_missing": fp_missing_rate,
            "ucl_valid": ucl_valid_rate,
            "ucl_missing": ucl_missing_rate,
            "total_valid": fp_valid_rate + ucl_valid_rate,
            "total_missing": fp_missing_rate + ucl_missing_rate,
        },
        "patient_counts": {
            "fp_unique_patients": len(fp_patients),
            "ucl_unique_patients": len(ucl_patients),
            "total_unique_patients": len(all_patients),
        },
        "split_counts": {
            "fp_train_images": len(fp_train),
            "fp_test_images": len(fp_test),
            "ucl_train_images": len(ucl_train),
            "ucl_test_images": len(ucl_test),
            "total_train_images": len(comb_train),
            "total_test_images": len(comb_test),
            "fp_train_patients": len(fp_train_pts),
            "fp_test_patients": len(fp_test_pts),
            "ucl_train_patients": len(ucl_train_pts),
            "ucl_test_patients": len(ucl_test_pts),
            "total_train_patients": len(all_train_pts),
            "total_test_patients": len(all_test_pts),
        },
        "leakage_check": {
            "fp_patient_overlap": fp_pt_overlap,
            "ucl_patient_overlap": ucl_pt_overlap,
            "total_patient_overlap": all_pt_overlap,
            "leakage_detected": bool(len(all_pt_overlap) > 0),
        },
        "landmark_quality": {
            "nan_coordinates": nan_coords,
            "negative_coordinates": neg_coords,
            "degenerate_landmarks": degenerate_landmarks,
            "out_of_bounds_landmarks": oob_count,
        },
        "image_dimensions": {
            "fp_width_min": int(fp_dims["width"].min()) if len(fp_dims) else None,
            "fp_width_max": int(fp_dims["width"].max()) if len(fp_dims) else None,
            "fp_height_min": int(fp_dims["height"].min()) if len(fp_dims) else None,
            "fp_height_max": int(fp_dims["height"].max()) if len(fp_dims) else None,
            "ucl_width_min": int(ucl_dims["width"].min()) if len(ucl_dims) else None,
            "ucl_width_max": int(ucl_dims["width"].max()) if len(ucl_dims) else None,
            "ucl_height_min": int(ucl_dims["height"].min()) if len(ucl_dims) else None,
            "ucl_height_max": int(ucl_dims["height"].max()) if len(ucl_dims) else None,
        },
        "femur_length_distribution_mm": {
            "fp": fl_fp_stats,
            "ucl": fl_ucl_stats,
            "combined": fl_comb_stats,
        },
        "ac_split_comparison": ac_comparison,
    }
    return report


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    print(f"=== FEMUR (FL) DATA INSPECTION AND VALIDATION ===")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {out_dir}")

    report = inspect_femur_data(data_dir)

    print("\n--- 1. File Paths ---")
    for k, v in report["file_paths"].items():
        print(f"  {k}: {v}")

    print("\n--- 2. Column Names ---")
    for k, v in report["raw_column_names"].items():
        print(f"  {k}: {v}")

    print("\n--- 3. Landmark Encoding ---")
    print(f"  Format:  {report['landmark_encoding']['format']}")
    print(f"  Columns: {report['landmark_encoding']['columns']}")
    print(f"  Formula: {report['landmark_encoding']['metric_formula']}")

    print("\n--- 4. Image & Patient Counts ---")
    print(f"  FP Images:           {report['image_counts']['fp_total']} ({report['patient_counts']['fp_unique_patients']} patients)")
    print(f"  UCL Images:          {report['image_counts']['ucl_calibrated']} ({report['patient_counts']['ucl_unique_patients']} patients)")
    print(f"  Total Images:        {report['image_counts']['total_calibrated']} ({report['patient_counts']['total_unique_patients']} patients)")

    print("\n--- 5. Calibration Availability (px_to_mm_rate) ---")
    print(f"  FP:   {report['px_to_mm_rate_availability']['fp_valid']} valid, {report['px_to_mm_rate_availability']['fp_missing']} missing")
    print(f"  UCL:  {report['px_to_mm_rate_availability']['ucl_valid']} valid, {report['px_to_mm_rate_availability']['ucl_missing']} missing")
    print(f"  All:  {report['px_to_mm_rate_availability']['total_valid']} valid, {report['px_to_mm_rate_availability']['total_missing']} missing")

    print("\n--- 6. Official Train/Test Split Counts ---")
    sc = report["split_counts"]
    print(f"  FP:    Train {sc['fp_train_images']} images ({sc['fp_train_patients']} patients) | Test {sc['fp_test_images']} images ({sc['fp_test_patients']} patients)")
    print(f"  UCL:   Train {sc['ucl_train_images']} images ({sc['ucl_train_patients']} patients) | Test {sc['ucl_test_images']} images ({sc['ucl_test_patients']} patients)")
    print(f"  Total: Train {sc['total_train_images']} images ({sc['total_train_patients']} patients) | Test {sc['total_test_images']} images ({sc['total_test_patients']} patients)")

    print("\n--- 7. Leakage Check (Patient Overlap Train vs Test) ---")
    lc = report["leakage_check"]
    print(f"  FP Overlap:     {len(lc['fp_patient_overlap'])} patients ({lc['fp_patient_overlap']})")
    print(f"  UCL Overlap:    {len(lc['ucl_patient_overlap'])} patients ({lc['ucl_patient_overlap']})")
    print(f"  Total Overlap:  {len(lc['total_patient_overlap'])} patients ({lc['total_patient_overlap']})")
    print(f"  Leakage Status: {'PASSED (Zero Leakage)' if not lc['leakage_detected'] else 'FAILED (LEAKAGE DETECTED)'}")

    print("\n--- 8. Landmark Quality Check ---")
    lq = report["landmark_quality"]
    print(f"  NaN Coordinates:           {lq['nan_coordinates']}")
    print(f"  Negative Coordinates:      {lq['negative_coordinates']}")
    print(f"  Degenerate Landmarks:      {lq['degenerate_landmarks']}")
    print(f"  Out of Bounds Coordinates: {lq['out_of_bounds_landmarks']}")

    print("\n--- 9. Image Dimensions (Width x Height) ---")
    idim = report["image_dimensions"]
    print(f"  FP:  Width [{idim['fp_width_min']}, {idim['fp_width_max']}], Height [{idim['fp_height_min']}, {idim['fp_height_max']}]")
    print(f"  UCL: Width [{idim['ucl_width_min']}, {idim['ucl_width_max']}], Height [{idim['ucl_height_min']}, {idim['ucl_height_max']}]")

    print("\n--- 10. Femur Length Distribution (mm) ---")
    fl = report["femur_length_distribution_mm"]
    print(f"  FP:       min={fl['fp']['min']:.2f}, median={fl['fp']['median']:.2f}, max={fl['fp']['max']:.2f}, mean={fl['fp']['mean']:.2f} +/- {fl['fp']['std']:.2f} mm")
    print(f"  UCL:      min={fl['ucl']['min']:.2f}, median={fl['ucl']['median']:.2f}, max={fl['ucl']['max']:.2f}, mean={fl['ucl']['mean']:.2f} +/- {fl['ucl']['std']:.2f} mm")
    print(f"  Combined: min={fl['combined']['min']:.2f}, median={fl['combined']['median']:.2f}, max={fl['combined']['max']:.2f}, mean={fl['combined']['mean']:.2f} +/- {fl['combined']['std']:.2f} mm")

    if report["ac_split_comparison"]:
        ac_c = report["ac_split_comparison"]
        print("\n--- 11. Cross-Anatomy Patient Overlap with AC ---")
        print(f"  FP Patients with both Femur and AC: {ac_c['fp_patients_shared_with_ac']} (out of {ac_c['fp_patients_total_femur']} Femur patients)")

    # Save to out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "femur_data_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to: {report_file}")


if __name__ == "__main__":
    main()
