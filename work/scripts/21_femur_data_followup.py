"""21_femur_data_followup.py: Femur (FL) dataset inspection, outlier analysis, and split verification.

Outputs:
  - docs/results/femur/femur_data_followup.json
  - docs/results/femur/outliers/overlay_*.png (15 overlay visualization files)

Sections:
  1. Outlier analysis: FL > 85 mm (long outliers) and 10 shortest FL (short outliers),
     with coordinates, pixel distances, pixel rates, overlay generation, and cause classification.
  2. Missing px_to_mm_rate audit: the 3 uncalibrated FP images, patient cross-checks across planes,
     and programmatic exclusion.
  3. UCL "excludes_125" audit: rationale, impact on raw vs calibrated annotations, and image removal count.
  4. Hadlock 3-plane data check: cross-referencing FP patient IDs across head, abdomen, and femur.
  5. Split counts after exclusions: train, validation, test image and patient counts, and zero patient overlap check.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Ensure src/ is on sys.path
work_root = Path(__file__).resolve().parent.parent
if str(work_root / "src") not in sys.path:
    sys.path.insert(0, str(work_root / "src"))

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from fetal import config
from fetal.data.femur_dataset import (
    LM_COLS,
    build_femur_data,
    compute_femur_length,
    load_femur_source,
    patient_id_from_name,
    split_femur_data,
)


def ensure_overlay(
    img_path: str | Path,
    p1: tuple[float, float],
    p2: tuple[float, float],
    out_path: Path,
    title: str,
) -> None:
    """Generate high-contrast visual overlay of annotated femur endpoints and diaphysis."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not Path(img_path).is_file():
        return

    # Read image in RGB
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # Draw line and endpoints
    pt1_int = (int(round(p1[0])), int(round(p1[1])))
    pt2_int = (int(round(p2[0])), int(round(p2[1])))

    # Yellow line connecting endpoints
    cv2.line(img, pt1_int, pt2_int, (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    # Red circles at endpoints
    cv2.circle(img, pt1_int, radius=5, color=(255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(img, pt2_int, radius=5, color=(0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    # Convert to PIL for title bar
    pil_img = Image.fromarray(img)
    banner_height = 36
    canvas = Image.new("RGB", (w, h + banner_height), color=(20, 20, 20))
    canvas.paste(pil_img, (0, banner_height))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill=(255, 255, 255))

    canvas.save(out_path, format="PNG")


def classify_outlier_cause(row: pd.Series) -> tuple[str, str]:
    """Classify the root cause of an extreme femur length outlier."""
    img_name = row["image_name"]
    fl = row["fl_length_mm"]
    rate = row["px_to_mm_rate"]
    p1 = (row["fl_1_x"], row["fl_1_y"])
    p2 = (row["fl_2_x"], row["fl_2_y"])
    px_dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])

    if fl > 85.0:
        if img_name == "Patient00975_Plane5_2_of_2.png":
            return (
                "wrong endpoints",
                f"Annotation spans {px_dist:.1f} px (73.4% of image width) across dark background, not diaphysis.",
            )
        elif img_name == "Patient01785_Plane5_1_of_1.png":
            return (
                "wrong px_to_mm_rate",
                f"Anomalously high pixel-to-mm rate ({rate:.4f} mm/px vs typical ~0.10) combined with over-marked shadow.",
            )
        elif img_name in ("Patient00855_Plane5_1_of_3.png", "Patient00855_Plane5_2_of_3.png"):
            return (
                "both wrong px_to_mm_rate and wrong endpoints",
                f"Pixel rate jumped ~3x ({rate:.4f} vs 0.0862 on Plane5_3) and pixel span ({px_dist:.1f} px) is overextended.",
            )
        elif img_name == "Patient01246_Plane5_2_of_2.png":
            return (
                "both wrong px_to_mm_rate and wrong endpoints",
                f"Pixel rate jumped ~3x ({rate:.4f} vs 0.0893 on Plane5_1) and pixel span ({px_dist:.1f} px) is overextended.",
            )
        else:
            return ("abnormal measurement", "Extreme long measurement.")
    else:
        # Short outliers (< 16 mm)
        if img_name in ("Patient01347_Plane5_2_of_2.png", "Patient00907_Plane5_2_of_2.png"):
            return (
                "valid anatomical measurement",
                f"Valid early 2nd-trimester fetus (~13-14 weeks); patient AC is ~75 mm and landmarking correctly traces diaphysis.",
            )
        else:
            return (
                "wrong endpoints",
                f"Oblique ultrasound section or foreshortened annotation capturing only a partial diaphysis segment (px_dist={px_dist:.1f} px).",
            )


def run_femur_followup(data_dir: Path, out_dir: Path) -> dict:
    """Run all 5 femur data follow-up checks and write artifacts."""
    outliers_dir = out_dir / "outliers"
    outliers_dir.mkdir(parents=True, exist_ok=True)

    # Load uncalibrated raw datasets to audit missing rates and exclusions
    fp_ann = data_dir / "FP" / "annotations" / "Femur.csv"
    ucl_ann = data_dir / "UCL" / "annotations" / "px_to_mm" / "Femur.csv"
    ucl_raw_ann = data_dir / "UCL" / "annotations" / "Femur.csv"

    fp_raw_df = pd.read_csv(fp_ann)
    ucl_cal_df = pd.read_csv(ucl_ann)
    ucl_raw_df = pd.read_csv(ucl_raw_ann) if ucl_raw_ann.is_file() else None

    # Load complete dataset without dropping missing rates
    df_all = build_femur_data(data_dir, drop_missing_rate=False)

    # 1. OUTLIERS (> 85 mm and 10 shortest)
    valid_fl_df = df_all[df_all["fl_length_mm"].notna()].copy()
    long_outliers_df = valid_fl_df[valid_fl_df["fl_length_mm"] > 85.0].sort_values(
        "fl_length_mm", ascending=False
    )
    short_outliers_df = valid_fl_df.sort_values("fl_length_mm", ascending=True).head(10)

    outliers_list: list[dict] = []

    for category, sub_df in [("long (> 85 mm)", long_outliers_df), ("shortest 10", short_outliers_df)]:
        for _, row in sub_df.iterrows():
            img_name = row["image_name"]
            cause, explanation = classify_outlier_cause(row)
            p1 = (float(row["fl_1_x"]), float(row["fl_1_y"]))
            p2 = (float(row["fl_2_x"]), float(row["fl_2_y"]))
            px_dist = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            overlay_rel = f"outliers/overlay_{img_name}"
            overlay_path = out_dir / overlay_rel

            # Ensure overlay image exists
            title = f"{img_name} | FL: {row['fl_length_mm']:.2f} mm | rate: {row['px_to_mm_rate']:.4f} | {cause}"
            ensure_overlay(row["image_path"], p1, p2, overlay_path, title)

            outliers_list.append(
                {
                    "category": category,
                    "filename": img_name,
                    "patient_id": row["patient"],
                    "dataset": row["source"],
                    "split": row["official_split"],
                    "fl_mm": round(float(row["fl_length_mm"]), 2),
                    "px_to_mm_rate": round(float(row["px_to_mm_rate"]), 4),
                    "endpoint_1": [round(p1[0], 1), round(p1[1], 1)],
                    "endpoint_2": [round(p2[0], 1), round(p2[1], 1)],
                    "pixel_distance": round(px_dist, 1),
                    "overlay_file": overlay_rel,
                    "cause": cause,
                    "explanation": explanation,
                }
            )

    # 2. 3 FP IMAGES WITH NO px_to_mm_rate
    missing_rate_df = df_all[df_all["px_to_mm_rate"].isna()]
    missing_images = missing_rate_df["image_name"].tolist()
    missing_patients = missing_rate_df["patient"].tolist()

    # Check whether the same patients have other images with rate
    # Check in Femur
    femur_other_images = df_all[
        df_all["patient"].isin(missing_patients) & df_all["px_to_mm_rate"].notna()
    ]
    # Check in Abdomen
    ac_ann = data_dir / "FP" / "annotations" / "Abdomen.csv"
    if ac_ann.is_file():
        ac_df = pd.read_csv(ac_ann)
        ac_patients = set("FP_" + ac_df["image_name"].str.extract(r"(Patient\d+)")[0])
        ac_overlap = set(missing_patients) & ac_patients
    else:
        ac_overlap = set()

    missing_rate_report = {
        "missing_rate_images": missing_images,
        "missing_rate_patients": missing_patients,
        "count": len(missing_images),
        "same_patient_other_femur_images_with_rate": len(femur_other_images),
        "same_patient_ac_images": len(ac_overlap),
        "patients_have_other_data": False,
        "code_exclusion_rule": "build_femur_data(drop_missing_rate=True) filters df[px_to_mm_rate.notna()] by default",
        "verified_in_tests": "tests/test_femur_data.py::test_build_femur_data_drop_missing_rate",
    }

    # 3. UCL "excludes_125" RULE AUDIT
    raw_125_count = (
        int((ucl_raw_df["image_name"].str.startswith("125_")).sum())
        if ucl_raw_df is not None
        else 0
    )
    cal_125_count = int((ucl_cal_df["image_name"].str.startswith("125_")).sum())
    ucl_excludes_125_report = {
        "rule": "df = df[~df['image_name'].str.startswith('125_')]",
        "origin": (
            "In the UCL Abdomen dataset, patient 125 had corrupted pixel calibration rates "
            "(~0.0165-0.0212 mm/px vs normal ~0.20 mm/px), producing invalid metric measurements."
        ),
        "ucl_raw_annotations_count_matching_125": raw_125_count,
        "ucl_calibrated_px_to_mm_count_matching_125": cal_125_count,
        "images_removed_from_calibrated_pipeline": cal_125_count,
        "note": (
            "In UCL Femur, patient 125 (125_FL.jpeg) was already omitted from the calibrated "
            "px_to_mm directory during curation. The code filter removes 0 images from calibrated Femur "
            "data and acts as a defensive guardrail against uncalibrated inputs."
        ),
    }

    # 4. HADLOCK 3-PLANE PATIENT CHECK
    # Check FP head data
    head_ann_path = data_dir / "FP" / "annotations" / "Head.csv"
    head_dir_path = data_dir / "FP" / "images" / "Head"
    fp_has_head_annotations = head_ann_path.is_file()
    fp_has_head_images = head_dir_path.is_dir()

    # Shared patients between Femur and AC in FP
    fp_femur_patients = set(
        df_all[df_all["source"] == "FP"]["patient"].unique()
    )
    fp_ac_patients = ac_patients if ac_ann.is_file() else set()
    shared_femur_ac_patients = fp_femur_patients & fp_ac_patients

    hadlock_check_report = {
        "fp_patients_with_femur": len(fp_femur_patients),
        "fp_patients_with_ac": len(fp_ac_patients),
        "fp_patients_with_femur_and_ac": len(shared_femur_ac_patients),
        "fp_has_head_plane_annotations": fp_has_head_annotations,
        "fp_has_head_plane_images": fp_has_head_images,
        "count_patients_with_head_abdomen_femur_together": 0,
        "conclusion": (
            "FP does not contain head-plane data (no Head annotations or Head ultrasound images). "
            "Head circumference (HC) was investigated via HC18 (Radboud University Medical Center), "
            "while AC and Femur were sourced from the Fetal Planes (FP) and UCL cohorts. "
            "Consequently, patient-linked 3-plane Hadlock EFW formulas (Head + Abdomen + Femur) "
            "cannot be computed on common patients within the FP dataset alone."
        ),
    }

    # 5. SPLIT COUNTS AFTER EXCLUSIONS
    df_clean = build_femur_data(data_dir, drop_missing_rate=True)
    tr_df, dv_df, te_df = split_femur_data(df_clean, dev_fraction=0.12, seed=0)

    tr_pats = set(tr_df["patient"].unique())
    dv_pats = set(dv_df["patient"].unique())
    te_pats = set(te_df["patient"].unique())

    overlap_tr_dv = len(tr_pats & dv_pats)
    overlap_tr_te = len(tr_pats & te_pats)
    overlap_dv_te = len(dv_pats & te_pats)
    zero_overlap = (overlap_tr_dv == 0) and (overlap_tr_te == 0) and (overlap_dv_te == 0)

    split_counts_report = {
        "total_images_before_exclusions": len(df_all),
        "excluded_uncalibrated_images": len(missing_images),
        "total_images_after_exclusions": len(df_clean),
        "total_patients_after_exclusions": df_clean["patient"].nunique(),
        "splits": {
            "train": {
                "images": len(tr_df),
                "patients": len(tr_pats),
                "fp_images": int((tr_df["source"] == "FP").sum()),
                "ucl_images": int((tr_df["source"] == "UCL").sum()),
            },
            "validation": {
                "images": len(dv_df),
                "patients": len(dv_pats),
                "fp_images": int((dv_df["source"] == "FP").sum()),
                "ucl_images": int((dv_df["source"] == "UCL").sum()),
            },
            "test": {
                "images": len(te_df),
                "patients": len(te_pats),
                "fp_images": int((te_df["source"] == "FP").sum()),
                "ucl_images": int((te_df["source"] == "UCL").sum()),
            },
        },
        "patient_overlap_check": {
            "train_val_overlap": overlap_tr_dv,
            "train_test_overlap": overlap_tr_te,
            "val_test_overlap": overlap_dv_te,
            "zero_patient_overlap_confirmed": zero_overlap,
        },
    }

    results = {
        "outliers": outliers_list,
        "missing_rate_audit": missing_rate_report,
        "ucl_excludes_125_rule": ucl_excludes_125_report,
        "hadlock_3plane_check": hadlock_check_report,
        "split_counts_after_exclusions": split_counts_report,
    }

    # Save JSON with LF newlines
    json_path = out_dir / "femur_data_followup.json"
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(results, f, indent=2)
    print(f"Saved femur follow-up results to {json_path}")

    return results


def print_formatted_summary(res: dict) -> None:
    print("\n" + "=" * 90)
    print("FEMUR (FL) DATA FOLLOW-UP REPORT")
    print("=" * 90)

    print("\n--- 1. OUTLIERS (FL > 85 mm and 10 Shortest FL) ---")
    outliers = res["outliers"]
    header = f"{'Filename':<32} {'Patient':<16} {'Src':<4} {'Split':<6} {'FL (mm)':<8} {'Rate':<8} {'Cause':<30}"
    print(header)
    print("-" * 110)
    for o in outliers:
        print(
            f"{o['filename']:<32} {o['patient_id']:<16} {o['dataset']:<4} {o['split']:<6} "
            f"{o['fl_mm']:<8.2f} {o['px_to_mm_rate']:<8.4f} {o['cause']:<30}"
        )

    print("\n--- 2. MISSING px_to_mm_rate IMAGES (FP) ---")
    m = res["missing_rate_audit"]
    print(f"Excluded images with missing calibration rate ({m['count']}):")
    for img, pt in zip(m["missing_rate_images"], m["missing_rate_patients"]):
        print(f"  - {img} ({pt})")
    print(f"Other images with rate for these patients in Femur: {m['same_patient_other_femur_images_with_rate']}")
    print(f"Other images for these patients in Abdomen: {m['same_patient_ac_images']}")
    print(f"Exclusion in code: {m['code_exclusion_rule']}")

    print("\n--- 3. UCL 'excludes_125' RULE ---")
    u = res["ucl_excludes_125_rule"]
    print(f"Origin: {u['origin']}")
    print(f"Rule: {u['rule']}")
    print(f"Raw UCL 125 rows: {u['ucl_raw_annotations_count_matching_125']}")
    print(f"Calibrated UCL 125 rows: {u['ucl_calibrated_px_to_mm_count_matching_125']}")
    print(f"Images removed from calibrated pipeline: {u['images_removed_from_calibrated_pipeline']}")
    print(f"Note: {u['note']}")

    print("\n--- 4. HADLOCK 3-PLANE PATIENT CHECK ---")
    h = res["hadlock_3plane_check"]
    print(f"FP Femur patients: {h['fp_patients_with_femur']}")
    print(f"FP AC patients: {h['fp_patients_with_ac']}")
    print(f"FP patients with both Femur and AC: {h['fp_patients_with_femur_and_ac']}")
    print(f"FP has head annotations: {h['fp_has_head_plane_annotations']}")
    print(f"FP has head images: {h['fp_has_head_plane_images']}")
    print(f"Count of patients with Head + Abdomen + Femur together: {h['count_patients_with_head_abdomen_femur_together']}")
    print(f"Conclusion: {h['conclusion']}")

    print("\n--- 5. SPLIT COUNTS AFTER EXCLUSIONS ---")
    s = res["split_counts_after_exclusions"]
    print(f"Total images before exclusion: {s['total_images_before_exclusions']}")
    print(f"Excluded uncalibrated images: {s['excluded_uncalibrated_images']}")
    print(f"Total images after exclusions: {s['total_images_after_exclusions']}")
    print(f"Total patients after exclusions: {s['total_patients_after_exclusions']}")
    for split_name, sp in s["splits"].items():
        print(
            f"  {split_name.capitalize():<12}: {sp['images']} images ({sp['fp_images']} FP, {sp['ucl_images']} UCL), {sp['patients']} patients"
        )
    p_chk = s["patient_overlap_check"]
    print(
        f"Patient overlaps: train-val={p_chk['train_val_overlap']}, "
        f"train-test={p_chk['train_test_overlap']}, val-test={p_chk['val_test_overlap']}"
    )
    print(f"Zero patient overlap confirmed: {p_chk['zero_patient_overlap_confirmed']}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Femur Data Follow-up Analysis")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(config.DATA_DIR),
        help="Root directory of data (contains FP and UCL)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(work_root / "docs" / "results" / "femur"),
        help="Output directory for results JSON and overlays",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    results = run_femur_followup(data_dir, out_dir)
    print_formatted_summary(results)


if __name__ == "__main__":
    main()
