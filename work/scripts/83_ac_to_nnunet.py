#!/usr/bin/env python3

# FIELD: AC
# MODEL: nnU-Net
# STEP: data_prep
# USED BY: work/docs/CODE_MAP.md, work/docs/report/PROOF_LOCATIONS.md

"""scripts/83_ac_to_nnunet.py

Convert Fetal Abdominal Circumference (AC) dataset to nnU-Net v2 format.
Fits an ellipse to sonographer landmarks (TAD and APAD calipers),
draws solid filled masks, matches the patient-grouped 5-fold CV split,
and packages Dataset502_AC into C:/fetal-data/nnunet_ac/Dataset502_AC/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
src_dir = REPO_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fetal import config
from fetal.data.ac_dataset import build_data, load_ac_folds
from fetal.eval.ellipse_fitting import fit_ellipse_robust, hc_bpd_from_ellipse_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AC dataset to nnU-Net v2 format")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("C:/fetal-data"),
        help="Path to fetal data directory containing FP and UCL",
    )
    parser.add_argument(
        "--folds-csv",
        type=Path,
        default=REPO_ROOT / "docs" / "results" / "ac_campaign" / "folds_v1.csv",
        help="Path to folds_v1.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("C:/fetal-data/nnunet_ac/Dataset502_AC"),
        help="Target output directory for nnU-Net dataset",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=REPO_ROOT / "docs" / "results" / "ac_campaign" / "ac_nnunet_prep_summary.json",
        help="Path to save execution summary JSON",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        default=True,
        help="Also export official test set (147 cases) to imagesTs",
    )
    return parser.parse_args()


def convert_ac_to_nnunet(
    data_dir: Path,
    folds_csv: Path,
    output_dir: Path,
    summary_json: Path,
    include_test: bool = True,
) -> dict:
    data_dir = Path(data_dir).resolve()
    folds_csv = Path(folds_csv).resolve()
    output_dir = Path(output_dir).resolve()
    summary_json = Path(summary_json).resolve()

    images_tr_dir = output_dir / "imagesTr"
    labels_tr_dir = output_dir / "labelsTr"
    images_tr_dir.mkdir(parents=True, exist_ok=True)
    labels_tr_dir.mkdir(parents=True, exist_ok=True)

    if include_test:
        images_ts_dir = output_dir / "imagesTs"
        images_ts_dir.mkdir(parents=True, exist_ok=True)

    print(f"[83_ac_to_nnunet] Loading folds from {folds_csv}...")
    folds_df = load_ac_folds(folds_csv=folds_csv, base=data_dir)
    print(f"[83_ac_to_nnunet] Loaded {len(folds_df)} training cases across {folds_df['patient'].nunique()} patients.")

    rejection_rule = (
        "Reject if semi-axis <= 0, aspect ratio > 4.0 or < 0.25, "
        "diameter midpoints distance > max(r_tad, r_apad), or contour < 5 points"
    )

    bad_fits = []
    diffs = []
    converted_train_cases = []

    for idx, row in folds_df.iterrows():
        img_name = str(row["image_name"])
        case_id = Path(img_name).stem
        converted_train_cases.append(case_id)

        p0 = np.array([row["tad_1_x"], row["tad_1_y"]], dtype=float)
        p1 = np.array([row["tad_2_x"], row["tad_2_y"]], dtype=float)
        p2 = np.array([row["apad_1_x"], row["apad_1_y"]], dtype=float)
        p3 = np.array([row["apad_2_x"], row["apad_2_y"]], dtype=float)

        d_tad = float(np.linalg.norm(p1 - p0))
        d_apad = float(np.linalg.norm(p3 - p2))
        mmpp = float(row["px_to_mm_rate"])

        # Reference AC from calipers using Ramanujan's formula
        a_ref = (d_tad * mmpp) / 2.0
        b_ref = (d_apad * mmpp) / 2.0
        ac_ref = math.pi * (3.0 * (a_ref + b_ref) - math.sqrt((3.0 * a_ref + b_ref) * (a_ref + 3.0 * b_ref)))

        # Caliper midpoints and ellipse centre
        m_tad = (p0 + p1) / 2.0
        m_apad = (p2 + p3) / 2.0
        center = (m_tad + m_apad) / 2.0

        r_tad = d_tad / 2.0
        r_apad = d_apad / 2.0
        angle_rad = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        angle_deg = math.degrees(angle_rad)

        # Check rejection rule
        is_bad = False
        reason = ""
        if r_tad <= 0 or r_apad <= 0:
            is_bad = True
            reason = "non-positive semi-axis"
        elif r_tad / r_apad > 4.0 or r_apad / r_tad > 4.0:
            is_bad = True
            reason = "extreme aspect ratio"
        elif float(np.linalg.norm(m_tad - m_apad)) > max(r_tad, r_apad):
            is_bad = True
            reason = "diameter midpoints distance too large"

        if is_bad:
            bad_fits.append({"case_id": case_id, "reason": reason})

        # Read original ultrasound image
        src_img_path = Path(row["image_path"])
        if not src_img_path.is_file():
            raise FileNotFoundError(f"Missing image file: {src_img_path}")
        img = cv2.imread(str(src_img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {src_img_path}")

        h, w = img.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw filled ellipse mask
        cv2.ellipse(
            mask,
            (int(round(center[0])), int(round(center[1]))),
            (int(round(r_tad)), int(round(r_apad))),
            angle_deg,
            0,
            360,
            1,
            -1,
        )

        # Measure circumference of drawn mask via robust ellipse fit on contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours or len(contours[0]) < 5:
            bad_fits.append({"case_id": case_id, "reason": "contour < 5 points"})
            ac_mask = 0.0
        else:
            box = fit_ellipse_robust(contours[0])
            (cx_m, cy_m), (axes_1, axes_2), angle_m = box
            semi_a = (axes_1 / 2.0) * mmpp
            semi_b = (axes_2 / 2.0) * mmpp
            ac_mask, _ = hc_bpd_from_ellipse_params(semi_a, semi_b)

        diff_mm = float(ac_mask - ac_ref)
        diffs.append(
            {
                "case_id": case_id,
                "ac_ref_mm": ac_ref,
                "ac_mask_mm": float(ac_mask),
                "diff_mm": diff_mm,
                "abs_diff_mm": abs(diff_mm),
            }
        )

        # Write nnU-Net input and label
        dst_img_path = images_tr_dir / f"{case_id}_0000.png"
        dst_lbl_path = labels_tr_dir / f"{case_id}.png"
        cv2.imwrite(str(dst_img_path), img)
        cv2.imwrite(str(dst_lbl_path), mask)

    diffs_df = pd.DataFrame(diffs)
    mean_signed_diff = float(diffs_df["diff_mm"].mean())
    mean_abs_diff = float(diffs_df["abs_diff_mm"].mean())
    max_abs_diff = float(diffs_df["abs_diff_mm"].max())

    print(f"[83_ac_to_nnunet] Processed {len(converted_train_cases)} training images.")
    print(f"  Bad ellipse fits count : {len(bad_fits)}")
    print(f"  Mean signed diff (mask - ref) : {mean_signed_diff:+.4f} mm")
    print(f"  Mean absolute diff            : {mean_abs_diff:.4f} mm")
    print(f"  Max absolute diff             : {max_abs_diff:.4f} mm")

    # Write dataset.json
    dataset_json_data = {
        "channel_names": {
            "0": "US"
        },
        "labels": {
            "background": 0,
            "ac": 1
        },
        "numTraining": len(converted_train_cases),
        "file_ending": ".png",
    }
    dataset_json_path = output_dir / "dataset.json"
    with open(dataset_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json_data, f, indent=2)
    print(f"[83_ac_to_nnunet] Wrote dataset.json -> {dataset_json_path}")

    # Build splits_final.json (5 folds matching folds_v1.csv, folds 0..4)
    splits = []
    oof_mapping = {}
    for fold_k in range(1, 6):
        nn_fold_idx = fold_k - 1
        val_sub = folds_df[folds_df["fold"] == fold_k]
        tr_sub = folds_df[folds_df["fold"] != fold_k]

        val_cases = sorted([Path(f).stem for f in val_sub["image_name"]])
        tr_cases = sorted([Path(f).stem for f in tr_sub["image_name"]])

        # Assert no patient overlap
        val_patients = set(val_sub["patient"])
        tr_patients = set(tr_sub["patient"])
        assert val_patients.isdisjoint(tr_patients), f"Patient leakage in fold {fold_k}!"
        assert set(val_cases).isdisjoint(set(tr_cases)), f"Case leakage in fold {fold_k}!"

        splits.append({"train": tr_cases, "val": val_cases})
        oof_mapping[f"fold_{nn_fold_idx}"] = {
            "ac_campaign_fold": fold_k,
            "val_cases_count": len(val_cases),
            "val_patients_count": len(val_patients),
            "train_cases_count": len(tr_cases),
            "train_patients_count": len(tr_patients),
        }

    splits_final_path = output_dir / "splits_final.json"
    with open(splits_final_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    print(f"[83_ac_to_nnunet] Wrote splits_final.json -> {splits_final_path}")

    # Optional: export test set to imagesTs
    n_test_exported = 0
    if include_test:
        all_data = build_data(base=data_dir)
        test_df = all_data[all_data["official_split"] == "test"].copy()
        for _, row in test_df.iterrows():
            img_name = str(row["image_name"])
            case_id = Path(img_name).stem
            src_p = Path(row["image_path"])
            dst_p = images_ts_dir / f"{case_id}_0000.png"
            if src_p.is_file():
                img = cv2.imread(str(src_p), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    cv2.imwrite(str(dst_p), img)
                    n_test_exported += 1
        print(f"[83_ac_to_nnunet] Exported {n_test_exported} test images to imagesTs/.")

    summary = {
        "dataset_name": "Dataset502_AC",
        "output_dir": str(output_dir),
        "num_training_images": len(converted_train_cases),
        "num_test_images": n_test_exported,
        "bad_fits_count": len(bad_fits),
        "bad_fits": bad_fits,
        "rejection_rule": rejection_rule,
        "mean_signed_diff_mm": round(mean_signed_diff, 4),
        "mean_abs_diff_mm": round(mean_abs_diff, 4),
        "max_abs_diff_mm": round(max_abs_diff, 4),
        "oof_mapping": oof_mapping,
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[83_ac_to_nnunet] Saved summary -> {summary_json}")

    return summary


def main():
    args = parse_args()
    summary = convert_ac_to_nnunet(
        data_dir=args.data_dir,
        folds_csv=args.folds_csv,
        output_dir=args.output_dir,
        summary_json=args.summary_json,
        include_test=args.include_test,
    )
    print("\n=== Summary Results ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
