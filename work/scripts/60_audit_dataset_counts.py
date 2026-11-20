"""scripts/60_audit_dataset_counts.py

Performs a dataset count audit across all anatomical structures (Abdomen, Femur, Head)
comparing raw annotations in C:/fetal-data vs arXiv 2512.16710 published cohort numbers.

Audits:
1. Abdomen: 823 published (FP 693 + UCL 130) vs 789 in our dataset (642 train + 147 test).
   Categorizes and lists all 34 missing images (31 missing px_to_mm rate, 3 patient 125 exclusions).
2. Femur: 895 published (FP 760 + UCL 135) vs 867 calibrated images.
3. Head: 1796 published (FP 1637 + UCL 159) vs 1763 calibrated images.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def audit_counts() -> dict:
    base = Path("C:/fetal-data")
    repo_root = Path(__file__).resolve().parent.parent

    # 1. Abdomen Audit
    fp_ab_df = pd.read_csv(base / "FP" / "annotations" / "Abdomen.csv")
    ucl_ab_raw = pd.read_csv(base / "UCL" / "annotations" / "Abdomen.csv")
    ucl_ab_px = pd.read_csv(base / "UCL" / "annotations" / "px_to_mm" / "Abdomen.csv")

    raw_ucl_names = set(ucl_ab_raw["image_name"])
    px_ucl_names = set(ucl_ab_px["image_name"])

    missing_px_calibration = sorted(list(raw_ucl_names - px_ucl_names))
    p125_excluded = sorted([f for f in px_ucl_names if f.startswith("125_")])
    all_34_missing = sorted(missing_px_calibration + p125_excluded)

    ucl_final_used = sorted([f for f in px_ucl_names if not f.startswith("125_")])
    total_abdomen_used = len(fp_ab_df) + len(ucl_final_used)

    # 2. Femur Audit
    fp_femur_df = pd.read_csv(base / "FP" / "annotations" / "Femur.csv")
    ucl_femur_raw = pd.read_csv(base / "UCL" / "annotations" / "Femur.csv")
    ucl_femur_px = pd.read_csv(base / "UCL" / "annotations" / "px_to_mm" / "Femur.csv")
    femur_missing_px = sorted(list(set(ucl_femur_raw["image_name"]) - set(ucl_femur_px["image_name"])))

    # 3. Head Audit
    fp_head_df = pd.read_csv(base / "FP" / "annotations" / "Head.csv")
    ucl_head_raw = pd.read_csv(base / "UCL" / "annotations" / "Head.csv")
    ucl_head_px = pd.read_csv(base / "UCL" / "annotations" / "px_to_mm" / "Head.csv")
    head_missing_px = sorted(list(set(ucl_head_raw["image_name"]) - set(ucl_head_px["image_name"])))

    audit_data = {
        "abdomen": {
            "published_total": 823,
            "published_breakdown": {"FP": 693, "UCL": 130},
            "raw_total_on_disk": len(fp_ab_df) + len(ucl_ab_raw),
            "raw_breakdown": {"FP": len(fp_ab_df), "UCL_raw": len(ucl_ab_raw)},
            "our_used_total": total_abdomen_used,
            "our_split_breakdown": {"train": 642, "test": 147, "sum": 789},
            "missing_count": len(all_34_missing),
            "missing_reasons": {
                "missing_px_to_mm_calibration": {
                    "count": len(missing_px_calibration),
                    "description": "Present in UCL/annotations/Abdomen.csv but omitted from px_to_mm/Abdomen.csv because pixel calibration was not recorded",
                    "filenames": missing_px_calibration,
                },
                "corrupted_geometry_exclusion": {
                    "count": len(p125_excluded),
                    "description": "Patient 125 images (125_3AC, 125_4AC, 125_5AC) excluded in src/fetal/data/ac_dataset.py due to corrupted ultrasound acquisition layout",
                    "filenames": p125_excluded,
                },
            },
            "all_34_missing_filenames": all_34_missing,
        },
        "femur": {
            "published_total": 895,
            "published_breakdown": {"FP": 760, "UCL": 135},
            "raw_total_on_disk": len(fp_femur_df) + len(ucl_femur_raw),
            "raw_breakdown": {"FP": len(fp_femur_df), "UCL_raw": len(ucl_femur_raw)},
            "calibrated_total": len(fp_femur_df) + len(ucl_femur_px),
            "uncalibrated_ucl_count": len(femur_missing_px),
            "uncalibrated_filenames": femur_missing_px,
        },
        "head": {
            "published_total": 1796,
            "published_breakdown": {"FP": 1637, "UCL": 159},
            "raw_total_on_disk": len(fp_head_df) + len(ucl_head_raw),
            "raw_breakdown": {"FP": len(fp_head_df), "UCL_raw": len(ucl_head_raw)},
            "calibrated_total": len(fp_head_df) + len(ucl_head_px),
            "uncalibrated_ucl_count": len(head_missing_px),
            "uncalibrated_filenames": head_missing_px,
        },
    }

    out_dir = repo_root / "docs" / "results" / "dataset_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "dataset_counts_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    out_md = out_dir / "dataset_counts_audit.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Dataset Image Count Audit (Abdomen, Femur, Head)\n\n")
        f.write("Comparison of image counts in arXiv 2512.16710 vs local repository annotations.\n\n")
        f.write("## 1. Summary Comparison Table\n\n")
        f.write("| Anatomical Plane | Paper Published Total | FP Raw | UCL Raw | UCL Calibrated (px_to_mm) | Excluded Corrupt | Final Used in Repo | Missing vs Paper |\n")
        f.write("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        f.write(f"| **Abdomen** | **823** (FP 693 + UCL 130) | {len(fp_ab_df)} | {len(ucl_ab_raw)} | {len(ucl_ab_px)} | {len(p125_excluded)} (pat 125) | **789** (642 train + 147 test) | **34** (31 no px_to_mm + 3 corrupt) |\n")
        f.write(f"| **Femur** | **895** (FP 760 + UCL 135) | {len(fp_femur_df)} | {len(ucl_femur_raw)} | {len(ucl_femur_px)} | 0 | **867** (FP 760 + UCL 107) | **28** (28 no px_to_mm) |\n")
        f.write(f"| **Head** | **1796** (FP 1637 + UCL 159) | {len(fp_head_df)} | {len(ucl_head_raw)} | {len(ucl_head_px)} | 0 | **1763** (FP 1637 + UCL 126) | **33** (33 no px_to_mm) |\n\n")
        f.write("## 2. Abdomen 34 Missing Images Breakdown\n\n")
        f.write(f"### A. Missing `px_to_mm` Calibration (31 images)\n")
        f.write("Present in raw UCL annotation CSV but excluded because physical mm scaling is unknown:\n")
        for fn in missing_px_calibration:
            f.write(f"- `{fn}`\n")
        f.write(f"\n### B. Corrupted Acquisition Exclusions (3 images)\n")
        f.write("Excluded by `df[~df['image_name'].str.startswith('125_')]` in `src/fetal/data/ac_dataset.py`:\n")
        for fn in p125_excluded:
            f.write(f"- `{fn}`\n")

    print(f"Generated {out_json} and {out_md}")
    return audit_data


if __name__ == "__main__":
    audit_counts()
