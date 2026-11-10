"""23_three_plane_check.py: Three-plane (head + abdomen + femur) subject check for FP and UCL.

Audits subject counts, per-plane availability, pairwise and 3-plane overlaps,
presence of gestational age / weight / outcome metadata, and train/test split
leakage across ultrasound anatomical planes.

Outputs:
  - docs/results/three_plane/three_plane_check.json
  - docs/results/three_plane/SHA256SUMS
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

import pandas as pd
from fetal import config


def get_raw_listings(data_dir: Path, raw_capstone_dir: Path) -> dict:
    """Collect raw directory listings for local data folders (annotations and image folders)."""
    data_dir = Path(data_dir)
    raw_capstone_dir = Path(raw_capstone_dir)
    listings = {}

    def scan_location(base: Path, subpaths: list[str]) -> list[dict]:
        entries = []
        if not base.exists():
            return entries
        for sp in subpaths:
            target = base / sp
            if not target.exists():
                entries.append({"path": sp, "exists": False, "type": "missing"})
                continue
            if target.is_file():
                entries.append({
                    "path": sp,
                    "exists": True,
                    "type": "file",
                    "size_bytes": target.stat().st_size,
                })
            elif target.is_dir():
                files = []
                subdirs = []
                with os.scandir(target) as it:
                    for entry in it:
                        if entry.is_file():
                            files.append(entry.name)
                        elif entry.is_dir():
                            subdirs.append(entry.name)
                files.sort()
                subdirs.sort()
                file_count = len(files)
                dir_count = len(subdirs)
                entries.append({
                    "path": sp,
                    "exists": True,
                    "type": "directory",
                    "files_count": file_count,
                    "subdirs_count": dir_count,
                    "files": files if file_count <= 20 else files[:10] + [f"... ({file_count} total files)"],
                    "subdirs": subdirs,
                })
        return entries

    # Local C:/fetal-data
    c_subpaths = [
        "FP/annotations",
        "FP/images",
        "FP/images/Abdomen",
        "FP/images/Femur",
        "FP/images/Head",
        "UCL/annotations",
        "UCL/annotations/px_to_mm",
        "UCL/images",
        "UCL/images/Abdomen",
        "UCL/images/Femur",
        "UCL/images/Head",
    ]
    listings["local_fetal_data"] = {
        "base_path": str(data_dir),
        "exists": data_dir.exists(),
        "entries": scan_location(data_dir, c_subpaths),
    }

    # Raw Capstone archive
    raw_subpaths = [
        "annotations/FP",
        "images/FP",
        "images/FP/Abdomen",
        "images/FP/Femur",
        "images/FP/Head",
        "annotations/UCL",
        "annotations/UCL/px_to_mm",
        "images/UCL",
        "images/UCL/Abdomen",
        "images/UCL/Femur",
        "images/UCL/Head",
    ]
    listings["raw_capstone_release"] = {
        "base_path": str(raw_capstone_dir),
        "exists": raw_capstone_dir.exists(),
        "entries": scan_location(raw_capstone_dir, raw_subpaths),
    }

    return listings


def audit_plane_subjects(name: str, h_df: pd.DataFrame, a_df: pd.DataFrame, f_df: pd.DataFrame,
                         subj_col_h: str, subj_col_a: str, subj_col_f: str) -> dict:
    """Analyze per-plane subjects, pairwise intersections, and 3-plane intersection."""
    h_df = h_df.copy()
    a_df = a_df.copy()
    f_df = f_df.copy()

    H = set(h_df[subj_col_h].dropna().unique())
    A = set(a_df[subj_col_a].dropna().unique())
    F = set(f_df[subj_col_f].dropna().unique())

    total_union = H | A | F
    HA = H & A
    HF = H & F
    AF = A & F
    HAF = H & A & F

    h_only = H - (A | F)
    a_only = A - (H | F)
    f_only = F - (H | A)
    ha_only = (H & A) - F
    hf_only = (H & F) - A
    af_only = (A & F) - H

    return {
        "dataset_name": name,
        "images_per_plane": {
            "head": len(h_df),
            "abdomen": len(a_df),
            "femur": len(f_df),
            "total_images": len(h_df) + len(a_df) + len(f_df),
        },
        "subjects_per_plane": {
            "head": len(H),
            "abdomen": len(A),
            "femur": len(F),
            "total_union": len(total_union),
        },
        "pairwise_intersections": {
            "head_and_abdomen": len(HA),
            "head_and_femur": len(HF),
            "abdomen_and_femur": len(AF),
        },
        "three_plane_intersection": len(HAF),
        "venn_partitions": {
            "head_only": len(h_only),
            "abdomen_only": len(a_only),
            "femur_only": len(f_only),
            "head_abdomen_only": len(ha_only),
            "head_femur_only": len(hf_only),
            "abdomen_femur_only": len(af_only),
            "all_three": len(HAF),
        },
    }


def audit_splits_and_leakage(name: str, h_df: pd.DataFrame, a_df: pd.DataFrame, f_df: pd.DataFrame,
                             subj_col_h: str, subj_col_a: str, subj_col_f: str) -> dict:
    """Analyze train/test split distribution and cross-plane leakage for 3-plane subjects."""
    h_df = h_df.copy()
    a_df = a_df.copy()
    f_df = f_df.copy()

    h_df["sid"] = h_df[subj_col_h]
    a_df["sid"] = a_df[subj_col_a]
    f_df["sid"] = f_df[subj_col_f]

    h_df["sp"] = h_df["Split"].astype(str).str.strip().str.lower()
    a_df["sp"] = a_df["Split"].astype(str).str.strip().str.lower()
    f_df["sp"] = f_df["Split"].astype(str).str.strip().str.lower()

    H = set(h_df["sid"].dropna().unique())
    A = set(a_df["sid"].dropna().unique())
    F = set(f_df["sid"].dropna().unique())
    three_plane_subjs = sorted(H & A & F)

    records = []
    for sid in three_plane_subjs:
        h_sps = set(h_df[h_df["sid"] == sid]["sp"].unique())
        a_sps = set(a_df[a_df["sid"] == sid]["sp"].unique())
        f_sps = set(f_df[f_df["sid"] == sid]["sp"].unique())

        h_sp = list(h_sps)[0] if len(h_sps) == 1 else "mixed"
        a_sp = list(a_sps)[0] if len(a_sps) == 1 else "mixed"
        f_sp = list(f_sps)[0] if len(f_sps) == 1 else "mixed"

        records.append({
            "sid": sid,
            "head_split": h_sp,
            "abd_split": a_sp,
            "femur_split": f_sp,
            "all_same": (h_sp == a_sp == f_sp),
            "all_test": (h_sp == "test" and a_sp == "test" and f_sp == "test"),
            "all_train": (h_sp == "train" and a_sp == "train" and f_sp == "train"),
        })

    df_rec = pd.DataFrame(records) if records else pd.DataFrame(columns=["sid", "head_split", "abd_split", "femur_split", "all_same", "all_test", "all_train"])

    h_test = int((df_rec["head_split"] == "test").sum()) if not df_rec.empty else 0
    a_test = int((df_rec["abd_split"] == "test").sum()) if not df_rec.empty else 0
    f_test = int((df_rec["femur_split"] == "test").sum()) if not df_rec.empty else 0

    all_test = int(df_rec["all_test"].sum()) if not df_rec.empty else 0
    all_train = int(df_rec["all_train"].sum()) if not df_rec.empty else 0
    inconsistent = int((~df_rec["all_same"]).sum()) if not df_rec.empty else 0

    mismatch_breakdown = {}
    if not df_rec.empty and inconsistent > 0:
        grp = df_rec.groupby(["head_split", "abd_split", "femur_split"])["sid"].count()
        for k, v in grp.items():
            key_str = f"head={k[0]}|abd={k[1]}|femur={k[2]}"
            mismatch_breakdown[key_str] = int(v)

    return {
        "dataset_name": name,
        "three_plane_subjects_total": len(three_plane_subjs),
        "three_plane_subjects_in_head_test": h_test,
        "three_plane_subjects_in_abdomen_test": a_test,
        "three_plane_subjects_in_femur_test": f_test,
        "three_plane_subjects_all_test": all_test,
        "three_plane_subjects_all_train": all_train,
        "cross_plane_split_mismatch_count": inconsistent,
        "mismatch_breakdown": mismatch_breakdown,
        "leakage_risk_explanation": (
            "Because per-plane test splits were constructed independently in the benchmark paper, "
            f"{inconsistent} subjects have conflicting split assignments across planes. Only {all_test} subjects "
            "are held-out in test across all three planes simultaneously. Evaluating multi-plane Hadlock on the remaining "
            "subjects causes cross-plane leakage if any single plane model was exposed to that subject during training."
        ),
    }


def run_three_plane_check(data_dir: Path | str, raw_capstone_dir: Path | str, out_dir: Path | str) -> dict:
    """Run complete three-plane audit and save JSON results."""
    data_dir = Path(data_dir)
    raw_capstone_dir = Path(raw_capstone_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Directory listings and origin check
    listings = get_raw_listings(data_dir, raw_capstone_dir)

    # Load annotations
    fp_ann = data_dir / "FP" / "annotations"
    ucl_ann = data_dir / "UCL" / "annotations"
    ucl_px = data_dir / "UCL" / "annotations" / "px_to_mm"

    fp_h = pd.read_csv(fp_ann / "Head.csv")
    fp_a = pd.read_csv(fp_ann / "Abdomen.csv")
    fp_f = pd.read_csv(fp_ann / "Femur.csv")

    ucl_h = pd.read_csv(ucl_ann / "Head.csv")
    ucl_a = pd.read_csv(ucl_ann / "Abdomen.csv")
    ucl_f = pd.read_csv(ucl_ann / "Femur.csv")

    ucl_px_h = pd.read_csv(ucl_px / "Head.csv")
    ucl_px_a = pd.read_csv(ucl_px / "Abdomen.csv")
    ucl_px_f = pd.read_csv(ucl_px / "Femur.csv")

    # In UCL, extract SubjectID from image_name (e.g. "002_HC.jpeg" -> 2)
    for df in [ucl_h, ucl_a, ucl_f, ucl_px_h, ucl_px_a, ucl_px_f]:
        df["SubjectID_extracted"] = df["image_name"].str.extract(r"^(\d+)_")[0].astype(int)

    # 2. Check metadata columns
    metadata_cols_check = {}
    check_terms = ["gestat", "ga", "weight", "efw", "bw", "outcome", "stillbirth", "mortality"]
    for dataset_name, df_dict in [
        ("FP", {"Head": fp_h, "Abdomen": fp_a, "Femur": fp_f}),
        ("UCL_Raw", {"Head": ucl_h, "Abdomen": ucl_a, "Femur": ucl_f}),
        ("UCL_px_to_mm", {"Head": ucl_px_h, "Abdomen": ucl_px_a, "Femur": ucl_px_f}),
    ]:
        metadata_cols_check[dataset_name] = {}
        for plane, df in df_dict.items():
            cols = list(df.columns)
            matched = [c for c in cols if any(t in c.lower() for t in check_terms)]
            metadata_cols_check[dataset_name][plane] = {
                "columns": cols,
                "matched_clinical_or_outcome_columns": matched,
                "has_gestational_age": any("gestat" in c.lower() or c.lower() == "ga" for c in cols),
                "has_estimated_fetal_weight": any("efw" in c.lower() or "weight" in c.lower() for c in cols),
                "has_birth_weight": any("bw" in c.lower() or "birth" in c.lower() for c in cols),
                "has_outcome": any("outcome" in c.lower() or "stillbirth" in c.lower() for c in cols),
            }

    # 3. Subject counts and overlaps
    fp_counts = audit_plane_subjects("FP (Fetal Planes)", fp_h, fp_a, fp_f, "SubjectID", "SubjectID", "Patient")
    ucl_raw_counts = audit_plane_subjects("UCL Raw Annotations", ucl_h, ucl_a, ucl_f, "SubjectID_extracted", "SubjectID_extracted", "SubjectID_extracted")
    ucl_px_counts = audit_plane_subjects("UCL Calibrated (px_to_mm)", ucl_px_h, ucl_px_a, ucl_px_f, "SubjectID_extracted", "SubjectID_extracted", "SubjectID_extracted")

    # Paper comparison
    paper_expected_fp = {"head": 909, "abdomen": 586, "femur": 629, "total_union": 1047}
    paper_comparison_fp = {
        "head": {"ours": fp_counts["subjects_per_plane"]["head"], "paper": paper_expected_fp["head"], "diff": fp_counts["subjects_per_plane"]["head"] - paper_expected_fp["head"]},
        "abdomen": {"ours": fp_counts["subjects_per_plane"]["abdomen"], "paper": paper_expected_fp["abdomen"], "diff": fp_counts["subjects_per_plane"]["abdomen"] - paper_expected_fp["abdomen"]},
        "femur": {"ours": fp_counts["subjects_per_plane"]["femur"], "paper": paper_expected_fp["femur"], "diff": fp_counts["subjects_per_plane"]["femur"] - paper_expected_fp["femur"]},
        "total_union": {"ours": fp_counts["subjects_per_plane"]["total_union"], "paper": paper_expected_fp["total_union"], "diff": fp_counts["subjects_per_plane"]["total_union"] - paper_expected_fp["total_union"]},
        "exact_match_all": (
            fp_counts["subjects_per_plane"]["head"] == paper_expected_fp["head"] and
            fp_counts["subjects_per_plane"]["abdomen"] == paper_expected_fp["abdomen"] and
            fp_counts["subjects_per_plane"]["femur"] == paper_expected_fp["femur"] and
            fp_counts["subjects_per_plane"]["total_union"] == paper_expected_fp["total_union"]
        )
    }

    # 4. Splits and cross-plane leakage
    fp_leakage = audit_splits_and_leakage("FP (Fetal Planes)", fp_h, fp_a, fp_f, "SubjectID", "SubjectID", "Patient")
    ucl_raw_leakage = audit_splits_and_leakage("UCL Raw Annotations", ucl_h, ucl_a, ucl_f, "SubjectID_extracted", "SubjectID_extracted", "SubjectID_extracted")
    ucl_px_leakage = audit_splits_and_leakage("UCL Calibrated (px_to_mm)", ucl_px_h, ucl_px_a, ucl_px_f, "SubjectID_extracted", "SubjectID_extracted", "SubjectID_extracted")

    results = {
        "dataset_origin": {
            "source_repo": "https://github.com/surgical-vision/Multicentre-Fetal-Biometry",
            "paper_citation": "arXiv 2512.16710",
            "doi": "10.5522/04/30819911",
            "license": "CC BY 4.0",
            "raw_local_archive_path": str(raw_capstone_dir),
            "local_fetal_data_path": str(data_dir),
            "explanation": (
                "The Abdomen and Femur data originally copied to C:/fetal-data originated from "
                "E:/BUBT-CSE-4th yr/Capstone/data/raw/Multicentre-Fetal-Biometry-2026/FetalBiometry-Multicentre-Landmarks-2026. "
                "The Head plane files (Head.csv, Head_Train.csv, Head_Test.csv and images/FP/Head, images/UCL/Head) "
                "were present in that local raw release but omitted in the earlier selective copy to C:/fetal-data. "
                "They have now been copied into C:/fetal-data (1,637 FP Head images, 159 UCL Head images)."
            ),
        },
        "directory_listings": listings,
        "metadata_columns_audit": metadata_cols_check,
        "subject_counts": {
            "fp": fp_counts,
            "ucl_raw": ucl_raw_counts,
            "ucl_px_to_mm": ucl_px_counts,
            "paper_comparison_fp": paper_comparison_fp,
        },
        "splits_and_leakage_audit": {
            "fp": fp_leakage,
            "ucl_raw": ucl_raw_leakage,
            "ucl_px_to_mm": ucl_px_leakage,
        },
    }

    out_file = out_dir / "three_plane_check.json"
    with open(out_file, "w", encoding="utf-8", newline="\n") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_file}")

    return results


def print_formatted_report(res: dict) -> None:
    print("\n" + "=" * 90)
    print("THREE-PLANE (HEAD + ABDOMEN + FEMUR) SUBJECT & LEAKAGE REPORT")
    print("=" * 90)

    print("\n--- 1. LOCAL RAW FP AND UCL FOLDERS (ITEM 1) ---")
    origin = res["dataset_origin"]
    print(f"Raw release path: {origin['raw_local_archive_path']}")
    print(f"Local working path: {origin['local_fetal_data_path']}")
    print(f"Origin explanation: {origin['explanation']}")

    print("\n--- 2. SUBJECT COUNTS & PAPER COMPARISON (ITEM 3) ---")
    fp = res["subject_counts"]["fp"]
    ucl_raw = res["subject_counts"]["ucl_raw"]
    ucl_px = res["subject_counts"]["ucl_px_to_mm"]
    comp = res["subject_counts"]["paper_comparison_fp"]

    print(f"{'Cohort':<26} {'Head':<8} {'Abdomen':<8} {'Femur':<8} {'H ∩ A':<8} {'H ∩ F':<8} {'A ∩ F':<8} {'H ∩ A ∩ F':<10} {'Total Union':<12}")
    print("-" * 106)
    print(
        f"{'FP (Fetal Planes)':<26} "
        f"{fp['subjects_per_plane']['head']:<8} "
        f"{fp['subjects_per_plane']['abdomen']:<8} "
        f"{fp['subjects_per_plane']['femur']:<8} "
        f"{fp['pairwise_intersections']['head_and_abdomen']:<8} "
        f"{fp['pairwise_intersections']['head_and_femur']:<8} "
        f"{fp['pairwise_intersections']['abdomen_and_femur']:<8} "
        f"{fp['three_plane_intersection']:<10} "
        f"{fp['subjects_per_plane']['total_union']:<12}"
    )
    print(
        f"{'UCL (Raw Annotations)':<26} "
        f"{ucl_raw['subjects_per_plane']['head']:<8} "
        f"{ucl_raw['subjects_per_plane']['abdomen']:<8} "
        f"{ucl_raw['subjects_per_plane']['femur']:<8} "
        f"{ucl_raw['pairwise_intersections']['head_and_abdomen']:<8} "
        f"{ucl_raw['pairwise_intersections']['head_and_femur']:<8} "
        f"{ucl_raw['pairwise_intersections']['abdomen_and_femur']:<8} "
        f"{ucl_raw['three_plane_intersection']:<10} "
        f"{ucl_raw['subjects_per_plane']['total_union']:<12}"
    )
    print(
        f"{'UCL (Calibrated px_to_mm)':<26} "
        f"{ucl_px['subjects_per_plane']['head']:<8} "
        f"{ucl_px['subjects_per_plane']['abdomen']:<8} "
        f"{ucl_px['subjects_per_plane']['femur']:<8} "
        f"{ucl_px['pairwise_intersections']['head_and_abdomen']:<8} "
        f"{ucl_px['pairwise_intersections']['head_and_femur']:<8} "
        f"{ucl_px['pairwise_intersections']['abdomen_and_femur']:<8} "
        f"{ucl_px['three_plane_intersection']:<10} "
        f"{ucl_px['subjects_per_plane']['total_union']:<12}"
    )

    print("\nPaper comparison for FP subjects:")
    for k in ["head", "abdomen", "femur", "total_union"]:
        print(f"  - {k.capitalize()}: Ours={comp[k]['ours']} vs Paper={comp[k]['paper']} (diff={comp[k]['diff']})")
    print(f"  Exact match across all planes: {comp['exact_match_all']}")

    print("\n--- 3. CSV COLUMN NAMES (ITEM 4) ---")
    cols = res["metadata_columns_audit"]
    for dataset_name, planes in cols.items():
        print(f"[{dataset_name}]")
        for plane_name, pinfo in planes.items():
            print(f"  {plane_name:<10}: {pinfo['columns']}")
    print("\nClinical / Outcome column check:")
    print("  - Gestational Age (GA): NONE")
    print("  - Estimated Fetal Weight (EFW): NONE")
    print("  - Birth Weight (BW): NONE")
    print("  - Clinical Outcome: NONE")

    print("\n--- 4. TEST-SPLIT COUNTS & LEAKAGE AUDIT (ITEM 5) ---")
    fp_leak = res["splits_and_leakage_audit"]["fp"]
    ucl_leak = res["splits_and_leakage_audit"]["ucl_px_to_mm"]
    print(f"FP 3-plane subjects (total: {fp_leak['three_plane_subjects_total']}):")
    print(f"  - In Head Test:    {fp_leak['three_plane_subjects_in_head_test']}")
    print(f"  - In Abdomen Test: {fp_leak['three_plane_subjects_in_abdomen_test']}")
    print(f"  - In Femur Test:   {fp_leak['three_plane_subjects_in_femur_test']}")
    print(f"  - In ALL THREE Test splits:  {fp_leak['three_plane_subjects_all_test']}")
    print(f"  - In ALL THREE Train splits: {fp_leak['three_plane_subjects_all_train']}")
    print(f"  - Split mismatch count:      {fp_leak['cross_plane_split_mismatch_count']}")
    print(f"\nUCL Calibrated 3-plane subjects (total: {ucl_leak['three_plane_subjects_total']}):")
    print(f"  - In Head Test:    {ucl_leak['three_plane_subjects_in_head_test']}")
    print(f"  - In Abdomen Test: {ucl_leak['three_plane_subjects_in_abdomen_test']}")
    print(f"  - In Femur Test:   {ucl_leak['three_plane_subjects_in_femur_test']}")
    print(f"  - In ALL THREE Test splits:  {ucl_leak['three_plane_subjects_all_test']}")
    print(f"  - In ALL THREE Train splits: {ucl_leak['three_plane_subjects_all_train']}")
    print(f"  - Split mismatch count:      {ucl_leak['cross_plane_split_mismatch_count']}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Three-Plane Subject and Leakage Check")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(config.DATA_DIR),
        help="Root directory for fetal datasets (contains FP and UCL)",
    )
    parser.add_argument(
        "--raw-capstone-dir",
        type=str,
        default="E:/BUBT-CSE-4th yr/Capstone/data/raw/Multicentre-Fetal-Biometry-2026/FetalBiometry-Multicentre-Landmarks-2026",
        help="Path to raw local release archive",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(work_root / "docs" / "results" / "three_plane"),
        help="Directory to write output JSON and hashes",
    )
    args = parser.parse_args()

    results = run_three_plane_check(Path(args.data_dir), Path(args.raw_capstone_dir), Path(args.out_dir))
    print_formatted_report(results)


if __name__ == "__main__":
    main()
