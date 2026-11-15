"""make_fassd_split_v2.py: Split leakage fix, group-level partitioning, and compact Kaggle dataset preparation.

Addresses data leakage in FASSD split_v1.csv where duplicate images across patients
spanned the train/test boundary (specifically P0119 in test vs P019 in train).

Steps:
1. Inspects split_v1.csv for the 3 duplicate patient pairs (P0119/P019, P058/P0128, P031/P033).
2. Computes SHA-256 hashes for all 1,588 PNGs, runs Union-Find over patients sharing ANY hash.
3. Splits by patient GROUP (seed 42, ~70/15/15), asserting zero patient and zero hash leakage.
4. Saves docs/results/fassd/split_v2.csv (filename, patient, group, split).
5. Prepares compact Kaggle dataset at E:\\BUBT-CSE-4th yr\\Capstone\\data\\kaggle_fassd_v2:
   - images/<stem>.png: byte-exact copy from IMAGES
   - masks/<stem>.png: 1-channel uint8 PNG with bit flags (artery=1, liver=2, stomach=4, vein=8)
   - meta.csv: filename, patient, group, split_v2
   - dataset-metadata.json: title 'FASSD fetal abdomen v2', id 'prantakumerpandit390/fassd-fetal-abdomen-v2'
6. Verifies on ALL 1,588 samples that decoding the bit-flag PNG exactly reproduces original GT masks.
7. Reports total dataset size in MB and prints the exact Kaggle CLI command to create the dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
src_dir = REPO_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import cv2
import numpy as np
import pandas as pd

from fetal.data.fassd import (
    STRUCTURE_BITFLAGS,
    STRUCTURE_NAMES,
    build_patient_groups_from_hashes,
    create_group_split_v2,
    decode_multilabel_mask,
    encode_multilabel_mask,
    load_fassd_sample,
    parse_patient_id,
)

DEFAULT_RAW_DIR = r"E:\BUBT-CSE-4th yr\Capstone\data\raw\FASSD--Fetal Abdominal Structures Segmentation Dataset Using Ultrasonic Images"
DEFAULT_KAGGLE_DIR = r"E:\BUBT-CSE-4th yr\Capstone\data\kaggle_fassd_v2"


def run_split_leakage_fix_and_kaggle_prep(
    raw_dir: str | Path,
    kaggle_dir: str | Path,
    output_split_csv: str | Path = "docs/results/fassd/split_v2.csv",
    split_v1_csv: str | Path = "docs/results/fassd/split_v1.csv",
    seed: int = 42,
) -> dict[str, Any]:
    raw_path = Path(raw_dir)
    img_dir = raw_path / "IMAGES"
    npy_dir = raw_path / "ARRAY_FORMAT"
    k_dir = Path(kaggle_dir)
    k_images_dir = k_dir / "images"
    k_masks_dir = k_dir / "masks"
    out_split_path = Path(output_split_csv)

    assert img_dir.is_dir(), f"Raw IMAGES dir not found: {img_dir}"
    assert npy_dir.is_dir(), f"Raw ARRAY_FORMAT dir not found: {npy_dir}"

    png_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".png")])
    assert len(png_files) == 1588, f"Expected 1588 PNG files, found {len(png_files)}"

    # =========================================================================
    # 1. Print split_v1.csv assignment for duplicate pairs
    # =========================================================================
    v1_pairs = [("P0119", "P019"), ("P058", "P0128"), ("P031", "P033")]
    v1_report: dict[str, Any] = {}
    if Path(split_v1_csv).is_file():
        df_v1 = pd.read_csv(split_v1_csv)
        pat_to_v1_split = dict(zip(df_v1["patient"], df_v1["split"]))
        print("=== Step 1: Duplicate Patient Pairs Split Status in split_v1.csv ===")
        for p1, p2 in v1_pairs:
            s1 = pat_to_v1_split.get(p1, "missing")
            s2 = pat_to_v1_split.get(p2, "missing")
            same = bool(s1 == s2)
            v1_report[f"{p1}_vs_{p2}"] = {"patient_1": p1, "split_1": s1, "patient_2": p2, "split_2": s2, "same_split": same}
            print(f"  {p1} (split: {s1}) vs {p2} (split: {s2}) -> Same split? {same}")
            if not same:
                print(f"  --> LEAKAGE DETECTED in split_v1.csv: {p1} and {p2} share identical images across splits!")
    else:
        print(f"split_v1.csv not found at {split_v1_csv}")

    # =========================================================================
    # 2. Build Union-Find Patient Groups from Image Hashes
    # =========================================================================
    patient_to_group, group_to_patients, file_to_hash = build_patient_groups_from_hashes(img_dir, png_files)
    n_unique_patients = len(patient_to_group)
    n_groups = len(group_to_patients)
    print(f"\n=== Step 2: Patient Grouping ===")
    print(f"Total Unique Patients: {n_unique_patients}")
    print(f"Total Patient Groups (Union-Find): {n_groups}")

    multi_pat_groups = {gid: pats for gid, pats in group_to_patients.items() if len(pats) > 1}
    print(f"Merged Multi-Patient Groups ({len(multi_pat_groups)} groups):")
    for gid, pats in multi_pat_groups.items():
        print(f"  Group {gid}: {pats}")

    # =========================================================================
    # 3. Create Group-Level Split (Seed 42, ~70/15/15)
    # =========================================================================
    # Cache structure presence for all images to check per-structure representation
    struct_presence: dict[str, dict[str, bool]] = {}
    for f in png_files:
        stem = Path(f).stem
        sample = load_fassd_sample(npy_dir / f"{stem}.npy")
        struct_presence[f] = {
            st: bool(np.sum(sample["structures"][st] > 0) > 0)
            for st in STRUCTURE_NAMES
        }

    selected_seed = seed
    selected_df = None
    selected_struct_counts = None

    for cand_seed in range(seed, seed + 10):
        df_cand = create_group_split_v2(png_files, patient_to_group, seed=cand_seed)

        # Check assertions
        split_pats = {sp: set(df_cand[df_cand["split"] == sp]["patient"]) for sp in ["train", "val", "test"]}
        split_hashes = {
            sp: set(df_cand[df_cand["split"] == sp]["filename"].map(file_to_hash))
            for sp in ["train", "val", "test"]
        }

        # Assert no patient in two splits
        assert len(split_pats["train"] & split_pats["val"]) == 0, "Patient overlap between train and val"
        assert len(split_pats["train"] & split_pats["test"]) == 0, "Patient overlap between train and test"
        assert len(split_pats["val"] & split_pats["test"]) == 0, "Patient overlap between val and test"

        # Assert no image hash in two splits
        assert len(split_hashes["train"] & split_hashes["val"]) == 0, "Hash overlap between train and val"
        assert len(split_hashes["train"] & split_hashes["test"]) == 0, "Hash overlap between train and test"
        assert len(split_hashes["val"] & split_hashes["test"]) == 0, "Hash overlap between val and test"

        # Check per-structure non-empty counts
        cand_counts = {sp: {st: 0 for st in STRUCTURE_NAMES} for sp in ["train", "val", "test"]}
        split_map = dict(zip(df_cand["filename"], df_cand["split"]))
        for f in png_files:
            sp = split_map[f]
            for st in STRUCTURE_NAMES:
                if struct_presence[f][st]:
                    cand_counts[sp][st] += 1

        all_ge_20 = all(cand_counts[sp][st] >= 20 for sp in ["train", "val", "test"] for st in STRUCTURE_NAMES)
        if all_ge_20:
            selected_seed = cand_seed
            selected_df = df_cand
            selected_struct_counts = cand_counts
            break

    assert selected_df is not None, "Failed to find seed where all structures >= 20"
    print(f"\n=== Step 3: Group Split Verified (Selected Seed: {selected_seed}) ===")
    print("Zero patient overlap asserted.")
    print("Zero image hash overlap asserted.")

    # Save docs/results/fassd/split_v2.csv
    out_split_path.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(out_split_path, index=False)
    print(f"Saved split_v2.csv to {out_split_path}")

    # Print summary tables
    patients_per_split = {sp: int(selected_df[selected_df["split"] == sp]["patient"].nunique()) for sp in ["train", "val", "test"]}
    images_per_split = {sp: int(len(selected_df[selected_df["split"] == sp])) for sp in ["train", "val", "test"]}
    groups_per_split = {sp: int(selected_df[selected_df["split"] == sp]["group"].nunique()) for sp in ["train", "val", "test"]}

    print("\nPer-Split Breakdown:")
    for sp in ["train", "val", "test"]:
        print(f"  [{sp.upper()}] Groups: {groups_per_split[sp]}, Patients: {patients_per_split[sp]}, Images: {images_per_split[sp]}")
        print(f"    Structure non-empty counts: {selected_struct_counts[sp]}")

    # Verify duplicate pairs in split_v2.csv
    pat_to_v2_split = dict(zip(selected_df["patient"], selected_df["split"]))
    print("\nVerification of Duplicate Pairs in split_v2.csv:")
    for p1, p2 in v1_pairs:
        s1 = pat_to_v2_split[p1]
        s2 = pat_to_v2_split[p2]
        print(f"  {p1} ({s1}) vs {p2} ({s2}) -> Same split? {s1 == s2} (RESOLVED)")
        assert s1 == s2, f"Leakage remains for {p1} and {p2} in split_v2!"

    # =========================================================================
    # 4. Prepare Compact Kaggle Dataset
    # =========================================================================
    k_dir.mkdir(parents=True, exist_ok=True)
    k_images_dir.mkdir(parents=True, exist_ok=True)
    k_masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Step 4: Building Compact Kaggle Dataset at {k_dir} ===")

    # Meta CSV: filename, patient, group, split_v2
    df_meta = selected_df.rename(columns={"split": "split_v2"})
    meta_csv_path = k_dir / "meta.csv"
    df_meta.to_csv(meta_csv_path, index=False)

    # dataset-metadata.json
    metadata = {
        "title": "FASSD fetal abdomen v2",
        "id": "prantakumerpandit390/fassd-fetal-abdomen-v2",
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(k_dir / "dataset-metadata.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    # Copy images byte-exact & encode bit-flag masks
    verification_failures = 0
    total_images_copied = 0
    total_masks_encoded = 0

    for idx, f in enumerate(png_files):
        stem = Path(f).stem
        p_src_img = img_dir / f
        p_dst_img = k_images_dir / f
        p_src_npy = npy_dir / f"{stem}.npy"
        p_dst_mask = k_masks_dir / f"{stem}.png"

        # 1. Byte-exact image copy
        if not p_dst_img.is_file() or p_dst_img.stat().st_size != p_src_img.stat().st_size:
            shutil.copy2(p_src_img, p_dst_img)
        total_images_copied += 1

        # 2. Encode bit-flag mask from NPY
        sample = load_fassd_sample(p_src_npy)
        gt_masks = sample["structures"]
        combined_mask = encode_multilabel_mask(gt_masks)
        cv2.imwrite(str(p_dst_mask), combined_mask)
        total_masks_encoded += 1

        # 3. Verify decoding reproduces all four original masks exactly
        decoded = decode_multilabel_mask(combined_mask)
        for st in STRUCTURE_NAMES:
            if not np.array_equal(gt_masks[st], decoded[st]):
                verification_failures += 1

        if (idx + 1) % 400 == 0 or idx == len(png_files) - 1:
            print(f"  Processed {idx + 1}/{len(png_files)} images & masks (failures: {verification_failures})")

    assert verification_failures == 0, f"Mask verification failed for {verification_failures} samples!"
    print(f"VERIFIED on ALL {len(png_files)} images: Decoding bit-flag masks reproduces all 4 original GT masks byte-for-byte!")

    # Compute total size in MB
    total_bytes = 0
    for root, _, files in os.walk(k_dir):
        for fl in files:
            total_bytes += (Path(root) / fl).stat().st_size
    total_size_mb = total_bytes / (1024 * 1024)

    print(f"\n=== Step 5: Dataset Summary & Kaggle Command ===")
    print(f"Total dataset size: {total_size_mb:.2f} MB ({total_bytes} bytes)")
    print(f"Total images: {total_images_copied}")
    print(f"Total masks: {total_masks_encoded}")
    print(f"Exact Kaggle upload command (no key):")
    print(f'kaggle datasets create -p "{k_dir}"')

    return {
        "selected_seed": selected_seed,
        "n_groups": n_groups,
        "n_patients": n_unique_patients,
        "patients_per_split": patients_per_split,
        "images_per_split": images_per_split,
        "groups_per_split": groups_per_split,
        "structure_counts": selected_struct_counts,
        "total_size_mb": total_size_mb,
        "kaggle_cmd": f'kaggle datasets create -p "{k_dir}"',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix FASSD split leakage, create split_v2, and prepare compact Kaggle dataset")
    parser.add_argument("--raw_dir", type=str, default=DEFAULT_RAW_DIR, help="Raw FASSD directory")
    parser.add_argument("--kaggle_dir", type=str, default=DEFAULT_KAGGLE_DIR, help="Compact Kaggle dataset directory")
    parser.add_argument("--output_split_csv", type=str, default="docs/results/fassd/split_v2.csv")
    parser.add_argument("--split_v1_csv", type=str, default="docs/results/fassd/split_v1.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_split_leakage_fix_and_kaggle_prep(
        raw_dir=args.raw_dir,
        kaggle_dir=args.kaggle_dir,
        output_split_csv=args.output_split_csv,
        split_v1_csv=args.split_v1_csv,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
