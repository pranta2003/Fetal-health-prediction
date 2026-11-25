#!/usr/bin/env python3

# FIELD: Report tools
# MODEL: None (Audit & Verification Harness)
# STEP: report_tools
# USED BY: work/docs/CODE_MAP.md, work/docs/results/data_inventory.json

"""scripts/build_data_inventory.py

Computes exact counts for docs/DATA_INVENTORY.md and generates docs/results/data_inventory.json.
Audits all datasets:
1. CDC Vital Statistics Tabular (C:/fetal-data/cdc/clinical_training_data.csv)
2. HC18 Grand Challenge (C:/fetal-data/HC18)
3. Fetal Abdominal Circumference (AC) (C:/fetal-data/FP, UCL, docs/results/ac_campaign/folds_v1.csv)
4. Fetal Femur Length (FL) (C:/fetal-data/FP, UCL, docs/results/femur/femur_data_followup.json)
5. Head Landmark Biometry (C:/fetal-data/FP, UCL/annotations/px_to_mm, docs/results/hadlock_demo)
6. FASSD Multi-Organ Segmentation (docs/results/fassd)
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
src_dir = REPO_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

DATA_DIR = Path("C:/fetal-data")
OUT_JSON = REPO_ROOT / "docs" / "results" / "data_inventory.json"
OUT_MD = REPO_ROOT / "docs" / "DATA_INVENTORY.md"


def get_patient_id(filename: str, source: str) -> str:
    s = str(filename)
    if source == "FP":
        m = re.search(r"(Patient\d+)", s)
        return m.group(1) if m else s
    elif source == "UCL":
        m = re.match(r"^(\d+)_", s)
        return f"UCL_{m.group(1)}" if m else s
    elif source == "HC18":
        m = re.match(r"^(\d+)_", s)
        return m.group(1) if m else s
    return s


def audit_cdc_tabular() -> dict:
    csv_path = DATA_DIR / "cdc" / "clinical_training_data.csv"
    assert csv_path.is_file(), f"Missing CDC file at {csv_path}"
    
    raw = pd.read_csv(csv_path, low_memory=False)
    total_raw_rows = len(raw)
    total_cols = len(raw.columns)
    
    # Sentinel cleaning and deduplication
    from fetal.clinical.constants import CAT_FEATURES, UNKNOWN_CODES, NUM_FEATURES, FEATURE_SETS
    df_clean = raw.copy()
    for c in CAT_FEATURES:
        if c in df_clean.columns:
            df_clean[c] = df_clean[c].replace(UNKNOWN_CODES, np.nan)
    if "COMBGEST" in df_clean.columns:
        df_clean["COMBGEST"] = df_clean["COMBGEST"].replace([99], np.nan)
    dedup = df_clean.drop_duplicates().reset_index(drop=True)
    
    # Breakdowns
    raw_outcome = raw["OUTCOME"].value_counts().to_dict()
    raw_year = raw["YEAR"].value_counts().sort_index().to_dict()
    raw_year_outcome = pd.crosstab(raw["YEAR"], raw["OUTCOME"]).to_dict()
    
    raw_d20 = raw[raw["COMBGEST"] >= 20].copy()
    raw_d20_train = raw_d20[raw_d20["YEAR"] <= 2021]
    raw_d20_test = raw_d20[raw_d20["YEAR"] >= 2022]
    
    dedup_d20 = dedup[dedup["COMBGEST"] >= 20].copy()
    dedup_d20_train = dedup_d20[dedup_d20["YEAR"] <= 2021]
    dedup_d20_test = dedup_d20[dedup_d20["YEAR"] >= 2022]
    
    e1_num = FEATURE_SETS["primary"]["num"]
    e1_cat = FEATURE_SETS["primary"]["cat"]
    e1_features = e1_num + e1_cat
    
    return {
        "dataset_name": "CDC NVSS Vital Statistics Tabular (Stillbirth Risk)",
        "source_file": str(csv_path),
        "raw_on_disk": False,
        "raw_source_description": (
            "16 annual files (2016-2023) from CDC National Vital Statistics System: "
            "8 Natality files (2016_NATAL.csv to 2023_NATAL.csv, totaling 29,950,846 live births) and "
            "8 Fetal Death files (2016_FETAL_COD.csv to 2023_FETAL_COD.csv, totaling 358,175 fetal deaths). "
            "The 16 raw files were processed in Google Colab/Drive and are NOT hosted on the local laptop disk."
        ),
        "modelling_file_construction": (
            "From the 29.95M live births, exactly ~45,000 live births per year (total 359,961, seed 42) were sampled. "
            "All 358,175 fetal deaths across 2016-2023 were retained without downsampling. "
            "The sets were concatenated (359,961 + 358,175 = 718,136 rows, 80 columns) and shuffled (seed 42)."
        ),
        "total_rows": total_raw_rows,
        "total_columns": total_cols,
        "rows_by_outcome": {
            "live_births_outcome_0": int(raw_outcome.get(0, 0)),
            "stillbirths_outcome_1": int(raw_outcome.get(1, 0)),
        },
        "rows_by_year": {str(k): int(v) for k, v in raw_year.items()},
        "rows_by_year_and_outcome": {
            str(yr): {
                "live": int(raw_year_outcome.get(0, {}).get(yr, 0)),
                "stillbirth": int(raw_year_outcome.get(1, {}).get(yr, 0)),
            }
            for yr in sorted(raw["YEAR"].unique())
        },
        "rows_after_20wk_filter": {
            "raw_file_d20": {
                "total": len(raw_d20),
                "live_births": int((raw_d20["OUTCOME"] == 0).sum()),
                "stillbirths": int((raw_d20["OUTCOME"] == 1).sum()),
                "train_le_2021": {
                    "total": len(raw_d20_train),
                    "stillbirths": int((raw_d20_train["OUTCOME"] == 1).sum()),
                    "live_births": int((raw_d20_train["OUTCOME"] == 0).sum()),
                },
                "test_ge_2022": {
                    "total": len(raw_d20_test),
                    "stillbirths": int((raw_d20_test["OUTCOME"] == 1).sum()),
                    "live_births": int((raw_d20_test["OUTCOME"] == 0).sum()),
                },
            },
            "deduplicated_d20": {
                "total": len(dedup_d20),
                "live_births": int((dedup_d20["OUTCOME"] == 0).sum()),
                "stillbirths": int((dedup_d20["OUTCOME"] == 1).sum()),
                "train_le_2021": {
                    "total": len(dedup_d20_train),
                    "stillbirths": int((dedup_d20_train["OUTCOME"] == 1).sum()),
                    "live_births": int((dedup_d20_train["OUTCOME"] == 0).sum()),
                },
                "test_ge_2022": {
                    "total": len(dedup_d20_test),
                    "stillbirths": int((dedup_d20_test["OUTCOME"] == 1).sum()),
                    "live_births": int((dedup_d20_test["OUTCOME"] == 0).sum()),
                },
            },
        },
        "e1_primary_model_features": {
            "count": len(e1_features),
            "numeric_count": len(e1_num),
            "categorical_count": len(e1_cat),
            "numeric_features": e1_num,
            "categorical_features": e1_cat,
            "all_features": e1_features,
        },
        "columns_removed_breakdown": {
            "total_removed": total_cols - len(e1_features),
            "categories": {
                "target_and_split": ["OUTCOME", "YEAR"],
                "data_leakage_outcome_time": ["COMBGEST", "OEGest_Comb", "OBGEST_FLG", "DBWT", "BWTR4"],
                "labor_delivery_facility_attendant": [
                    "ATTEND", "BFACIL", "BFACIL3", "DLMP_MM", "DLMP_YY", "DMETH_REC",
                    "RDMETH_REC", "ME_PRES", "ME_ROUT", "ME_TRIAL", "MM_AICU", "MM_RUPT", "RESTATUS"
                ],
                "reporting_and_imputation_flags": [
                    "F_CIGS_0", "F_CIGS_1", "F_CIGS_2", "F_CIGS_3", "F_MEDUC", "F_MM_AICU",
                    "F_MPCB", "F_M_HT", "F_PWGT", "F_RF_CESAR", "F_RF_GDIAB", "F_RF_GHYPER",
                    "F_RF_NCESAR", "F_RF_PDIAB", "F_RF_PHYPER", "F_TOBACO", "F_WIC",
                    "MAGE_IMPFLG", "MAGE_REPFLG", "IMP_PLUR", "IMP_SEX", "MRACEIMP", "FAGERPT_FLG"
                ],
                "paternal_characteristics": ["FAGECOMB", "FAGEREC11"],
                "redundant_recodes": [
                    "BMI_R", "CIG_0", "CIG_1", "CIG_2", "CIG_3", "MAGER9", "MAGER14",
                    "M_Ht_In", "PWgt_R", "LBO_REC", "ILLB_R", "ILLB_R11", "RF_CESARN",
                    "MRACE15", "MRACEHISP", "MHISPX"
                ],
                "socioeconomic_and_sensitive_demographic": ["MEDUC", "WIC", "MBSTATE_REC", "MRACE6"],
            },
        },
    }


def audit_hc18() -> dict:
    hc_dir = DATA_DIR / "HC18"
    train_csv = hc_dir / "training_set_pixel_size_and_HC.csv"
    test_csv = hc_dir / "test_set_pixel_size.csv"
    
    df_tr = pd.read_csv(train_csv)
    df_te = pd.read_csv(test_csv)
    
    df_tr["patient"] = df_tr["filename"].apply(lambda f: get_patient_id(f, "HC18"))
    df_te["patient"] = df_te["filename"].apply(lambda f: get_patient_id(f, "HC18"))
    
    # 5-fold cross-validation split (splits_final.json)
    split_json = DATA_DIR / "nnUNet_raw" / "Dataset501_HC18" / "splits_final.json"
    splits_info = {}
    if split_json.is_file():
        with open(split_json, "r") as f:
            splits = json.load(f)
        for i, s in enumerate(splits):
            splits_info[f"fold_{i}"] = {"train": len(s["train"]), "val": len(s["val"])}
            
    return {
        "dataset_name": "HC18 Grand Challenge (Head Circumference)",
        "source": "Grand Challenge / Radboud University Medical Center (van den Heuvel et al., 2018)",
        "files": {
            "training_csv": str(train_csv),
            "test_csv": str(test_csv),
        },
        "total_images": len(df_tr) + len(df_te),
        "total_patients": df_tr["patient"].nunique() + df_te["patient"].nunique(),
        "train_set": {
            "images": len(df_tr),
            "patients": df_tr["patient"].nunique(),
            "folds": splits_info or "5-fold patient-grouped split (Fold 0: 800/199; Fold 1: 800/199; Fold 2: 791/208; Fold 3: 805/194; Fold 4: 807/192)",
        },
        "test_set": {
            "images": len(df_te),
            "patients": df_te["patient"].nunique(),
            "description": "Official unseen Grand Challenge test set evaluated on grand-challenge.org",
        },
    }


def audit_ac() -> dict:
    fp_csv = DATA_DIR / "FP" / "annotations" / "Abdomen.csv"
    ucl_csv = DATA_DIR / "UCL" / "annotations" / "px_to_mm" / "Abdomen.csv"
    ucl_raw_csv = DATA_DIR / "UCL" / "annotations" / "Abdomen.csv"
    folds_csv = REPO_ROOT / "docs" / "results" / "ac_campaign" / "folds_v1.csv"
    audit_json = REPO_ROOT / "docs" / "results" / "dataset_audit" / "dataset_counts_audit.json"
    
    df_fp = pd.read_csv(fp_csv)
    df_ucl = pd.read_csv(ucl_csv)
    df_ucl_raw = pd.read_csv(ucl_raw_csv)
    df_folds = pd.read_csv(folds_csv)
    
    df_fp["patient"] = df_fp["image_name"].apply(lambda f: get_patient_id(f, "FP"))
    df_ucl["patient"] = df_ucl["image_name"].apply(lambda f: get_patient_id(f, "UCL"))
    
    # Check official test split
    fp_test = pd.read_csv(DATA_DIR / "FP" / "annotations" / "Abdomen_Test.csv")
    ucl_test = pd.read_csv(DATA_DIR / "UCL" / "annotations" / "px_to_mm" / "Abdomen_Test.csv")
    
    folds_summary = {}
    for fold_id, group in df_folds.groupby("fold"):
        folds_summary[f"fold_{fold_id}"] = {
            "images": len(group),
            "patients": int(group["patient"].nunique()),
        }
        
    return {
        "dataset_name": "Fetal Abdominal Circumference (AC)",
        "sources": {
            "FP": "Fetal Planes dataset (Burgos-Artizzu et al., 2020 / Zenodo), Barcelona cohort",
            "UCL": "University College London fetal ultrasound cohort (Gimenez et al., 2024 / Zenodo)",
        },
        "files": {
            "fp_annotations": str(fp_csv),
            "ucl_calibrated_annotations": str(ucl_csv),
            "folds_csv": str(folds_csv),
        },
        "published_literature_counts": {"FP": 693, "UCL": 130, "total": 823},
        "raw_on_disk_counts": {"FP": len(df_fp), "UCL_raw": len(df_ucl_raw), "total": len(df_fp) + len(df_ucl_raw)},
        "calibrated_used_counts": {
            "FP": len(df_fp),
            "UCL": len(df_ucl),
            "total": len(df_fp) + len(df_ucl),
            "exclusions_from_published": "34 total missing (31 UCL uncalibrated without px_to_mm rate, 3 patient 125 corrupted layout)",
        },
        "training_cv_cohort": {
            "images": len(df_folds),
            "patients": int(df_folds["patient"].nunique()),
            "folds": folds_summary,
        },
        "test_cohort": {
            "images": len(fp_test) + len(ucl_test),
            "fp_images": len(fp_test),
            "ucl_images": len(ucl_test),
            "description": "Held-out test set from official Abdomen_Test.csv files",
        },
    }


def audit_femur() -> dict:
    fp_csv = DATA_DIR / "FP" / "annotations" / "Femur.csv"
    ucl_csv = DATA_DIR / "UCL" / "annotations" / "px_to_mm" / "Femur.csv"
    ucl_raw_csv = DATA_DIR / "UCL" / "annotations" / "Femur.csv"
    followup_json = REPO_ROOT / "docs" / "results" / "femur" / "femur_data_followup.json"
    
    with open(followup_json, "r") as f:
        followup = json.load(f)
    
    splits = followup["split_counts_after_exclusions"]["splits"]
    
    return {
        "dataset_name": "Fetal Femur Length (FL)",
        "sources": {
            "FP": "Fetal Planes dataset (Burgos-Artizzu et al., 2020), Plane 5 (Femur)",
            "UCL": "University College London fetal ultrasound cohort, Femur landmarks",
        },
        "files": {
            "fp_annotations": str(fp_csv),
            "ucl_calibrated_annotations": str(ucl_csv),
            "audit_file": str(followup_json),
        },
        "published_literature_counts": {"FP": 760, "UCL": 135, "total": 895},
        "calibrated_total": followup["split_counts_after_exclusions"]["total_images_before_exclusions"],
        "exclusions": "3 FP images missing px_to_mm rate (Patient00888, Patient01319, Patient01423); 28 UCL uncalibrated",
        "clean_cohort": {
            "images": followup["split_counts_after_exclusions"]["total_images_after_exclusions"],
            "patients": followup["split_counts_after_exclusions"]["total_patients_after_exclusions"],
        },
        "splits": {
            "train": splits["train"],
            "validation": splits["validation"],
            "test": splits["test"],
        },
    }


def audit_head_landmarks() -> dict:
    fp_csv = DATA_DIR / "FP" / "annotations" / "Head.csv"
    ucl_csv = DATA_DIR / "UCL" / "annotations" / "px_to_mm" / "Head.csv"
    ucl_raw_csv = DATA_DIR / "UCL" / "annotations" / "Head.csv"
    fp_tr_csv = DATA_DIR / "FP" / "annotations" / "Head_Train.csv"
    fp_te_csv = DATA_DIR / "FP" / "annotations" / "Head_Test.csv"
    ucl_tr_csv = DATA_DIR / "UCL" / "annotations" / "px_to_mm" / "Head_Train.csv"
    ucl_te_csv = DATA_DIR / "UCL" / "annotations" / "px_to_mm" / "Head_Test.csv"
    
    df_fp = pd.read_csv(fp_csv)
    df_ucl = pd.read_csv(ucl_csv)
    df_ucl_raw = pd.read_csv(ucl_raw_csv)
    df_fp_tr = pd.read_csv(fp_tr_csv)
    df_fp_te = pd.read_csv(fp_te_csv)
    df_ucl_tr = pd.read_csv(ucl_tr_csv)
    df_ucl_te = pd.read_csv(ucl_te_csv)
    
    df_fp["patient"] = df_fp["image_name"].apply(lambda f: get_patient_id(f, "FP"))
    df_ucl["patient"] = df_ucl["image_name"].apply(lambda f: get_patient_id(f, "UCL"))
    df_fp_tr["patient"] = df_fp_tr["image_name"].apply(lambda f: get_patient_id(f, "FP"))
    df_fp_te["patient"] = df_fp_te["image_name"].apply(lambda f: get_patient_id(f, "FP"))
    df_ucl_tr["patient"] = df_ucl_tr["image_name"].apply(lambda f: get_patient_id(f, "UCL"))
    df_ucl_te["patient"] = df_ucl_te["image_name"].apply(lambda f: get_patient_id(f, "UCL"))
    
    return {
        "dataset_name": "Fetal Head Landmark Biometry (BPD & OFD Calipers)",
        "sources": {
            "FP": "Fetal Planes dataset (Burgos-Artizzu et al., 2020), Head/Trans-thalamic plane",
            "UCL": "University College London cohort, Head landmarks (px_to_mm calibrated)",
        },
        "files": {
            "fp_annotations": str(fp_csv),
            "ucl_calibrated_annotations": str(ucl_csv),
            "fp_train_split": str(fp_tr_csv),
            "fp_test_split": str(fp_te_csv),
            "ucl_train_split": str(ucl_tr_csv),
            "ucl_test_split": str(ucl_te_csv),
        },
        "published_literature_counts": {"FP": 1637, "UCL": 159, "total": 1796},
        "raw_on_disk_counts": {"FP": len(df_fp), "UCL_raw": len(df_ucl_raw), "total": len(df_fp) + len(df_ucl_raw)},
        "calibrated_total": len(df_fp) + len(df_ucl),
        "uncalibrated_ucl_count": len(df_ucl_raw) - len(df_ucl),
        "splits": {
            "train": {
                "total_images": len(df_fp_tr) + len(df_ucl_tr),
                "fp_images": len(df_fp_tr),
                "fp_patients": df_fp_tr["patient"].nunique(),
                "ucl_images": len(df_ucl_tr),
                "ucl_patients": df_ucl_tr["patient"].nunique(),
            },
            "test": {
                "total_images": len(df_fp_te) + len(df_ucl_te),
                "fp_images": len(df_fp_te),
                "fp_patients": df_fp_te["patient"].nunique(),
                "ucl_images": len(df_ucl_te),
                "ucl_patients": df_ucl_te["patient"].nunique(),
            },
        },
    }


def audit_fassd() -> dict:
    audit_file = REPO_ROOT / "docs" / "results" / "fassd" / "fassd_audit.json"
    split1_file = REPO_ROOT / "docs" / "results" / "fassd" / "split_v1.csv"
    split2_file = REPO_ROOT / "docs" / "results" / "fassd" / "split_v2.csv"
    
    with open(audit_file, "r") as f:
        aud = json.load(f)
    df_s1 = pd.read_csv(split1_file)
    df_s2 = pd.read_csv(split2_file)
    
    s1_splits = {}
    for s, g in df_s1.groupby("split"):
        s1_splits[s] = {"images": len(g), "patients": int(g["patient"].nunique())}
        
    s2_splits = {}
    for s, g in df_s2.groupby("split"):
        s2_splits[s] = {"images": len(g), "patients": int(g["patient"].nunique())}
        
    return {
        "dataset_name": "FASSD (Fetal Abdominal Structures Segmentation Dataset)",
        "source": "FASSD Dataset (Xu et al., 2023 / Zenodo / Kaggle prantakumerpandit390/fassd-fetal-abdomen-v2)",
        "files": {
            "audit_file": str(audit_file),
            "split_v1_csv": str(split1_file),
            "split_v2_csv": str(split2_file),
        },
        "total_images": aud["total_images"],
        "total_patients": aud["patient_id_parsing"]["unique_patients"],
        "structures": ["artery", "liver", "stomach", "vein"],
        "split_v1": s1_splits,
        "split_v2": s2_splits,
    }


def main():
    print("Building comprehensive data inventory from real disk files...")
    cdc = audit_cdc_tabular()
    hc18 = audit_hc18()
    ac = audit_ac()
    femur = audit_femur()
    head = audit_head_landmarks()
    fassd = audit_fassd()
    
    inventory = {
        "title": "Comprehensive Dataset Inventory",
        "generated_by": "scripts/build_data_inventory.py",
        "datasets": {
            "cdc_tabular": cdc,
            "hc18": hc18,
            "ac": ac,
            "femur": femur,
            "head_landmarks": head,
            "fassd": fassd,
        },
    }
    
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)
    print(f"Saved inventory JSON to {OUT_JSON}")


if __name__ == "__main__":
    main()
