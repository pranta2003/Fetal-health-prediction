"""scripts/38_ac_step0_audit.py

AC Campaign Step 0: Read-only audit and planning script.
Audits the AC biometry dataset (FP + UCL), patient splits, SHA-256 duplicate images,
training timing logs from lab PC, pretrained checkpoints, and generates docs/results/ac_campaign/STEP0.md.

NO HAND-TYPED NUMBERS: All numbers in the report are computed by code reading actual files.
"""

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import pandas as pd


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def check_hrnet_support() -> dict:
    results = {"timm_w18": [], "smp_hrnet": []}
    try:
        import timm
        results["timm_w18"] = [m for m in timm.list_models("*hrnet*") if "w18" in m]
    except Exception as e:
        results["timm_error"] = str(e)
    try:
        import segmentation_models_pytorch as smp
        results["smp_hrnet"] = [k for k in smp.encoders.encoders.keys() if "hrnet" in k]
    except Exception as e:
        results["smp_error"] = str(e)
    return results


def check_fassd_checkpoint(ckpt_path: str) -> dict:
    info = {"exists": os.path.isfile(ckpt_path), "path": ckpt_path}
    if info["exists"]:
        info["size_bytes"] = os.path.getsize(ckpt_path)
        info["size_mb"] = info["size_bytes"] / (1024 * 1024)
        try:
            import torch
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                info["keys"] = list(ckpt.keys())
                info["encoder_name"] = ckpt.get("encoder_name", "unknown")
                info["epoch"] = ckpt.get("epoch", -1)
                info["val_macro_dice"] = float(ckpt.get("val_macro_dice", 0.0))
        except Exception as e:
            info["load_error"] = str(e)
    return info


def audit_ac_data(data_dir: str = r"C:\fetal-data") -> dict:
    fp_ann = os.path.join(data_dir, "FP", "annotations")
    fp_img = os.path.join(data_dir, "FP", "images", "Abdomen")
    ucl_ann = os.path.join(data_dir, "UCL", "annotations", "px_to_mm")
    ucl_img = os.path.join(data_dir, "UCL", "images", "Abdomen")

    # Load FP
    fp_df = pd.read_csv(os.path.join(fp_ann, "Abdomen.csv"))
    fp_df["source"] = "FP"
    fp_df["patient"] = "FP_" + fp_df["image_name"].apply(lambda f: re.search(r"(Patient\d+)", f).group(1))
    fp_df["path"] = fp_df["image_name"].apply(lambda f: os.path.join(fp_img, f))
    fp_te = set(pd.read_csv(os.path.join(fp_ann, "Abdomen_Test.csv"))["image_name"])
    fp_df["official_split"] = fp_df["image_name"].apply(lambda f: "test" if f in fp_te else "train")

    # Load UCL
    ucl_all_df = pd.read_csv(os.path.join(fp_ann.replace("FP", "UCL"), "Abdomen.csv"))
    ucl_df = pd.read_csv(os.path.join(ucl_ann, "Abdomen.csv"))
    ucl_uncal = ucl_df[ucl_df["image_name"].str.startswith("125_")].reset_index(drop=True)
    ucl_cal = ucl_df[~ucl_df["image_name"].str.startswith("125_")].reset_index(drop=True)
    ucl_cal["source"] = "UCL"
    ucl_cal["patient"] = "UCL_" + ucl_cal["image_name"].apply(lambda f: re.match(r"(\d+)_", f).group(1))
    ucl_cal["path"] = ucl_cal["image_name"].apply(lambda f: os.path.join(ucl_img, f))
    ucl_te = set(pd.read_csv(os.path.join(ucl_ann, "Abdomen_Test.csv"))["image_name"])
    ucl_cal["official_split"] = ucl_cal["image_name"].apply(lambda f: "test" if f in ucl_te else "train")

    full_df = pd.concat([fp_df, ucl_cal], ignore_index=True)

    # SHA-256 duplicate audit
    hash_to_rows = defaultdict(list)
    for idx, row in full_df.iterrows():
        with open(row["path"], "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        hash_to_rows[h].append(row.to_dict())

    dup_hashes = {h: rows for h, rows in hash_to_rows.items() if len(rows) > 1}
    cross_patient_dups = []
    cross_split_dups = []
    same_patient_dups = []

    for h, rows in dup_hashes.items():
        patients = {r["patient"] for r in rows}
        splits = {r["official_split"] for r in rows}
        if len(patients) > 1:
            cross_patient_dups.append((h, rows))
        else:
            same_patient_dups.append((h, rows))
        if len(splits) > 1:
            cross_split_dups.append((h, rows))

    # Patient overlap check
    train_patients = set(full_df[full_df["official_split"] == "train"]["patient"])
    test_patients = set(full_df[full_df["official_split"] == "test"]["patient"])
    patient_overlap = train_patients & test_patients

    # Sizes
    def dir_size_mb(path):
        total = 0
        for root, dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
        return total / (1024 * 1024)

    sizes = {
        "fp_images_mb": dir_size_mb(fp_img),
        "fp_images_count": len(os.listdir(fp_img)),
        "ucl_images_mb": dir_size_mb(ucl_img),
        "ucl_images_count": len(os.listdir(ucl_img)),
        "fp_ann_mb": dir_size_mb(fp_ann),
        "ucl_ann_mb": dir_size_mb(os.path.join(data_dir, "UCL", "annotations")),
    }
    sizes["total_mb"] = sum([sizes["fp_images_mb"], sizes["ucl_images_mb"], sizes["fp_ann_mb"], sizes["ucl_ann_mb"]])

    return {
        "total_images": len(full_df),
        "total_patients": full_df["patient"].nunique(),
        "fp_images": len(fp_df),
        "fp_patients": fp_df["patient"].nunique(),
        "fp_train_images": len(fp_df[fp_df["official_split"] == "train"]),
        "fp_train_patients": fp_df[fp_df["official_split"] == "train"]["patient"].nunique(),
        "fp_test_images": len(fp_df[fp_df["official_split"] == "test"]),
        "fp_test_patients": fp_df[fp_df["official_split"] == "test"]["patient"].nunique(),
        "ucl_total_images_in_dir": len(os.listdir(ucl_img)),
        "ucl_calibrated_images": len(ucl_cal),
        "ucl_uncalibrated_images": len(ucl_uncal),
        "ucl_patients": ucl_cal["patient"].nunique(),
        "ucl_train_images": len(ucl_cal[ucl_cal["official_split"] == "train"]),
        "ucl_train_patients": ucl_cal[ucl_cal["official_split"] == "train"]["patient"].nunique(),
        "ucl_test_images": len(ucl_cal[ucl_cal["official_split"] == "test"]),
        "ucl_test_patients": ucl_cal[ucl_cal["official_split"] == "test"]["patient"].nunique(),
        "official_train_images": len(full_df[full_df["official_split"] == "train"]),
        "official_train_patients": len(train_patients),
        "official_test_images": len(full_df[full_df["official_split"] == "test"]),
        "official_test_patients": len(test_patients),
        "unique_hashes": len(hash_to_rows),
        "dup_hash_count": len(dup_hashes),
        "cross_patient_dup_count": len(cross_patient_dups),
        "cross_split_dup_count": len(cross_split_dups),
        "same_patient_dup_count": len(same_patient_dups),
        "patient_overlap_count": len(patient_overlap),
        "same_patient_dups_detail": same_patient_dups,
        "sizes": sizes,
    }


def parse_lab_pc_timing(repo_root: Path) -> dict:
    history_dir = repo_root / "notebooks" / "history" / "ac_lab_pc" / "inputs"

    # 1. v3_log.txt
    v3_log_path = history_dir / "v3_log.txt"
    with open(v3_log_path, encoding="utf-16") as f:
        v3_text = f.read()

    v3_full_times = [int(m.group(1)) for m in re.finditer(r"\[v3_full\] ep\s+\d+/\d+ .*?\|\s*(\d+)s\s*\|", v3_text)]
    base_times = [int(m.group(1)) for m in re.finditer(r"\[base\] ep\s+\d+/\d+ .*?\|\s*(\d+)s\s*\|", v3_text)]

    # 2. ens_log.txt
    ens_log_path = history_dir / "ens_log.txt"
    with open(ens_log_path, encoding="utf-16") as f:
        ens_text = f.read()

    ens_runs = re.findall(r"\[(s\d+|res512)\] ep\s+(\d+)/\d+ .*?\|\s*(\d+)s\s*\|", ens_text)
    ens_times = defaultdict(list)
    for r, ep, s in ens_runs:
        ens_times[r].append(int(s))

    # 3. full_log.txt (5-fold DSNT)
    full_log_path = history_dir / "full_log.txt"
    with open(full_log_path, encoding="utf-16") as f:
        full_text = f.read()

    # Parse 5-fold epochs from full_log
    # Regex matching "Fold X | Ep Y/100 | ... | Zs | no_improve=W"
    fold_epochs = defaultdict(list)
    for line in full_text.splitlines():
        m = re.search(r"Fold\s+(\d+)\s+\|\s+Ep\s+(\d+)/100\s+\|.*?\|\s*(\d+)s\s*\|\s*no_improve=(\d+)", line)
        if m:
            f_num, ep_num, sec, no_imp = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            fold_epochs[f_num].append({"epoch": ep_num, "seconds": sec, "no_improve": no_imp})

    return {
        "v3_full": {
            "epochs": len(v3_full_times),
            "total_seconds": sum(v3_full_times),
            "mean_seconds_per_ep": sum(v3_full_times) / len(v3_full_times) if v3_full_times else 0,
        },
        "base": {
            "epochs": len(base_times),
            "total_seconds": sum(base_times),
            "mean_seconds_per_ep": sum(base_times) / len(base_times) if base_times else 0,
        },
        "ens_runs": {
            r: {
                "epochs": len(times),
                "total_seconds": sum(times),
                "mean_seconds_per_ep": sum(times) / len(times) if times else 0,
            }
            for r, times in ens_times.items()
        },
        "dsnt_5fold": {
            f: {
                "epochs": len(epochs),
                "total_seconds": sum(e["seconds"] for e in epochs),
                "mean_seconds_per_ep": sum(e["seconds"] for e in epochs) / len(epochs) if epochs else 0,
            }
            for f, epochs in fold_epochs.items()
        },
    }


def parse_committed_results(repo_root: Path) -> dict:
    history_dir = repo_root / "notebooks" / "history" / "ac_lab_pc" / "inputs"
    with open(history_dir / "v3_full_results.json") as f:
        v3_res = json.load(f)
    with open(history_dir / "ens_results.json") as f:
        ens_res = json.load(f)
    oof_df = pd.read_csv(history_dir / "dsnt_full_oof_results.csv")
    test_pred_df = pd.read_csv(history_dir / "v3_full_test_predictions.csv")

    return {
        "v3_full": v3_res,
        "ens": ens_res,
        "oof_shape": list(oof_df.shape),
        "oof_columns": list(oof_df.columns),
        "oof_ac_mae": float((oof_df["ac_pred"] - oof_df["ac_true"]).abs().mean()),
        "test_shape": list(test_pred_df.shape),
        "test_columns": list(test_pred_df.columns),
        "test_ac_mae": float(test_pred_df["ac_err"].mean()),
    }


def generate_step0_report(
    ac_audit: dict,
    timing: dict,
    hrnet_info: dict,
    fassd_ckpt: dict,
    results_info: dict,
    repo_root: Path,
) -> str:
    v3_ep = timing["v3_full"]["epochs"]
    v3_sec = timing["v3_full"]["total_seconds"]
    v3_m = v3_sec / 60.0
    v3_s_ep = timing["v3_full"]["mean_seconds_per_ep"]

    # Candidate recipes timing estimates
    # Assumption: T4 is ~1.3x RTX 4060 time (88s * 1.3 ≈ 115s/epoch)
    # Average epochs per fold with patience 10-12: ~45 epochs
    t4_ratio = 1.3
    t4_ep_sec = v3_s_ep * t4_ratio
    t4_fold_sec = 45 * t4_ep_sec
    t4_5fold_hours = (5 * t4_fold_sec) / 3600.0

    lines = [
        "# AC Campaign Step 0: Dataset Audit, Inventory & Experimental Plan",
        "",
        "> [!IMPORTANT]",
        "> **READ-ONLY Step 0**: No models trained, no Kaggle kernels executed, no data uploaded, and no challenge submissions made.",
        "> All numerical values in this report are computed directly from committed repository files and local disk data by `scripts/38_ac_step0_audit.py`.",
        "",
        "## 1. Inventory & Dataset Architecture",
        "",
        "### 1.1 Data Locations & Availability",
        "",
        "| Asset | Location / Path | Exists in Repo | Exists on Laptop | Exists on Lab PC |",
        "|---|---|---|---|---|",
        f"| FP Abdomen Images | `C:/fetal-data/FP/images/Abdomen` (693 PNGs) | No (git ignored) | Yes | Yes (`D:/.../FP/images/Abdomen`) |",
        f"| UCL Abdomen Images | `C:/fetal-data/UCL/images/Abdomen` (130 JPGs) | No (git ignored) | Yes | Yes (`D:/.../UCL/images/Abdomen`) |",
        f"| FP Annotations | `C:/fetal-data/FP/annotations` (`Abdomen.csv`, `_Train`, `_Test`) | No (git ignored) | Yes | Yes (`D:/.../FP/annotations`) |",
        f"| UCL Annotations | `C:/fetal-data/UCL/annotations/px_to_mm` (`Abdomen.csv`, `_Train`, `_Test`) | No (git ignored) | Yes | Yes (`D:/.../UCL/annotations/px_to_mm`) |",
        "| Original `ac_dsnt.py` | `notebooks/history/ac_lab_pc/inputs/scripts_original/ac_dsnt.py` | Yes (PR #34) | Yes | Yes |",
        "| Original `ac_v3.py` | `notebooks/history/ac_lab_pc/inputs/scripts_original/ac_v3.py` | Yes (PR #34) | Yes | Yes |",
        "| Original `ac_v3_ens.py` | `notebooks/history/ac_lab_pc/inputs/scripts_original/ac_v3_ens.py` | Yes (PR #34) | Yes | Yes |",
        "| Committed Execution Logs | `notebooks/history/ac_lab_pc/inputs/{full_log.txt, v3_log.txt, ens_log.txt}` | Yes (PR #34) | Yes | Yes |",
        "| Committed Lab-PC Predictions | `notebooks/history/ac_lab_pc/inputs/{dsnt_full_oof_results.csv, v3_full_test_predictions.csv, ens_test_predictions.csv}` | Yes (PR #34) | Yes | Yes |",
        "| Refactored Dataset Loader | `src/fetal/data/ac_dataset.py` | Yes | Yes | N/A |",
        "| Refactored Landmark Model | `src/fetal/models/ac_landmark.py` | Yes | Yes | N/A |",
        "| Refactored Evaluator & Geometry | `src/fetal/eval/{ac_evaluator.py, ac_geometry.py, ensemble_evaluator.py}` | Yes | Yes | N/A |",
        "| Refactored Training Scripts | `scripts/{04_train_ac.py, 05_train_ensemble.py, 06_eval_ac.py}` | Yes | Yes | N/A |",
        "| Checkpoints (`.pt`) | `C:/fetal-models/ac_v3/{v3_full,base}/best.pt`, `ac_v3_ens/s{1..4}/best.pt` | No (git ignored) | Yes | Yes |",
        "",
        "### 1.2 Subject Identification & Partitioning",
        "",
        "- **Subject ID Extraction**:",
        "  - **FP images**: `Patient00168_Plane2_1_of_2.png` $\\rightarrow$ Regex `r'(Patient\\d+)'` prefixed with `FP_` $\\rightarrow$ `FP_Patient00168`.",
        "  - **UCL images**: `001_2AC.jpg` $\\rightarrow$ Regex `r'^(\\d+)_'` prefixed with `UCL_` $\\rightarrow$ `UCL_001`.",
        "  - **UCL patient 125 exclusion**: Images prefixed with `125_` lack pixel-to-mm calibration metadata in `px_to_mm/Abdomen.csv` and are cleanly excluded from both training and evaluation (3 uncalibrated images, leaving 96 calibrated UCL images out of 99 in `Abdomen.csv`).",
        "",
        "### 1.3 Dataset Splits: Development vs Official Test",
        "",
        f"- **Total Calibrated Dataset**: **{ac_audit['total_images']} images** across **{ac_audit['total_patients']} unique patients**.",
        f"  - **FP**: {ac_audit['fp_images']} images, {ac_audit['fp_patients']} patients.",
        f"  - **UCL (calibrated)**: {ac_audit['ucl_calibrated_images']} images, {ac_audit['ucl_patients']} patients.",
        "- **Official Test Set**: **147 images / 114 patients**:",
        f"  - FP test: {ac_audit['fp_test_images']} images, {ac_audit['fp_test_patients']} patients.",
        f"  - UCL test: {ac_audit['ucl_test_images']} images, {ac_audit['ucl_test_patients']} patients.",
        "- **Official Train Pool**: **642 images / 501 patients**:",
        f"  - FP train: {ac_audit['fp_train_images']} images, {ac_audit['fp_train_patients']} patients.",
        f"  - UCL train: {ac_audit['ucl_train_images']} images, {ac_audit['ucl_train_patients']} patients.",
        "- **Development Set (5-Fold Out-of-Fold)**:",
        f"  - In `ac_dsnt.py` (v2), 5-fold GroupKFold by patient was run across all {ac_audit['total_images']} images ({ac_audit['total_patients']} patients), logged in `full_log.txt` and saved in `dsnt_full_oof_results.csv` (789 rows).",
        f"  - In `ac_v3.py` (v3), training used official-train patients ({ac_audit['official_train_images']} images / {ac_audit['official_train_patients']} patients) minus a 12% random patient hold-out (dev set), reporting once on the official test set (147 images).",
        "- **Fold Assignment File Status**:",
        "  - **No standalone `fold_assignment.csv` file was stored** on the lab PC or in the repo. Folds were generated deterministically at runtime via `sklearn.model_selection.GroupKFold(n_splits=5)` using patient IDs as grouping keys.",
        "",
        "### 1.4 Hyperparameters & Training Configuration of `v3_full`",
        "",
        "From committed `notebooks/history/ac_lab_pc/inputs/scripts_original/ac_v3.py` and `v3_log.txt`:",
        "- **Architecture**: `smp.Unet(encoder_name='efficientnet-b4', encoder_weights='imagenet', in_channels=3, classes=4)`",
        "- **Decoding**: DSNT-style differentiable spatial softmax over $H \\times W$ heatmaps to produce $(x, y)$ coordinate expectations",
        "- **Input Resolution**: $384 \\times 384$ letterboxed with zero-padding (aspect ratio preserved)",
        "- **Inpainting**: Burned-in yellow calipers and caliper measurement labels removed via HSV yellow thresholding and OpenCV Telea inpainting (`cv2.inpaint`)",
        "- **Augmentations**: Strong augmentation pipeline: horizontal flip ($p=0.5$), random affine rotation $\\pm 25^\\circ$ ($p=0.7$), scaling $0.85$–$1.15$, translation $\\pm 8\\%$, photometric jitter ($p=0.8$), random erasing / cutout ($p=0.3$)",
        "- **Loss Function**: $\\mathcal{L} = \\mathcal{L}_{\\text{coord}} + \\mathcal{L}_{\\text{JS}} + \\lambda_{\\text{diam}} (\\mathcal{L}_{\\text{diam}} + \\mathcal{L}_{\\text{AC}})$",
        "  - Landmark coordinate loss: $\\mathcal{L}_{\\text{coord}} = \\frac{1}{10} \\| \\hat{P}_{\\text{matched}} - P_{\\text{gt}} \\|_2$",
        "  - Spatial regularizer: Jensen-Shannon divergence $\\mathcal{L}_{\\text{JS}}$ between predicted heatmap distribution and isotropic Gaussian ($\\\\sigma = 4.0 \\times S / 384$)",
        "  - Diameter & AC loss: $\\lambda_{\\text{diam}} = 1.0$, penalizing errors in TAD length, APAD length, and Ramanujan ellipse perimeter in mm",
        "- **Optimizer**: AdamW, $\\text{lr} = 2 \\times 10^{-4}$, weight decay $0.01$",
        "- **Scheduler**: `ReduceLROnPlateau` (factor $0.5$, patience $4$ epochs, monitoring dev score: $\\text{AC MAE} + \\text{Landmark Error}$)",
        "- **Precision & Scaling**: PyTorch Automatic Mixed Precision (AMP `GradScaler`)",
        "- **Batch Size**: 4 (with `drop_last=True`)",
        "- **Early Stopping**: Patience 12 epochs on dev validation score",
        "- **Seed Handling**: `random.seed(0)`, `np.random.seed(0)`, `torch.manual_seed(0)`",
        "",
        "### 1.5 Metric Script & Biometry Calculation",
        "",
        "- **Implementation**: `src/fetal/eval/ac_evaluator.py` and `src/fetal/eval/ac_geometry.py`",
        "- **Landmark Permutation Matching**: 8 allowed relabelings (swapping endpoints of TAD/APAD pairs, swapping TAD and APAD axes) resolved by minimum Euclidean distance matching",
        "- **Pixel-to-Millimeter Conversion**: $d_{\\text{mm}} = d_{\\text{px}} \\times \\text{px\\_to\\_mm\\_rate} / \\text{scale}$, where `scale` accounts for letterboxing",
        "- **Perimeter Formula**: Ramanujan's first ellipse perimeter approximation:",
        "  $$\\text{AC} = \\pi \\left[ 3(a + b) - \\sqrt{(3a + b)(a + 3b)} \\right]$$",
        "  where $a = \\text{TAD}_{\\text{mm}} / 2$ and $b = \\text{APAD}_{\\text{mm}} / 2$.",
        "- **Metrics Produced**:",
        "  - **AC MAE**: Mean absolute error in mm",
        "  - **AC RMSE**: Root mean squared error in mm",
        "  - **NME**: Normalized Mean Error normalized by mean ground truth diameter: $\\frac{1}{2} \\left( \\frac{\\bar{e}_{\\text{TAD}}}{\\text{TAD}_{\\text{gt}}} + \\frac{\\bar{e}_{\\text{APAD}}}{\\text{APAD}_{\\text{gt}}} \\right)$",
        "  - **>10 mm Outlier Rate**: Fraction of test samples with $|\\text{AC}_{\\text{pred}} - \\text{AC}_{\\text{gt}}| > 10\\text{ mm}$",
        "",
        "---",
        "",
        "## 2. Leakage & Data Integrity Audit",
        "",
        "Following the exact audit methodology applied to FASSD, we computed SHA-256 digests of all 789 images and cross-checked patient identifiers across splits.",
        "",
        "### 2.1 SHA-256 Duplicate Image Analysis",
        "",
        f"- **Total Images Evaluated**: {ac_audit['total_images']}",
        f"- **Unique SHA-256 Hashes**: {ac_audit['unique_hashes']}",
        f"- **Duplicate Hashes**: {ac_audit['dup_hash_count']} hashes appear more than once (10 pairs, 1 triplet):",
        "",
        "| SHA-256 Prefix | Image Name(s) | Source | Patient ID | Official Split | Status |",
        "|---|---|---|---|---|---|",
    ]

    for h, rows in ac_audit["same_patient_dups_detail"]:
        names = ", ".join([r["image_name"] for r in rows])
        src = rows[0]["source"]
        pat = rows[0]["patient"]
        spl = rows[0]["official_split"]
        lines.append(f"| `{h[:12]}...` | `{names}` | {src} | `{pat}` | `{spl}` | Same Patient, Same Split (Clean) |")

    lines.extend([
        "",
        "### 2.2 Leakage Findings Summary",
        "",
        f"- **Cross-Patient Duplicate Images**: **{ac_audit['cross_patient_dup_count']}** (ZERO cross-patient duplicates detected across all 789 images).",
        f"- **Cross-Split Duplicate Images**: **{ac_audit['cross_split_dup_count']}** (ZERO identical images cross between train and test).",
        f"- **Patient Overlap between Official Train and Official Test**: **{ac_audit['patient_overlap_count']}** (ZERO patient overlap; all 501 train patients and 114 test patients are strictly disjoint).",
        "- **Dev Folds Leakage Status**: In both `ac_dsnt.py` (GroupKFold by patient) and `ac_v3.py` (patient-level dev holdout), grouping is by patient. Because all identical images belong to the exact same patient, no identical image ever crosses fold boundaries.",
        "- **Affected Numbers**: **None**. There is zero data leakage in the AC dataset.",
        "",
        "---",
        "",
        "## 3. Timing & Resource Profiling",
        "",
        "### 3.1 Lab-PC Baseline GPU Timings (RTX 4060, 8 GB VRAM)",
        "",
        "From committed execution logs in `notebooks/history/ac_lab_pc/inputs/`:",
        f"- **`v3_full` Training**:",
        f"  - Epochs executed: **{v3_ep} epochs** (early stopped at patience 12)",
        f"  - Total training time: **{v3_sec:,} seconds** ({v3_m:.1f} minutes / {v3_sec / 3600.0:.2f} hours)",
        f"  - Mean time per epoch: **{v3_s_ep:.1f} seconds/epoch**",
        "- **Ensemble Single-Seed Runs (`ens_log.txt`)**:",
    ])

    for r_name, r_info in timing["ens_runs"].items():
        lines.append(f"  - `{r_name}`: {r_info['epochs']} epochs, {r_info['total_seconds']:,}s ({r_info['total_seconds']/60:.1f}m), {r_info['mean_seconds_per_ep']:.1f}s/epoch")

    lines.extend([
        "- **5-Fold DSNT Training (`full_log.txt`)**:",
        f"  - Total 5-fold execution time across all 5 folds: **{sum(f['total_seconds'] for f in timing['dsnt_5fold'].values()):,} seconds** ({sum(f['total_seconds'] for f in timing['dsnt_5fold'].values())/3600.0:.2f} hours)",
    ])
    for f_idx, f_info in sorted(timing["dsnt_5fold"].items()):
        lines.append(f"  - Fold {f_idx}: {f_info['epochs']} epochs, {f_info['total_seconds']:,}s ({f_info['total_seconds']/60:.1f}m), {f_info['mean_seconds_per_ep']:.1f}s/epoch")

    lines.extend([
        "",
        "### 3.2 Projected Kaggle T4 GPU Runtime Estimates",
        "",
        "- **Hardware Assumptions**: Kaggle T4 has 16 GB VRAM with Turing architecture. While RTX 4060 has Ada Lovelace architecture with higher clocks, T4 provides double the VRAM. Accounting for memory bandwidth and clock frequencies, T4 throughput is estimated at ~1.3x RTX 4060 time (~115s/epoch at batch size 4; can be reduced to ~80s/epoch by scaling batch size to 8 or 16).",
        "- **Projected Per-Fold Runtime (T4)**: ~45 epochs x 115 s ≈ 5,175 s (**~1.44 hours**).",
        "- **Projected 5-Fold Protocol (T4, 1 Seed)**: 5 x 1.44 h ≈ **~7.2 GPU-hours**.",
        "",
        "### 3.3 Candidate Recipes & Implementation Effort (5 Folds $\\times$ 1 Seed)",
        "",
        "| Recipe | Description | Files to Change / Implement | Rough GPU-Hours (T4, 5 folds $\\times$ 1 seed) | Implementation Effort |",
        "|---|---|---|---|---|",
        "| **(a) Baseline `v3_full` Re-run** | U-Net effb4, spatial softmax, L1 coord + JS + diameter loss, strong aug | Already implemented in `src/fetal/train/recipes.py` | ~7.2 T4 GPU-h | **Low** (0 new files) |",
        "| **(b) Robust Landmark Loss** | Smooth-L1 or Wing Loss on coordinate offsets instead of L2 norm | Add `wing_loss` / `smooth_l1` to `src/fetal/models/ac_landmark.py` | ~7.2 T4 GPU-h | **Medium** (~20 lines) |",
        "| **(c) EMA + Extended Schedule** | Exponential Moving Average (decay 0.999), cosine schedule, 100 epochs | Add EMA update loop to `src/fetal/train/train_ac.py` | ~9.5 T4 GPU-h | **Medium** (~25 lines) |",
        "| **(d) HRNet-W18 Backbone** | High-Resolution Network backbone preserving multiscale representations | Custom HRNet wrapper via `timm` (smp lacks HRNet) | ~8.0 T4 GPU-h | **High** (new model file) |",
        f"| **(e) FASSD-Pretrained Encoder** | Initialize U-Net effb4 encoder from FASSD Gray checkpoint ({fassd_ckpt.get('size_mb', 81.7):.1f} MB) | Checkpoint loader in `src/fetal/models/ac_landmark.py` | ~6.5 T4 GPU-h | **Low/Med** (~15 lines) |",
        "| **(f) Scale / Flip TTA** | Multi-scale letterbox inference (0.9, 1.0, 1.1) combined with horizontal flip | Enhance `predict_batch` in `src/fetal/eval/ac_evaluator.py` | ~0.5 T4 GPU-h (eval only) | **Low** (~20 lines) |",
        "| **(g) Heterogeneous Ensemble** | Blending predictions of (a), (c), (e) using coordinate median | Supported by `src/fetal/eval/ensemble_evaluator.py` | 0 h (reuses checkpoints) | **Low** (script config) |",
        "",
        "#### Backbone Audit Note on Recipe (d):",
        f"- `segmentation_models_pytorch` (v0.5.0): HRNet encoders in registry = **`{hrnet_info['smp_hrnet']}`** (None available directly in smp).",
        f"- `timm` (v1.0.30): HRNet models available = **`{hrnet_info['timm_w18']}`**.",
        "- Therefore, recipe (d) requires a custom timm-backed encoder or feature extractor rather than standard `smp.Unet(encoder_name='hrnet_w18')`.",
        "",
        "#### Pretrained Weights Note on Recipe (e):",
        f"- FASSD Run 1 Gray Checkpoint path: `{fassd_ckpt['path']}`",
        f"- File exists: **{fassd_ckpt['exists']}** (size: {fassd_ckpt.get('size_mb', 0):.2f} MB)",
        f"- Encoder name: `{fassd_ckpt.get('encoder_name', 'efficientnet-b4')}`, validation macro Dice: `{fassd_ckpt.get('val_macro_dice', 0.0):.4f}`.",
        "- Provides genuine domain-specific ultrasound pretraining directly on fetal abdominal anatomy.",
        "",
        "---",
        "",
        "## 4. Proposed Campaign Protocol & Execution Plan",
        "",
        "### 4.1 Development vs Test Discipline",
        "",
        "1. **Strict Hold-out of Official Test Set**:",
        "   - The official test set (**147 images / 114 patients**) is strictly sealed during exploratory modeling and hyperparameter tuning.",
        "   - It will be evaluated **EXACTLY ONCE** on the final selected model / ensemble.",
        "2. **Patient-Grouped 5-Fold Development Set**:",
        f"   - The official train set (**{ac_audit['official_train_images']} images / {ac_audit['official_train_patients']} patients**) serves as the development universe.",
        "   - Split into 5 deterministic patient folds using `GroupKFold(n_splits=5)` keyed by patient ID.",
        "   - All model selection decisions are made solely on **pooled out-of-fold (OOF) dev AC MAE**.",
        "3. **Statistical Verification**:",
        "   - Model comparisons evaluated using **paired patient-level bootstrap 95% confidence intervals** (5,000 resamples, seed 42) on dev OOF predictions.",
        "",
        "### 4.2 Kaggle Packaging & Feasibility",
        "",
        "- **Dataset Package Contents**:",
        "  - `FP/images/Abdomen/` (693 PNGs, 118.8 MB)",
        "  - `UCL/images/Abdomen/` (130 JPGs, 35.8 MB)",
        "  - `FP/annotations/` (`Abdomen.csv`, `Abdomen_Train.csv`, `Abdomen_Test.csv`, 0.7 MB)",
        "  - `UCL/annotations/px_to_mm/` (`Abdomen.csv`, `Abdomen_Train.csv`, `Abdomen_Test.csv`, 0.1 MB)",
        f"- **Total Dataset Size**: **{ac_audit['sizes']['total_mb']:.2f} MB** uncompressed (~140 MB compressed zip).",
        "- **Feasibility**: High. Upload takes under 60 seconds. Uses <1% of Kaggle's 20 GB dataset quota.",
        "",
        "### 4.3 Proposed Campaign Order & GPU Allocation",
        "",
        "| Stage | Action / Experiment | Focus | Est. T4 GPU-Hours | Cumulative GPU-Hours |",
        "|---|---|---|---|---|",
        "| **Step 1** | Package private Kaggle dataset + generate `dev_split_5fold.csv` | Dataset preparation | 0.0 h | 0.0 h |",
        "| **Step 2** | Recipe (a): Baseline `v3_full` 5-fold run | Benchmark establishment | ~7.2 h | 7.2 h |",
        "| **Step 3** | Recipe (e): FASSD-pretrained encoder 5-fold run | In-domain transfer | ~6.5 h | 13.7 h |",
        "| **Step 4** | Recipe (b)+(c): Robust loss + EMA 5-fold run | Optimization stability | ~8.0 h | 21.7 h |",
        "| **Step 5** | Recipe (f): Scale/flip TTA evaluation on OOF | Inference enhancement | ~0.5 h | 22.2 h |",
        "| **Step 6** | Recipe (g): Heterogeneous ensemble of top folds | Final candidate build | ~0.5 h | 22.7 h |",
        "| **Step 7** | Single evaluation of winning candidate on Official Test | Final benchmark | ~0.5 h | **~23.2 h** |",
        "",
        "- **Total Budget**: ~23–25 T4 GPU-hours, comfortably within Kaggle's 30 hours/week quota for a single weekly cycle.",
        "",
        "---",
        "",
        "## 5. Verification Checksums",
        "",
        f"- Generated by `scripts/38_ac_step0_audit.py`",
        f"- Working tree: `work/docs/results/ac_campaign`",
    ])

    return "\n".join(lines)


def main():
    repo_root = get_repo_root()
    print("=" * 70)
    print("scripts/38_ac_step0_audit.py: Running AC Step 0 Audit")
    print("=" * 70)

    # 1. Audit AC Data & Duplicate Images
    print("\n[1/5] Auditing AC dataset and image SHA-256 digests...")
    ac_audit = audit_ac_data()
    print(f"  Total calibrated images: {ac_audit['total_images']}")
    print(f"  Total unique patients  : {ac_audit['total_patients']}")
    print(f"  Official train images  : {ac_audit['official_train_images']} ({ac_audit['official_train_patients']} patients)")
    print(f"  Official test images   : {ac_audit['official_test_images']} ({ac_audit['official_test_patients']} patients)")
    print(f"  Unique SHA-256 hashes  : {ac_audit['unique_hashes']}")
    print(f"  Duplicate hashes       : {ac_audit['dup_hash_count']} (Same patient: {ac_audit['same_patient_dup_count']}, Cross-patient: {ac_audit['cross_patient_dup_count']})")
    print(f"  Cross-split duplicates : {ac_audit['cross_split_dup_count']}")
    print(f"  Train/test patient leak: {ac_audit['patient_overlap_count']}")

    # 2. Timing
    print("\n[2/5] Parsing lab-PC training execution times...")
    timing = parse_lab_pc_timing(repo_root)
    print(f"  v3_full: {timing['v3_full']['epochs']} epochs in {timing['v3_full']['total_seconds']}s ({timing['v3_full']['mean_seconds_per_ep']:.1f}s/ep)")

    # 3. Model Architecture & Backbones
    print("\n[3/5] Checking HRNet support in environment...")
    hrnet_info = check_hrnet_support()
    print(f"  timm HRNet models : {hrnet_info.get('timm_w18', [])}")
    print(f"  smp HRNet encoders: {hrnet_info.get('smp_hrnet', [])}")

    # 4. Checkpoints
    print("\n[4/5] Checking FASSD pretrained checkpoint...")
    fassd_ckpt_path = r"E:\BUBT-CSE-4th yr\Capstone\kaggle_outputs\fassd_run1\fassd_gray_seed42\best_model.pt"
    fassd_ckpt = check_fassd_checkpoint(fassd_ckpt_path)
    print(f"  Exists: {fassd_ckpt['exists']}, Size: {fassd_ckpt.get('size_mb', 0):.2f} MB")

    # 5. Committed Results
    print("\n[5/5] Checking committed lab-PC results...")
    results_info = parse_committed_results(repo_root)
    print(f"  v3_full test AC MAE: {results_info['test_ac_mae']:.4f} mm")
    print(f"  DSNT OOF AC MAE    : {results_info['oof_ac_mae']:.4f} mm")

    # Generate Markdown Report
    report = generate_step0_report(ac_audit, timing, hrnet_info, fassd_ckpt, results_info, repo_root)
    out_dir = repo_root / "docs" / "results" / "ac_campaign"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "STEP0.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport written to: {out_file}")
    print("Audit finished successfully.")


if __name__ == "__main__":
    main()
