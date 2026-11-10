"""Hadlock Demo Step 1: Count usable subjects and run external validation on FP/UCL head images.

Audits cross-plane usable subjects for 3-plane Hadlock estimation, then applies
the HC18 5-fold ensemble (Run B weights) with hflip TTA to FP and UCL head images.
Compares predictions against annotated BPD landmarks and Ramanujan approximate HC reference.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch

from fetal.eval.ellipse_fitting import hc_bpd_from_ellipse_params
from fetal.eval.hc18_eval import fit_submission_ellipse, predict_native_prob
from fetal.models.hc18_unet import build_segmentation_model


def ramanujan_hc(semi_major_mm: float, semi_minor_mm: float) -> float:
    """Calculate approximate ellipse perimeter using Ramanujan's second formula."""
    a, b = float(semi_major_mm), float(semi_minor_mm)
    return float(np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))))


def audit_usable_subjects(fp_dir: Path, ucl_dir: Path) -> dict:
    """Audit usable subjects across head, abdomen, and femur planes for FP and UCL."""
    fp_h = pd.read_csv(fp_dir / "annotations" / "Head.csv")
    fp_a = pd.read_csv(fp_dir / "annotations" / "Abdomen.csv")
    fp_f = pd.read_csv(fp_dir / "annotations" / "Femur.csv")
    fp_h["SubjectID"] = fp_h["SubjectID"].astype(int)
    fp_a["SubjectID"] = fp_a["SubjectID"].astype(int)
    fp_f["SubjectID"] = fp_f["Patient"].astype(int)

    ucl_h = pd.read_csv(ucl_dir / "annotations" / "px_to_mm" / "Head.csv")
    ucl_a = pd.read_csv(ucl_dir / "annotations" / "px_to_mm" / "Abdomen.csv")
    ucl_f = pd.read_csv(ucl_dir / "annotations" / "px_to_mm" / "Femur.csv")
    for df in [ucl_h, ucl_a, ucl_f]:
        df["SubjectID"] = df["image_name"].str.extract(r"^(\d+)_")[0].astype(int)

    def analyze_cohort(name: str, dh: pd.DataFrame, da: pd.DataFrame, df: pd.DataFrame) -> dict:
        subjs_a_test = set(da[da["Split"] == "Test"]["SubjectID"])
        subjs_f_test = set(df[df["Split"] == "Test"]["SubjectID"])
        subjs_h_test = set(dh[dh["Split"] == "Test"]["SubjectID"])
        subjs_h_all = set(dh["SubjectID"])

        subjs_1a = sorted(list(subjs_a_test & subjs_f_test & subjs_h_all))
        subjs_1b = sorted(list(subjs_a_test & subjs_f_test & subjs_h_test))

        def get_stats(subjs: list[int]) -> dict:
            if not subjs:
                return {}
            h_cnt = dh[dh["SubjectID"].isin(subjs)].groupby("SubjectID")["image_name"].count()
            a_cnt = da[da["SubjectID"].isin(subjs)].groupby("SubjectID")["image_name"].count()
            f_cnt = df[df["SubjectID"].isin(subjs)].groupby("SubjectID")["image_name"].count()
            return {
                "head": {"min": int(h_cnt.min()), "median": float(h_cnt.median()), "max": int(h_cnt.max())},
                "abdomen": {"min": int(a_cnt.min()), "median": float(a_cnt.median()), "max": int(a_cnt.max())},
                "femur": {"min": int(f_cnt.min()), "median": float(f_cnt.median()), "max": int(f_cnt.max())},
            }

        return {
            "cohort": name,
            "subjects_1a_count": len(subjs_1a),
            "subjects_1a": subjs_1a,
            "images_per_plane_1a": get_stats(subjs_1a),
            "subjects_1b_count": len(subjs_1b),
            "subjects_1b": subjs_1b,
            "images_per_plane_1b": get_stats(subjs_1b),
        }

    return {
        "fp": analyze_cohort("FP", fp_h, fp_a, fp_f),
        "ucl_calibrated": analyze_cohort("UCL (calibrated)", ucl_h, ucl_a, ucl_f),
    }


def patient_level_bootstrap_ci(
    df: pd.DataFrame,
    val_col: str,
    subject_col: str = "SubjectID",
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Compute 1000-iteration patient-level bootstrap 95% confidence intervals."""
    df = df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    unique_subjs = np.array(df[subject_col].unique())
    n_subjs = len(unique_subjs)
    if n_subjs <= 1:
        val = float(df[val_col].mean())
        return {"mae": [val, val], "rmse": [val, val], "bias": [val, val], "median": [val, val]}

    subj_indices = {s: df.index[df[subject_col] == s].to_numpy() for s in unique_subjs}

    boot_mae = []
    boot_rmse = []
    boot_bias = []
    boot_median = []

    errs = df[val_col].to_numpy()
    abs_errs = np.abs(errs)

    for _ in range(n_boot):
        sampled_subjs = rng.choice(unique_subjs, size=n_subjs, replace=True)
        idx_list = [subj_indices[s] for s in sampled_subjs]
        all_idx = np.concatenate(idx_list)
        b_errs = errs[all_idx]
        b_abs = abs_errs[all_idx]

        boot_mae.append(float(np.mean(b_abs)))
        boot_rmse.append(float(np.sqrt(np.mean(b_errs**2))))
        boot_bias.append(float(np.mean(b_errs)))
        boot_median.append(float(np.median(b_abs)))

    return {
        "mae": [float(np.percentile(boot_mae, 2.5)), float(np.percentile(boot_mae, 97.5))],
        "rmse": [float(np.percentile(boot_rmse, 2.5)), float(np.percentile(boot_rmse, 97.5))],
        "bias": [float(np.percentile(boot_bias, 2.5)), float(np.percentile(boot_bias, 97.5))],
        "median": [float(np.percentile(boot_median, 2.5)), float(np.percentile(boot_median, 97.5))],
    }


def compute_group_metrics(
    df: pd.DataFrame,
    group_name: str,
    subject_col: str = "SubjectID",
) -> dict:
    """Compute MAE, RMSE, signed bias, median absolute error, and bootstrap CIs."""
    hc_errs = df["hc_err"].to_numpy()
    hc_abs = df["hc_abs_err"].to_numpy()
    bpd_errs = df["bpd_err"].to_numpy()
    bpd_abs = df["bpd_abs_err"].to_numpy()

    hc_ci = patient_level_bootstrap_ci(df, "hc_err", subject_col=subject_col)
    bpd_ci = patient_level_bootstrap_ci(df, "bpd_err", subject_col=subject_col)

    return {
        "group": group_name,
        "n_images": int(len(df)),
        "n_subjects": int(df[subject_col].nunique()),
        "hc": {
            "mae": float(np.mean(hc_abs)),
            "mae_ci_95": hc_ci["mae"],
            "rmse": float(np.sqrt(np.mean(hc_errs**2))),
            "rmse_ci_95": hc_ci["rmse"],
            "bias": float(np.mean(hc_errs)),
            "bias_ci_95": hc_ci["bias"],
            "median": float(np.median(hc_abs)),
            "median_ci_95": hc_ci["median"],
        },
        "bpd": {
            "mae": float(np.mean(bpd_abs)),
            "mae_ci_95": bpd_ci["mae"],
            "rmse": float(np.sqrt(np.mean(bpd_errs**2))),
            "rmse_ci_95": bpd_ci["rmse"],
            "bias": float(np.mean(bpd_errs)),
            "bias_ci_95": bpd_ci["bias"],
            "median": float(np.median(bpd_abs)),
            "median_ci_95": bpd_ci["median"],
        },
    }


def generate_markdown_report(item1: dict, metrics: dict, worst_overall: list[dict]) -> str:
    """Format results into a markdown report with tables."""
    lines = [
        "# External Validation of HC18 5-Fold Ensemble on FP and UCL Head Ultrasound",
        "",
        "## 1. Usable Subjects for Hadlock Demo",
        "",
        "| Cohort | Criterion 1(a): Abd Test & Femur Test & Head (any) | Criterion 1(b): Abd Test & Femur Test & Head Test | Images/Subj 1(a) [Head, Abd, Femur] (min, med, max) | Images/Subj 1(b) [Head, Abd, Femur] (min, med, max) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for c_key, c_data in [("FP", item1["fp"]), ("UCL (calibrated)", item1["ucl_calibrated"])]:
        s1a = c_data["images_per_plane_1a"]
        s1b = c_data["images_per_plane_1b"]
        str_1a = f"H: ({s1a['head']['min']},{s1a['head']['median']},{s1a['head']['max']}) A: ({s1a['abdomen']['min']},{s1a['abdomen']['median']},{s1a['abdomen']['max']}) F: ({s1a['femur']['min']},{s1a['femur']['median']},{s1a['femur']['max']})"
        str_1b = f"H: ({s1b['head']['min']},{s1b['head']['median']},{s1b['head']['max']}) A: ({s1b['abdomen']['min']},{s1b['abdomen']['median']},{s1b['abdomen']['max']}) F: ({s1b['femur']['min']},{s1b['femur']['median']},{s1b['femur']['max']})"
        lines.append(f"| **{c_key}** | **{c_data['subjects_1a_count']}** | **{c_data['subjects_1b_count']}** | {str_1a} | {str_1b} |")

    lines.extend([
        "",
        "## 2. External Validation Metrics (HC & BPD against Reference Annotations)",
        "",
        "> Note: Reference BPD is derived from annotated BPD landmarks × px_to_mm_rate. Approximate HC reference is computed from annotated OFD and BPD axes using the Ramanujan ellipse perimeter formula.",
        "",
        "| Subgroup | N (Images / Subj) | Metric | MAE [95% CI] (mm) | RMSE [95% CI] (mm) | Bias (Signed Mean) [95% CI] (mm) | Median Abs Error [95% CI] (mm) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for grp_name, grp_res in metrics.items():
        n_str = f"{grp_res['n_images']} / {grp_res['n_subjects']}"
        hc = grp_res["hc"]
        hc_mae_ci = f"{hc['mae']:.3f} [{hc['mae_ci_95'][0]:.3f}, {hc['mae_ci_95'][1]:.3f}]"
        hc_rmse_ci = f"{hc['rmse']:.3f} [{hc['rmse_ci_95'][0]:.3f}, {hc['rmse_ci_95'][1]:.3f}]"
        hc_bias_ci = f"{hc['bias']:+.3f} [{hc['bias_ci_95'][0]:+.3f}, {hc['bias_ci_95'][1]:+.3f}]"
        hc_med_ci = f"{hc['median']:.3f} [{hc['median_ci_95'][0]:.3f}, {hc['median_ci_95'][1]:.3f}]"
        lines.append(f"| **{grp_name}** | {n_str} | **HC** | {hc_mae_ci} | {hc_rmse_ci} | {hc_bias_ci} | {hc_med_ci} |")

        bpd = grp_res["bpd"]
        bpd_mae_ci = f"{bpd['mae']:.3f} [{bpd['mae_ci_95'][0]:.3f}, {bpd['mae_ci_95'][1]:.3f}]"
        bpd_rmse_ci = f"{bpd['rmse']:.3f} [{bpd['rmse_ci_95'][0]:.3f}, {bpd['rmse_ci_95'][1]:.3f}]"
        bpd_bias_ci = f"{bpd['bias']:+.3f} [{bpd['bias_ci_95'][0]:+.3f}, {bpd['bias_ci_95'][1]:+.3f}]"
        bpd_med_ci = f"{bpd['median']:.3f} [{bpd['median_ci_95'][0]:.3f}, {bpd['median_ci_95'][1]:.3f}]"
        lines.append(f"| | | **BPD** | {bpd_mae_ci} | {bpd_rmse_ci} | {bpd_bias_ci} | {bpd_med_ci} |")

    lines.extend([
        "",
        "## 3. 10 Worst Images by HC Absolute Error",
        "",
        "| # | Dataset | Image Name | Subject ID | Split | Ref HC (mm) | Pred HC (mm) | HC Abs Error (mm) | Ref BPD (mm) | Pred BPD (mm) | BPD Abs Error (mm) |",
        "| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for rank, item in enumerate(worst_overall, start=1):
        lines.append(
            f"| {rank} | {item['dataset']} | `{item['image_name']}` | {item['SubjectID']} | {item['Split']} | "
            f"{item['ref_hc_mm']:.2f} | {item['pred_hc_mm']:.2f} | **{item['hc_abs_err']:.2f}** | "
            f"{item['ref_bpd_mm']:.2f} | {item['pred_bpd_mm']:.2f} | {item['bpd_abs_err']:.2f} |"
        )

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Hadlock Demo Step 1: Usable subjects & head external validation")
    parser.add_argument("--data-dir", type=Path, default=Path("C:/fetal-data"), help="Root fetal-data path")
    parser.add_argument("--model-dir", type=Path, default=Path("E:/BUBT-CSE-4th yr/Capstone/fm_B"), help="Model checkpoints dir")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/results/hadlock_demo"), help="Output directory")
    parser.add_argument("--predictions-csv", type=Path, default=None, help="Precomputed predictions CSV")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--smoke", action="store_true", help="Run on small subset for testing")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HADLOCK DEMO STEP 1: USABLE SUBJECTS & HEAD EXTERNAL VALIDATION")
    print("=" * 70)

    # 1. Usable subjects audit
    print("\nAuditing usable subjects across 3 planes...")
    item1 = audit_usable_subjects(args.data_dir / "FP", args.data_dir / "UCL")

    # 2. Predictions
    if args.predictions_csv and args.predictions_csv.exists():
        print(f"\nLoading existing predictions from {args.predictions_csv}...")
        all_preds_df = pd.read_csv(args.predictions_csv)
    else:
        # Check if predictions already exist in out_dir
        cached_preds = args.out_dir / "head_predictions.csv"
        if cached_preds.exists():
            print(f"\nLoading cached predictions from {cached_preds}...")
            all_preds_df = pd.read_csv(cached_preds)
        else:
            print(f"\nRunning model inference on device: {args.device}...")
            weight_files = [args.model_dir / f"hc18_seg_fold{fold}_eff_b4_384.pt" for fold in range(1, 6)]
            models = []
            for w in weight_files:
                m = build_segmentation_model(encoder_name="efficientnet-b4", encoder_weights=None, classes=1)
                ckpt = torch.load(w, map_location=args.device, weights_only=False)
                m.load_state_dict(ckpt)
                m.eval()
                m.to(args.device)
                models.append(m)

            def eval_cohort(name: str, csv_path: Path, img_dir: Path, is_ucl: bool = False):
                df = pd.read_csv(csv_path)
                valid_mask = df["px_to_mm_rate"].notna() & (df["px_to_mm_rate"] > 0)
                df_valid = df[valid_mask].copy().reset_index(drop=True)
                if is_ucl:
                    df_valid["SubjectID"] = df_valid["image_name"].str.extract(r"^(\d+)_")[0].astype(int)
                else:
                    df_valid["SubjectID"] = df_valid["SubjectID"].astype(int)

                if args.smoke:
                    df_valid = df_valid.head(5)

                records = []
                for _, row in df_valid.iterrows():
                    img_name = row["image_name"]
                    img_path = img_dir / img_name
                    rate = float(row["px_to_mm_rate"])

                    ofd_px = np.sqrt((row["ofd_1_x"] - row["ofd_2_x"])**2 + (row["ofd_1_y"] - row["ofd_2_y"])**2)
                    bpd_px = np.sqrt((row["bpd_1_x"] - row["bpd_2_x"])**2 + (row["bpd_1_y"] - row["bpd_2_y"])**2)
                    ref_bpd = float(bpd_px * rate)
                    ref_ofd = float(ofd_px * rate)
                    a_ref = max(ref_ofd, ref_bpd) / 2.0
                    b_ref = min(ref_ofd, ref_bpd) / 2.0
                    ref_hc = ramanujan_hc(a_ref, b_ref)

                    img_pil = Image.open(img_path).convert("L")
                    probs = [predict_native_prob(m, img_pil, target_size=384, tta=True, device=args.device) for m in models]
                    mean_prob = np.mean(probs, axis=0)

                    fit_res = fit_submission_ellipse(mean_prob, pixel_size=rate, filename=img_name)
                    pred_hc, pred_bpd = hc_bpd_from_ellipse_params(fit_res["semi_axes_a_mm"], fit_res["semi_axes_b_mm"])

                    records.append({
                        "dataset": name,
                        "image_name": img_name,
                        "SubjectID": int(row["SubjectID"]),
                        "Split": str(row["Split"]),
                        "px_to_mm_rate": rate,
                        "ref_ofd_mm": ref_ofd,
                        "ref_bpd_mm": ref_bpd,
                        "ref_hc_mm": ref_hc,
                        "pred_bpd_mm": float(pred_bpd),
                        "pred_hc_mm": float(pred_hc),
                        "bpd_err": float(pred_bpd - ref_bpd),
                        "bpd_abs_err": float(abs(pred_bpd - ref_bpd)),
                        "hc_err": float(pred_hc - ref_hc),
                        "hc_abs_err": float(abs(pred_hc - ref_hc)),
                    })
                return pd.DataFrame(records)

            fp_df = eval_cohort("FP", args.data_dir / "FP" / "annotations" / "Head.csv", args.data_dir / "FP" / "images" / "Head", is_ucl=False)
            ucl_df = eval_cohort("UCL", args.data_dir / "UCL" / "annotations" / "px_to_mm" / "Head.csv", args.data_dir / "UCL" / "images" / "Head", is_ucl=True)
            all_preds_df = pd.concat([fp_df, ucl_df], ignore_index=True)
            all_preds_df.to_csv(args.out_dir / "head_predictions.csv", index=False)

    fp_df = all_preds_df[all_preds_df["dataset"] == "FP"].copy()
    ucl_df = all_preds_df[all_preds_df["dataset"] == "UCL"].copy()

    # Metrics calculation
    subgroups = [
        ("FP All", fp_df),
        ("FP Train", fp_df[fp_df["Split"] == "Train"]),
        ("FP Test", fp_df[fp_df["Split"] == "Test"]),
        ("UCL All", ucl_df),
        ("UCL Train", ucl_df[ucl_df["Split"] == "Train"]),
        ("UCL Test", ucl_df[ucl_df["Split"] == "Test"]),
    ]

    metrics = {name: compute_group_metrics(df_sub, name) for name, df_sub in subgroups}
    worst_overall = all_preds_df.sort_values("hc_abs_err", ascending=False).head(10).to_dict(orient="records")
    worst_fp = fp_df.sort_values("hc_abs_err", ascending=False).head(10).to_dict(orient="records")
    worst_ucl = ucl_df.sort_values("hc_abs_err", ascending=False).head(10).to_dict(orient="records")

    payload = {
        "item1_usable_subjects": item1,
        "item2_metrics": metrics,
        "worst_10_images_overall": worst_overall,
        "worst_10_images_fp": worst_fp,
        "worst_10_images_ucl": worst_ucl,
    }

    with open(args.out_dir / "hadlock_demo_step1.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md_report = generate_markdown_report(item1, metrics, worst_overall)
    with open(args.out_dir / "head_external_table.md", "w", encoding="utf-8") as f:
        f.write(md_report)

    print("\n" + "=" * 70)
    print("HADLOCK DEMO STEP 1 SUMMARY:")
    print("=" * 70)
    print(md_report)


if __name__ == "__main__":
    main()
