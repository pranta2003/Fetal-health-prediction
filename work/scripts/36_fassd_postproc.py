"""FASSD Post-Processing Study Script.

Runs inference on VAL (254) and TEST (271) images for FASSD Run 1 models (gray and rgb seed 42),
saves float16 predicted probability maps outside the repository, tunes post-processing
hyperparameters on VAL only, applies the chosen settings to TEST once, and computes
patient-level paired bootstrap 95% confidence intervals (5000 resamples, seed 42).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.fetal.seg.fassd_postproc import (
    apply_morphology,
    apply_postprocessing,
    evaluate_structure_predictions,
    run_patient_paired_bootstrap,
    tune_val_grid,
)
from src.fetal.seg.fassd_train import (
    STRUCTURE_NAMES,
    STRUCTURE_BITFLAGS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_fassd_model,
    compute_hd95,
)


def compute_sha256(filepath: Path | str) -> str:
    """Compute sha256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_model_inference_and_cache(
    model: torch.nn.Module,
    df_subset: pd.DataFrame,
    images_dir: Path,
    cache_dir: Path,
    input_mode: str,
    device: torch.device,
    target_size: tuple[int, int] = (384, 512),
    native_size: tuple[int, int] = (768, 1024),
) -> None:
    """Run model inference and save float16 probability maps to cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    th, tw = target_size
    nh, nw = native_size

    for idx, row in df_subset.iterrows():
        fname = str(row["filename"])
        stem = Path(fname).stem
        save_path = cache_dir / f"{stem}.npz"
        if save_path.exists():
            continue

        img_path = images_dir / fname
        if not img_path.exists():
            img_path = images_dir / f"{stem}.png"

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image at {img_path}")

        if input_mode == "gray":
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img_resized = cv2.resize(img_rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
        img_float = img_resized.astype(np.float32) / 255.0
        img_norm = (img_float - IMAGENET_MEAN) / IMAGENET_STD
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits)
            probs_orig = F.interpolate(
                probs, size=(nh, nw), mode="bilinear", align_corners=False
            ).squeeze(0).cpu().numpy().astype(np.float16)

        np.savez_compressed(save_path, prob=probs_orig)


def load_cached_probabilities(
    df_subset: pd.DataFrame,
    cache_dir: Path,
) -> dict[str, np.ndarray]:
    """Load cached float16 probabilities for images in df_subset."""
    probs_dict = {}
    for _, row in df_subset.iterrows():
        fname = str(row["filename"])
        stem = Path(fname).stem
        p_file = cache_dir / f"{stem}.npz"
        if not p_file.exists():
            raise FileNotFoundError(f"Missing cached probabilities at {p_file}")
        with np.load(p_file) as data:
            probs_dict[fname] = data["prob"]
    return probs_dict


def load_ground_truth_masks(
    df_subset: pd.DataFrame,
    masks_dir: Path,
) -> dict[str, np.ndarray]:
    """Load raw ground truth masks unpacked into (4, H, W) binary uint8 arrays."""
    gt_dict = {}
    for _, row in df_subset.iterrows():
        fname = str(row["filename"])
        stem = Path(fname).stem
        m_file = masks_dir / f"{stem}.png"
        if not m_file.exists():
            m_file = masks_dir / fname
        raw_mask = cv2.imread(str(m_file), cv2.IMREAD_UNCHANGED)
        if raw_mask is None:
            raise FileNotFoundError(f"Missing mask at {m_file}")

        gt_channels = []
        for name in STRUCTURE_NAMES:
            flag = STRUCTURE_BITFLAGS[name]
            gt_channels.append(((raw_mask & flag) > 0).astype(np.uint8))
        gt_dict[fname] = np.stack(gt_channels, axis=0)  # (4, H, W)
    return gt_dict


def evaluate_dataset_full(
    df_subset: pd.DataFrame,
    probs_dict: dict[str, np.ndarray],
    gts_dict: dict[str, np.ndarray],
    config_per_struct: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate dataset using per-structure postprocessing configurations."""
    patients = sorted(list(df_subset["patient"].unique()))

    per_image_struct_metrics: list[dict[str, Any]] = []
    patient_struct_dices: dict[str, dict[str, list[float]]] = {p: {s: [] for s in STRUCTURE_NAMES} for p in patients}
    patient_struct_hd95s: dict[str, dict[str, list[float]]] = {p: {s: [] for s in STRUCTURE_NAMES} for p in patients}
    patient_macro_dices: dict[str, list[float]] = {p: [] for p in patients}
    patient_macro_hd95s: dict[str, list[float]] = {p: [] for p in patients}

    for _, row in df_subset.iterrows():
        fname = str(row["filename"])
        pat = str(row["patient"])
        prob_4c = probs_dict[fname]  # (4, H, W) float16
        gt_4c = gts_dict[fname]      # (4, H, W) uint8

        sample_record = {"filename": fname, "patient": pat}
        sample_dices = []
        sample_hd95s = []

        for idx, s in enumerate(STRUCTURE_NAMES):
            cfg = config_per_struct[s]
            t = cfg.get("threshold", 0.50)
            m_type = cfg.get("morph_type", "none")
            m_area = cfg.get("min_area", 0)

            p_chan = prob_4c[idx].astype(np.float32)
            g_chan = gt_4c[idx]

            p_bin = apply_postprocessing(p_chan, threshold=t, morph_type=m_type, min_area=m_area)
            g_sum = int(g_chan.sum())
            p_sum = int(p_bin.sum())

            if g_sum == 0:
                is_detected = (p_sum == 0)
                sample_record[s] = {
                    "gt_empty": True,
                    "pred_empty": is_detected,
                    "empty_detected": is_detected,
                    "dice": None,
                    "iou": None,
                    "hd95": None,
                }
            else:
                inter = int(np.logical_and(p_bin, g_chan).sum())
                union = int(np.logical_or(p_bin, g_chan).sum())
                dice = float((2.0 * inter) / (p_sum + g_sum))
                iou = float(inter / union) if union > 0 else 0.0
                hd = compute_hd95(p_bin, g_chan)

                sample_record[s] = {
                    "gt_empty": False,
                    "pred_empty": (p_sum == 0),
                    "empty_detected": None,
                    "dice": dice,
                    "iou": iou,
                    "hd95": hd,
                }
                sample_dices.append(dice)
                sample_hd95s.append(hd)
                patient_struct_dices[pat][s].append(dice)
                patient_struct_hd95s[pat][s].append(hd)

        if sample_dices:
            mean_d = float(np.mean(sample_dices))
            mean_h = float(np.mean(sample_hd95s))
            patient_macro_dices[pat].append(mean_d)
            patient_macro_hd95s[pat].append(mean_h)
            sample_record["per_image_macro_dice"] = mean_d
            sample_record["per_image_macro_hd95"] = mean_h
        else:
            sample_record["per_image_macro_dice"] = None
            sample_record["per_image_macro_hd95"] = None

        per_image_struct_metrics.append(sample_record)

    # Aggregate by structure
    by_structure = {}
    macro_dice_means = []
    macro_iou_means = []
    macro_hd95_means = []

    for s in STRUCTURE_NAMES:
        dices = [m[s]["dice"] for m in per_image_struct_metrics if m[s]["dice"] is not None]
        ious = [m[s]["iou"] for m in per_image_struct_metrics if m[s]["iou"] is not None]
        hd95s = [m[s]["hd95"] for m in per_image_struct_metrics if m[s]["hd95"] is not None]

        n_empty_gt = sum(1 for m in per_image_struct_metrics if m[s]["gt_empty"])
        empty_detected = sum(1 for m in per_image_struct_metrics if m[s]["empty_detected"] is True)
        empty_det_rate = float(empty_detected / n_empty_gt) if n_empty_gt > 0 else 1.0
        non_empty_rate = 1.0 - empty_det_rate if n_empty_gt > 0 else 0.0

        d_mean = float(np.mean(dices)) if dices else 0.0
        i_mean = float(np.mean(ious)) if ious else 0.0
        h_mean = float(np.mean(hd95s)) if hd95s else 0.0

        by_structure[s] = {
            "n_non_empty": len(dices),
            "n_empty_gt": n_empty_gt,
            "empty_gt_detected_count": empty_detected,
            "empty_gt_detection_rate": empty_det_rate,
            "non_empty_pred_rate_on_empty_gt": non_empty_rate,
            "dice_mean": d_mean,
            "dice_std": float(np.std(dices)) if dices else 0.0,
            "iou_mean": i_mean,
            "iou_std": float(np.std(ious)) if ious else 0.0,
            "hd95_mean": h_mean,
            "hd95_std": float(np.std(hd95s)) if hd95s else 0.0,
        }
        if dices:
            macro_dice_means.append(d_mean)
            macro_iou_means.append(i_mean)
            macro_hd95_means.append(h_mean)

    per_image_macros = [
        m["per_image_macro_dice"]
        for m in per_image_struct_metrics
        if m["per_image_macro_dice"] is not None
    ]
    mean_of_per_image_macro_dice = float(np.mean(per_image_macros)) if per_image_macros else 0.0

    patient_means_dice = {
        p: float(np.mean(ds)) for p, ds in patient_macro_dices.items() if len(ds) > 0
    }
    patient_means_hd95 = {
        p: float(np.mean(hs)) for p, hs in patient_macro_hd95s.items() if len(hs) > 0
    }

    patient_macro_dice_avg = float(np.mean(list(patient_means_dice.values()))) if patient_means_dice else 0.0
    patient_macro_hd95_avg = float(np.mean(list(patient_means_hd95.values()))) if patient_means_hd95 else 0.0

    # Per-structure patient means
    patient_struct_means = {s: {} for s in STRUCTURE_NAMES}
    for s in STRUCTURE_NAMES:
        for p, ds in patient_struct_dices.items():
            if len(ds[s]) > 0:
                patient_struct_means[s][p] = {
                    "dice": float(np.mean(ds[s])),
                    "hd95": float(np.mean(patient_struct_hd95s[p][s])),
                }

    return {
        "by_structure": by_structure,
        "structure_macro_dice_mean": float(np.mean(macro_dice_means)) if macro_dice_means else 0.0,
        "structure_macro_iou_mean": float(np.mean(macro_iou_means)) if macro_iou_means else 0.0,
        "structure_macro_hd95_mean": float(np.mean(macro_hd95_means)) if macro_hd95_means else 0.0,
        "image_level_macro_dice_mean": mean_of_per_image_macro_dice,
        "patient_level_macro_dice_mean": patient_macro_dice_avg,
        "patient_level_macro_hd95_mean": patient_macro_hd95_avg,
        "patient_means_dice": patient_means_dice,
        "patient_means_hd95": patient_means_hd95,
        "patient_struct_means": patient_struct_means,
        "per_image_metrics": per_image_struct_metrics,
    }


def compute_paired_differences(
    eval_base: dict[str, Any],
    eval_post: dict[str, Any],
    n_bootstraps: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute patient-level paired differences and 95% bootstrap CIs."""
    results = {}

    # Overall macro Dice difference
    p_base_dice = eval_base["patient_means_dice"]
    p_post_dice = eval_post["patient_means_dice"]
    common_patients = sorted(list(set(p_base_dice.keys()) & set(p_post_dice.keys())))

    diffs_macro_dice = {p: p_post_dice[p] - p_base_dice[p] for p in common_patients}
    m_diff, ci_l, ci_u = run_patient_paired_bootstrap(diffs_macro_dice, n_bootstraps=n_bootstraps, seed=seed)
    results["macro_dice"] = {
        "mean_diff": m_diff,
        "ci_lower_95": ci_l,
        "ci_upper_95": ci_u,
        "n_patients": len(common_patients),
    }

    # Overall macro HD95 difference
    p_base_hd = eval_base["patient_means_hd95"]
    p_post_hd = eval_post["patient_means_hd95"]
    diffs_macro_hd = {p: p_post_hd[p] - p_base_hd[p] for p in common_patients}
    m_diff_h, ci_l_h, ci_u_h = run_patient_paired_bootstrap(diffs_macro_hd, n_bootstraps=n_bootstraps, seed=seed)
    results["macro_hd95"] = {
        "mean_diff": m_diff_h,
        "ci_lower_95": ci_l_h,
        "ci_upper_95": ci_u_h,
        "n_patients": len(common_patients),
    }

    # Per-structure differences
    results["by_structure"] = {}
    for s in STRUCTURE_NAMES:
        b_s = eval_base["patient_struct_means"][s]
        p_s = eval_post["patient_struct_means"][s]
        s_pats = sorted(list(set(b_s.keys()) & set(p_s.keys())))

        d_diffs = {p: p_s[p]["dice"] - b_s[p]["dice"] for p in s_pats}
        h_diffs = {p: p_s[p]["hd95"] - b_s[p]["hd95"] for p in s_pats}

        d_m, d_l, d_u = run_patient_paired_bootstrap(d_diffs, n_bootstraps=n_bootstraps, seed=seed)
        h_m, h_l, h_u = run_patient_paired_bootstrap(h_diffs, n_bootstraps=n_bootstraps, seed=seed)

        results["by_structure"][s] = {
            "n_patients": len(s_pats),
            "dice": {"mean_diff": d_m, "ci_lower_95": d_l, "ci_upper_95": d_u},
            "hd95": {"mean_diff": h_m, "ci_lower_95": h_l, "ci_upper_95": h_u},
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="FASSD Post-Processing Study")
    parser.add_argument("--data-root", type=str, default=r"E:\BUBT-CSE-4th yr\Capstone\data\kaggle_fassd_v2")
    parser.add_argument("--split-csv", type=str, default="docs/results/fassd/split_v2.csv")
    parser.add_argument("--run1-outputs", type=str, default=r"E:\BUBT-CSE-4th yr\Capstone\kaggle_outputs\fassd_run1")
    parser.add_argument("--prob-cache-dir", type=str, default=r"E:\BUBT-CSE-4th yr\Capstone\data\fassd_pred_prob")
    parser.add_argument("--output-dir", type=str, default="docs/results/fassd/postproc")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"
    split_csv = Path(args.split_csv)
    run1_dir = Path(args.run1_outputs)
    cache_root = Path(args.prob_cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading split definition from {split_csv}...")
    df_split = pd.read_csv(split_csv)
    df_val = df_split[df_split["split"] == "val"].reset_index(drop=True)
    df_test = df_split[df_split["split"] == "test"].reset_index(drop=True)
    print(f"VAL images: {len(df_val)} (patients: {df_val['patient'].nunique()})")
    print(f"TEST images: {len(df_test)} (patients: {df_test['patient'].nunique()})")

    # Load ground truths
    print("Loading native ground truth masks...")
    val_gts = load_ground_truth_masks(df_val, masks_dir)
    test_gts = load_ground_truth_masks(df_test, masks_dir)

    # Models to evaluate
    models_to_eval = [
        ("gray", run1_dir / "fassd_gray_seed42" / "best_model.pt", "fassd_gray_seed42"),
        ("rgb", run1_dir / "fassd_rgb_seed42" / "best_model.pt", "fassd_rgb_seed42"),
    ]

    all_models_summary: dict[str, Any] = {}
    device = torch.device("cpu")

    for input_mode, ckpt_path, tag in models_to_eval:
        print(f"\n=======================================================")
        print(f"Evaluating Model: {tag} ({input_mode})")
        print(f"Checkpoint: {ckpt_path}")
        print(f"=======================================================")

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        encoder_name = ckpt.get("encoder_name", "efficientnet-b4")
        model = build_fassd_model(encoder_name=encoder_name, in_channels=3, classes=4)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        model_cache_val = cache_root / tag / "val"
        model_cache_test = cache_root / tag / "test"

        # Step 1: Run inference & cache probabilities
        print(f"Checking/generating float16 probabilities for VAL in {model_cache_val}...")
        t0 = time.time()
        run_model_inference_and_cache(model, df_val, images_dir, model_cache_val, input_mode, device)
        print(f"Checking/generating float16 probabilities for TEST in {model_cache_test}...")
        run_model_inference_and_cache(model, df_test, images_dir, model_cache_test, input_mode, device)
        print(f"Inference & caching done in {time.time() - t0:.1f}s.")

        val_probs = load_cached_probabilities(df_val, model_cache_val)
        test_probs = load_cached_probabilities(df_test, model_cache_test)

        # Step 2: Tune post-processing on VAL ONLY
        print("Tuning post-processing hyperparameters on VAL (thresholds & morphology)...")
        val_probs_by_struct = {
            s: [val_probs[row["filename"]][idx].astype(np.float32) for _, row in df_val.iterrows()]
            for idx, s in enumerate(STRUCTURE_NAMES)
        }
        val_gts_by_struct = {
            s: [val_gts[row["filename"]][idx] for _, row in df_val.iterrows()]
            for idx, s in enumerate(STRUCTURE_NAMES)
        }

        tuning_results = tune_val_grid(
            val_probs_by_struct,
            val_gts_by_struct,
            threshold_grid=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
            min_area_grid=[50, 100, 200, 500, 1000, 2000],
        )

        selected_config_per_struct = {}
        for s in STRUCTURE_NAMES:
            best_cfg = tuning_results[s]["best"]
            selected_config_per_struct[s] = best_cfg
            print(f"Structure '{s}': best VAL settings -> Threshold={best_cfg['threshold']}, "
                  f"Morph={best_cfg['morph_name']}, VAL Dice={best_cfg['val_dice']:.4f}")

        # Baseline config: threshold 0.50, morph none
        baseline_config = {
            s: {"threshold": 0.50, "morph_type": "none", "min_area": 0, "morph_name": "none"}
            for s in STRUCTURE_NAMES
        }

        # Step 3: Evaluate VAL before & after
        val_eval_base = evaluate_dataset_full(df_val, val_probs, val_gts, baseline_config)
        val_eval_post = evaluate_dataset_full(df_val, val_probs, val_gts, selected_config_per_struct)
        val_diffs = compute_paired_differences(val_eval_base, val_eval_post, n_bootstraps=5000, seed=42)

        # Step 4: Evaluate TEST ONCE before & after
        print("Evaluating TEST ONCE with selected configurations...")
        test_eval_base = evaluate_dataset_full(df_test, test_probs, test_gts, baseline_config)
        test_eval_post = evaluate_dataset_full(df_test, test_probs, test_gts, selected_config_per_struct)
        test_diffs = compute_paired_differences(test_eval_base, test_eval_post, n_bootstraps=5000, seed=42)

        all_models_summary[tag] = {
            "tag": tag,
            "input_mode": input_mode,
            "encoder_name": encoder_name,
            "checkpoint_path": str(ckpt_path),
            "selected_config": selected_config_per_struct,
            "val": {
                "baseline": val_eval_base,
                "postproc": val_eval_post,
                "paired_diffs_5000boot_ci": val_diffs,
            },
            "test": {
                "baseline": test_eval_base,
                "postproc": test_eval_post,
                "paired_diffs_5000boot_ci": test_diffs,
            },
        }

    # Save JSON summary
    json_path = out_dir / "fassd_postproc_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_models_summary, f, indent=2)
    print(f"\nSaved results JSON to {json_path}")

    # Generate Markdown Report
    md_path = out_dir / "fassd_postproc_report.md"
    generate_markdown_report(all_models_summary, md_path)
    print(f"Saved Markdown report to {md_path}")

    # Update SHA256SUMS
    sha_path = out_dir / "SHA256SUMS"
    with open(sha_path, "w", encoding="utf-8") as f:
        for fpath in (json_path, md_path):
            sha = compute_sha256(fpath)
            f.write(f"{sha}  {fpath.name}\n")
            print(f"SHA256 ({fpath.name}): {sha}")
    print(f"Updated SHA256SUMS at {sha_path}")


def generate_markdown_report(summary: dict[str, Any], md_path: Path) -> None:
    lines = [
        "# FASSD Multi-Label Post-Processing Benchmark & Validation Report",
        "",
        "## 1. Study Overview",
        "",
        "- **Models Evaluated**:",
        "  - `fassd_gray_seed42`: U-Net (`efficientnet-b4`) trained with single-channel grayscale input.",
        "  - `fassd_rgb_seed42`: U-Net (`efficientnet-b4`) trained with standard 3-channel RGB input.",
        "- **Data Splits** (Leakage-Free `split_v2.csv`):",
        "  - **Validation Set**: 254 images across 26 patient groups.",
        "  - **Test Set**: 271 images across 26 patient groups.",
        "- **Resolution**: Inference performed at native $(768 \\times 1024)$ resolution.",
        "- **Empty Ground Truth Protocol**:",
        "  - For each anatomical structure, images with 0 annotated pixels (`gt_empty == True`) are evaluated separately.",
        "  - Detection status is scored as `empty_detected = (pred_pixels == 0)`. The non-empty prediction rate (`1.0 - detection_rate`) measures false positive blobs on empty ground truth.",
        "  - Empty images are excluded from standard Dice, IoU, and HD95 averages to prevent skewing and ensure clinical validity.",
        "- **Macro Dice Metric Clarification**:",
        "  - **Per-Image Macro Dice**: Average Dice across non-empty structures present on each individual image.",
        "  - **Image-Level Mean Macro Dice**: Mean of per-image macro Dice values across all images in the split.",
        "  - **Structure-Level Macro Dice**: Unweighted arithmetic mean of the 4 individual structure Dice averages.",
        "  - **Patient-Level Mean Macro Dice**: Grouped mean of per-image macro Dice values by patient, used for bootstrap resampling.",
        "- **Post-Processing Protocol**:",
        "  - Hyperparameters tuned **strictly on VAL**: threshold $T \\in [0.30..0.70]$ and morphology (none, largest component, min area filter).",
        "  - Selected per-structure settings applied **once** to TEST.",
        "  - Statistical significance assessed via **patient-level paired bootstrap** (5,000 resamples, seed 42) for $\\Delta = \\text{postproc} - \\text{baseline}$.",
        "",
    ]

    for tag, mdata in summary.items():
        lines.append(f"## 2. Model: `{tag}`")
        lines.append("")
        lines.append("### 2.1 Selected Validation Configurations (Tuned on VAL Only)")
        lines.append("")
        lines.append("| Structure | Selected Threshold | Morphology Filter | Min Area (px) | VAL Dice (Tuned) |")
        lines.append("|---|---|---|---|---|")
        for s in STRUCTURE_NAMES:
            cfg = mdata["selected_config"][s]
            lines.append(f"| **{s.capitalize()}** | {cfg['threshold']:.2f} | `{cfg['morph_name']}` | {cfg['min_area']} | {cfg['val_dice']:.4f} |")
        lines.append("")

        for split in ("val", "test"):
            sdata = mdata[split]
            b_eval = sdata["baseline"]
            p_eval = sdata["postproc"]
            diffs = sdata["paired_diffs_5000boot_ci"]

            split_title = "Validation Set (254 images)" if split == "val" else "Test Set (271 images - Evaluated Once)"
            lines.append(f"### 2.2 {split_title} Results")
            lines.append("")
            lines.append("#### Structure-by-Structure Performance")
            lines.append("")
            lines.append("| Structure | Baseline Dice | PostProc Dice | Paired $\\Delta$ Dice (95% CI) | Baseline HD95 (px) | PostProc HD95 (px) | Paired $\\Delta$ HD95 (95% CI) |")
            lines.append("|---|---|---|---|---|---|---|")

            for s in STRUCTURE_NAMES:
                b_s = b_eval["by_structure"][s]
                p_s = p_eval["by_structure"][s]
                s_diff = diffs["by_structure"][s]

                d_diff_str = f"{s_diff['dice']['mean_diff']:+.4f} [{s_diff['dice']['ci_lower_95']:+.4f}, {s_diff['dice']['ci_upper_95']:+.4f}]"
                h_diff_str = f"{s_diff['hd95']['mean_diff']:+.2f} [{s_diff['hd95']['ci_lower_95']:+.2f}, {s_diff['hd95']['ci_upper_95']:+.2f}]"

                lines.append(
                    f"| **{s.capitalize()}** | {b_s['dice_mean']:.4f} ± {b_s['dice_std']:.4f} | "
                    f"{p_s['dice_mean']:.4f} ± {p_s['dice_std']:.4f} | {d_diff_str} | "
                    f"{b_s['hd95_mean']:.2f} ± {b_s['hd95_std']:.2f} | "
                    f"{p_s['hd95_mean']:.2f} ± {p_s['hd95_std']:.2f} | {h_diff_str} |"
                )

            lines.append("")
            lines.append("#### Empty Ground Truth & False Positive Rates")
            lines.append("")
            lines.append("| Structure | Total Empty GT Images | Baseline Detected Empty | Baseline FP Rate | PostProc Detected Empty | PostProc FP Rate |")
            lines.append("|---|---|---|---|---|---|")
            for s in STRUCTURE_NAMES:
                b_s = b_eval["by_structure"][s]
                p_s = p_eval["by_structure"][s]
                lines.append(
                    f"| **{s.capitalize()}** | {b_s['n_empty_gt']} | "
                    f"{b_s['empty_gt_detected_count']} / {b_s['n_empty_gt']} ({b_s['empty_gt_detection_rate']*100:.1f}%) | {b_s['non_empty_pred_rate_on_empty_gt']*100:.1f}% | "
                    f"{p_s['empty_gt_detected_count']} / {p_s['n_empty_gt']} ({p_s['empty_gt_detection_rate']*100:.1f}%) | {p_s['non_empty_pred_rate_on_empty_gt']*100:.1f}% |"
                )

            lines.append("")
            lines.append("#### Overall Summary & Patient-Level Bootstrap")
            lines.append("")
            m_d_diff = diffs["macro_dice"]
            m_h_diff = diffs["macro_hd95"]
            lines.append(f"- **Structure-Mean Macro Dice**: Baseline = {b_eval['structure_macro_dice_mean']:.4f} -> PostProc = {p_eval['structure_macro_dice_mean']:.4f}")
            lines.append(f"- **Per-Image Macro Dice Mean**: Baseline = {b_eval['image_level_macro_dice_mean']:.4f} -> PostProc = {p_eval['image_level_macro_dice_mean']:.4f}")
            lines.append(f"- **Patient-Level Macro Dice Mean**: Baseline = {b_eval['patient_level_macro_dice_mean']:.4f} -> PostProc = {p_eval['patient_level_macro_dice_mean']:.4f}")
            lines.append(f"- **Patient-Level Paired $\\Delta$ Dice (5000 resamples, 95% CI)**: `{m_d_diff['mean_diff']:+.4f} [{m_d_diff['ci_lower_95']:+.4f}, {m_d_diff['ci_upper_95']:+.4f}]`")
            lines.append(f"- **Patient-Level Macro HD95 Mean**: Baseline = {b_eval['patient_level_macro_hd95_mean']:.2f} px -> PostProc = {p_eval['patient_level_macro_hd95_mean']:.2f} px")
            lines.append(f"- **Patient-Level Paired $\\Delta$ HD95 (5000 resamples, 95% CI)**: `{m_h_diff['mean_diff']:+.2f} px [{m_h_diff['ci_lower_95']:+.2f}, {m_h_diff['ci_upper_95']:+.2f}]`")
            lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
