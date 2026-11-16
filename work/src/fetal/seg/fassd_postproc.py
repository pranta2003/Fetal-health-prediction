"""FASSD Post-Processing & Validation Module.

Implements candidate post-processing methods for FASSD multi-label abdominal
segmentation:
  (a) none (baseline threshold 0.5)
  (b) keep largest connected component
  (c) remove components below min area threshold
  (d) per-structure probability threshold tuning in [0.3..0.7]

Performs parameter selection on VAL only, evaluates chosen settings once on TEST,
and computes patient-level paired bootstrap 95% confidence intervals for Dice
and HD95 differences.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd

from src.fetal.seg.fassd_train import (
    STRUCTURE_NAMES,
    STRUCTURE_BITFLAGS,
    compute_hd95,
    compute_structure_metrics,
)


def apply_morphology(bin_mask: np.ndarray, morph_type: str, min_area: int = 0) -> np.ndarray:
    """Apply morphological filtering to binary uint8/bool mask (H, W).

    Args:
        bin_mask: 2D binary array (values 0 and 1).
        morph_type: 'none', 'largest', or 'min_area'.
        min_area: Minimum pixel area threshold for 'min_area' mode.

    Returns:
        Filtered uint8 binary mask of shape (H, W).
    """
    if morph_type == "none" or morph_type is None:
        return bin_mask.astype(np.uint8)

    if bin_mask.sum() == 0:
        return np.zeros_like(bin_mask, dtype=np.uint8)

    mask_u8 = (bin_mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    if num_labels <= 1:
        return np.zeros_like(mask_u8, dtype=np.uint8)

    out = np.zeros_like(mask_u8, dtype=np.uint8)

    if morph_type == "largest":
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + int(np.argmax(areas))
        out[labels == largest_label] = 1
        return out

    if morph_type == "min_area":
        lookup = (stats[:, cv2.CC_STAT_AREA] >= min_area).astype(np.uint8)
        lookup[0] = 0
        return lookup[labels]

    raise ValueError(f"Unknown morph_type '{morph_type}'. Expected 'none', 'largest', or 'min_area'.")


def apply_postprocessing(
    prob: np.ndarray,
    threshold: float = 0.5,
    morph_type: str = "none",
    min_area: int = 0,
) -> np.ndarray:
    """Threshold probability map and apply morphological filtering."""
    bin_mask = (prob > threshold).astype(np.uint8)
    return apply_morphology(bin_mask, morph_type=morph_type, min_area=min_area)


def run_patient_paired_bootstrap(
    patient_diffs: dict[str, float],
    n_bootstraps: int = 5000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute patient-level bootstrap 95% CI for paired differences (postproc - baseline).

    Args:
        patient_diffs: Dict mapping patient_id to mean difference across their images.
        n_bootstraps: Number of resamples.
        seed: Random seed.

    Returns:
        obs_mean, ci_lower, ci_upper
    """
    patients = sorted(list(patient_diffs.keys()))
    if len(patients) == 0:
        return 0.0, 0.0, 0.0

    values = np.array([patient_diffs[p] for p in patients], dtype=np.float64)
    obs_mean = float(np.mean(values))

    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_bootstraps, dtype=np.float64)
    for b in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = np.mean(values[idx])

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return obs_mean, ci_lower, ci_upper


def evaluate_structure_predictions(
    probs_list: Sequence[np.ndarray],
    gts_list: Sequence[np.ndarray],
    threshold: float,
    morph_type: str,
    min_area: int,
    compute_hd: bool = True,
) -> dict[str, Any]:
    """Evaluate a single structure across a list of images under given postproc settings."""
    dices = []
    ious = []
    hd95s = []
    empty_gt_total = 0
    empty_gt_detected = 0

    for prob, gt in zip(probs_list, gts_list):
        p_bin = apply_postprocessing(prob, threshold=threshold, morph_type=morph_type, min_area=min_area)
        g_bin = (gt > 0).astype(np.uint8)

        p_sum = int(p_bin.sum())
        g_sum = int(g_bin.sum())

        if g_sum == 0:
            empty_gt_total += 1
            if p_sum == 0:
                empty_gt_detected += 1
        else:
            intersection = int(np.logical_and(p_bin, g_bin).sum())
            union = int(np.logical_or(p_bin, g_bin).sum())
            dice = float((2.0 * intersection) / (p_sum + g_sum))
            iou = float(intersection / union) if union > 0 else 0.0
            hd = compute_hd95(p_bin, g_bin) if compute_hd else 0.0

            dices.append(dice)
            ious.append(iou)
            if compute_hd:
                hd95s.append(hd)

    empty_rate = float(empty_gt_detected / empty_gt_total) if empty_gt_total > 0 else 1.0
    non_empty_pred_rate = 1.0 - empty_rate if empty_gt_total > 0 else 0.0

    return {
        "n_non_empty": len(dices),
        "n_empty_gt": empty_gt_total,
        "empty_gt_detected": empty_gt_detected,
        "empty_gt_detection_rate": empty_rate,
        "non_empty_pred_rate_on_empty_gt": non_empty_pred_rate,
        "dice_mean": float(np.mean(dices)) if dices else 0.0,
        "dice_std": float(np.std(dices)) if dices else 0.0,
        "iou_mean": float(np.mean(ious)) if ious else 0.0,
        "iou_std": float(np.std(ious)) if ious else 0.0,
        "hd95_mean": float(np.mean(hd95s)) if hd95s else 0.0,
        "hd95_std": float(np.std(hd95s)) if hd95s else 0.0,
    }


def tune_val_grid(
    val_probs_by_struct: dict[str, list[np.ndarray]],
    val_gts_by_struct: dict[str, list[np.ndarray]],
    threshold_grid: Sequence[float] = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70),
    min_area_grid: Sequence[int] = (50, 100, 200, 500, 1000, 2000),
) -> dict[str, dict[str, Any]]:
    """Tune post-processing settings per structure on validation data to maximize Dice."""
    selected_configs = {}

    morph_candidates: list[tuple[str, int, str]] = [("none", 0, "none"), ("largest", 0, "largest")]
    for a in min_area_grid:
        morph_candidates.append(("min_area", a, f"min_area_{a}"))

    for struct in STRUCTURE_NAMES:
        print(f"  Tuning structure: {struct}...", flush=True)
        probs = val_probs_by_struct[struct]
        gts = val_gts_by_struct[struct]

        best_dice = -1.0
        best_cfg: dict[str, Any] = {}
        all_grid_results = []

        for t in threshold_grid:
            for m_type, m_area, m_name in morph_candidates:
                res = evaluate_structure_predictions(
                    probs, gts, threshold=t, morph_type=m_type, min_area=m_area, compute_hd=False
                )
                dice = res["dice_mean"]
                candidate_record = {
                    "structure": struct,
                    "threshold": t,
                    "morph_type": m_type,
                    "min_area": m_area,
                    "morph_name": m_name,
                    "val_dice": dice,
                    "val_iou": res["iou_mean"],
                    "val_hd95": res["hd95_mean"],
                }
                all_grid_results.append(candidate_record)

                if dice > best_dice:
                    best_dice = dice
                    best_cfg = candidate_record

        selected_configs[struct] = {
            "best": best_cfg,
            "grid_results": all_grid_results,
        }

    return selected_configs
