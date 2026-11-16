"""FASSD Multilabel Segmentation Ensembling Utilities.

Provides probability map averaging, ensemble candidate selection,
and evaluation functions with patient-level bootstrap statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from src.fetal.seg.fassd_postproc import apply_postprocessing, run_patient_paired_bootstrap
from src.fetal.seg.fassd_train import compute_hd95

STRUCTURE_NAMES = ["artery", "liver", "stomach", "vein"]


def average_probabilities(prob_maps: list[np.ndarray]) -> np.ndarray:
    """Compute arithmetic mean of multiple probability maps.

    Args:
        prob_maps: List of numpy arrays, each of shape (C, H, W) or (H, W).

    Returns:
        Averaged probability map as float16 array.
    """
    if not prob_maps:
        raise ValueError("prob_maps list cannot be empty.")
    if len(prob_maps) == 1:
        return prob_maps[0].astype(np.float16)

    total = np.zeros_like(prob_maps[0], dtype=np.float32)
    for p in prob_maps:
        total += p.astype(np.float32)
    total /= len(prob_maps)
    return total.astype(np.float16)


def select_candidate_models(
    val_scores: dict[str, float], candidate_type: str
) -> list[str]:
    """Select model keys based on validation scores and candidate rule.

    Args:
        val_scores: Dict mapping model key to validation macro Dice.
        candidate_type: 'all', 'top3', or 'top2'.

    Returns:
        List of selected model keys.
    """
    sorted_models = sorted(val_scores.keys(), key=lambda k: val_scores[k], reverse=True)
    if candidate_type == "all":
        return sorted_models
    elif candidate_type == "top3":
        return sorted_models[: min(3, len(sorted_models))]
    elif candidate_type == "top2":
        return sorted_models[: min(2, len(sorted_models))]
    else:
        raise ValueError(
            f"Unknown candidate_type '{candidate_type}'. Expected 'all', 'top3', or 'top2'."
        )


def evaluate_ensemble_slice(
    probs_dict: dict[str, np.ndarray],
    gts_dict: dict[str, np.ndarray],
    df_subset: pd.DataFrame,
    threshold: float | dict[str, float] = 0.5,
    morph_type: str | dict[str, str] = "none",
    min_area: int | dict[str, int] = 0,
    compute_hd: bool = True,
) -> dict[str, Any]:
    """Evaluate predictions against ground-truth masks.

    Args:
        probs_dict: Mapping stem -> (4, H, W) float16 probabilities.
        gts_dict: Mapping stem -> (4, H, W) uint8 ground truth masks.
        df_subset: DataFrame containing at least 'filename' and 'patient'.
        threshold: Single float or dict per structure.
        morph_type: Single string ('none', 'largest', 'min_area') or dict per structure.
        min_area: Single int or dict per structure.
        compute_hd: Whether to compute 95% Hausdorff Distance.

    Returns:
        Dict with structure breakdown, macro metrics, and per-patient averages.
    """
    stems = [Path(f).stem for f in df_subset["filename"]]
    patient_map = {Path(row["filename"]).stem: str(row["patient"]) for _, row in df_subset.iterrows()}

    struct_results: dict[str, dict[str, Any]] = {}
    patient_dices_by_struct: dict[str, dict[str, list[float]]] = {s: {} for s in STRUCTURE_NAMES}
    patient_hd95s_by_struct: dict[str, dict[str, list[float]]] = {s: {} for s in STRUCTURE_NAMES}
    patient_macro_dices: dict[str, list[float]] = {}
    patient_macro_hd95s: dict[str, list[float]] = {}

    for c_idx, struct in enumerate(STRUCTURE_NAMES):
        t = threshold[struct] if isinstance(threshold, dict) else threshold
        m_type = morph_type[struct] if isinstance(morph_type, dict) else morph_type
        m_area = min_area[struct] if isinstance(min_area, dict) else min_area

        dices: list[float] = []
        ious: list[float] = []
        hd95s: list[float] = []
        empty_gt_total = 0
        empty_gt_detected = 0

        for stem in stems:
            prob_c = probs_dict[stem][c_idx].astype(np.float32)
            gt_c = gts_dict[stem][c_idx]
            pid = patient_map[stem]

            p_bin = apply_postprocessing(prob_c, threshold=t, morph_type=m_type, min_area=m_area)
            g_bin = (gt_c > 0).astype(np.uint8)

            p_sum = int(p_bin.sum())
            g_sum = int(g_bin.sum())

            if g_sum == 0:
                empty_gt_total += 1
                if p_sum == 0:
                    empty_gt_detected += 1
                continue

            intersection = int(np.logical_and(p_bin, g_bin).sum())
            union = int(np.logical_or(p_bin, g_bin).sum())
            dice = float((2.0 * intersection) / (p_sum + g_sum))
            iou = float(intersection / union) if union > 0 else 0.0

            dices.append(dice)
            ious.append(iou)
            if pid not in patient_dices_by_struct[struct]:
                patient_dices_by_struct[struct][pid] = []
            patient_dices_by_struct[struct][pid].append(dice)

            if compute_hd:
                hd = compute_hd95(p_bin, g_bin)
                hd95s.append(hd)
                if pid not in patient_hd95s_by_struct[struct]:
                    patient_hd95s_by_struct[struct][pid] = []
                patient_hd95s_by_struct[struct][pid].append(hd)

        empty_rate = float(empty_gt_detected / empty_gt_total) if empty_gt_total > 0 else 1.0
        struct_results[struct] = {
            "n_non_empty": len(dices),
            "n_empty_gt": empty_gt_total,
            "empty_gt_detected_count": empty_gt_detected,
            "empty_gt_detection_rate": empty_rate,
            "empty_gt_fp_rate": 1.0 - empty_rate if empty_gt_total > 0 else 0.0,
            "dice_mean": float(np.mean(dices)) if dices else 0.0,
            "dice_std": float(np.std(dices)) if dices else 0.0,
            "iou_mean": float(np.mean(ious)) if ious else 0.0,
            "iou_std": float(np.std(ious)) if ious else 0.0,
            "hd95_mean": float(np.mean(hd95s)) if hd95s else 0.0,
            "hd95_std": float(np.std(hd95s)) if hd95s else 0.0,
        }

    # Per-image macro Dice and HD95
    per_image_macro_dices = []
    per_image_macro_hd95s = []
    for stem in stems:
        pid = patient_map[stem]
        img_dices = []
        img_hds = []
        for c_idx, struct in enumerate(STRUCTURE_NAMES):
            t = threshold[struct] if isinstance(threshold, dict) else threshold
            m_type = morph_type[struct] if isinstance(morph_type, dict) else morph_type
            m_area = min_area[struct] if isinstance(min_area, dict) else min_area

            prob_c = probs_dict[stem][c_idx].astype(np.float32)
            gt_c = gts_dict[stem][c_idx]

            p_bin = apply_postprocessing(prob_c, threshold=t, morph_type=m_type, min_area=m_area)
            g_bin = (gt_c > 0).astype(np.uint8)

            p_sum = int(p_bin.sum())
            g_sum = int(g_bin.sum())
            if g_sum == 0:
                continue

            intersection = int(np.logical_and(p_bin, g_bin).sum())
            dice = float((2.0 * intersection) / (p_sum + g_sum))
            img_dices.append(dice)

            if compute_hd:
                img_hds.append(compute_hd95(p_bin, g_bin))

        if img_dices:
            mean_d = float(np.mean(img_dices))
            per_image_macro_dices.append(mean_d)
            if pid not in patient_macro_dices:
                patient_macro_dices[pid] = []
            patient_macro_dices[pid].append(mean_d)

        if img_hds:
            mean_h = float(np.mean(img_hds))
            per_image_macro_hd95s.append(mean_h)
            if pid not in patient_macro_hd95s:
                patient_macro_hd95s[pid] = []
            patient_macro_hd95s[pid].append(mean_h)

    # Patient-level mean aggregations
    patient_mean_macro_dices = {p: float(np.mean(vals)) for p, vals in patient_macro_dices.items()}
    patient_mean_macro_hd95s = {p: float(np.mean(vals)) for p, vals in patient_macro_hd95s.items()}
    patient_mean_dices_by_struct = {
        s: {p: float(np.mean(vals)) for p, vals in p_dict.items()}
        for s, p_dict in patient_dices_by_struct.items()
    }
    patient_mean_hd95s_by_struct = {
        s: {p: float(np.mean(vals)) for p, vals in p_dict.items()}
        for s, p_dict in patient_hd95s_by_struct.items()
    }

    struct_mean_dice = float(np.mean([struct_results[s]["dice_mean"] for s in STRUCTURE_NAMES]))
    struct_mean_iou = float(np.mean([struct_results[s]["iou_mean"] for s in STRUCTURE_NAMES]))
    struct_mean_hd95 = float(np.mean([struct_results[s]["hd95_mean"] for s in STRUCTURE_NAMES])) if compute_hd else 0.0

    return {
        "by_structure": struct_results,
        "structure_mean_macro_dice": struct_mean_dice,
        "structure_mean_macro_iou": struct_mean_iou,
        "structure_mean_macro_hd95": struct_mean_hd95,
        "per_image_macro_dice_mean": float(np.mean(per_image_macro_dices)) if per_image_macro_dices else 0.0,
        "per_image_macro_hd95_mean": float(np.mean(per_image_macro_hd95s)) if per_image_macro_hd95s else 0.0,
        "patient_macro_dice_mean": float(np.mean(list(patient_mean_macro_dices.values()))) if patient_mean_macro_dices else 0.0,
        "patient_macro_hd95_mean": float(np.mean(list(patient_mean_macro_hd95s.values()))) if patient_mean_macro_hd95s else 0.0,
        "patient_mean_macro_dices": patient_mean_macro_dices,
        "patient_mean_macro_hd95s": patient_mean_macro_hd95s,
        "patient_mean_dices_by_struct": patient_mean_dices_by_struct,
        "patient_mean_hd95s_by_struct": patient_mean_hd95s_by_struct,
    }
