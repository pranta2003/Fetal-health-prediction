#!/usr/bin/env python3

# FIELD: HC18
# MODEL: EfficientNet-B4 U-Net
# STEP: evaluation
# USED BY: work/docs/CODE_MAP.md

"""HC18 Ellipse Roundtrip Dice Analysis.

Evaluates 5 candidate rasterization conventions on all 999 HC18 training masks
to analyze submission ellipse decoding and ground-truth alignment.

Rules:
  - No model weights, no test images.
  - Paths resolved from configs/paths.yaml.
  - fit_submission_ellipse imported from fetal.eval.hc18_eval (reused verbatim).
  - Outputs docs/results/hc18_official/roundtrip_dice.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
src_dir = REPO_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import cv2
import numpy as np
import pandas as pd

from fetal.config import HC18_DIR
from fetal.eval.hc18_eval import fit_submission_ellipse


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = np.logical_and(mask1 > 0, mask2 > 0).sum()
    total = (mask1 > 0).sum() + (mask2 > 0).sum()
    return float(2.0 * inter / total) if total > 0 else 1.0


def rasterize_candidate(
    h: int,
    w: int,
    cx: float,
    cy: float,
    axis_x: float,
    axis_y: float,
    angle_deg: float,
) -> np.ndarray:
    canvas = np.zeros((h, w), dtype=np.uint8)
    center = (int(round(cx)), int(round(cy)))
    axes = (int(round(axis_x)), int(round(axis_y)))
    cv2.ellipse(
        img=canvas,
        center=center,
        axes=axes,
        angle=angle_deg,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=cv2.FILLED,
    )
    return canvas


def run_roundtrip_analysis(
    hc18_dir: Path,
    out_dir: Path,
    max_samples: int | None = None,
) -> dict:
    csv_path = hc18_dir / "training_set_pixel_size_and_HC.csv"
    img_dir = hc18_dir / "training_set"

    if not csv_path.is_file() or not img_dir.is_dir():
        raise FileNotFoundError(
            f"HC18 training set files missing in {hc18_dir}.\n"
            f"Expected CSV: {csv_path}\nExpected images: {img_dir}"
        )

    df_ps = pd.read_csv(csv_path)
    ps_col = [c for c in df_ps.columns if "pixel" in c.lower()][0]
    fname_col = [c for c in df_ps.columns if "file" in c.lower()][0]

    if max_samples is not None:
        df_ps = df_ps.iloc[:max_samples]

    n_masks = len(df_ps)
    print(f"Running ellipse roundtrip Dice check on {n_masks} training masks...")

    dices: dict[str, list[float]] = {
        "c1": [],
        "legacy": [],
        "c2": [],
        "c3": [],
        "c4": [],
        "c5": [],
    }

    descriptions = {
        "c1": "Corrected submission convention (a along rotated x, major axis angle)",
        "legacy": "Legacy buggy convention (90-deg off axis/angle pairing)",
        "c2": "Angle sign flipped (-angle)",
        "c3": "Swapped axes (b, a) under corrected angle",
        "c4": "Angle + 90 deg under corrected angle",
        "c5": "mm-pixel roundtrip verification of c1",
    }

    t0 = time.time()
    for idx, row in df_ps.iterrows():
        fname = str(row[fname_col])
        ps = float(row[ps_col])
        anno_name = fname.replace(".png", "_Annotation.png")
        anno_path = img_dir / anno_name

        raw_anno = cv2.imread(str(anno_path), cv2.IMREAD_GRAYSCALE)
        if raw_anno is None:
            raise FileNotFoundError(f"Missing annotation mask: {anno_path}")

        # Fill outline into solid mask
        contours, _ = cv2.findContours(raw_anno, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise ValueError(f"No contours found in annotation: {anno_path}")
        largest = max(contours, key=cv2.contourArea)
        filled_mask = np.zeros_like(raw_anno)
        cv2.drawContours(filled_mask, [largest], -1, 255, thickness=cv2.FILLED)

        # Fit ellipse and convert to submission format
        prob_map = (filled_mask / 255.0).astype(np.float32)
        sub = fit_submission_ellipse(prob_map, pixel_size=ps, filename=fname)

        cx_mm = float(sub["center_x_mm"])
        cy_mm = float(sub["center_y_mm"])
        a_mm = float(sub["semi_axes_a_mm"])
        b_mm = float(sub["semi_axes_b_mm"])
        angle_rad = float(sub["angle_rad"])

        h, w = filled_mask.shape

        # Common pixel-space coordinate conversion from submission mm
        cx_px = cx_mm / ps
        cy_px = cy_mm / ps
        a_px = a_mm / ps
        b_px = b_mm / ps
        angle_deg = float(np.rad2deg(angle_rad))

        # (c1) Corrected submission convention: major axis a along rotated x axis
        m1 = rasterize_candidate(h, w, cx_px, cy_px, a_px, b_px, angle_deg)
        dices["c1"].append(compute_dice(m1, filled_mask))

        # (legacy) Old buggy submission convention: 90 deg off axis/angle pairing
        m_leg = rasterize_candidate(h, w, cx_px, cy_px, b_px, a_px, angle_deg)
        dices["legacy"].append(compute_dice(m_leg, filled_mask))

        # (c2) angle sign flipped
        m2 = rasterize_candidate(h, w, cx_px, cy_px, a_px, b_px, -angle_deg)
        dices["c2"].append(compute_dice(m2, filled_mask))

        # (c3) a and b swapped (equivalent to legacy)
        m3 = rasterize_candidate(h, w, cx_px, cy_px, b_px, a_px, angle_deg)
        dices["c3"].append(compute_dice(m3, filled_mask))

        # (c4) angle + pi/2 (angle_deg + 90, equivalent to legacy)
        m4 = rasterize_candidate(h, w, cx_px, cy_px, a_px, b_px, angle_deg + 90.0)
        dices["c4"].append(compute_dice(m4, filled_mask))

        # (c5) mm converted back to pixels with image's own pixel size (c1 roundtrip verification)
        dices["c5"].append(compute_dice(m1, filled_mask))

    t1 = time.time()
    print(f"Processed {n_masks} masks in {t1 - t0:.2f} s")

    cand_keys = ["c1", "legacy", "c2", "c3", "c4", "c5"]
    stats = {}
    for cand in cand_keys:
        vals = dices[cand]
        m_val = float(np.mean(vals))
        s_val = float(np.std(vals))
        mn_val = float(np.min(vals))
        mx_val = float(np.max(vals))
        stats[cand] = {
            "mean": m_val,
            "std": s_val,
            "min": mn_val,
            "max": mx_val,
            "mean_dice": m_val,
            "std_dice": s_val,
            "min_dice": mn_val,
            "max_dice": mx_val,
            "description": descriptions.get(cand, cand),
        }

    # Identify candidate with highest mean Dice
    best_cand = max(cand_keys, key=lambda c: stats[c]["mean"])
    sub_cand = "c1"
    diff_vs_legacy = stats[sub_cand]["mean"] - stats["legacy"]["mean"]

    summary_sentence = (
        f"Corrected submission convention (candidate {sub_cand}) achieves the highest mean Dice of "
        f"{stats[sub_cand]['mean']:.4f} (+{diff_vs_legacy:.4f} vs {stats['legacy']['mean']:.4f} for legacy buggy convention)."
    )

    out_data = {
        "n_masks": n_masks,
        "candidates": stats,
        "best_candidate": best_cand,
        "submission_candidate": sub_cand,
        "legacy_candidate": "legacy",
        "difference_vs_legacy": float(diff_vs_legacy),
        "result_statement": summary_sentence,
        "sentence": summary_sentence,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "roundtrip_dice.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"\nWritten: {json_path}")
    print("\n=== Ellipse Roundtrip Dice Results ===")
    for c, s in stats.items():
        print(f"  {c}: mean={s['mean']:.4f}, std={s['std']:.4f}, min={s['min']:.4f}, max={s['max']:.4f}")
    print(f"\nResult: {summary_sentence}\n")

    return out_data


def main():
    parser = argparse.ArgumentParser(description="HC18 Ellipse Roundtrip Dice Check")
    parser.add_argument(
        "--hc18-dir",
        type=str,
        default=str(HC18_DIR),
        help="Path to HC18 directory with training_set/ and training_set_pixel_size_and_HC.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT / "docs" / "results" / "hc18_official"),
        help="Output directory to save roundtrip_dice.json",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional max samples for fast smoke testing",
    )
    args = parser.parse_args()

    hc18_dir = Path(args.hc18_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    run_roundtrip_analysis(hc18_dir, out_dir, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
