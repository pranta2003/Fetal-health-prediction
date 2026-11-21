import os
import sys
import hashlib
import json
import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fetal import config
from fetal.data.ac_dataset import load_ac_folds, get_fold_data, AbdomenDS
from fetal.models.ac_landmark import build_model, decode
from fetal.eval.ac_geometry import match, lengths_mm, ac_t

def run():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = Path("C:/fetal-data")
    ckpt_base = Path(r"E:\BUBT-CSE-4th yr\Capstone\kaggle_outputs\ac_baseline_s42\ac-baseline-s42")
    folds_csv = repo_root / "docs/results/ac_campaign/folds_v1.csv"
    out_worst8_dir = repo_root / "docs/results/ac_campaign/worst8"
    out_worst8_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load pooled OOF predictions
    oof_csv = ckpt_base / "oof_predictions_pooled.csv"
    oof_df = pd.read_csv(oof_csv)
    folds_df = load_ac_folds(folds_csv, base=str(data_dir))

    # Merge fold info
    oof_df = oof_df.merge(folds_df[["filename", "fold", "patient", "image_path", "px_to_mm_rate"]], left_on="image_name", right_on="filename", how="left")
    
    # 2. Yellow Caliper Analysis across all 642 images
    print("--- Running Yellow Caliper Analysis on all 642 images ---")
    yellow_results = []
    for idx, row in folds_df.iterrows():
        img_path = row["image_path"]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Yellow detection in HSV: Hue ~20-35, Saturation > 100, Value > 100
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_pixel_count = int(np.count_nonzero(mask))
        
        # Threshold: if > 50 pixels, considered to have yellow caliper/text
        has_yellow = yellow_pixel_count > 50
        yellow_results.append({
            "image_name": row["filename"],
            "yellow_pixels": yellow_pixel_count,
            "has_yellow": has_yellow
        })

    yellow_df = pd.DataFrame(yellow_results)
    merged_yellow = oof_df.merge(yellow_df, on="image_name")
    
    # Quantitative counts and MAE
    with_y = merged_yellow[merged_yellow["has_yellow"]]
    without_y = merged_yellow[~merged_yellow["has_yellow"]]
    
    print(f"Total images: {len(merged_yellow)}")
    print(f"With yellow pixels: N={len(with_y)}, MAE={with_y['ac_err'].mean():.4f} mm, median={with_y['ac_err'].median():.4f} mm")
    print(f"Without yellow pixels: N={len(without_y)}, MAE={without_y['ac_err'].mean():.4f} mm, median={without_y['ac_err'].median():.4f} mm")

    # 3. 8 Worst Images Visualization
    worst8 = oof_df.sort_values("ac_err", ascending=False).head(8)
    print("\n--- 8 Worst OOF Images ---")
    for _, r in worst8.iterrows():
        print(f"{r['image_name']}: fold {r['fold']}, source {r['source']}, ac_err={r['ac_err']:.2f} mm, tad_err={r['tad_err']:.2f} mm, apad_err={r['apad_err']:.2f} mm")

    # Load models per fold
    models = {}
    for fold in worst8["fold"].unique():
        m = build_model().to("cpu")
        ckpt = torch.load(ckpt_base / f"fold_{fold}" / "best.pt", map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt)
        m.eval()
        models[fold] = m

    # Plot each of the 8 worst images
    for _, r in worst8.iterrows():
        img_name = r["image_name"]
        fold = int(r["fold"])
        model = models[fold]
        
        # Build DS item
        single_df = folds_df[folds_df["filename"] == img_name].copy()
        ds = AbdomenDS(single_df, 384, False, False, False)
        x, gt, mmpp, _ = ds[0]
        
        with torch.no_grad():
            lo = model(x.unsqueeze(0))
            pred_pts, _ = decode(lo)
            pred_pts = pred_pts[0] # (4, 2)
            gt_pts = gt # (4, 2)
            pred_matched, _ = match(pred_pts.unsqueeze(0), gt_pts.unsqueeze(0))
            pred_matched = pred_matched[0]

        # Convert back to original image space
        orig_img_bgr = cv2.imread(r["image_path"])
        orig_h, orig_w = orig_img_bgr.shape[:2]
        orig_img_rgb = cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2RGB)

        gt_orig = gt_pts.numpy().copy()
        gt_orig[:, 0] = gt_orig[:, 0] * (orig_w / 384.0)
        gt_orig[:, 1] = gt_orig[:, 1] * (orig_h / 384.0)

        pred_orig = pred_matched.numpy().copy()
        pred_orig[:, 0] = pred_orig[:, 0] * (orig_w / 384.0)
        pred_orig[:, 1] = pred_orig[:, 1] * (orig_h / 384.0)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(orig_img_rgb)
        
        # Ground truth: Green line between 0-1 (TAD) and 2-3 (APAD)
        ax.plot([gt_orig[0, 0], gt_orig[1, 0]], [gt_orig[0, 1], gt_orig[1, 1]], 'g-', lw=2, label="GT TAD")
        ax.plot([gt_orig[2, 0], gt_orig[3, 0]], [gt_orig[2, 1], gt_orig[3, 1]], 'g--', lw=2, label="GT APAD")
        ax.scatter(gt_orig[:, 0], gt_orig[:, 1], c='lime', s=40, zorder=5)

        # Predicted: Red line between 0-1 and 2-3
        ax.plot([pred_orig[0, 0], pred_orig[1, 0]], [pred_orig[0, 1], pred_orig[1, 1]], 'r-', lw=2, label="Pred TAD")
        ax.plot([pred_orig[2, 0], pred_orig[3, 0]], [pred_orig[2, 1], pred_orig[3, 1]], 'r--', lw=2, label="Pred APAD")
        ax.scatter(pred_orig[:, 0], pred_orig[:, 1], c='red', s=40, zorder=5)

        ax.set_title(f"{img_name} ({r['source']}) | Err: {r['ac_err']:.2f}mm (TAD: {r['tad_err']:.2f}, APAD: {r['apad_err']:.2f})", fontsize=11)
        ax.legend(loc="upper right")
        ax.axis("off")
        
        save_path = out_worst8_dir / f"{Path(img_name).stem}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved figure: {save_path}")

    # 4. Generate RESULTS_LEDGER.md
    json_path = repo_root / "docs/results/ac_campaign/ac_baseline_s42.json"
    with open(json_path, "rb") as f:
        sha256_hash = hashlib.sha256(f.read()).hexdigest()
    
    with open(json_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    
    s = bench_data["oof_summary"]
    
    ledger_content = f"""# Experimental Results Ledger

| Date | Run ID / Recipe | Split / Protocol | N (Pts) | AC MAE (mm) | AC RMSE (mm) | AC Median (mm) | AC Bias (mm) | TAD MAE (mm) | APAD MAE (mm) | NME | >10mm (%) | Artifact SHA-256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-19 | `ac-baseline-s42` (v3_full baseline, seed 42) | 5-fold CV (`folds_v1.csv`) | 642 (501) | {s['ac_mae']:.4f} | {s['ac_rmse']:.4f} | {s['ac_median']:.4f} | {s['ac_bias']:+.4f} | {s['tad_mae']:.4f} | {s['apad_mae']:.4f} | {s['nme']:.4f} | {100*s['over10']:.2f}% | `{sha256_hash}` |

### Detailed Notes & Checkpoint Provenance
- **`ac-baseline-s42`**:
  - Kaggle Kernel: `prantakumerpandit390/ac-baseline-s42`
  - Output Archive: `E:\\BUBT-CSE-4th yr\\Capstone\\kaggle_outputs\\ac_baseline_s42`
  - Summary JSON: `docs/results/ac_campaign/ac_baseline_s42.json` (`{sha256_hash}`)
  - Protocol: Patient-grouped 5-fold cross-validation on 642 images across 501 patients, evaluated strictly out-of-fold.
"""
    ledger_path = repo_root / "docs/results/RESULTS_LEDGER.md"
    with open(ledger_path, "w", encoding="utf-8") as f:
        f.write(ledger_content)
    print(f"\nWritten {ledger_path} with SHA-256 `{sha256_hash}`")

if __name__ == "__main__":
    run()
