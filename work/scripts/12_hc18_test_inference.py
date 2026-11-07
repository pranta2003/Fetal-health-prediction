"""Official test set inference for HC18 Grand Challenge submission format.

Generates two submission CSV files:
  1. `submission_single_fold1.csv`: Single model (Fold 1) predictions.
  2. `submission_ensemble_5fold.csv`: 5-model ensemble (mean probability map, then ellipse fit).

Columns required by HC18 Grand Challenge:
  filename,center_x_mm,center_y_mm,semi_axes_a_mm,semi_axes_b_mm,angle_rad

NOTE: Submission files are generated locally for verification; DO NOT upload anywhere.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
import pandas as pd
from PIL import Image
import torch

from fetal.eval.hc18_eval import fit_submission_ellipse, predict_native_prob
from fetal.models.hc18_unet import build_segmentation_model


def parse_args():
    parser = argparse.ArgumentParser(description="HC18 Official Test Set Inference")
    parser.add_argument(
        "--test-dir",
        type=str,
        default="C:/fetal-data/HC18/test_set/test_set",
        help="Path to official 335 test set image directory",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="C:/fetal-data/HC18/test_set_pixel_size.csv",
        help="Path to test set pixel size CSV",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="E:/BUBT-CSE-4th yr/Capstone/fm_B",
        help="Directory containing 5-fold checkpoints",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="docs/results",
        help="Directory to save submission CSVs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of test samples to process (for smoke testing)",
    )
    return parser.parse_args()


def load_fold_models(model_dir: str, device: torch.device, n_folds: int = 5) -> list[torch.nn.Module]:
    """Load all 5 trained fold models into evaluation mode."""
    models: list[torch.nn.Module] = []
    for f in range(1, n_folds + 1):
        ckpt_path = Path(model_dir) / f"hc18_seg_fold{f}_eff_b4_384.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Fold {f} checkpoint missing: {ckpt_path}")
        model = build_segmentation_model(
            encoder_name="efficientnet-b4",
            encoder_weights=None,
            classes=1,
        ).to(device)
        model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
        model.eval()
        models.append(model)
        print(f"  Loaded checkpoint: {ckpt_path.name}")
    return models


def run_test_inference(
    test_dir: str,
    test_csv: str,
    model_dir: str,
    out_dir: str,
    device: str | None = None,
    max_samples: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run inference on test images and write single-model and ensemble submission files."""
    test_dir_path = Path(test_dir)
    test_csv_path = Path(test_csv)

    if not test_dir_path.exists() or not test_csv_path.exists():
        print(
            f"ERROR: HC18 official test set or pixel size CSV missing!\n"
            f"  Test images path: {test_dir_path} (exists={test_dir_path.exists()})\n"
            f"  Test CSV path:    {test_csv_path} (exists={test_csv_path.exists()})\n"
            f"STOPPING."
        )
        sys.exit(1)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_test = pd.read_csv(test_csv_path)
    # Detect pixel size column name
    ps_col = [c for c in df_test.columns if "pixel" in c.lower()][0]
    filename_col = [c for c in df_test.columns if "file" in c.lower()][0]

    if max_samples is not None:
        df_test = df_test.iloc[:max_samples]

    dev = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {dev}")
    print(f"Loading 5 fold checkpoints from {model_dir}...")
    models = load_fold_models(model_dir, dev, n_folds=5)

    single_rows: list[dict] = []
    ensemble_rows: list[dict] = []

    print(f"\nProcessing {len(df_test)} test images...")
    for idx, row in df_test.iterrows():
        fname = str(row[filename_col])
        ps = float(row[ps_col])
        img_file = test_dir_path / fname
        if not img_file.exists():
            print(f"Warning: test image {img_file} not found, skipping")
            continue

        with Image.open(img_file) as img:
            img_rgb = img.convert("RGB")

            # 1. Single model (Fold 1)
            prob_fold1 = predict_native_prob(models[0], img_rgb, target_size=384, device=dev)
            single_ellipse = fit_submission_ellipse(prob_fold1, pixel_size=ps, filename=fname)
            single_rows.append(single_ellipse)

            # 2. 5-Model ensemble: average native probability maps across folds
            ensemble_probs = [prob_fold1]
            for m in models[1:]:
                p = predict_native_prob(m, img_rgb, target_size=384, device=dev)
                ensemble_probs.append(p)
            avg_prob = np.mean(ensemble_probs, axis=0)

            ensemble_ellipse = fit_submission_ellipse(avg_prob, pixel_size=ps, filename=fname)
            ensemble_rows.append(ensemble_ellipse)

        if (idx + 1) % 25 == 0 or (idx + 1) == len(df_test):
            print(f"  Processed [{idx + 1}/{len(df_test)}] images")

    df_single = pd.DataFrame(single_rows)
    df_ensemble = pd.DataFrame(ensemble_rows)

    single_csv_out = out_path / "submission_single_fold1.csv"
    ensemble_csv_out = out_path / "submission_ensemble_5fold.csv"

    df_single.to_csv(single_csv_out, index=False)
    df_ensemble.to_csv(ensemble_csv_out, index=False)

    print("\n" + "=" * 60)
    print("===== HC18 TEST SET INFERENCE COMPLETE =====")
    print(f"Single-model submission: {single_csv_out} ({len(df_single)} rows)")
    print(f"5-model ensemble:        {ensemble_csv_out} ({len(df_ensemble)} rows)")
    print("=" * 60)
    print("CONFIRMATION: No submission files were uploaded.")

    return df_single, df_ensemble


def main():
    args = parse_args()
    run_test_inference(
        test_dir=args.test_dir,
        test_csv=args.test_csv,
        model_dir=args.model_dir,
        out_dir=args.out_dir,
        device=args.device,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
