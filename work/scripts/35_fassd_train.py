"""35_fassd_train.py: CLI training script for FASSD multilabel segmentation.

Trains a 4-channel sigmoid U-Net (artery, liver, stomach, vein) on the compact Kaggle
FASSD dataset using BCE + Soft-Dice loss, AdamW, cosine schedule, AMP, and Model EMA.
Metrics are evaluated at native (768, 1024) resolution with surface HD95 and
patient-level paired bootstrap 95% confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

# Add src/ to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
src_dir = REPO_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fetal.seg.fassd_train import train_fassd


def get_git_commit_hash() -> str:
    """Retrieve current git commit hash, or 'unknown' if not in a git repo."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FASSD Multilabel Segmentation Model")

    # Kaggle default vs local default
    kaggle_input = Path("/kaggle/input/fassd-fetal-abdomen-v2")
    local_default = Path(r"E:\BUBT-CSE-4th yr\Capstone\data\kaggle_fassd_v2")
    if kaggle_input.exists():
        default_data_root = str(kaggle_input)
    elif local_default.exists():
        default_data_root = str(local_default)
    else:
        default_data_root = "data/kaggle_fassd_v2"

    parser.add_argument(
        "--data-root",
        type=str,
        default=default_data_root,
        help="Path to compact Kaggle FASSD dataset folder containing images/ and masks/",
    )
    parser.add_argument(
        "--split-csv",
        type=str,
        default=str(REPO_ROOT / "docs/results/fassd/split_v2.csv"),
        help="Path to split_v2.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "docs/results/fassd/train_run"),
        help="Directory to save logs, checkpoints, and summary JSON",
    )
    parser.add_argument(
        "--input",
        type=str,
        choices=["rgb", "gray"],
        default="rgb",
        help="Input mode: 'rgb' or 'gray' (grayscale replicated to 3 channels)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["unet", "unetplusplus", "fpn"],
        default="unet",
        help="Model architecture ('unet', 'unetplusplus', 'fpn')",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="efficientnet-b4",
        help="Encoder architecture name for smp (e.g. 'efficientnet-b4', 'resnet34', 'convnext_tiny')",
    )
    parser.add_argument(
        "--encoder-weights",
        type=str,
        default="imagenet",
        help="Pretrained encoder weights (e.g. 'imagenet' or 'none')",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs (default: 40)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience epochs on val macro Dice (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Initial learning rate for AdamW (default: 3e-4)",
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay for AdamW (default: 1e-2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--cache-in-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache resized 384x512 images and masks in RAM (default: True)",
    )
    parser.add_argument(
        "--batch-gpu-aug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply fast GPU batch augmentations (default: True)",
    )
    parser.add_argument(
        "--channels-last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use channels_last memory format on GPU (default: True)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run quick CPU smoke mode (8 images, resnet18, no pretrained weights, 2 epochs)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on ('cuda' or 'cpu')",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("==================================================================")
    print("FASSD Multilabel Segmentation Training")
    print("==================================================================")
    print(f"Data Root:       {args.data_root}")
    print(f"Split CSV:       {args.split_csv}")
    print(f"Output Dir:      {args.output_dir}")
    print(f"Input Mode:      {args.input}")
    print(f"Arch:            {args.arch}")
    print(f"Encoder:         {args.encoder} (weights={args.encoder_weights})")
    print(f"Epochs:          {args.epochs} (patience={args.patience})")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Learning Rate:   {args.lr}")
    print(f"Weight Decay:    {args.weight_decay}")
    print(f"Seed:            {args.seed}")
    print(f"Cache RAM:       {args.cache_in_memory}")
    print(f"GPU Aug:         {args.batch_gpu_aug}")
    print(f"Channels Last:   {args.channels_last}")
    print(f"Smoke Mode:      {args.smoke}")
    print("==================================================================")

    enc_weights = None if args.encoder_weights.lower() in ("none", "null") or args.smoke else args.encoder_weights

    summary = train_fassd(
        data_root=args.data_root,
        output_dir=args.output_dir,
        split_csv=args.split_csv,
        input_mode=args.input,
        arch=args.arch,
        encoder_name="resnet18" if args.smoke else args.encoder,
        encoder_weights=enc_weights,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        smoke=args.smoke,
        cache_in_memory=args.cache_in_memory,
        batch_gpu_aug=args.batch_gpu_aug,
        channels_last=args.channels_last,
    )

    git_hash = get_git_commit_hash()
    summary["git_commit"] = git_hash

    summary_file = Path(args.output_dir) / "train_summary.json"
    with open(summary_file, "w") as fp:
        json.dump(summary, fp, indent=2)

    print("\nTraining summary updated with git commit hash:", git_hash)
    print("Summary JSON saved to:", summary_file)


if __name__ == "__main__":
    main()
