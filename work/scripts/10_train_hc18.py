"""10_train_hc18.py: CLI runner for 5-fold HC18 segmentation model training.

Runs EfficientNet-B4 U-Net training with patient-grouped folds, computes out-of-fold
predictions with TTA, and outputs biometric metrics (MAE, RMSE) and checkpoints.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fetal.train.train_hc18 import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    run_hc18_training,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="10_train_hc18: Train HC18 5-fold segmentation ensemble."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Path to HC18 dataset directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save models and metrics (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        choices=[1, 2, 3, 4, 5],
        help="Train only a single 1-indexed fold (default: run all 5 folds)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip training for folds where checkpoint file already exists",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke test (2 epochs on 16 images on CPU)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs per fold (default: 100)",
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
        default=1e-4,
        help="Learning rate for AdamW (default: 1e-4)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early stopping patience (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for patient split (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on ('cuda' or 'cpu')",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("10_train_hc18: HC18 5-Fold Segmentation Pipeline")
    print(f"Data Dir:   {args.data_dir}")
    print(f"Model Dir:  {args.model_dir}")
    print(f"Fold:       {args.fold if args.fold else 'All (1..5)'}")
    print(f"Smoke:      {args.smoke}")
    print(f"Resume:     {args.resume}")
    print("=" * 60)

    summary = run_hc18_training(
        data_dir=args.data_dir,
        output_dir=args.model_dir,
        target_fold=args.fold,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
        smoke=args.smoke,
    )

    print("\n[10_train_hc18] Complete!")
    return summary


if __name__ == "__main__":
    main()
