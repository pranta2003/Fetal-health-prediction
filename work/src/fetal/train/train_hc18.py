"""HC18 5-fold training pipeline with patient-grouped splits.

Trains EfficientNet-B4 U-Net models across patient-grouped cross-validation folds,
evaluates out-of-fold predictions with test-time augmentation (TTA), and saves
checkpoints and biometric metrics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fetal.data.hc18_dataset import HC18SegDataset
from fetal.data.hc18_split import make_grouped_folds, split_patient_inner_holdout
from fetal.eval.hc18_eval import (
    compute_metrics,
    evaluate_solo_with_tta,
    measure_from_prob,
    predict_prob_with_tta,
)
from fetal.models.hc18_unet import HC18Loss, build_segmentation_model

DEFAULT_DATA_DIR = "C:/fetal-data/HC18"
DEFAULT_OUTPUT_DIR = "models/hc18"
N_FOLDS = 5
IMG_SIZE = 384
BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 0.01
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 4
MAX_EPOCHS = 100
PATIENCE = 8
SEED = 42
ENCODER_NAME = "efficientnet-b4"
ENCODER_WEIGHTS = "imagenet"


def train_one_fold(
    train_loader: DataLoader,
    val_loader: DataLoader,
    fold_num: int,
    max_epochs: int = MAX_EPOCHS,
    lr: float = LR,
    patience: int = PATIENCE,
    device: torch.device | str | None = None,
    model: torch.nn.Module | None = None,
    loss_fn: torch.nn.Module | None = None,
) -> torch.nn.Module:
    """Train segmentation model on one fold with ReduceLROnPlateau and early stopping.

    Args:
        train_loader: DataLoader for training subset.
        val_loader: DataLoader for validation subset.
        fold_num: Fold index identifier (1-indexed for logging).
        max_epochs: Maximum training epochs.
        lr: Learning rate for AdamW.
        patience: Epochs without val_loss improvement before early stopping.
        device: Device to train on.
        model: Optional pre-constructed model; if None, creates default U-Net.
        loss_fn: Optional loss module; if None, creates HC18Loss.

    Returns:
        Trained model with weights loaded from best validation epoch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if model is None:
        model = build_segmentation_model(encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )
    if loss_fn is None:
        loss_fn = HC18Loss(mode="binary")

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state: dict | None = None
    stopped_epoch = max_epochs

    for epoch in range(max_epochs):
        epoch_start = time.time()

        model.train()
        train_loss_sum = 0.0
        for images, masks, _, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                preds = model(images)
                loss = loss_fn(preds, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item()
        train_loss_avg = train_loss_sum / len(train_loader) if len(train_loader) > 0 else 0.0

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks, _, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    preds = model(images)
                    val_loss += loss_fn(preds, masks).item()
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        print(
            f"  Fold {fold_num} | Epoch {epoch+1:3d}/{max_epochs} | "
            f"train_loss={train_loss_avg:.4f} val_loss={val_loss:.4f} | "
            f"{epoch_time:.1f}s | no_improve={epochs_no_improve}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            stopped_epoch = epoch + 1
            print(f"  Early stopping fold {fold_num} at epoch {stopped_epoch}")
            break

    print(f"Fold {fold_num}: best val_loss = {best_val_loss:.4f} (stopped at epoch {stopped_epoch})")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_hc18_training(
    data_dir: str = DEFAULT_DATA_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    n_folds: int = N_FOLDS,
    target_fold: int | None = None,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    lr: float = LR,
    patience: int = PATIENCE,
    seed: int = SEED,
    num_workers: int = 2,
    device: str | None = None,
    resume: bool = False,
    smoke: bool = False,
    inner_holdout: float = 0.0,
) -> dict[str, float]:
    """Execute patient-grouped cross-validation training on HC18 dataset.

    Args:
        data_dir: Directory containing HC18 images and CSV.
        output_dir: Directory to save fold checkpoints and metrics.
        n_folds: Number of patient-grouped folds (default: 5).
        target_fold: If specified, run only this 1-indexed fold.
        img_size: Input resolution (default: 384; 128 in smoke).
        batch_size: Mini-batch size.
        max_epochs: Maximum epochs per fold.
        lr: Initial learning rate.
        patience: Early stopping patience.
        seed: Random seed for patient split.
        num_workers: DataLoader worker count.
        device: Computation device ('cuda' or 'cpu').
        resume: If True, reuse existing fold checkpoints when found.
        smoke: If True or HC18_SMOKE=1, run 2 epochs on 16 images on CPU.
        inner_holdout: Fraction of train patients to hold out for early stopping (default: 0.0).

    Returns:
        Summary metrics dictionary.
    """
    if smoke or os.environ.get("HC18_SMOKE") == "1":
        smoke = True
        img_size = 128
        batch_size = 4
        max_epochs = 2
        patience = 2
        device = "cpu"
        num_workers = 0
        encoder_weights = None
    else:
        encoder_weights = ENCODER_WEIGHTS

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = os.path.join(data_dir, "training_set_pixel_size_and_HC.csv")
    image_dir = os.path.join(data_dir, "training_set")

    val_dataset = HC18SegDataset(
        csv_path=csv_path, image_dir=image_dir, img_size=img_size, augment=False
    )
    train_dataset = HC18SegDataset(
        csv_path=csv_path, image_dir=image_dir, img_size=img_size, augment=True
    )

    if smoke:
        n_samples = min(16, len(val_dataset))
        val_dataset.data = val_dataset.data.iloc[:n_samples].reset_index(drop=True)
        train_dataset.data = train_dataset.data.iloc[:n_samples].reset_index(drop=True)

    names = val_dataset.names
    folds = make_grouped_folds(names, n_splits=n_folds, seed=seed)

    all_hc_errors: list[float] = []
    all_bpd_errors: list[float] = []
    per_fold_metrics: dict[str, dict[str, float]] = {}

    folds_to_run = (
        [target_fold - 1]
        if (target_fold is not None and 1 <= target_fold <= n_folds)
        else list(range(n_folds))
    )

    for fold_idx in folds_to_run:
        fold_num = fold_idx + 1
        print(f"\n===== HC18 Fold {fold_num}/{n_folds} =====")
        train_idx, val_idx = folds[fold_idx]

        ckpt_file = out_path / f"hc18_seg_fold{fold_num}_eff_b4_{img_size}.pt"
        pred_file = out_path / f"fold{fold_num}_predictions.json"
        metric_file = out_path / f"fold{fold_num}_metrics.json"

        model = build_segmentation_model(
            encoder_name=ENCODER_NAME,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        ).to(dev)

        if resume and ckpt_file.exists():
            print(f"Loading existing checkpoint for fold {fold_num} from {ckpt_file}")
            model.load_state_dict(torch.load(str(ckpt_file), map_location=dev))
        else:
            if inner_holdout > 0.0:
                inner_train_idx, inner_holdout_idx = split_patient_inner_holdout(
                    train_indices=train_idx,
                    names=names,
                    holdout_fraction=inner_holdout,
                    seed=seed + fold_num,
                )
                print(
                    f"Fold {fold_num}: inner holdout enabled ({inner_holdout * 100:.1f}%) -> "
                    f"train images={len(inner_train_idx)}, inner val images={len(inner_holdout_idx)}, "
                    f"outer val images={len(val_idx)}"
                )
                train_subset = Subset(train_dataset, inner_train_idx)
                val_subset = Subset(val_dataset, inner_holdout_idx)
            else:
                train_subset = Subset(train_dataset, train_idx)
                val_subset = Subset(val_dataset, val_idx)

            train_loader = DataLoader(
                train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
            )
            val_loader = DataLoader(
                val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )

            model = train_one_fold(
                train_loader=train_loader,
                val_loader=val_loader,
                fold_num=fold_num,
                max_epochs=max_epochs,
                lr=lr,
                patience=patience,
                device=dev,
                model=model,
            )
            torch.save(model.state_dict(), str(ckpt_file))
            print(f"Saved fold {fold_num} checkpoint to {ckpt_file}")

        # Solo fold TTA evaluation on outer held-out fold
        hc_err, bpd_err = evaluate_solo_with_tta(model, val_dataset, val_idx, device=dev)
        fold_metrics = compute_metrics(hc_err, bpd_err)
        per_fold_metrics[f"fold_{fold_num}"] = fold_metrics
        with open(metric_file, "w", encoding="utf-8") as f:
            json.dump(fold_metrics, f, indent=2)

        # Save predictions per sample
        fold_preds = []
        for i, idx in enumerate(val_idx):
            if i < len(hc_err):
                fold_preds.append({
                    "index": int(idx),
                    "filename": str(val_dataset.names[idx]),
                    "hc_error": float(hc_err[i]),
                    "bpd_error": float(bpd_err[i]),
                })
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump(fold_preds, f, indent=2)

        print(
            f"Fold {fold_num} solo (TTA) -> HC MAE: {fold_metrics['hc_mae']:.2f}mm, "
            f"BPD MAE: {fold_metrics['bpd_mae']:.2f}mm ({len(hc_err)} evaluated)"
        )

        all_hc_errors.extend(hc_err)
        all_bpd_errors.extend(bpd_err)

    pooled_metrics = compute_metrics(all_hc_errors, all_bpd_errors)
    pooled_metrics["per_fold"] = per_fold_metrics  # type: ignore[assignment]

    summary_file = out_path / "metrics.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(pooled_metrics, f, indent=2)

    print("\n===== HC18 Training Summary =====\n")
    print(f"Pooled HC MAE: {pooled_metrics['hc_mae']:.2f}mm, RMSE: {pooled_metrics['hc_rmse']:.2f}mm")
    print(f"Pooled BPD MAE: {pooled_metrics['bpd_mae']:.2f}mm, RMSE: {pooled_metrics['bpd_rmse']:.2f}mm")
    print(f"Saved summary metrics to {summary_file}")

    return pooled_metrics
