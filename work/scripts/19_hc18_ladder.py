"""19_hc18_ladder.py: Train HC18 historical approach ladder on grouped fold 1.

Trains rows A (cell 22), B (cell 27), C (cell 25), and D (cell 31 recipe),
evaluates each on the 206 held-out fold-1 images, computes row E from oof_predictions.csv,
and generates the final comparison table and per-row prediction/metric artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fetal.data.hc18_split import make_grouped_folds, split_patient_inner_holdout, patient_id_from_name
from fetal.eval.ellipse_fitting import fit_ellipse_robust, hc_bpd_from_ellipse_params
from fetal.eval.hc18_eval import predict_native_prob, load_raw_filled_mask
from fetal.models.hc18_old_approaches import (
    RowADataset,
    RowBDataset,
    RowCDataset,
    RowDDataset,
    build_model_row_a,
    build_model_row_b,
    build_model_row_c,
    build_model_row_d,
    get_transforms_row_a,
    get_transforms_row_b,
    get_transforms_row_c,
)
import segmentation_models_pytorch as smp


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def compute_model_hash(state_dict: dict) -> str:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Training Row A (Direct Regression - Cell 22)
# ---------------------------------------------------------------------------
def train_and_eval_row_a(
    df: pd.DataFrame,
    image_dir: str,
    inner_tr_idx: Sequence[int],
    inner_val_idx: Sequence[int],
    outer_val_idx: Sequence[int],
    device: torch.device,
    max_epochs: int = 100,
    patience: int = 10,
    smoke: bool = False,
) -> tuple[pd.DataFrame, dict]:
    print("\n" + "=" * 60)
    print("=== ROW A: Direct Regression (Cell 22) ===")
    print("=" * 60)
    start_time = time.time()
    train_tf, val_tf = get_transforms_row_a()

    # Pre-extract target stats on inner_tr_idx
    inner_tr_df = df.iloc[list(inner_tr_idx)].reset_index(drop=True)
    raw_ds_tr = RowADataset(inner_tr_df, image_dir, transform=None)
    all_targets = torch.stack([raw_ds_tr[i][1] for i in range(len(raw_ds_tr))])
    target_mean = all_targets.mean(dim=0).to(device)
    target_std = all_targets.std(dim=0).to(device)
    print(f"Row A target mean: {target_mean.cpu().tolist()}")
    print(f"Row A target std:  {target_std.cpu().tolist()}")

    train_ds = RowADataset(df.iloc[list(inner_tr_idx)], image_dir, transform=train_tf)
    val_ds = RowADataset(df.iloc[list(inner_val_idx)], image_dir, transform=val_tf)

    batch_size = 4 if smoke else 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model_row_a(pretrained=not smoke).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None
    epochs_trained = 0

    actual_max_epochs = 2 if smoke else max_epochs
    for epoch in range(actual_max_epochs):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            targets_norm = (targets - target_mean) / target_std
            optimizer.zero_grad()
            preds_norm = model(images)
            loss = criterion(preds_norm, targets_norm)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                targets_norm = (targets - target_mean) / target_std
                preds_norm = model(images)
                val_loss += criterion(preds_norm, targets_norm).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        epochs_trained = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0 or epochs_no_improve >= patience or epoch + 1 == actual_max_epochs:
            print(f"Epoch {epoch+1:3d}/{actual_max_epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.6f} no_improve={epochs_no_improve}")

        if epochs_no_improve >= patience:
            print(f"Stopping early at epoch {epoch+1} (patience={patience}).")
            break

    model.load_state_dict(best_state)
    model.eval()
    train_time = round(time.time() - start_time, 2)
    m_hash = compute_model_hash(best_state)

    # Evaluate on OUTER VAL (fold 1 held-out)
    outer_df = df.iloc[list(outer_val_idx)].reset_index(drop=True)
    outer_ds = RowADataset(outer_df, image_dir, transform=val_tf)
    outer_loader = DataLoader(outer_ds, batch_size=1, shuffle=False)

    preds_rows: list[dict] = []
    with torch.no_grad():
        for i, (images, targets) in enumerate(outer_loader):
            images = images.to(device)
            preds_norm = model(images)
            preds = preds_norm * target_std + target_mean
            pred_hc = float(preds[0, 0].item())
            pred_bpd = float(preds[0, 1].item())
            true_hc = float(targets[0, 0].item())
            true_bpd = float(targets[0, 1].item())

            row_info = outer_df.iloc[i]
            preds_rows.append({
                "image": row_info["filename"],
                "patient_id": patient_id_from_name(row_info["filename"]),
                "true_HC_mm": round(true_hc, 4),
                "pred_HC_mm": round(pred_hc, 4),
                "abs_err_HC": round(abs(pred_hc - true_hc), 4),
                "true_BPD": round(true_bpd, 4),
                "pred_BPD": round(pred_bpd, 4),
                "abs_err_BPD": round(abs(pred_bpd - true_bpd), 4),
            })

    pred_df = pd.DataFrame(preds_rows)
    metrics = {
        "model_name": "Row A: Direct Regression (Cell 22)",
        "approach": "Direct continuous scalar regression",
        "backbone": "ConvNeXt-Tiny",
        "input_res": "224x224 (direct)",
        "hc_mae": round(float(pred_df["abs_err_HC"].mean()), 4),
        "hc_rmse": round(float(np.sqrt(np.mean(pred_df["abs_err_HC"] ** 2))), 4),
        "bpd_mae": round(float(pred_df["abs_err_BPD"].mean()), 4),
        "bpd_rmse": round(float(np.sqrt(np.mean(pred_df["abs_err_BPD"] ** 2))), 4),
        "epochs_trained": epochs_trained,
        "train_time_sec": train_time,
        "model_hash": m_hash,
    }
    print(f"Row A -> Fold 1 HC MAE: {metrics['hc_mae']} mm, BPD MAE: {metrics['bpd_mae']} mm in {train_time}s")
    return pred_df, metrics


# ---------------------------------------------------------------------------
# Training Row B (Geometry v3 - Cell 27)
# ---------------------------------------------------------------------------
def train_and_eval_row_b(
    df: pd.DataFrame,
    image_dir: str,
    inner_tr_idx: Sequence[int],
    inner_val_idx: Sequence[int],
    outer_val_idx: Sequence[int],
    device: torch.device,
    max_epochs: int = 100,
    patience: int = 10,
    smoke: bool = False,
) -> tuple[pd.DataFrame, dict]:
    print("\n" + "=" * 60)
    print("=== ROW B: Geometry Approach v3 (Cell 27) ===")
    print("=" * 60)
    start_time = time.time()
    train_tf, val_tf = get_transforms_row_b()

    inner_tr_df = df.iloc[list(inner_tr_idx)].reset_index(drop=True)
    raw_ds_tr = RowBDataset(inner_tr_df, image_dir, transform=None)
    all_targets = torch.stack([raw_ds_tr[i][1] for i in range(len(raw_ds_tr))])
    target_mean = all_targets.mean(dim=0).to(device)
    target_std = all_targets.std(dim=0).to(device)
    print(f"Row B target mean: {target_mean.cpu().tolist()}")
    print(f"Row B target std:  {target_std.cpu().tolist()}")

    train_ds = RowBDataset(df.iloc[list(inner_tr_idx)], image_dir, transform=train_tf)
    val_ds = RowBDataset(df.iloc[list(inner_val_idx)], image_dir, transform=val_tf)

    batch_size = 4 if smoke else 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model_row_b(pretrained=not smoke).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None
    epochs_trained = 0

    actual_max_epochs = 2 if smoke else max_epochs
    for epoch in range(actual_max_epochs):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            targets_norm = (targets - target_mean) / target_std
            optimizer.zero_grad()
            preds_norm = model(images)
            loss = criterion(preds_norm, targets_norm)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                targets_norm = (targets - target_mean) / target_std
                preds_norm = model(images)
                val_loss += criterion(preds_norm, targets_norm).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        epochs_trained = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0 or epochs_no_improve >= patience or epoch + 1 == actual_max_epochs:
            print(f"Epoch {epoch+1:3d}/{actual_max_epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.6f} no_improve={epochs_no_improve}")

        if epochs_no_improve >= patience:
            print(f"Stopping early at epoch {epoch+1} (patience={patience}).")
            break

    model.load_state_dict(best_state)
    model.eval()
    train_time = round(time.time() - start_time, 2)
    m_hash = compute_model_hash(best_state)

    # Evaluate on OUTER VAL (fold 1 held-out)
    outer_df = df.iloc[list(outer_val_idx)].reset_index(drop=True)
    outer_ds = RowBDataset(outer_df, image_dir, transform=val_tf)
    outer_loader = DataLoader(outer_ds, batch_size=1, shuffle=False)

    preds_rows: list[dict] = []
    with torch.no_grad():
        for i, (images, targets) in enumerate(outer_loader):
            images = images.to(device)
            preds_norm = model(images)
            preds = preds_norm * target_std + target_mean

            pred_a = float(preds[0, 2].item())
            pred_b = float(preds[0, 3].item())
            pred_hc, pred_bpd = hc_bpd_from_ellipse_params(pred_a, pred_b)

            true_a = float(targets[0, 2].item())
            true_b = float(targets[0, 3].item())
            # Ground truth HC comes straight from CSV
            row_info = outer_df.iloc[i]
            true_hc = float(row_info["head circumference (mm)"])
            _, true_bpd = hc_bpd_from_ellipse_params(true_a, true_b)

            preds_rows.append({
                "image": row_info["filename"],
                "patient_id": patient_id_from_name(row_info["filename"]),
                "true_HC_mm": round(true_hc, 4),
                "pred_HC_mm": round(pred_hc, 4),
                "abs_err_HC": round(abs(pred_hc - true_hc), 4),
                "true_BPD": round(true_bpd, 4),
                "pred_BPD": round(pred_bpd, 4),
                "abs_err_BPD": round(abs(pred_bpd - true_bpd), 4),
            })

    pred_df = pd.DataFrame(preds_rows)
    metrics = {
        "model_name": "Row B: Geometry Approach v3 (Cell 27)",
        "approach": "Ellipse parameter regression (Huber loss)",
        "backbone": "ConvNeXt-Tiny",
        "input_res": "224x224 (letterbox)",
        "hc_mae": round(float(pred_df["abs_err_HC"].mean()), 4),
        "hc_rmse": round(float(np.sqrt(np.mean(pred_df["abs_err_HC"] ** 2))), 4),
        "bpd_mae": round(float(pred_df["abs_err_BPD"].mean()), 4),
        "bpd_rmse": round(float(np.sqrt(np.mean(pred_df["abs_err_BPD"] ** 2))), 4),
        "epochs_trained": epochs_trained,
        "train_time_sec": train_time,
        "model_hash": m_hash,
    }
    print(f"Row B -> Fold 1 HC MAE: {metrics['hc_mae']} mm, BPD MAE: {metrics['bpd_mae']} mm in {train_time}s")
    return pred_df, metrics


# ---------------------------------------------------------------------------
# Training Row C (Multi-Task U-Net - Cell 25)
# ---------------------------------------------------------------------------
def train_and_eval_row_c(
    df: pd.DataFrame,
    image_dir: str,
    inner_tr_idx: Sequence[int],
    inner_val_idx: Sequence[int],
    outer_val_idx: Sequence[int],
    device: torch.device,
    max_epochs: int = 100,
    patience: int = 10,
    smoke: bool = False,
) -> tuple[pd.DataFrame, dict]:
    print("\n" + "=" * 60)
    print("=== ROW C: Multi-Task U-Net (Cell 25) ===")
    print("=" * 60)
    start_time = time.time()
    train_tf, val_tf = get_transforms_row_c()

    inner_tr_df = df.iloc[list(inner_tr_idx)].reset_index(drop=True)
    raw_ds_tr = RowCDataset(inner_tr_df, image_dir, transform=None)
    all_targets = torch.stack([raw_ds_tr[i][2] for i in range(len(raw_ds_tr))])
    target_mean = all_targets.mean(dim=0).to(device)
    target_std = all_targets.std(dim=0).to(device)
    print(f"Row C target mean: {target_mean.cpu().tolist()}")
    print(f"Row C target std:  {target_std.cpu().tolist()}")

    train_ds = RowCDataset(df.iloc[list(inner_tr_idx)], image_dir, transform=train_tf)
    val_ds = RowCDataset(df.iloc[list(inner_val_idx)], image_dir, transform=val_tf)

    batch_size = 4 if smoke else 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model_row_c(pretrained=not smoke).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    dice_loss_fn = smp.losses.DiceLoss(mode="binary")
    bce_loss_fn = nn.BCEWithLogitsLoss()
    geo_loss_fn = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None
    epochs_trained = 0

    actual_max_epochs = 2 if smoke else max_epochs
    for epoch in range(actual_max_epochs):
        model.train()
        train_loss = 0.0
        for images, masks, geo_targets in train_loader:
            images, masks, geo_targets = images.to(device), masks.to(device), geo_targets.to(device)
            targets_norm = (geo_targets - target_mean) / target_std
            optimizer.zero_grad()
            pred_masks, pred_geos_norm = model(images)
            loss_mask = dice_loss_fn(pred_masks, masks) + bce_loss_fn(pred_masks, masks)
            loss_geo = geo_loss_fn(pred_geos_norm, targets_norm)
            loss = loss_mask + loss_geo
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks, geo_targets in val_loader:
                images, masks, geo_targets = images.to(device), masks.to(device), geo_targets.to(device)
                targets_norm = (geo_targets - target_mean) / target_std
                pred_masks, pred_geos_norm = model(images)
                loss_mask = dice_loss_fn(pred_masks, masks) + bce_loss_fn(pred_masks, masks)
                loss_geo = geo_loss_fn(pred_geos_norm, targets_norm)
                val_loss += (loss_mask + loss_geo).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        epochs_trained = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0 or epochs_no_improve >= patience or epoch + 1 == actual_max_epochs:
            print(f"Epoch {epoch+1:3d}/{actual_max_epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.6f} no_improve={epochs_no_improve}")

        if epochs_no_improve >= patience:
            print(f"Stopping early at epoch {epoch+1} (patience={patience}).")
            break

    model.load_state_dict(best_state)
    model.eval()
    train_time = round(time.time() - start_time, 2)
    m_hash = compute_model_hash(best_state)

    # Evaluate on OUTER VAL (fold 1 held-out)
    outer_df = df.iloc[list(outer_val_idx)].reset_index(drop=True)
    outer_ds = RowCDataset(outer_df, image_dir, transform=val_tf)
    outer_loader = DataLoader(outer_ds, batch_size=1, shuffle=False)

    preds_rows: list[dict] = []
    with torch.no_grad():
        for i, (images, masks, geo_targets) in enumerate(outer_loader):
            images = images.to(device)
            _, pred_geos_norm = model(images)
            preds = pred_geos_norm * target_std + target_mean

            pred_a = float(preds[0, 2].item())
            pred_b = float(preds[0, 3].item())
            pred_hc, pred_bpd = hc_bpd_from_ellipse_params(pred_a, pred_b)

            true_a = float(geo_targets[0, 2].item())
            true_b = float(geo_targets[0, 3].item())
            row_info = outer_df.iloc[i]
            true_hc = float(row_info["head circumference (mm)"])
            _, true_bpd = hc_bpd_from_ellipse_params(true_a, true_b)

            preds_rows.append({
                "image": row_info["filename"],
                "patient_id": patient_id_from_name(row_info["filename"]),
                "true_HC_mm": round(true_hc, 4),
                "pred_HC_mm": round(pred_hc, 4),
                "abs_err_HC": round(abs(pred_hc - true_hc), 4),
                "true_BPD": round(true_bpd, 4),
                "pred_BPD": round(pred_bpd, 4),
                "abs_err_BPD": round(abs(pred_bpd - true_bpd), 4),
            })

    pred_df = pd.DataFrame(preds_rows)
    metrics = {
        "model_name": "Row C: Multi-Task U-Net (Cell 25)",
        "approach": "Segmentation + shared-encoder ellipse regression",
        "backbone": "ResNet-34",
        "input_res": "224x224 (direct)",
        "hc_mae": round(float(pred_df["abs_err_HC"].mean()), 4),
        "hc_rmse": round(float(np.sqrt(np.mean(pred_df["abs_err_HC"] ** 2))), 4),
        "bpd_mae": round(float(pred_df["abs_err_BPD"].mean()), 4),
        "bpd_rmse": round(float(np.sqrt(np.mean(pred_df["abs_err_BPD"] ** 2))), 4),
        "epochs_trained": epochs_trained,
        "train_time_sec": train_time,
        "model_hash": m_hash,
    }
    print(f"Row C -> Fold 1 HC MAE: {metrics['hc_mae']} mm, BPD MAE: {metrics['bpd_mae']} mm in {train_time}s")
    return pred_df, metrics


# ---------------------------------------------------------------------------
# Training Row D (ResNet-34 U-Net - Cell 31)
# ---------------------------------------------------------------------------
def train_and_eval_row_d(
    df: pd.DataFrame,
    image_dir: str,
    inner_tr_idx: Sequence[int],
    inner_val_idx: Sequence[int],
    outer_val_idx: Sequence[int],
    device: torch.device,
    max_epochs: int = 100,
    patience: int = 10,
    smoke: bool = False,
) -> tuple[pd.DataFrame, dict]:
    print("\n" + "=" * 60)
    print("=== ROW D: ResNet-34 U-Net (Cell 31 recipe) ===")
    print("=" * 60)
    start_time = time.time()

    train_ds = RowDDataset(df.iloc[list(inner_tr_idx)], image_dir, augment=True, mask_size=224)
    val_ds = RowDDataset(df.iloc[list(inner_val_idx)], image_dir, augment=False, mask_size=224)

    batch_size = 4 if smoke else 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model_row_d(pretrained=not smoke).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    dice_loss_fn = smp.losses.DiceLoss(mode="binary")
    bce_loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None
    epochs_trained = 0

    actual_max_epochs = 2 if smoke else max_epochs
    for epoch in range(actual_max_epochs):
        model.train()
        train_loss = 0.0
        for images, masks, _, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = dice_loss_fn(preds, masks) + bce_loss_fn(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks, _, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                val_loss += (dice_loss_fn(preds, masks) + bce_loss_fn(preds, masks)).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        epochs_trained = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0 or epochs_no_improve >= patience or epoch + 1 == actual_max_epochs:
            print(f"Epoch {epoch+1:3d}/{actual_max_epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.6f} no_improve={epochs_no_improve}")

        if epochs_no_improve >= patience:
            print(f"Stopping early at epoch {epoch+1} (patience={patience}).")
            break

    model.load_state_dict(best_state)
    model.eval()
    train_time = round(time.time() - start_time, 2)
    m_hash = compute_model_hash(best_state)

    # Evaluate on OUTER VAL using repo's robust native-resolution measurement (predict_native_prob at 224)
    outer_df = df.iloc[list(outer_val_idx)].reset_index(drop=True)
    preds_rows: list[dict] = []

    for i in range(len(outer_df)):
        row = outer_df.iloc[i]
        img_path = os.path.join(image_dir, row["filename"])
        image_pil = Image.open(img_path).convert("RGB")
        pixel_size = float(row["pixel size(mm)"])

        native_prob = predict_native_prob(model, image_pil, target_size=224, device=device)
        binary = (native_prob > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 5:
            continue

        (cx, cy), (axis1_px, axis2_px), _ = fit_ellipse_robust(largest)
        major_px = max(axis1_px, axis2_px)
        minor_px = min(axis1_px, axis2_px)
        pred_hc, pred_bpd = hc_bpd_from_ellipse_params((major_px / 2.0) * pixel_size, (minor_px / 2.0) * pixel_size)

        true_hc = float(row["head circumference (mm)"])
        raw_mask = load_raw_filled_mask(row, image_dir)
        true_contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not true_contours:
            continue
        true_largest = max(true_contours, key=cv2.contourArea)
        if len(true_largest) < 5:
            continue
        (_, _), (t_axis1, t_axis2), _ = fit_ellipse_robust(true_largest)
        t_major = max(t_axis1, t_axis2)
        t_minor = min(t_axis1, t_axis2)
        _, true_bpd = hc_bpd_from_ellipse_params((t_major / 2.0) * pixel_size, (t_minor / 2.0) * pixel_size)

        preds_rows.append({
            "image": row["filename"],
            "patient_id": patient_id_from_name(row["filename"]),
            "true_HC_mm": round(true_hc, 4),
            "pred_HC_mm": round(pred_hc, 4),
            "abs_err_HC": round(abs(pred_hc - true_hc), 4),
            "true_BPD": round(true_bpd, 4),
            "pred_BPD": round(pred_bpd, 4),
            "abs_err_BPD": round(abs(pred_bpd - true_bpd), 4),
        })

    pred_df = pd.DataFrame(preds_rows)
    metrics = {
        "model_name": "Row D: ResNet-34 U-Net (Cell 31 recipe)",
        "approach": "Segmentation mask + robust native ellipse fit",
        "backbone": "ResNet-34",
        "input_res": "224x224 (letterbox)",
        "hc_mae": round(float(pred_df["abs_err_HC"].mean()), 4),
        "hc_rmse": round(float(np.sqrt(np.mean(pred_df["abs_err_HC"] ** 2))), 4),
        "bpd_mae": round(float(pred_df["abs_err_BPD"].mean()), 4),
        "bpd_rmse": round(float(np.sqrt(np.mean(pred_df["abs_err_BPD"] ** 2))), 4),
        "epochs_trained": epochs_trained,
        "train_time_sec": train_time,
        "model_hash": m_hash,
    }
    print(f"Row D -> Fold 1 HC MAE: {metrics['hc_mae']} mm, BPD MAE: {metrics['bpd_mae']} mm in {train_time}s")
    return pred_df, metrics


# ---------------------------------------------------------------------------
# Load Row E from oof_predictions.csv
# ---------------------------------------------------------------------------
def load_row_e_metrics(oof_csv_path: str | Path) -> tuple[pd.DataFrame, dict]:
    oof_df = pd.read_csv(oof_csv_path)
    f1_df = oof_df[oof_df["fold"] == 1].copy()
    hc_err = f1_df["abs_err_HC"].to_numpy(dtype=float)
    bpd_err = f1_df["abs_err_BPD"].to_numpy(dtype=float)

    metrics = {
        "model_name": "Row E: EfficientNet-B4 U-Net 384 (Stream 1 Final)",
        "approach": "High-res segmentation + TTA + fitEllipseAMS",
        "backbone": "EfficientNet-B4",
        "input_res": "384x384 (letterbox)",
        "hc_mae": round(float(np.mean(hc_err)), 4),
        "hc_rmse": round(float(np.sqrt(np.mean(hc_err ** 2))), 4),
        "bpd_mae": round(float(np.mean(bpd_err)), 4),
        "bpd_rmse": round(float(np.sqrt(np.mean(bpd_err ** 2))), 4),
        "epochs_trained": 100,
        "train_time_sec": None,
        "model_hash": "repo_checkpoint",
    }
    return f1_df, metrics


def generate_ladder_table(all_metrics: list[dict]) -> str:
    md_lines = [
        "# HC18 Historical Approach Ladder (Grouped Fold 1)",
        "",
        "Evaluation on the exact 206 held-out fold-1 images (`make_grouped_folds`, seed 42) with inner 10% patient-isolation validation.",
        "",
        "| Row | Approach | Model / Backbone | Input Res | Target Type | Fold 1 HC MAE (mm) | Fold 1 HC RMSE (mm) | Fold 1 BPD MAE (mm) | Fold 1 BPD RMSE (mm) | Epochs | Train Time (s) |",
        "|:---:|:---|:---|:---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row_label, m in zip(["A", "B", "C", "D", "E"], all_metrics):
        ttime = f"{m['train_time_sec']:.1f}" if m.get("train_time_sec") is not None else "-"
        md_lines.append(
            f"| {row_label} | {m['approach']} | {m['backbone']} | {m['input_res']} | {m['model_name'].split('(')[0].strip()} | **{m['hc_mae']:.4f}** | {m['hc_rmse']:.4f} | **{m['bpd_mae']:.4f}** | {m['bpd_rmse']:.4f} | {m['epochs_trained']} | {ttime} |"
        )
    md_lines.append("")
    return "\n".join(md_lines)


def main():
    parser = argparse.ArgumentParser(description="Train HC18 old approach ladder on grouped fold 1.")
    parser.add_argument("--data-dir", type=str, default="data/raw/HC-18", help="Path to HC18 dataset dir")
    parser.add_argument("--oof-csv", type=str, default="docs/results/hc18/oof_predictions.csv", help="Path to oof_predictions.csv")
    parser.add_argument("--out-dir", type=str, default="docs/results/hc18_ladder", help="Path to save ladder results")
    parser.add_argument("--device", type=str, default=None, help="Device to train on (cuda/cpu)")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs per model (default: 100)")
    parser.add_argument("--smoke", action="store_true", help="Run tiny smoke test (2 epochs)")
    args = parser.parse_args()

    set_seed(42)
    dev = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {dev}")

    data_dir = Path(args.data_dir).resolve()
    csv_path = data_dir / "training_set_pixel_size_and_HC.csv"
    if not csv_path.exists():
        # Fallback search if nested
        for p in data_dir.rglob("training_set_pixel_size_and_HC.csv"):
            csv_path = p
            break
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find training_set_pixel_size_and_HC.csv in {data_dir}")

    image_dir = csv_path.parent / "training_set"
    if not image_dir.exists():
        image_dir = csv_path.parent
    print(f"HC18 CSV: {csv_path}")
    print(f"HC18 Image Dir: {image_dir}")

    df = pd.read_csv(csv_path)
    names = df["filename"].tolist()

    # Reconstruct patient-grouped folds (seed 42)
    folds = make_grouped_folds(names, n_splits=5, seed=42)
    outer_tr_idx, outer_val_idx = folds[0]
    print(f"Fold 1 outer train: {len(outer_tr_idx)}, outer val: {len(outer_val_idx)}")

    # Split outer_tr_idx into inner train and inner val (10% patient holdout)
    inner_tr_idx, inner_val_idx = split_patient_inner_holdout(outer_tr_idx, names, holdout_fraction=0.10, seed=42)
    print(f"Inner train: {len(inner_tr_idx)}, inner val: {len(inner_val_idx)}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []

    # 1. Row A
    df_pred_a, met_a = train_and_eval_row_a(
        df, str(image_dir), inner_tr_idx, inner_val_idx, outer_val_idx, dev, max_epochs=args.epochs, smoke=args.smoke
    )
    df_pred_a.to_csv(out_dir / "row_A_predictions.csv", index=False)
    with open(out_dir / "row_A_metrics.json", "w", encoding="utf-8") as f:
        json.dump(met_a, f, indent=2)
    all_metrics.append(met_a)

    # 2. Row B
    df_pred_b, met_b = train_and_eval_row_b(
        df, str(image_dir), inner_tr_idx, inner_val_idx, outer_val_idx, dev, max_epochs=args.epochs, smoke=args.smoke
    )
    df_pred_b.to_csv(out_dir / "row_B_predictions.csv", index=False)
    with open(out_dir / "row_B_metrics.json", "w", encoding="utf-8") as f:
        json.dump(met_b, f, indent=2)
    all_metrics.append(met_b)

    # 3. Row C
    df_pred_c, met_c = train_and_eval_row_c(
        df, str(image_dir), inner_tr_idx, inner_val_idx, outer_val_idx, dev, max_epochs=args.epochs, smoke=args.smoke
    )
    df_pred_c.to_csv(out_dir / "row_C_predictions.csv", index=False)
    with open(out_dir / "row_C_metrics.json", "w", encoding="utf-8") as f:
        json.dump(met_c, f, indent=2)
    all_metrics.append(met_c)

    # 4. Row D
    df_pred_d, met_d = train_and_eval_row_d(
        df, str(image_dir), inner_tr_idx, inner_val_idx, outer_val_idx, dev, max_epochs=args.epochs, smoke=args.smoke
    )
    df_pred_d.to_csv(out_dir / "row_D_predictions.csv", index=False)
    with open(out_dir / "row_D_metrics.json", "w", encoding="utf-8") as f:
        json.dump(met_d, f, indent=2)
    all_metrics.append(met_d)

    # 5. Row E (from oof_predictions.csv)
    oof_csv_path = Path(args.oof_csv).resolve()
    df_pred_e, met_e = load_row_e_metrics(oof_csv_path)
    with open(out_dir / "row_E_metrics.json", "w", encoding="utf-8") as f:
        json.dump(met_e, f, indent=2)
    all_metrics.append(met_e)

    # Summary table and JSON
    table_md = generate_ladder_table(all_metrics)
    with open(out_dir / "ladder_table.md", "w", encoding="utf-8", newline="\n") as f:
        f.write(table_md)

    with open(out_dir / "ladder_results.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "held_out_fold": 1,
            "seed": 42,
            "metrics": all_metrics,
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("=== HC18 LADDER BENCHMARK SUMMARY ===")
    print("=" * 60)
    print(table_md)


if __name__ == "__main__":
    main()
