# FIELD: FASSD
# MODEL: EfficientNet-B4 Multilabel U-Net / U-Net++
# STEP: training
# USED BY: pipeline imports across work/scripts and work/tests

"""FASSD Multilabel Segmentation Training & Evaluation Module.

Implements 4-channel sigmoid multilabel segmentation for fetal abdominal structures
(artery, liver, stomach, vein) using segmentation_models_pytorch U-Net backbones,
combined BCE + Soft-Dice loss, AdamW, cosine annealing, AMP, and Model EMA.
Metrics are computed at native (768, 1024) resolution with surface HD95 and
patient-level paired bootstrap confidence intervals.
"""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp

STRUCTURE_NAMES = ("artery", "liver", "stomach", "vein")
STRUCTURE_BITFLAGS = {
    "artery": 1,
    "liver": 2,
    "stomach": 4,
    "vein": 8,
}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =============================================================================
# 1. Dataset & Data Augmentation
# =============================================================================

def apply_affine_augmentation(
    img: np.ndarray,
    mask: np.ndarray,
    max_angle: float = 10.0,
    max_translate: float = 0.05,
    max_scale: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply random affine transformation (rotation, translation, scale) to image and mask."""
    if rng is None:
        rng = np.random.default_rng()

    h, w = img.shape[:2]
    angle = rng.uniform(-max_angle, max_angle)
    scale = rng.uniform(1.0 - max_scale, 1.0 + max_scale)
    tx = rng.uniform(-max_translate, max_translate) * w
    ty = rng.uniform(-max_translate, max_translate) * h

    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty

    aug_img = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    aug_mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return aug_img, aug_mask


def apply_color_jitter(
    img: np.ndarray,
    alpha_range: tuple[float, float] = (0.85, 1.15),
    beta_range: tuple[float, float] = (-20.0, 20.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply brightness (beta) and contrast (alpha) jitter to uint8 image."""
    if rng is None:
        rng = np.random.default_rng()

    alpha = rng.uniform(*alpha_range)
    beta = rng.uniform(*beta_range)
    jittered = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return jittered


def apply_gpu_batch_augmentations(
    imgs: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast batch GPU augmentations: horizontal flip, affine, brightness/contrast, gaussian noise."""
    b, _, _, _ = imgs.shape
    device = imgs.device

    # 1. Random horizontal flip (p=0.5)
    flips = torch.rand(b, 1, 1, 1, device=device) < 0.5
    imgs = torch.where(flips, torch.flip(imgs, dims=[3]), imgs)
    masks = torch.where(flips, torch.flip(masks, dims=[3]), masks)

    # 2. Random Affine: rotation (-10 to 10 deg), scale (0.95 to 1.05), translate (-0.05 to 0.05)
    angles = (torch.rand(b, device=device) * 20.0 - 10.0) * (math.pi / 180.0)
    scales = torch.rand(b, device=device) * 0.10 + 0.95
    tx = torch.rand(b, device=device) * 0.10 - 0.05
    ty = torch.rand(b, device=device) * 0.10 - 0.05

    cos_a = torch.cos(angles) * scales
    sin_a = torch.sin(angles) * scales

    theta = torch.zeros(b, 2, 3, device=device)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, imgs.size(), align_corners=False)
    imgs = F.grid_sample(imgs, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    masks = F.grid_sample(masks, grid, mode="nearest", padding_mode="zeros", align_corners=False)

    # 3. Brightness & contrast jitter
    alpha = torch.empty(b, 1, 1, 1, device=device).uniform_(0.85, 1.15)
    beta = torch.empty(b, 1, 1, 1, device=device).uniform_(-0.08, 0.08)
    imgs = imgs * alpha + beta

    # 4. Mild Gaussian noise (p=0.3)
    if torch.rand(1, device=device).item() < 0.3:
        noise = torch.randn_like(imgs) * 0.02
        imgs = imgs + noise

    return imgs, masks


class FASSDKaggleDataset(Dataset):
    """Dataset for FASSD compact Kaggle format with bit-flag masks."""

    def __init__(
        self,
        images_dir: Path | str,
        masks_dir: Path | str,
        df: pd.DataFrame,
        input_mode: str = "rgb",
        target_size: tuple[int, int] = (384, 512),
        is_train: bool = True,
        seed: int | None = None,
        cache_in_memory: bool = False,
        batch_gpu_aug: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.df = df.reset_index(drop=True)
        self.input_mode = input_mode.lower()
        if self.input_mode not in ("rgb", "gray"):
            raise ValueError(f"input_mode must be 'rgb' or 'gray', got '{input_mode}'")
        self.target_size = target_size  # (H, W)
        self.is_train = is_train
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.cache_in_memory = cache_in_memory
        self.batch_gpu_aug = batch_gpu_aug

        self.cached_images: list[np.ndarray] = []
        self.cached_masks: list[np.ndarray] = []

        if self.cache_in_memory:
            th, tw = self.target_size
            for idx in range(len(self.df)):
                row = self.df.iloc[idx]
                fname = str(row["filename"])
                stem = Path(fname).stem

                img_path = self.images_dir / fname
                if not img_path.exists():
                    img_path = self.images_dir / f"{stem}.png"
                img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    raise FileNotFoundError(f"Failed to read image at {img_path}")

                if self.input_mode == "gray":
                    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                else:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                mask_path = self.masks_dir / f"{stem}.png"
                if not mask_path.exists():
                    mask_path = self.masks_dir / fname
                mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if mask_raw is None:
                    raise FileNotFoundError(f"Failed to read mask at {mask_path}")

                c_masks = []
                for name in STRUCTURE_NAMES:
                    flag = STRUCTURE_BITFLAGS[name]
                    c_masks.append(((mask_raw & flag) > 0).astype(np.uint8))
                multi_mask = np.stack(c_masks, axis=-1)

                img_resized = cv2.resize(img_rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(multi_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
                if mask_resized.ndim == 2:
                    mask_resized = mask_resized[:, :, None]

                self.cached_images.append(img_resized)
                self.cached_masks.append(mask_resized)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        fname = str(row["filename"])
        stem = Path(fname).stem
        patient = str(row["patient"])
        group = str(row.get("group", patient))

        if self.cache_in_memory:
            img_rgb = self.cached_images[idx]
            mask_resized = self.cached_masks[idx]

            if self.is_train and not self.batch_gpu_aug:
                if self.rng.random() < 0.5:
                    img_rgb = np.fliplr(img_rgb).copy()
                    mask_resized = np.fliplr(mask_resized).copy()
                img_rgb, mask_resized = apply_affine_augmentation(img_rgb, mask_resized, rng=self.rng)
                img_rgb = apply_color_jitter(img_rgb, rng=self.rng)

            img_float = img_rgb.astype(np.float32) / 255.0
            img_norm = (img_float - IMAGENET_MEAN) / IMAGENET_STD
            img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
            mask_tensor = torch.from_numpy(mask_resized.transpose(2, 0, 1)).float()

            return {
                "image": img_tensor,
                "mask": mask_tensor,
                "filename": fname,
                "patient": patient,
                "group": group,
                "native_size": (768, 1024),
            }

        img_path = self.images_dir / fname
        if not img_path.exists():
            img_path = self.images_dir / f"{stem}.png"
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image at {img_path}")

        if self.input_mode == "gray":
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask_path = self.masks_dir / f"{stem}.png"
        if not mask_path.exists():
            mask_path = self.masks_dir / fname
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask_raw is None:
            raise FileNotFoundError(f"Failed to read mask at {mask_path}")

        # Augmentation on native resolution
        if self.is_train and not self.batch_gpu_aug:
            # Horizontal flip
            if self.rng.random() < 0.5:
                img_rgb = np.fliplr(img_rgb).copy()
                mask_raw = np.fliplr(mask_raw).copy()

            # Random affine
            img_rgb, mask_raw = apply_affine_augmentation(img_rgb, mask_raw, rng=self.rng)

            # Brightness and contrast jitter
            img_rgb = apply_color_jitter(img_rgb, rng=self.rng)

        # Decode bit flags to 4 channels: (H, W, 4)
        c_masks = []
        for name in STRUCTURE_NAMES:
            flag = STRUCTURE_BITFLAGS[name]
            c_masks.append(((mask_raw & flag) > 0).astype(np.uint8))
        multi_mask = np.stack(c_masks, axis=-1)  # (H, W, 4)

        # Store native mask for native evaluation if needed
        native_h, native_w = img_rgb.shape[:2]

        # Resize to target size (384, 512)
        th, tw = self.target_size
        img_resized = cv2.resize(img_rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(multi_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
        if mask_resized.ndim == 2:
            mask_resized = mask_resized[:, :, None]

        # Normalize image to [0, 1] then ImageNet mean/std
        img_float = img_resized.astype(np.float32) / 255.0
        img_norm = (img_float - IMAGENET_MEAN) / IMAGENET_STD
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()

        mask_tensor = torch.from_numpy(mask_resized.transpose(2, 0, 1)).float()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "filename": fname,
            "patient": patient,
            "group": group,
            "native_size": (native_h, native_w),
        }


# =============================================================================
# 2. Loss Function: Multilabel BCE + Soft Dice
# =============================================================================

class FASSDMultiLabelLoss(nn.Module):
    """Combined BCE + Soft Dice Loss for multilabel sigmoid segmentation."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute compound loss over 4 channels.

        Args:
            logits: Shape (B, C, H, W).
            targets: Shape (B, C, H, W) in [0, 1].

        Returns:
            total_loss, bce_loss, dice_loss
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets)

        probs = torch.sigmoid(logits)
        # Spatial dimensions are (2, 3)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        dice_loss = 1.0 - dice.mean()

        total = self.bce_weight * bce + self.dice_weight * dice_loss
        return total, bce, dice_loss


# =============================================================================
# 3. Model Construction & EMA
# =============================================================================

def build_fassd_model(
    encoder_name: str = "efficientnet-b4",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 4,
    arch: str = "unet",
) -> nn.Module:
    """Build segmentation network with smp supporting unet, unetplusplus, and fpn backbones."""
    arch_norm = arch.lower().replace("-", "").replace("_", "")
    enc_name = encoder_name.lower()
    if enc_name == "convnext_tiny":
        enc_name = "tu-convnext_tiny"

    if arch_norm == "unet":
        return smp.Unet(
            encoder_name=enc_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif arch_norm in ("unetplusplus", "unet++"):
        return smp.UnetPlusPlus(
            encoder_name=enc_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif arch_norm == "fpn":
        return smp.FPN(
            encoder_name=enc_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    else:
        raise ValueError(f"Unsupported arch: '{arch}'. Choose from: 'unet', 'unetplusplus', 'fpn'.")


class ModelEMA:
    """Exponential Moving Average of model parameters and buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for param in self.module.parameters():
            param.requires_grad = False

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.module.state_dict().items():
                if k in msd:
                    if v.dtype.is_floating_point:
                        v.copy_(v * self.decay + (1.0 - self.decay) * msd[k].to(v.device))
                    else:
                        v.copy_(msd[k].to(v.device))


# =============================================================================
# 4. Metrics at Original Resolution (768, 1024)
# =============================================================================

def compute_hd95(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
    """Compute 95th percentile Hausdorff Distance (HD95) in pixels using surface EDT.

    If either mask is empty, returns diagonal penalty distance sqrt(H^2 + W^2).
    """
    h, w = pred_binary.shape
    diag = float(math.hypot(h, w))

    p_sum = int(pred_binary.sum())
    g_sum = int(gt_binary.sum())

    if p_sum == 0 and g_sum == 0:
        return 0.0
    if p_sum == 0 or g_sum == 0:
        return diag

    # Surface contour extraction with explicit boolean masks
    p_bool = (pred_binary > 0)
    g_bool = (gt_binary > 0)

    p_border = p_bool ^ binary_erosion(p_bool)
    g_border = g_bool ^ binary_erosion(g_bool)

    if not np.any(p_border):
        p_border = p_bool
    if not np.any(g_border):
        g_border = g_bool

    p_border = (p_border > 0)
    g_border = (g_border > 0)

    dt_gt = distance_transform_edt(~g_border)
    dt_pred = distance_transform_edt(~p_border)

    d_p2g = dt_gt[p_border]
    d_g2p = dt_pred[g_border]

    d_all = np.concatenate([d_p2g, d_g2p])
    return float(np.percentile(d_all, 95))


def compute_structure_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> dict[str, Any]:
    """Compute per-structure Dice, IoU, HD95 and empty detection status.

    Args:
        pred_mask: (4, H, W) binary uint8 / bool array.
        gt_mask: (4, H, W) binary uint8 / bool array.

    Returns:
        dict containing per-structure metrics and flags.
    """
    results: dict[str, Any] = {}

    for idx, name in enumerate(STRUCTURE_NAMES):
        p = pred_mask[idx] > 0
        g = gt_mask[idx] > 0

        p_sum = int(p.sum())
        g_sum = int(g.sum())

        is_gt_empty = (g_sum == 0)
        is_pred_empty = (p_sum == 0)

        struct_res: dict[str, Any] = {
            "gt_empty": is_gt_empty,
            "pred_empty": is_pred_empty,
            "gt_pixels": g_sum,
            "pred_pixels": p_sum,
        }

        if is_gt_empty:
            # Evaluated in empty-GT detection count
            struct_res["empty_detected"] = is_pred_empty
            struct_res["dice"] = None
            struct_res["iou"] = None
            struct_res["hd95"] = None
        else:
            intersection = int(np.logical_and(p, g).sum())
            union = int(np.logical_or(p, g).sum())

            dice = float((2.0 * intersection) / (p_sum + g_sum))
            iou = float(intersection / union) if union > 0 else 0.0
            hd95 = compute_hd95(p, g)

            struct_res["empty_detected"] = None
            struct_res["dice"] = dice
            struct_res["iou"] = iou
            struct_res["hd95"] = hd95

        results[name] = struct_res

    return results


def run_patient_bootstrap_ci(
    patient_scores: dict[str, float],
    n_bootstraps: int = 5000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute patient-level bootstrap 95% confidence interval for mean metric.

    Returns:
        mean, ci_lower, ci_upper
    """
    patients = sorted(list(patient_scores.keys()))
    if len(patients) == 0:
        return 0.0, 0.0, 0.0

    values = np.array([patient_scores[p] for p in patients], dtype=np.float64)
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


# =============================================================================
# 5. Dataset Evaluation Routine
# =============================================================================

def evaluate_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
    masks_dir: Path | str,
    orig_shape: tuple[int, int] = (768, 1024),
) -> dict[str, Any]:
    """Evaluate model on a dataset at original (768, 1024) resolution.

    Reads exact ground truth masks from masks_dir to avoid any interpolation artifacts.
    """
    model.eval()
    masks_path = Path(masks_dir)

    all_sample_metrics: list[dict[str, Any]] = []
    patient_dice_accum: dict[str, list[float]] = {}

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            filenames = batch["filename"]
            patients = batch["patient"]

            logits = model(images)  # (B, 4, 384, 512)
            probs = torch.sigmoid(logits)

            # Upsample probabilities to original shape (768, 1024)
            probs_orig = F.interpolate(
                probs,
                size=orig_shape,
                mode="bilinear",
                align_corners=False,
            ).cpu().numpy()  # (B, 4, 768, 1024)

            pred_bin = (probs_orig > 0.5).astype(np.uint8)

            for b in range(len(filenames)):
                fname = filenames[b]
                stem = Path(fname).stem
                pat = patients[b]

                # Load raw native ground truth mask
                m_file = masks_path / f"{stem}.png"
                if not m_file.exists():
                    m_file = masks_path / fname
                raw_mask = cv2.imread(str(m_file), cv2.IMREAD_UNCHANGED)
                if raw_mask is None:
                    # Fallback to batch mask if raw mask not found
                    gt_bin = (batch["mask"][b].numpy() > 0.5).astype(np.uint8)
                else:
                    gt_channels = []
                    for name in STRUCTURE_NAMES:
                        flag = STRUCTURE_BITFLAGS[name]
                        gt_channels.append(((raw_mask & flag) > 0).astype(np.uint8))
                    gt_bin = np.stack(gt_channels, axis=0)  # (4, 768, 1024)

                p_bin = pred_bin[b]
                sample_m = compute_structure_metrics(p_bin, gt_bin)
                sample_m["filename"] = fname
                sample_m["patient"] = pat

                all_sample_metrics.append(sample_m)

                # Collect per-image mean Dice across non-empty structures
                sample_dices = [
                    sample_m[name]["dice"]
                    for name in STRUCTURE_NAMES
                    if sample_m[name]["dice"] is not None
                ]
                if sample_dices:
                    img_mean_dice = float(np.mean(sample_dices))
                    patient_dice_accum.setdefault(pat, []).append(img_mean_dice)

    # Compute per-structure aggregates over non-empty GT images
    summary_by_structure: dict[str, Any] = {}
    macro_dice_list = []
    macro_iou_list = []
    macro_hd95_list = []

    for name in STRUCTURE_NAMES:
        dices = [m[name]["dice"] for m in all_sample_metrics if m[name]["dice"] is not None]
        ious = [m[name]["iou"] for m in all_sample_metrics if m[name]["iou"] is not None]
        hd95s = [m[name]["hd95"] for m in all_sample_metrics if m[name]["hd95"] is not None]

        empty_gt_total = sum(1 for m in all_sample_metrics if m[name]["gt_empty"])
        empty_detected = sum(1 for m in all_sample_metrics if m[name]["empty_detected"] is True)

        s_dict = {
            "n_non_empty": len(dices),
            "n_empty_gt": empty_gt_total,
            "empty_gt_detected_count": empty_detected,
            "empty_gt_detection_rate": float(empty_detected / empty_gt_total) if empty_gt_total > 0 else 1.0,
            "dice_mean": float(np.mean(dices)) if dices else 0.0,
            "dice_std": float(np.std(dices)) if dices else 0.0,
            "iou_mean": float(np.mean(ious)) if ious else 0.0,
            "iou_std": float(np.std(ious)) if ious else 0.0,
            "hd95_mean": float(np.mean(hd95s)) if hd95s else 0.0,
            "hd95_std": float(np.std(hd95s)) if hd95s else 0.0,
        }
        summary_by_structure[name] = s_dict

        if dices:
            macro_dice_list.append(s_dict["dice_mean"])
            macro_iou_list.append(s_dict["iou_mean"])
            macro_hd95_list.append(s_dict["hd95_mean"])

    macro_dice = float(np.mean(macro_dice_list)) if macro_dice_list else 0.0
    macro_iou = float(np.mean(macro_iou_list)) if macro_iou_list else 0.0
    macro_hd95 = float(np.mean(macro_hd95_list)) if macro_hd95_list else 0.0

    # Patient-level mean Dice for bootstrap
    patient_mean_dice = {pat: float(np.mean(ds)) for pat, ds in patient_dice_accum.items()}
    p_mean, ci_low, ci_high = run_patient_bootstrap_ci(patient_mean_dice, n_bootstraps=5000, seed=42)

    return {
        "macro_mean_dice": macro_dice,
        "macro_mean_iou": macro_iou,
        "macro_mean_hd95": macro_hd95,
        "patient_bootstrap_ci": {
            "mean": p_mean,
            "ci_lower_95": ci_low,
            "ci_upper_95": ci_high,
            "n_patients": len(patient_mean_dice),
        },
        "by_structure": summary_by_structure,
        "n_samples": len(all_sample_metrics),
    }


# =============================================================================
# 6. Full Training Routine
# =============================================================================

def train_fassd(
    data_root: str | Path,
    output_dir: str | Path,
    split_csv: str | Path,
    input_mode: str = "rgb",
    arch: str = "unet",
    encoder_name: str = "efficientnet-b4",
    encoder_weights: str | None = "imagenet",
    epochs: int = 40,
    patience: int = 10,
    batch_size: int = 8,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    seed: int = 42,
    device: str | None = None,
    smoke: bool = False,
    ema_decay: float = 0.999,
    cache_in_memory: bool = True,
    batch_gpu_aug: bool = True,
    channels_last: bool = True,
) -> dict[str, Any]:
    """Train FASSD multilabel segmentation model with AMP, EMA, caching, and batch GPU aug."""
    # Determinism
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_log_path = out_path / "train_log.csv"
    summary_json_path = out_path / "train_summary.json"
    best_model_path = out_path / "best_model.pt"

    root_path = Path(data_root)
    img_dir = root_path / "images"
    mask_dir = root_path / "masks"
    df_splits = pd.read_csv(split_csv)

    if smoke:
        # Fast smoke configuration
        train_df = df_splits[df_splits["split"] == "train"].head(6)
        val_df = df_splits[df_splits["split"] == "val"].head(2)
        test_df = df_splits[df_splits["split"] == "test"].head(2)
        epochs = min(epochs, 2)
        batch_size = min(batch_size, 2)
        encoder_name = "resnet18"
        encoder_weights = None
        arch = "unet"
        cache_in_memory = False
        batch_gpu_aug = False
    else:
        train_df = df_splits[df_splits["split"] == "train"]
        val_df = df_splits[df_splits["split"] == "val"]
        test_df = df_splits[df_splits["split"] == "test"]

    train_ds = FASSDKaggleDataset(
        img_dir,
        mask_dir,
        train_df,
        input_mode=input_mode,
        is_train=True,
        seed=seed,
        cache_in_memory=cache_in_memory,
        batch_gpu_aug=batch_gpu_aug,
    )
    val_ds = FASSDKaggleDataset(
        img_dir,
        mask_dir,
        val_df,
        input_mode=input_mode,
        is_train=False,
        seed=seed,
        cache_in_memory=cache_in_memory,
        batch_gpu_aug=False,
    )
    test_ds = FASSDKaggleDataset(
        img_dir,
        mask_dir,
        test_df,
        input_mode=input_mode,
        is_train=False,
        seed=seed,
        cache_in_memory=cache_in_memory,
        batch_gpu_aug=False,
    )

    num_workers = 0 if os.name == "nt" or smoke else 2
    pin_memory = (device_obj.type == "cuda")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    model = build_fassd_model(
        arch=arch, encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=3, classes=4
    )
    model = model.to(device_obj)
    if channels_last and device_obj.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    ema = ModelEMA(model, decay=ema_decay)

    criterion = FASSDMultiLabelLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    use_amp = (device_obj.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_dice = -1.0
    best_epoch = -1
    no_improve_epochs = 0
    log_rows: list[dict[str, Any]] = []

    print(f"Starting training on {device_obj}: {epochs} epochs, patience={patience}, arch={arch}, input={input_mode}, encoder={encoder_name}, weights={encoder_weights}, wd={weight_decay}")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_bce_accum = 0.0
        train_dice_accum = 0.0
        n_train_batches = 0

        for batch in train_loader:
            imgs = batch["image"].to(device_obj)
            masks = batch["mask"].to(device_obj)

            if batch_gpu_aug and device_obj.type == "cuda":
                imgs, masks = apply_gpu_batch_augmentations(imgs, masks)

            if channels_last and device_obj.type == "cuda":
                imgs = imgs.to(memory_format=torch.channels_last)

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(imgs)
                    loss, bce, d_loss = criterion(logits, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss, bce, d_loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()

            ema.update(model)

            train_loss_accum += float(loss.item())
            train_bce_accum += float(bce.item())
            train_dice_accum += float(d_loss.item())
            n_train_batches += 1

        scheduler.step()

        avg_train_loss = train_loss_accum / max(1, n_train_batches)
        avg_train_bce = train_bce_accum / max(1, n_train_batches)
        avg_train_d_loss = train_dice_accum / max(1, n_train_batches)

        # Validation with EMA model
        val_eval = evaluate_dataset(ema.module, val_loader, device_obj, mask_dir)
        val_dice = val_eval["macro_mean_dice"]
        val_iou = val_eval["macro_mean_iou"]
        val_hd95 = val_eval["macro_mean_hd95"]

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": avg_train_loss,
            "train_bce": avg_train_bce,
            "train_dice_loss": avg_train_d_loss,
            "val_macro_dice": val_dice,
            "val_macro_iou": val_iou,
            "val_macro_hd95": val_hd95,
        }
        log_rows.append(row)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | Val HD95: {val_hd95:.2f}")

        # Best checkpoint strictly by VAL mean Dice
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": ema.module.state_dict(),
                "val_macro_dice": val_dice,
                "val_eval": val_eval,
                "arch": arch,
                "encoder_name": encoder_name,
                "input_mode": input_mode,
            }, str(best_model_path))
        else:
            no_improve_epochs += 1
            if patience > 0 and no_improve_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch}: no improvement for {patience} epochs.")
                break

    pd.DataFrame(log_rows).to_csv(csv_log_path, index=False)
    print(f"Training completed in {time.time() - start_time:.2f}s. Best epoch: {best_epoch} (Val Dice: {best_val_dice:.4f})")

    # Evaluate TEST set once at the end with the best-val checkpoint
    checkpoint = torch.load(str(best_model_path), map_location=device_obj)
    best_model = build_fassd_model(
        arch=arch, encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=4
    )
    best_model.load_state_dict(checkpoint["model_state_dict"])
    best_model = best_model.to(device_obj)

    print("Evaluating TEST set once with best-val checkpoint...")
    test_eval = evaluate_dataset(best_model, test_loader, device_obj, mask_dir)
    print(f"Test Macro Dice: {test_eval['macro_mean_dice']:.4f} | Test Macro IoU: {test_eval['macro_mean_iou']:.4f} | Test Macro HD95: {test_eval['macro_mean_hd95']:.2f}")
    print(f"Test Patient Bootstrap 95% CI: [{test_eval['patient_bootstrap_ci']['ci_lower_95']:.4f}, {test_eval['patient_bootstrap_ci']['ci_upper_95']:.4f}]")

    summary: dict[str, Any] = {
        "config": {
            "data_root": str(data_root),
            "split_csv": str(split_csv),
            "input_mode": input_mode,
            "arch": arch,
            "encoder_name": encoder_name,
            "encoder_weights": encoder_weights,
            "epochs": epochs,
            "patience": patience,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
            "smoke": smoke,
            "ema_decay": ema_decay,
            "cache_in_memory": cache_in_memory,
            "batch_gpu_aug": batch_gpu_aug,
            "channels_last": channels_last,
        },
        "best_epoch": best_epoch,
        "best_val_macro_dice": best_val_dice,
        "test_results": test_eval,
    }

    import json
    with open(summary_json_path, "w") as fp:
        json.dump(summary, fp, indent=2)

    return summary
