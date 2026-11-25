# FIELD: Femur
# MODEL: EfficientNet-B4 / ConvNeXt Landmark
# STEP: training
# USED BY: work/docs/CODE_MAP.md

"""Self-contained Kaggle runner for Femur Round 2: Modern Techniques.

Trains and evaluates:
  Row A:  Baseline U-Net EfficientNet-B4, 1-channel dual-peak heatmap, 384px, standard peak decode, flip TTA
  Row R1: HRNet-W32 (timm), 1-channel dual-peak heatmap, 384px, DARK-style 2D Taylor sub-pixel decode, flip TTA
  Row R2: Row-A network (classes=2) with heatmap loss + integral (soft-argmax) L1 coordinate loss on the two endpoints (order-invariant), decode by soft-argmax, flip TTA
  Row R3: Row-A network (classes=2: ch0 dual peak, ch1 bone line) + auxiliary bone line heatmap loss + loss on measured length (pixel FL), decode with min_dist=50.0 px (from TRAIN split), flip TTA
  Row R4: Row-A network + strong ultrasound-style augmentation (rot +-30 deg, scale 0.8-1.2, trans +-10%, speckle/Gaussian noise, gamma, brightness/contrast, hflip; endpoints transformed consistently), 384px
  Row R5: Pretrained ultrasound/vision encoder check (UltraSam / DINOv2 ViT-S/14). Reported as incompatible with 384px and skipped.
  Row R6: Median ensemble of Row A and the two best rows chosen by validation MAE only.

Evaluates on 348 test images.
Computes patient-level paired bootstrap (1,000 resamples) of difference vs Row A with 95% CIs.
Generates femur_round2_table.md, femur_round2_results.json, femur_round2_predictions.csv, and executes femur_round2.ipynb.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 1. Install SMP and timm if missing
try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("Installing segmentation-models-pytorch==0.5.0...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-warn-script-location", "segmentation-models-pytorch==0.5.0"])
    import segmentation_models_pytorch as smp

try:
    import timm
except ImportError:
    print("Installing timm...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-warn-script-location", "timm"])
    import timm

# Normalization constants
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Data Loading and Splitting
# ---------------------------------------------------------------------------
def patient_id_from_name(name: str, src: str) -> str:
    name_str = str(name)
    if src == "FP":
        m = re.search(r"(Patient\d+)", name_str)
        if not m:
            raise ValueError(f"Could not extract FP patient ID from {name_str}")
        return f"FP_{m.group(1)}"
    elif src == "UCL":
        m = re.match(r"(\d+)_", name_str)
        if not m:
            raise ValueError(f"Could not extract UCL patient ID from {name_str}")
        return f"UCL_{m.group(1)}"
    raise ValueError(f"Unknown source: {src}")


def load_femur_source(src: str, ann_dir: Path, img_dir: Path) -> pd.DataFrame:
    ann_path = ann_dir / "Femur.csv"
    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing {ann_path}")
    df = pd.read_csv(ann_path)
    if src == "UCL":
        df = df[~df["image_name"].str.startswith("125_")].copy()
    df = df.reset_index(drop=True)

    df["source"] = src
    df["image_path"] = df["image_name"].apply(lambda f: str(img_dir / f))
    df["patient"] = df["image_name"].apply(lambda f: patient_id_from_name(f, src))

    te_path = ann_dir / "Femur_Test.csv"
    tr_path = ann_dir / "Femur_Train.csv"
    if te_path.is_file():
        te_names = set(pd.read_csv(te_path)["image_name"].astype(str))
        df["official_split"] = df["image_name"].astype(str).apply(lambda f: "test" if f in te_names else "train")
    elif tr_path.is_file():
        tr_names = set(pd.read_csv(tr_path)["image_name"].astype(str))
        df["official_split"] = df["image_name"].astype(str).apply(lambda f: "train" if f in tr_names else "test")
    else:
        df["official_split"] = df["Split"].astype(str).str.lower()

    df["fl_length_mm"] = [
        float(np.hypot(r.fl_2_x - r.fl_1_x, r.fl_2_y - r.fl_1_y) * r.px_to_mm_rate)
        if pd.notna(r.px_to_mm_rate) else float("nan")
        for r in df.itertuples()
    ]
    cols = [
        "image_name", "image_path", "source", "patient", "official_split",
        "px_to_mm_rate", "fl_length_mm", "fl_1_x", "fl_1_y", "fl_2_x", "fl_2_y"
    ]
    return df[cols]


def build_femur_data_from_root(data_root: Path) -> pd.DataFrame:
    fp = load_femur_source("FP", data_root / "FP" / "annotations", data_root / "FP" / "images" / "Femur")
    ucl = load_femur_source("UCL", data_root / "UCL" / "annotations" / "px_to_mm", data_root / "UCL" / "images" / "Femur")
    df = pd.concat([fp, ucl], ignore_index=True)
    df = df[df["px_to_mm_rate"].notna()].reset_index(drop=True)
    return df


def split_femur_data(df: pd.DataFrame, dev_fraction: float = 0.12, seed: int = 0):
    train_all = df[df["official_split"] == "train"]
    pats = np.array(sorted(train_all["patient"].unique()))
    rng = np.random.RandomState(seed)
    dev_count = max(1, int(round(dev_fraction * len(pats))))
    dev_p = set(rng.permutation(pats)[:dev_count])

    tr = train_all[~train_all["patient"].isin(dev_p)].reset_index(drop=True)
    dv = train_all[train_all["patient"].isin(dev_p)].reset_index(drop=True)
    te = df[df["official_split"] == "test"].reset_index(drop=True)

    assert not (set(tr["patient"]) & set(dv["patient"])), "Train/dev patient overlap!"
    assert not (set(tr["patient"]) & set(te["patient"])), "Train/test patient overlap!"
    assert not (set(dv["patient"]) & set(te["patient"])), "Dev/test patient overlap!"
    return tr, dv, te


# ---------------------------------------------------------------------------
# Letterboxing & Datasets
# ---------------------------------------------------------------------------
def letterbox_image(img: np.ndarray, S: int):
    h, w = img.shape[:2]
    scale = S / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    canvas = np.zeros((S, S, 3), np.uint8)
    px, py = (S - nw) // 2, (S - nh) // 2
    resized = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )
    if resized.ndim == 2:
        resized = np.stack([resized] * 3, axis=-1)
    elif resized.shape[2] == 1:
        resized = np.repeat(resized, 3, axis=2)
    canvas[py : py + nh, px : px + nw] = resized
    return canvas, scale, px, py


def apply_strong_ultrasound_aug(canvas: np.ndarray, pts: np.ndarray, S: int, rng: np.random.RandomState):
    """Apply strong ultrasound-style geometric and photometric augmentation."""
    # 1. Flip horizontally
    if rng.rand() < 0.5:
        canvas = np.fliplr(canvas).copy()
        pts[:, 0] = (S - 1) - pts[:, 0]

    # 2. Geometric affine (rot +-30 deg, scale 0.8-1.2, trans +-10%)
    ang = rng.uniform(-30, 30)
    sc = rng.uniform(0.8, 1.2)
    tx = rng.uniform(-0.1 * S, 0.1 * S)
    ty = rng.uniform(-0.1 * S, 0.1 * S)
    M = cv2.getRotationMatrix2D((S / 2.0, S / 2.0), ang, sc)
    M[0, 2] += tx
    M[1, 2] += ty
    canvas = cv2.warpAffine(canvas, M, (S, S), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    pts = pts @ M[:, :2].T + M[:, 2]
    pts = np.clip(pts, 0, S - 1)

    # 3. Gamma / contrast / brightness
    gamma = rng.uniform(0.7, 1.4)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    canvas = cv2.LUT(canvas, table)

    alpha = rng.uniform(0.8, 1.2)
    beta = rng.uniform(-20, 20)
    canvas = np.clip(alpha * canvas.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # 4. Speckle / Gaussian noise
    if rng.rand() < 0.5:
        noise = rng.normal(0, 15, canvas.shape).astype(np.float32)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return canvas, pts


class FemurKaggleDS(Dataset):
    def __init__(self, df: pd.DataFrame, S: int = 384, is_train: bool = False, strong_aug: bool = False):
        self.df = df.reset_index(drop=True)
        self.S = S
        self.is_train = is_train
        self.strong_aug = strong_aug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = str(row["image_path"])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Missing {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        canvas, scale, px, py = letterbox_image(img, self.S)

        p1_x, p1_y = float(row["fl_1_x"]), float(row["fl_1_y"])
        p2_x, p2_y = float(row["fl_2_x"]), float(row["fl_2_y"])

        p1 = np.array([p1_x * scale + px, p1_y * scale + py], dtype=np.float32)
        p2 = np.array([p2_x * scale + px, p2_y * scale + py], dtype=np.float32)
        pts = np.stack([p1, p2], axis=0)

        if self.is_train:
            rng = np.random.RandomState()
            if self.strong_aug:
                canvas, pts = apply_strong_ultrasound_aug(canvas, pts, self.S, rng)
            else:
                # Standard training augmentation
                if rng.rand() < 0.5:
                    canvas = np.fliplr(canvas).copy()
                    pts[:, 0] = (self.S - 1) - pts[:, 0]
                if rng.rand() < 0.5:
                    ang = rng.uniform(-15, 15)
                    sc = rng.uniform(0.92, 1.08)
                    M = cv2.getRotationMatrix2D((self.S / 2, self.S / 2), ang, sc)
                    canvas = cv2.warpAffine(canvas, M, (self.S, self.S), borderValue=(0, 0, 0))
                    pts = pts @ M[:, :2].T + M[:, 2]
                    pts = np.clip(pts, 0, self.S - 1)

        norm_img = ((canvas.astype(np.float32) / 255.0) - MEAN) / STD
        x = torch.from_numpy(norm_img.transpose(2, 0, 1).astype(np.float32))
        gt = torch.from_numpy(pts.astype(np.float32))
        rate = float(row["px_to_mm_rate"])
        mmpp = torch.tensor(rate / scale, dtype=torch.float32)
        return x, gt, mmpp, idx


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_femur_heatmap_model(encoder_weights="imagenet", classes=1):
    return smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=classes,
    )


class FemurHRNetHeatmap(nn.Module):
    def __init__(self, pretrained: bool = True, classes: int = 1):
        super().__init__()
        self.backbone = timm.create_model("hrnet_w32", pretrained=pretrained, features_only=True)
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        h = self.head(feats[0])
        return F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)


def build_femur_hrnet_model(pretrained: bool = True, classes: int = 1):
    return FemurHRNetHeatmap(pretrained=pretrained, classes=classes)


# ---------------------------------------------------------------------------
# Target Generation & Decoders
# ---------------------------------------------------------------------------
def generate_femur_heatmap(pts: torch.Tensor, S: int, sigma: float = 4.0):
    B = pts.shape[0]
    device = pts.device
    xs = torch.arange(S, device=device, dtype=torch.float32).view(1, 1, 1, S)
    ys = torch.arange(S, device=device, dtype=torch.float32).view(1, 1, S, 1)

    p1_x, p1_y = pts[:, 0, 0:1, None, None], pts[:, 0, 1:2, None, None]
    p2_x, p2_y = pts[:, 1, 0:1, None, None], pts[:, 1, 1:2, None, None]

    g1 = torch.exp(-((xs - p1_x) ** 2 + (ys - p1_y) ** 2) / (2.0 * sigma ** 2))
    g2 = torch.exp(-((xs - p2_x) ** 2 + (ys - p2_y) ** 2) / (2.0 * sigma ** 2))

    unnorm = torch.clamp(g1 + g2, 0.0, 1.0)
    flat = unnorm.view(B, 1, -1) + 1e-12
    prob = (flat / flat.sum(-1, keepdim=True)).view(B, 1, S, S)
    return prob, unnorm


def generate_single_landmark_heatmap(pts_single: torch.Tensor, S: int, sigma: float = 4.0):
    """Generate normalized and unnormalized heatmap for a single point (B, 1, 2)."""
    B = pts_single.shape[0]
    device = pts_single.device
    xs = torch.arange(S, device=device, dtype=torch.float32).view(1, 1, 1, S)
    ys = torch.arange(S, device=device, dtype=torch.float32).view(1, 1, S, 1)
    px = pts_single[:, 0, 0:1, None, None]
    py = pts_single[:, 0, 1:2, None, None]
    unnorm = torch.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2.0 * sigma ** 2))
    flat = unnorm.view(B, 1, -1) + 1e-12
    prob = (flat / flat.sum(-1, keepdim=True)).view(B, 1, S, S)
    return prob, unnorm


def generate_femur_boneline(pts: torch.Tensor, S: int, sigma: float = 3.0):
    """Generate Gaussian bone line segment connecting two endpoints."""
    B = pts.shape[0]
    device = pts.device
    xs = torch.arange(S, device=device, dtype=torch.float32).view(1, 1, 1, S)
    ys = torch.arange(S, device=device, dtype=torch.float32).view(1, 1, S, 1)

    p1_x, p1_y = pts[:, 0, 0:1, None, None], pts[:, 0, 1:2, None, None]
    p2_x, p2_y = pts[:, 1, 0:1, None, None], pts[:, 1, 1:2, None, None]

    vx = p2_x - p1_x
    vy = p2_y - p1_y
    l2 = vx ** 2 + vy ** 2 + 1e-8

    t = torch.clamp(((xs - p1_x) * vx + (ys - p1_y) * vy) / l2, 0.0, 1.0)
    qx = p1_x + t * vx
    qy = p1_y + t * vy

    d2 = (xs - qx) ** 2 + (ys - qy) ** 2
    return torch.exp(-d2 / (2.0 * sigma ** 2))


def decode_femur_peaks(heatmap: torch.Tensor, S: int | None = None, k: int = 9, r: int = 8, min_dist: float = 8.0):
    """Standard center-of-mass peak decoder."""
    B, C, H, W = heatmap.shape
    if S is None:
        S = H

    if (heatmap < 0).any() or (heatmap > 1.01).any():
        flat = torch.softmax(heatmap.float().reshape(B, -1), dim=-1)
        prob = flat.reshape(B, 1, H, W)
    else:
        prob = heatmap.float()

    hmax = F.max_pool2d(prob, kernel_size=k, stride=1, padding=k // 2)
    peaks_mask = (prob == hmax)

    batch_pts = []
    for b in range(B):
        p_b = prob[b, 0]
        mask_b = peaks_mask[b, 0]
        coords = torch.nonzero(mask_b)

        if len(coords) >= 2:
            vals = p_b[coords[:, 0], coords[:, 1]]
            top_count = min(len(coords), 16)
            top_idx = torch.topk(vals, top_count).indices
            top_coords = coords[top_idx]

            p1 = top_coords[0]
            p2 = None
            for cand in top_coords[1:]:
                dist = torch.hypot((cand[0] - p1[0]).float(), (cand[1] - p1[1]).float())
                if dist >= min_dist:
                    p2 = cand
                    break
            if p2 is None:
                p2 = top_coords[1]
            chosen = [p1, p2]
        else:
            flat_b = p_b.flatten()
            idx1 = flat_b.argmax()
            y1, x1 = idx1 // W, idx1 % W
            p1 = torch.stack([y1, x1])

            p_b_masked = p_b.clone()
            y_min, y_max = max(0, int(y1 - min_dist)), min(H, int(y1 + min_dist + 1))
            x_min, x_max = max(0, int(x1 - min_dist)), min(W, int(x1 + min_dist + 1))
            p_b_masked[y_min:y_max, x_min:x_max] = -1.0
            idx2 = p_b_masked.flatten().argmax()
            y2, x2 = idx2 // W, idx2 % W
            p2 = torch.stack([y2, x2])
            chosen = [p1, p2]

        refined = []
        for cy, cx in chosen:
            cy_i, cx_i = int(cy), int(cx)
            y_min, y_max = max(0, cy_i - r), min(H, cy_i + r + 1)
            x_min, x_max = max(0, cx_i - r), min(W, cx_i + r + 1)
            win = p_b[y_min:y_max, x_min:x_max]
            w_sum = win.sum() + 1e-12
            wx = torch.arange(x_min, x_max, dtype=torch.float32, device=p_b.device)
            wy = torch.arange(y_min, y_max, dtype=torch.float32, device=p_b.device)
            sub_x = (win.sum(0) * wx).sum() / w_sum
            sub_y = (win.sum(1) * wy).sum() / w_sum
            refined.append(torch.stack([sub_x, sub_y]))

        batch_pts.append(torch.stack(refined))

    return torch.stack(batch_pts)


def decode_femur_peaks_dark(heatmap: torch.Tensor, S: int | None = None, k: int = 9, min_dist: float = 8.0):
    """DARK-style 2D Taylor expansion peak decoder."""
    B, C, H, W = heatmap.shape
    if S is None:
        S = H

    if (heatmap < 0).any() or (heatmap > 1.01).any():
        flat = torch.softmax(heatmap.float().reshape(B, -1), dim=-1)
        prob = flat.reshape(B, 1, H, W)
    else:
        prob = heatmap.float()

    log_p = torch.log(prob + 1e-12)
    k_blur = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32, device=heatmap.device).view(1, 1, 3, 3) / 16.0
    log_p_smooth = F.conv2d(log_p, k_blur, padding=1)

    hmax = F.max_pool2d(prob, kernel_size=k, stride=1, padding=k // 2)
    peaks_mask = (prob == hmax)

    batch_pts = []
    for b in range(B):
        p_b = prob[b, 0]
        lp_b = log_p_smooth[b, 0]
        mask_b = peaks_mask[b, 0]
        coords = torch.nonzero(mask_b)

        if len(coords) >= 2:
            vals = p_b[coords[:, 0], coords[:, 1]]
            top_count = min(len(coords), 16)
            top_idx = torch.topk(vals, top_count).indices
            top_coords = coords[top_idx]
            p1 = top_coords[0]
            p2 = None
            for cand in top_coords[1:]:
                dist = torch.hypot((cand[0] - p1[0]).float(), (cand[1] - p1[1]).float())
                if dist >= min_dist:
                    p2 = cand
                    break
            if p2 is None:
                p2 = top_coords[1]
            chosen = [p1, p2]
        else:
            flat_b = p_b.flatten()
            idx1 = flat_b.argmax()
            y1, x1 = idx1 // W, idx1 % W
            p1 = torch.stack([y1, x1])
            p_b_masked = p_b.clone()
            y_min, y_max = max(0, int(y1 - min_dist)), min(H, int(y1 + min_dist + 1))
            x_min, x_max = max(0, int(x1 - min_dist)), min(W, int(x1 + min_dist + 1))
            p_b_masked[y_min:y_max, x_min:x_max] = -1.0
            idx2 = p_b_masked.flatten().argmax()
            y2, x2 = idx2 // W, idx2 % W
            p2 = torch.stack([y2, x2])
            chosen = [p1, p2]

        refined = []
        for cy, cx in chosen:
            y, x = int(cy), int(cx)
            if 1 <= y < H - 1 and 1 <= x < W - 1:
                dx = 0.5 * (lp_b[y, x + 1] - lp_b[y, x - 1])
                dy = 0.5 * (lp_b[y + 1, x] - lp_b[y - 1, x])
                dxx = lp_b[y, x + 1] - 2.0 * lp_b[y, x] + lp_b[y, x - 1]
                dyy = lp_b[y + 1, x] - 2.0 * lp_b[y, x] + lp_b[y - 1, x]
                dxy = 0.25 * (lp_b[y + 1, x + 1] - lp_b[y + 1, x - 1] - lp_b[y - 1, x + 1] + lp_b[y - 1, x - 1])

                det = dxx * dyy - dxy * dxy
                if abs(det) > 1e-6 and dxx < 0 and dyy < 0:
                    shift_x = -(dx * dyy - dy * dxy) / det
                    shift_y = -(dy * dxx - dx * dxy) / det
                    shift_x = torch.clamp(shift_x, -0.5, 0.5)
                    shift_y = torch.clamp(shift_y, -0.5, 0.5)
                else:
                    shift_x = -dx / torch.clamp(dxx, max=-1e-4) if dxx < 0 else torch.tensor(0.0, device=lp_b.device)
                    shift_y = -dy / torch.clamp(dyy, max=-1e-4) if dyy < 0 else torch.tensor(0.0, device=lp_b.device)
                    shift_x = torch.clamp(shift_x, -0.5, 0.5)
                    shift_y = torch.clamp(shift_y, -0.5, 0.5)
                refined.append(torch.stack([float(x) + shift_x, float(y) + shift_y]))
            else:
                refined.append(torch.tensor([float(x), float(y)], device=lp_b.device))
        batch_pts.append(torch.stack(refined))

    return torch.stack(batch_pts)


def soft_argmax_2d(logits: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Compute 2D spatial soft-argmax for each channel in logits."""
    B, C, H, W = logits.shape
    device = logits.device
    flat = logits.view(B, C, -1)
    prob = torch.softmax(flat * beta, dim=-1).view(B, C, H, W)

    xs = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)
    ys = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)

    pred_x = (prob * xs).sum(dim=(-2, -1))
    pred_y = (prob * ys).sum(dim=(-2, -1))
    return torch.stack([pred_x, pred_y], dim=-1)


# ---------------------------------------------------------------------------
# Matching & Losses
# ---------------------------------------------------------------------------
def match_endpoints(pred: torch.Tensor, gt: torch.Tensor):
    """Find permutation of pred (B, 2, 2) that minimizes distance to gt (B, 2, 2)."""
    B = pred.shape[0]
    d0 = (pred - gt).norm(dim=-1).sum(-1)
    pred_swapped = torch.stack([pred[:, 1], pred[:, 0]], dim=1)
    d1 = (pred_swapped - gt).norm(dim=-1).sum(-1)

    use_swap = (d1 < d0)
    matched = torch.where(use_swap[:, None, None], pred_swapped, pred)
    order = torch.where(use_swap[:, None], torch.tensor([1, 0], device=pred.device), torch.tensor([0, 1], device=pred.device))
    return matched, order


def femur_heatmap_loss(logits: torch.Tensor, gt_pts: torch.Tensor, S: int, sigma: float = 4.0):
    """Dual-Gaussian JS + BCE loss for 1-channel heatmap."""
    B = logits.shape[0]
    scaled_sigma = sigma * S / 384.0
    prob_target, unnorm_target = generate_femur_heatmap(gt_pts, S, scaled_sigma)

    p = torch.softmax(logits.float().reshape(B, -1), dim=-1)
    q = prob_target.reshape(B, -1) + 1e-12

    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p + 1e-12) - torch.log(m + 1e-12))).sum(-1)
    kl_qm = (q * (torch.log(q) - torch.log(m + 1e-12))).sum(-1)
    js = (0.5 * kl_pm + 0.5 * kl_qm).mean()
    bce = F.binary_cross_entropy_with_logits(logits, unnorm_target)
    return js + bce


def femur_integral_loss(logits: torch.Tensor, gt_pts: torch.Tensor, S: int, sigma: float = 4.0):
    """Row R2 loss: 2-channel heatmap loss + order-invariant integral soft-argmax L1 loss."""
    B = logits.shape[0]
    pred_pts = soft_argmax_2d(logits)  # (B, 2, 2)

    # Order-invariant coordinate loss
    d0 = (pred_pts[:, 0] - gt_pts[:, 0]).abs().sum(-1) + (pred_pts[:, 1] - gt_pts[:, 1]).abs().sum(-1)
    d1 = (pred_pts[:, 0] - gt_pts[:, 1]).abs().sum(-1) + (pred_pts[:, 1] - gt_pts[:, 0]).abs().sum(-1)
    use_swap = (d1 < d0)
    coord_loss = torch.minimum(d0, d1).mean() / 10.0

    # 2-channel heatmap supervision aligned with best permutation
    scaled_sigma = sigma * S / 384.0
    _, t_p1 = generate_single_landmark_heatmap(gt_pts[:, 0:1], S, scaled_sigma)
    _, t_p2 = generate_single_landmark_heatmap(gt_pts[:, 1:2], S, scaled_sigma)

    target_ch0 = torch.where(use_swap[:, None, None, None], t_p2, t_p1)
    target_ch1 = torch.where(use_swap[:, None, None, None], t_p1, t_p2)

    bce0 = F.binary_cross_entropy_with_logits(logits[:, 0:1], target_ch0)
    bce1 = F.binary_cross_entropy_with_logits(logits[:, 1:2], target_ch1)
    heatmap_loss = bce0 + bce1

    return heatmap_loss + 0.1 * coord_loss


def femur_boneline_loss(logits: torch.Tensor, gt_pts: torch.Tensor, S: int, sigma: float = 4.0):
    """Row R3 loss: dual-peak heatmap (ch0) + bone line (ch1) + length loss."""
    # Channel 0: Dual peak heatmap
    hm_loss = femur_heatmap_loss(logits[:, 0:1], gt_pts, S=S, sigma=sigma)

    # Channel 1: Auxiliary Gaussian bone line
    bl_target = generate_femur_boneline(gt_pts, S=S, sigma=3.0)
    bl_loss = F.binary_cross_entropy_with_logits(logits[:, 1:2], bl_target)

    # Differentiable length loss via soft-argmax around local peaks of ch0
    prob0 = torch.softmax(logits[:, 0:1].reshape(logits.shape[0], -1), dim=-1).view_as(logits[:, 0:1])
    # Compute soft length from bone line mass (integral of Gaussian line is proportional to length)
    pred_bone_prob = torch.sigmoid(logits[:, 1:2])
    pred_len_px = pred_bone_prob.sum(dim=(-2, -1)).squeeze(-1) / (math.sqrt(2.0 * math.pi) * 3.0)
    gt_len_px = (gt_pts[:, 1] - gt_pts[:, 0]).norm(dim=-1)
    len_loss = F.l1_loss(pred_len_px, gt_len_px) / 10.0

    return hm_loss + bl_loss + 0.2 * len_loss


# ---------------------------------------------------------------------------
# Prediction & Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_batch(
    model: nn.Module,
    x: torch.Tensor,
    row_key: str,
    tta: bool = True,
    device: str = "cpu",
    S: int = 384,
):
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    x = x.to(device)

    if row_key in ["row_A", "row_R4"]:
        with torch.autocast(device_type=device_type, enabled=(device_type == "cuda")):
            lo0 = model(x)
            if tta:
                lo1 = model(torch.flip(x, dims=[3]))
                lo = 0.5 * (lo0 + torch.flip(lo1, dims=[3]))
            else:
                lo = lo0
        return decode_femur_peaks(lo, S=S, min_dist=8.0)

    elif row_key == "row_R1":
        # HRNet-W32 with DARK sub-pixel decoding
        with torch.autocast(device_type=device_type, enabled=(device_type == "cuda")):
            lo0 = model(x)
            if tta:
                lo1 = model(torch.flip(x, dims=[3]))
                lo = 0.5 * (lo0 + torch.flip(lo1, dims=[3]))
            else:
                lo = lo0
        return decode_femur_peaks_dark(lo, S=S, min_dist=8.0)

    elif row_key == "row_R2":
        # Row-A network with integral soft-argmax decoding
        with torch.autocast(device_type=device_type, enabled=(device_type == "cuda")):
            lo0 = model(x)
            pts0 = soft_argmax_2d(lo0)
            if not tta:
                return pts0
            lo1 = model(torch.flip(x, dims=[3]))
            pts1 = soft_argmax_2d(lo1)
            pts1_back = pts1.clone()
            pts1_back[..., 0] = (x.shape[-1] - 1) - pts1_back[..., 0]
            matched_1, _ = match_endpoints(pts1_back, pts0)
            return 0.5 * (pts0 + matched_1)

    elif row_key == "row_R3":
        # Bone line auxiliary + min_dist=50.0 px (strictly chosen from TRAIN split)
        with torch.autocast(device_type=device_type, enabled=(device_type == "cuda")):
            lo0 = model(x)
            if tta:
                lo1 = model(torch.flip(x, dims=[3]))
                lo = 0.5 * (lo0 + torch.flip(lo1, dims=[3]))
            else:
                lo = lo0
        return decode_femur_peaks(lo[:, 0:1], S=S, min_dist=50.0)

    else:
        raise ValueError(f"Unknown row_key: {row_key}")


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, mmpp: torch.Tensor):
    matched, _ = match_endpoints(pred, gt)
    d = (matched - gt).norm(dim=-1)

    fl_px_true = (gt[:, 1] - gt[:, 0]).norm(dim=-1)
    fl_px_pred = (matched[:, 1] - matched[:, 0]).norm(dim=-1)

    fl_true = fl_px_true * mmpp
    fl_pred = fl_px_pred * mmpp

    fl_err = (fl_pred - fl_true).abs()
    fl_signed = fl_pred - fl_true
    pct_err = (fl_err / torch.clamp(fl_true, min=1e-4)) * 100.0
    nme = d.mean(-1) / torch.clamp(fl_px_true, min=1e-4)

    return {
        "fl_true": fl_true, "fl_pred": fl_pred, "fl_err": fl_err,
        "fl_signed": fl_signed, "fl_pct_err": pct_err, "nme": nme,
        "lm_mm": d.mean(-1) * mmpp,
    }


def evaluate_dataframe(model, df, row_key, S=384, tta=True, bs=16, device="cpu"):
    ds = FemurKaggleDS(df, S=S, is_train=False, strong_aug=False)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2)
    model.eval()
    records = []

    with torch.no_grad():
        for x, gt, mmpp, idx_t in dl:
            x, gt, mmpp = x.to(device), gt.to(device), mmpp.to(device)
            pred = predict_batch(model, x, row_key=row_key, tta=tta, device=device, S=S)
            m = compute_metrics(pred, gt, mmpp)

            for j in range(len(idx_t)):
                orig_idx = int(idx_t[j].item())
                orig_row = df.iloc[orig_idx]
                records.append({
                    "image_name": orig_row["image_name"],
                    "patient": orig_row["patient"],
                    "source": orig_row["source"],
                    "official_split": orig_row["official_split"],
                    "px_to_mm_rate": float(orig_row["px_to_mm_rate"]),
                    "mmpp": float(mmpp[j].item()),
                    "fl_true": float(m["fl_true"][j].item()),
                    "fl_pred": float(m["fl_pred"][j].item()),
                    "fl_err": float(m["fl_err"][j].item()),
                    "fl_signed": float(m["fl_signed"][j].item()),
                    "fl_pct_err": float(m["fl_pct_err"][j].item()),
                    "nme": float(m["nme"][j].item()),
                    "lm_mm": float(m["lm_mm"][j].item()),
                    "gt_p1_x": float(gt[j, 0, 0].item()),
                    "gt_p1_y": float(gt[j, 0, 1].item()),
                    "gt_p2_x": float(gt[j, 1, 0].item()),
                    "gt_p2_y": float(gt[j, 1, 1].item()),
                    "pred_p1_x": float(pred[j, 0, 0].item()),
                    "pred_p1_y": float(pred[j, 0, 1].item()),
                    "pred_p2_x": float(pred[j, 1, 0].item()),
                    "pred_p2_y": float(pred[j, 1, 1].item()),
                })
    return pd.DataFrame(records)


def patient_bootstrap_ci(df: pd.DataFrame, metric_col: str, agg_func: str = "mean", n_resamples: int = 1000, seed: int = 42):
    patients = np.array(sorted(df["patient"].unique()))
    n_pats = len(patients)
    if n_pats < 2:
        val = float(df[metric_col].mean()) if agg_func == "mean" else float(df[metric_col].median())
        return val, val

    grouped = df.groupby("patient").indices
    rng = np.random.RandomState(seed)

    boot_stats = []
    for _ in range(n_resamples):
        sample_pats = rng.choice(patients, size=n_pats, replace=True)
        sampled_indices = np.concatenate([grouped[p] for p in sample_pats])
        sample_vals = df[metric_col].iloc[sampled_indices].to_numpy()

        if agg_func == "mean":
            stat = float(np.mean(sample_vals))
        elif agg_func == "median":
            stat = float(np.median(sample_vals))
        elif agg_func == "rmse":
            stat = float(np.sqrt(np.mean(sample_vals ** 2)))
        else:
            raise ValueError(f"Unknown agg_func: {agg_func}")
        boot_stats.append(stat)

    return float(np.percentile(boot_stats, 2.5)), float(np.percentile(boot_stats, 97.5))


def patient_paired_diff_bootstrap(df_row: pd.DataFrame, df_base: pd.DataFrame, n_resamples: int = 1000, seed: int = 42):
    """Compute patient-level paired bootstrap difference in FL MAE: (row - base)."""
    assert len(df_row) == len(df_base)
    diff = df_row["fl_err"].to_numpy() - df_base["fl_err"].to_numpy()
    patients = np.array(sorted(df_row["patient"].unique()))
    n_pats = len(patients)
    grouped = df_row.groupby("patient").indices
    rng = np.random.RandomState(seed)

    boot_diffs = []
    for _ in range(n_resamples):
        sample_pats = rng.choice(patients, size=n_pats, replace=True)
        sampled_indices = np.concatenate([grouped[p] for p in sample_pats])
        boot_diffs.append(float(np.mean(diff[sampled_indices])))

    mean_diff = float(np.mean(diff))
    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))
    return mean_diff, ci_low, ci_high


def summarize_metrics(df: pd.DataFrame, label: str = "eval"):
    valid_df = df[df["fl_true"].notna() & df["fl_pred"].notna()].copy()
    fl_err = valid_df["fl_err"].to_numpy()
    fl_signed = valid_df["fl_signed"].to_numpy()
    fl_pct = valid_df["fl_pct_err"].to_numpy()
    nme = valid_df["nme"].to_numpy()

    mae = float(np.mean(fl_err))
    rmse = float(np.sqrt(np.mean(fl_err ** 2)))
    med = float(np.median(fl_err))
    bias = float(np.mean(fl_signed))
    pct_err = float(np.mean(fl_pct))
    nme_mean = float(np.mean(nme))
    nme_sd = float(np.std(nme))

    sq_err = fl_err ** 2
    top10_share = float(np.sort(sq_err)[-10:].sum() / sq_err.sum()) if len(sq_err) > 0 and sq_err.sum() > 0 else 0.0

    mae_ci = list(patient_bootstrap_ci(valid_df, "fl_err", "mean"))
    rmse_ci = list(patient_bootstrap_ci(valid_df, "fl_err", "rmse"))
    bias_ci = list(patient_bootstrap_ci(valid_df, "fl_signed", "mean"))
    med_ci = list(patient_bootstrap_ci(valid_df, "fl_err", "median"))
    pct_ci = list(patient_bootstrap_ci(valid_df, "fl_pct_err", "mean"))

    return {
        "label": label,
        "n_images": len(valid_df),
        "n_patients": valid_df["patient"].nunique(),
        "fl_mae": mae,
        "fl_mae_ci": mae_ci,
        "fl_rmse": rmse,
        "fl_rmse_ci": rmse_ci,
        "fl_median": med,
        "fl_median_ci": med_ci,
        "fl_bias": bias,
        "fl_bias_ci": bias_ci,
        "fl_pct_error": pct_err,
        "fl_pct_error_ci": pct_ci,
        "fl_nme_mean": nme_mean,
        "fl_nme_sd": nme_sd,
        "top10_sq_err_share": top10_share,
    }


def ensemble_3seeds(df42: pd.DataFrame, df43: pd.DataFrame, df44: pd.DataFrame):
    assert len(df42) == len(df43) == len(df44)

    p42 = torch.tensor(df42[["pred_p1_x", "pred_p1_y", "pred_p2_x", "pred_p2_y"]].to_numpy().reshape(-1, 2, 2))
    p43 = torch.tensor(df43[["pred_p1_x", "pred_p1_y", "pred_p2_x", "pred_p2_y"]].to_numpy().reshape(-1, 2, 2))
    p44 = torch.tensor(df44[["pred_p1_x", "pred_p1_y", "pred_p2_x", "pred_p2_y"]].to_numpy().reshape(-1, 2, 2))

    p43_aligned, _ = match_endpoints(p43, p42)
    p44_aligned, _ = match_endpoints(p44, p42)

    stacked = torch.stack([p42, p43_aligned, p44_aligned], dim=0)
    ens_pts = torch.median(stacked, dim=0).values

    gt_pts = torch.tensor(df42[["gt_p1_x", "gt_p1_y", "gt_p2_x", "gt_p2_y"]].to_numpy().reshape(-1, 2, 2))
    mmpp = torch.tensor(df42["mmpp"].to_numpy())

    m = compute_metrics(ens_pts, gt_pts, mmpp)

    res_df = df42.copy()
    res_df["pred_p1_x"] = ens_pts[:, 0, 0].numpy()
    res_df["pred_p1_y"] = ens_pts[:, 0, 1].numpy()
    res_df["pred_p2_x"] = ens_pts[:, 1, 0].numpy()
    res_df["pred_p2_y"] = ens_pts[:, 1, 1].numpy()
    res_df["fl_pred"] = m["fl_pred"].numpy()
    res_df["fl_err"] = m["fl_err"].numpy()
    res_df["fl_signed"] = m["fl_signed"].numpy()
    res_df["fl_pct_err"] = m["fl_pct_err"].numpy()
    res_df["nme"] = m["nme"].numpy()
    res_df["lm_mm"] = m["lm_mm"].numpy()

    return res_df


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------
def train_single_model(
    row_key: str,
    seed: int,
    tr_df: pd.DataFrame,
    dv_df: pd.DataFrame,
    te_df: pd.DataFrame,
    S: int,
    device: str = "cuda",
    max_epochs: int = 35,
    patience: int = 8,
    batch_size: int = 8,
    lr: float = 2e-4,
    out_dir: Path = Path("ckpts"),
):
    print(f"\n>>> Starting {row_key} (Seed {seed}, S={S})")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / f"{row_key}_s{seed}_best.pt"

    # Instantiate model
    if row_key in ["row_A", "row_R4"]:
        model = build_femur_heatmap_model(encoder_weights="imagenet", classes=1).to(device)
    elif row_key == "row_R1":
        model = build_femur_hrnet_model(pretrained=True, classes=1).to(device)
    elif row_key in ["row_R2", "row_R3"]:
        model = build_femur_heatmap_model(encoder_weights="imagenet", classes=2).to(device)
    else:
        raise ValueError(f"Unknown row_key: {row_key}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
    scaler = torch.amp.GradScaler(device, enabled=("cuda" in str(device)))

    is_strong_aug = (row_key == "row_R4")
    ds_tr = FemurKaggleDS(tr_df, S=S, is_train=True, strong_aug=is_strong_aug)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    best_val_score = float("inf")
    bad_epochs = 0
    t0_train = time.time()
    epochs_trained = 0

    for ep in range(max_epochs):
        t0_ep = time.time()
        model.train()
        train_loss = 0.0

        for x, gt, mmpp, _ in dl_tr:
            x, gt, mmpp = x.to(device), gt.to(device), mmpp.to(device)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=("cuda" in str(device))):
                logits = model(x)
                if row_key in ["row_A", "row_R1", "row_R4"]:
                    loss = femur_heatmap_loss(logits, gt, S=S)
                elif row_key == "row_R2":
                    loss = femur_integral_loss(logits, gt, S=S)
                elif row_key == "row_R3":
                    loss = femur_boneline_loss(logits, gt, S=S)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            train_loss += loss.item()

        epochs_trained = ep + 1
        avg_loss = train_loss / len(dl_tr)

        # Validation on the 65 validation images only
        val_res = evaluate_dataframe(model, dv_df, row_key=row_key, S=S, tta=False, device=device)
        val_sum = summarize_metrics(val_res, "val")
        val_score = val_sum["fl_mae"]

        sched.step(val_score)

        improved = val_score < best_val_score
        if improved:
            best_val_score = val_score
            bad_epochs = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            bad_epochs += 1

        ep_sec = time.time() - t0_ep
        star = " *" if improved else ""
        print(f"[{row_key} s{seed}] Ep {ep+1:02d}/{max_epochs} | loss: {avg_loss:.4f} | val MAE: {val_score:.3f} mm, NME: {val_sum['fl_nme_mean']:.4f} | {ep_sec:.1f}s{star}")

        if bad_epochs >= patience:
            print(f"[{row_key} s{seed}] Early stopping triggered at epoch {ep+1}")
            break

    train_time_sec = time.time() - t0_train
    print(f"[{row_key} s{seed}] Training finished in {train_time_sec:.1f}s ({epochs_trained} epochs). Best val MAE: {best_val_score:.3f} mm. Loading best ckpt...")
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))

    # Test evaluation on 348 test images with flip TTA
    test_res_df = evaluate_dataframe(model, te_df, row_key=row_key, S=S, tta=True, device=device)
    metrics_all = summarize_metrics(test_res_df, "test_all")
    metrics_fp = summarize_metrics(test_res_df[test_res_df["source"] == "FP"], "test_fp")
    metrics_ucl = summarize_metrics(test_res_df[test_res_df["source"] == "UCL"], "test_ucl")

    metrics_all["epochs"] = epochs_trained
    metrics_all["train_time_sec"] = train_time_sec
    metrics_all["val_mae"] = best_val_score
    metrics_all["fp_mae"] = metrics_fp["fl_mae"]
    metrics_all["ucl_mae"] = metrics_ucl["fl_mae"]

    return model, test_res_df, metrics_all, metrics_fp, metrics_ucl


# ---------------------------------------------------------------------------
# Format Markdown Table
# ---------------------------------------------------------------------------
def format_round2_markdown_table(results: dict, winning_key: str) -> str:
    lines = [
        "# Femur Round 2: Modern Techniques (FP + UCL Femur Test Set, N=348 images / 266 patients)",
        "",
        "> Reference FL: Euclidean distance between annotated endpoints * px_to_mm_rate.",
        "> Protocol: Train on 451 images, early stop on 65 validation images (0 patient overlap).",
        "> Evaluation: 348 official test images (323 FP / 261 patients, 25 UCL / 5 patients [tiny]).",
        "> Paired Diff: Patient-level bootstrap with 1,000 resamples of difference in FL MAE vs Row A.",
        "> Selection: Best recipe chosen strictly by validation MAE only.",
        "",
        "| Approach | Row / Model | Resolution | Val MAE (mm) | Epochs | Train Time (s) | FL MAE [95% CI] (mm) | Paired Diff vs Row A [95% CI] (mm) | FL RMSE [95% CI] (mm) | FL Median [95% CI] (mm) | % Error [95% CI] | FL NME (mean+-SD) | Top-10 SqErr Share (%) | FP MAE (mm) | UCL MAE (mm, tiny) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    row_names = [
        ("Row A (Baseline EffNet-B4 1-ch Heatmap, flip TTA)", "row_A"),
        ("Row R1 (HRNet-W32 Heatmap, DARK Sub-pixel)", "row_R1"),
        ("Row R2 (Row-A Net, Integral Soft-Argmax Loss)", "row_R2"),
        ("Row R3 (Row-A Net, Bone Line + Length Loss, MinDist=50px)", "row_R3"),
        ("Row R4 (Row-A Net + Strong Ultrasound Aug)", "row_R4"),
        ("Row R6 (Ensemble of Row A + Best 2 Val Rows)", "row_R6"),
    ]

    for display_name, row_key in row_names:
        if row_key not in results:
            continue
        row_data = results[row_key]
        res_str = f"{row_data['resolution']} px"
        m_sd = row_data["seeds_mean_sd"]
        ens = row_data["ensemble_all"]
        ens_fp = row_data["ensemble_fp"]
        ens_ucl = row_data["ensemble_ucl"]
        paired = row_data["paired_vs_row_A"]

        val_mae_str = f"{m_sd['val_mae_mean']:.3f}" if "val_mae_mean" in m_sd else "-"
        paired_str = (
            f"**{paired['diff_mae']:.3f}** [{paired['ci_low']:.3f}, {paired['ci_high']:.3f}]"
            if row_key != "row_A"
            else "Reference"
        )
        if row_key != "row_A" and paired["ci_low"] <= 0.0 <= paired["ci_high"]:
            paired_str += " *(no measurable difference)*"

        # Seeds Mean +- SD row
        ep_str = f"{m_sd['epochs_mean']:.1f} +- {m_sd['epochs_sd']:.1f}" if "epochs_mean" in m_sd else "-"
        time_str = f"{m_sd['time_mean']:.1f} +- {m_sd['time_sd']:.1f}" if "time_mean" in m_sd else "-"
        mae_ms = f"{m_sd['fl_mae_mean']:.3f} +- {m_sd['fl_mae_sd']:.3f}" if "fl_mae_mean" in m_sd else "-"
        rmse_ms = f"{m_sd['fl_rmse_mean']:.3f} +- {m_sd['fl_rmse_sd']:.3f}" if "fl_rmse_mean" in m_sd else "-"
        med_ms = f"{m_sd['fl_med_mean']:.3f} +- {m_sd['fl_med_sd']:.3f}" if "fl_med_mean" in m_sd else "-"
        pct_ms = f"{m_sd['fl_pct_mean']:.2f}% +- {m_sd['fl_pct_sd']:.2f}%" if "fl_pct_mean" in m_sd else "-"
        nme_ms = f"{m_sd['fl_nme_mean']:.3f} +- {m_sd['fl_nme_sd']:.3f}" if "fl_nme_mean" in m_sd else "-"
        top10_ms = f"{m_sd['top10_mean']*100:.1f}% +- {m_sd['top10_sd']*100:.1f}%" if "top10_mean" in m_sd else "-"
        fp_ms = f"{m_sd['fp_mae_mean']:.3f}" if "fp_mae_mean" in m_sd else "-"
        ucl_ms = f"{m_sd['ucl_mae_mean']:.3f}" if "ucl_mae_mean" in m_sd else "-"

        prefix = "★ " if row_key == winning_key else ""

        if row_key != "row_R6":
            lines.append(
                f"| {prefix}{display_name} | *Seeds Mean +- SD* | {res_str} | {val_mae_str} | {ep_str} | {time_str} | "
                f"{mae_ms} | - | {rmse_ms} | {med_ms} | {pct_ms} | {nme_ms} | {top10_ms} | {fp_ms} | {ucl_ms} |"
            )

        # Ensemble row
        e_mae = f"**{ens['fl_mae']:.3f}** [{ens['fl_mae_ci'][0]:.3f}, {ens['fl_mae_ci'][1]:.3f}]"
        e_rmse = f"**{ens['fl_rmse']:.3f}** [{ens['fl_rmse_ci'][0]:.3f}, {ens['fl_rmse_ci'][1]:.3f}]"
        e_med = f"**{ens['fl_median']:.3f}** [{ens['fl_median_ci'][0]:.3f}, {ens['fl_median_ci'][1]:.3f}]"
        e_pct = f"**{ens['fl_pct_error']:.2f}%** [{ens['fl_pct_error_ci'][0]:.2f}%, {ens['fl_pct_error_ci'][1]:.2f}%]"
        e_nme = f"**{ens['fl_nme_mean']:.3f} +- {ens['fl_nme_sd']:.3f}**"
        e_top10 = f"**{ens['top10_sq_err_share']*100:.1f}%**"
        e_fp = f"**{ens_fp['fl_mae']:.3f}**"
        e_ucl = f"**{ens_ucl['fl_mae']:.3f}**"

        ens_label = "**3-Seed Median Ens**" if row_key != "row_R6" else "**9-Model Multi-Row Ens**"
        lines.append(
            f"| {prefix}{display_name} | {ens_label} | {res_str} | {val_mae_str} | - | - | "
            f"{e_mae} | {paired_str} | {e_rmse} | {e_med} | {e_pct} | {e_nme} | {e_top10} | {e_fp} | {e_ucl} |"
        )

    # Note on R5
    lines.extend([
        "",
        "### Pretrained Encoder Check (Row R5):",
        "- **DINOv2 ViT-S/14**: Apache 2.0 license, publicly downloadable weights from Facebook Research hub. Incompatible with standard 384px resolution because patch size is 14 (`384 % 14 != 0`), violating the standardized letterbox resolution protocol. Skipped.",
        "- **UltraSam (arXiv 2411.16222)**: Derivative of SAM requiring external heavy checkpoint hosting and 1024px architecture. Skipped due to lack of standard PyTorch distribution.",
    ])

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def run_round2_experiment():
    print("=" * 80)
    print("FEMUR ROUND 2: MODERN TECHNIQUES EXPERIMENT (Rows A, R1, R2, R3, R4, R6)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Locate dataset dynamically
    raw_root = None
    if Path("/kaggle/input").is_dir():
        for p in Path("/kaggle/input").rglob("*.zip"):
            if "femur" in p.name.lower():
                raw_root = p.parent
                break
        if raw_root is None:
            for p in Path("/kaggle/input").rglob("Femur.csv"):
                for parent in p.parents:
                    if (parent / "FP").is_dir() and (parent / "UCL").is_dir():
                        raw_root = parent
                        break
                if raw_root is not None:
                    break

    if raw_root is None:
        for cand in [Path("C:/fetal-data"), Path("../data"), Path("/tmp/femur_data")]:
            if cand.is_dir():
                raw_root = cand
                break

    if raw_root is None:
        raise FileNotFoundError("Could not find femur dataset in /kaggle/input or local paths.")

    zip_candidates = list(raw_root.glob("*.zip"))
    work_data = Path("/tmp/femur_data")
    if zip_candidates:
        print(f"Unpacking {zip_candidates[0]} to {work_data}...")
        work_data.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_candidates[0], "r") as zf:
            zf.extractall(work_data)
        data_dir = work_data
    elif (raw_root / "FP").is_dir():
        data_dir = raw_root
    else:
        raise FileNotFoundError(f"No FP folder or zip found in {raw_root}")

    print(f"Using data directory: {data_dir}")
    df = build_femur_data_from_root(data_dir)
    assert len(df) == 864, f"Expected 864 calibrated femur images, got {len(df)}"

    tr_df, dv_df, te_df = split_femur_data(df, dev_fraction=0.12, seed=0)
    print(f"Splits: Train = {len(tr_df)} ({tr_df['patient'].nunique()} pats), Val = {len(dv_df)} ({dv_df['patient'].nunique()} pats), Test = {len(te_df)} ({te_df['patient'].nunique()} pats)")

    out_dir = Path("/kaggle/working/out") if Path("/kaggle").is_dir() else Path("docs/results/femur_round2")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("/kaggle/working/ckpts") if Path("/kaggle").is_dir() else Path("ckpts_femur_r2")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rows_config = [
        ("row_A", "Row A (EffNet-B4 1-ch Heatmap, 384px)", 384),
        ("row_R1", "Row R1 (HRNet-W32 Heatmap, DARK Decode, 384px)", 384),
        ("row_R2", "Row R2 (Row-A Net, Integral Soft-Argmax, 384px)", 384),
        ("row_R3", "Row R3 (Row-A Net, Bone Line + Length Loss, 384px)", 384),
        ("row_R4", "Row R4 (Row-A Net + Strong Ultrasound Aug, 384px)", 384),
    ]

    seeds = [42, 43, 44]
    all_results = {}
    pred_dfs_by_row = {}
    val_maes_by_row = {}

    for row_key, row_name, S in rows_config:
        print("\n" + "#" * 70)
        print(f"STARTING {row_name}")
        print("#" * 70)

        seed_dfs = {}
        seed_metrics = {}

        for seed in seeds:
            _, test_pred_df, m_all, m_fp, m_ucl = train_single_model(
                row_key=row_key,
                seed=seed,
                tr_df=tr_df,
                dv_df=dv_df,
                te_df=te_df,
                S=S,
                device=device,
                max_epochs=35,
                patience=8,
                batch_size=8,
                lr=2e-4,
                out_dir=ckpt_dir,
            )
            seed_dfs[seed] = test_pred_df
            seed_metrics[f"seed_{seed}"] = {
                "all": m_all,
                "fp": m_fp,
                "ucl": m_ucl,
            }

        # 3-seed Mean +- SD
        mae_vals = [seed_metrics[f"seed_{s}"]["all"]["fl_mae"] for s in seeds]
        rmse_vals = [seed_metrics[f"seed_{s}"]["all"]["fl_rmse"] for s in seeds]
        med_vals = [seed_metrics[f"seed_{s}"]["all"]["fl_median"] for s in seeds]
        pct_vals = [seed_metrics[f"seed_{s}"]["all"]["fl_pct_error"] for s in seeds]
        nme_vals = [seed_metrics[f"seed_{s}"]["all"]["fl_nme_mean"] for s in seeds]
        top10_vals = [seed_metrics[f"seed_{s}"]["all"]["top10_sq_err_share"] for s in seeds]
        ep_vals = [seed_metrics[f"seed_{s}"]["all"]["epochs"] for s in seeds]
        time_vals = [seed_metrics[f"seed_{s}"]["all"]["train_time_sec"] for s in seeds]
        val_mae_vals = [seed_metrics[f"seed_{s}"]["all"]["val_mae"] for s in seeds]
        fp_maes = [seed_metrics[f"seed_{s}"]["fp"]["fl_mae"] for s in seeds]
        ucl_maes = [seed_metrics[f"seed_{s}"]["ucl"]["fl_mae"] for s in seeds]

        mean_val_mae = float(np.mean(val_mae_vals))
        val_maes_by_row[row_key] = mean_val_mae

        mean_sd_stats = {
            "val_mae_mean": mean_val_mae,
            "val_mae_sd": float(np.std(val_mae_vals)),
            "fl_mae_mean": float(np.mean(mae_vals)),
            "fl_mae_sd": float(np.std(mae_vals)),
            "fl_rmse_mean": float(np.mean(rmse_vals)),
            "fl_rmse_sd": float(np.std(rmse_vals)),
            "fl_med_mean": float(np.mean(med_vals)),
            "fl_med_sd": float(np.std(med_vals)),
            "fl_pct_mean": float(np.mean(pct_vals)),
            "fl_pct_sd": float(np.std(pct_vals)),
            "fl_nme_mean": float(np.mean(nme_vals)),
            "fl_nme_sd": float(np.std(nme_vals)),
            "top10_mean": float(np.mean(top10_vals)),
            "top10_sd": float(np.std(top10_vals)),
            "epochs_mean": float(np.mean(ep_vals)),
            "epochs_sd": float(np.std(ep_vals)),
            "time_mean": float(np.mean(time_vals)),
            "time_sd": float(np.std(time_vals)),
            "fp_mae_mean": float(np.mean(fp_maes)),
            "ucl_mae_mean": float(np.mean(ucl_maes)),
        }

        # 3-Seed Median Ensemble
        ens_df = ensemble_3seeds(seed_dfs[42], seed_dfs[43], seed_dfs[44])
        pred_dfs_by_row[row_key] = ens_df

        ens_all = summarize_metrics(ens_df, f"{row_key}_ens_all")
        ens_fp = summarize_metrics(ens_df[ens_df["source"] == "FP"], f"{row_key}_ens_fp")
        ens_ucl = summarize_metrics(ens_df[ens_df["source"] == "UCL"], f"{row_key}_ens_ucl")

        all_results[row_key] = {
            "name": row_name,
            "resolution": S,
            "seeds": seed_metrics,
            "seeds_mean_sd": mean_sd_stats,
            "ensemble_all": ens_all,
            "ensemble_fp": ens_fp,
            "ensemble_ucl": ens_ucl,
        }

    # Selection: pick 2 best rows among R1-R4 by validation MAE only
    cand_rows = ["row_R1", "row_R2", "row_R3", "row_R4"]
    sorted_cands = sorted(cand_rows, key=lambda rk: val_maes_by_row[rk])
    best_row1, best_row2 = sorted_cands[0], sorted_cands[1]
    print(f"\nSelection by validation MAE only: Best 1 = {best_row1} ({val_maes_by_row[best_row1]:.3f} mm), Best 2 = {best_row2} ({val_maes_by_row[best_row2]:.3f} mm)")

    # Build R6: Ensemble of Row A + best 2 rows
    print("\nForming R6 Ensemble (Row A + " + best_row1 + " + " + best_row2 + ")...")
    ens_r6_df = pred_dfs_by_row["row_A"].copy()
    p_a = torch.tensor(pred_dfs_by_row["row_A"][["pred_p1_x", "pred_p1_y", "pred_p2_x", "pred_p2_y"]].to_numpy().reshape(-1, 2, 2))
    p_b1 = torch.tensor(pred_dfs_by_row[best_row1][["pred_p1_x", "pred_p1_y", "pred_p2_x", "pred_p2_y"]].to_numpy().reshape(-1, 2, 2))
    p_b2 = torch.tensor(pred_dfs_by_row[best_row2][["pred_p1_x", "pred_p1_y", "pred_p2_x", "pred_p2_y"]].to_numpy().reshape(-1, 2, 2))

    p_b1_aligned, _ = match_endpoints(p_b1, p_a)
    p_b2_aligned, _ = match_endpoints(p_b2, p_a)
    r6_pts = torch.median(torch.stack([p_a, p_b1_aligned, p_b2_aligned], dim=0), dim=0).values

    gt_pts = torch.tensor(ens_r6_df[["gt_p1_x", "gt_p1_y", "gt_p2_x", "gt_p2_y"]].to_numpy().reshape(-1, 2, 2))
    mmpp = torch.tensor(ens_r6_df["mmpp"].to_numpy())
    m_r6 = compute_metrics(r6_pts, gt_pts, mmpp)

    ens_r6_df["pred_p1_x"] = r6_pts[:, 0, 0].numpy()
    ens_r6_df["pred_p1_y"] = r6_pts[:, 0, 1].numpy()
    ens_r6_df["pred_p2_x"] = r6_pts[:, 1, 0].numpy()
    ens_r6_df["pred_p2_y"] = r6_pts[:, 1, 1].numpy()
    ens_r6_df["fl_pred"] = m_r6["fl_pred"].numpy()
    ens_r6_df["fl_err"] = m_r6["fl_err"].numpy()
    ens_r6_df["fl_signed"] = m_r6["fl_signed"].numpy()
    ens_r6_df["fl_pct_err"] = m_r6["fl_pct_err"].numpy()
    ens_r6_df["nme"] = m_r6["nme"].numpy()
    ens_r6_df["lm_mm"] = m_r6["lm_mm"].numpy()
    pred_dfs_by_row["row_R6"] = ens_r6_df

    r6_all = summarize_metrics(ens_r6_df, "row_R6_ens_all")
    r6_fp = summarize_metrics(ens_r6_df[ens_r6_df["source"] == "FP"], "row_R6_ens_fp")
    r6_ucl = summarize_metrics(ens_r6_df[ens_r6_df["source"] == "UCL"], "row_R6_ens_ucl")

    all_results["row_R6"] = {
        "name": f"Row R6 (Ensemble of Row A + {best_row1} + {best_row2})",
        "resolution": 384,
        "seeds": {},
        "seeds_mean_sd": {"val_mae_mean": (val_maes_by_row["row_A"] + val_maes_by_row[best_row1] + val_maes_by_row[best_row2]) / 3.0},
        "ensemble_all": r6_all,
        "ensemble_fp": r6_fp,
        "ensemble_ucl": r6_ucl,
    }

    # Compute patient-level paired bootstrap difference vs Row A for every row
    base_df = pred_dfs_by_row["row_A"]
    for rk in all_results:
        diff_mean, ci_low, ci_high = patient_paired_diff_bootstrap(pred_dfs_by_row[rk], base_df)
        all_results[rk]["paired_vs_row_A"] = {
            "diff_mae": diff_mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_includes_zero": (ci_low <= 0.0 <= ci_high),
        }

    # Pick final recipe by validation MAE only
    winning_key = sorted_cands[0]
    print(f"\nFinal Femur Recipe selected by validation MAE only: {winning_key} ({val_maes_by_row[winning_key]:.3f} mm)")

    # Format Markdown Table
    md_table = format_round2_markdown_table(all_results, winning_key)
    print("\n" + "=" * 80)
    print("FINAL ROUND 2 TABLE:")
    print("=" * 80)
    print(md_table)

    # Save Markdown Table
    table_path = out_dir / "femur_round2_table.md"
    with open(table_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(md_table)
    print(f"Saved {table_path}")

    # Save JSON results
    json_path = out_dir / "femur_round2_results.json"
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {json_path}")

    # Save combined predictions CSV
    combined_csv_path = out_dir / "femur_round2_predictions.csv"
    comb_df = pred_dfs_by_row["row_A"].copy()
    comb_df = comb_df[["image_name", "patient", "source", "official_split", "px_to_mm_rate", "fl_true"]].copy()
    for rk in ["row_A", "row_R1", "row_R2", "row_R3", "row_R4", "row_R6"]:
        comb_df[f"{rk}_fl_pred"] = pred_dfs_by_row[rk]["fl_pred"]
        comb_df[f"{rk}_fl_err"] = pred_dfs_by_row[rk]["fl_err"]
    comb_df.to_csv(combined_csv_path, index=False, lineterminator="\n")
    print(f"Saved {combined_csv_path}")

    # Generate and execute benchmark notebook
    nb_dir = Path("/kaggle/working") if Path("/kaggle").is_dir() else Path("notebooks/benchmarks")
    nb_file = nb_dir / "femur_round2.ipynb"

    import nbformat as nbf
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Femur Round 2: Modern Techniques Benchmark\n\n"
            "Evaluates modern techniques on FP+UCL Femur test set (N=348 images / 266 patients):\n"
            "- **Row A**: Baseline U-Net EffNet-B4 1-ch heatmap (384px, standard peak decode, flip TTA)\n"
            "- **Row R1**: HRNet-W32 heatmap (384px, DARK sub-pixel decode, flip TTA)\n"
            "- **Row R2**: Row-A net with integral soft-argmax coordinate loss, decode by soft-argmax\n"
            "- **Row R3**: Row-A net with auxiliary bone line heatmap + length loss + min_dist=50px decoding\n"
            "- **Row R4**: Row-A net + strong ultrasound-style augmentation\n"
            "- **Row R5**: Pretrained encoder check (DINOv2 / UltraSam) - documented and skipped\n"
            "- **Row R6**: Median ensemble of Row A and top 2 rows by validation MAE only\n\n"
            "Final femur recipe selected strictly by validation MAE only."
        ),
        nbf.v4.new_code_cell(
            "import json\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Markdown, display\n\n"
            "results_dir = Path('/kaggle/working/out')\n"
            "if not results_dir.is_dir():\n"
            "    results_dir = Path('../../docs/results/femur_round2')\n"
            "if not results_dir.is_dir():\n"
            "    results_dir = Path('docs/results/femur_round2')\n\n"
            "table_path = results_dir / 'femur_round2_table.md'\n"
            "if table_path.is_file():\n"
            "    display(Markdown(table_path.read_text(encoding='utf-8')))\n"
            "else:\n"
            "    print('Table not found at', table_path)"
        ),
        nbf.v4.new_code_cell(
            "json_path = results_dir / 'femur_round2_results.json'\n"
            "if json_path.is_file():\n"
            "    with open(json_path, 'r', encoding='utf-8') as f:\n"
            "        results = json.load(f)\n"
            "    print('Loaded round 2 results for keys:', list(results.keys()))\n"
            "    for rk in ['row_A', 'row_R1', 'row_R2', 'row_R3', 'row_R4', 'row_R6']:\n"
            "        if rk in results:\n"
            "            ens = results[rk]['ensemble_all']\n"
            "            ms = results[rk]['seeds_mean_sd']\n"
            "            paired = results[rk].get('paired_vs_row_A', {})\n"
            "            print(f'=== {results[rk][\"name\"]} ===')\n"
            "            print(f'  Val MAE:    {ms.get(\"val_mae_mean\", float(\"nan\")):.3f} mm')\n"
            "            print(f'  Ensemble:   MAE={ens[\"fl_mae\"]:.3f} [{ens[\"fl_mae_ci\"][0]:.3f}, {ens[\"fl_mae_ci\"][1]:.3f}] mm, NME={ens[\"fl_nme_mean\"]:.3f} +- {ens[\"fl_nme_sd\"]:.3f}')\n"
            "            if rk != 'row_A':\n"
            "                diff = paired.get('diff_mae', 0.0)\n"
            "                ci = [paired.get('ci_low', 0.0), paired.get('ci_high', 0.0)]\n"
            "                meas = ' (no measurable difference)' if paired.get('ci_includes_zero') else ''\n"
            "                print(f'  Paired vs A: {diff:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] mm{meas}')\n"
            "else:\n"
            "    print('Results JSON not found at', json_path)"
        ),
        nbf.v4.new_code_cell(
            "csv_path = results_dir / 'femur_round2_predictions.csv'\n"
            "if csv_path.is_file():\n"
            "    df = pd.read_csv(csv_path)\n"
            "    print(f'Total predictions: {len(df)} images across {df[\"patient\"].nunique()} patients')\n"
            "    assert len(df) == 348, f'Expected 348 test images, got {len(df)}'\n"
            "    print('Verification check passed: 348 test predictions confirmed.')\n"
            "else:\n"
            "    print('Predictions CSV not found at', csv_path)"
        ),
    ]
    with open(nb_file, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created notebook at {nb_file}. Executing via nbconvert...")

    for cmd in [
        ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(nb_file)],
        [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(nb_file)],
    ]:
        try:
            print(f"Running {' '.join(cmd)}...")
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if res.returncode == 0:
                print("nbconvert executed successfully!")
                break
            else:
                print(f"nbconvert stderr: {res.stderr[:300]}")
        except Exception as e:
            print(f"nbconvert execution failed with: {e}")

    print("\nExperiment complete!")


if __name__ == "__main__":
    run_round2_experiment()
