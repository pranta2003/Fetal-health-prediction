"""hc18_old_approaches.py: Historical HC18 model architectures and datasets.

Re-implements the 4 historical HC18 model families from the Capstone notebook:
- Row A (Cell 22): Direct Regression (ConvNeXt-Tiny -> [HC, BPD] in mm, normalized MSE loss)
- Row B (Cell 27): Geometry Approach v3 (ConvNeXt-Tiny -> [cx, cy, a, b, angle], Smooth L1 loss)
- Row C (Cell 25): Multi-Task U-Net (ResNet-34 U-Net + shared encoder regression head -> mask & 5 ellipse params)
- Row D (Cell 31): ResNet-34 U-Net pure segmentation (224x224 letterbox, Dice + BCE loss)

Differences from the original Colab cells:
1. Held-out validation set is strictly Fold 1 of `make_grouped_folds(names, 5, seed=42)` (206 outer images).
2. Early stopping and learning rate scheduling use a disjoint inner validation slice
   (10% of training patients, patient-grouped, zero leakage), never the held-out fold.
3. Epoch cap is 100 with fixed seed 42.
4. Uses CSV pixel size directly; Row D uses the repository's robust native-resolution
   biometric evaluation (`predict_native_prob` + `fit_ellipse_robust`).
"""

from __future__ import annotations

import os
from typing import Sequence
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import segmentation_models_pytorch as smp
import timm

from fetal.eval.ellipse_fitting import fit_ellipse_robust, hc_bpd_from_ellipse_params


# ---------------------------------------------------------------------------
# Coordinate and Ellipse Helpers (from cells 20, 23, 25, 27)
# ---------------------------------------------------------------------------

class LetterboxResize:
    """Resizes to fit inside target_size x target_size preserving aspect ratio with black padding."""

    def __init__(self, target_size: int = 224):
        self.target_size = target_size

    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))
        return canvas


class LetterboxWithCoords:
    """Letterbox resize that returns canvas, scale, and padding offsets."""

    def __init__(self, target_size: int = 224):
        self.target_size = target_size

    def __call__(self, image: Image.Image) -> tuple[Image.Image, float, int, int]:
        w, h = image.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))
        return canvas, scale, paste_x, paste_y


def extract_full_ellipse_from_mask(mask_path: str, pixel_size_mm: float) -> tuple[float, float, float, float, float]:
    """Extract ellipse parameters [cx_mm, cy_mm, semi_major_mm, semi_minor_mm, angle] from annotation mask."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask at {mask_path}")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    (cx, cy), (axis1_px, axis2_px), angle = fit_ellipse_robust(largest)
    major_px = max(axis1_px, axis2_px)
    minor_px = min(axis1_px, axis2_px)
    semi_major_mm = (major_px / 2.0) * pixel_size_mm
    semi_minor_mm = (minor_px / 2.0) * pixel_size_mm
    cx_mm = cx * pixel_size_mm
    cy_mm = cy * pixel_size_mm
    return cx_mm, cy_mm, semi_major_mm, semi_minor_mm, float(angle)


def extract_bpd_from_mask(mask_path: str, pixel_size_mm: float) -> float:
    """Extract ground-truth BPD in mm from annotation mask (Cell 20 & 22)."""
    _, _, _, semi_minor_mm, _ = extract_full_ellipse_from_mask(mask_path, pixel_size_mm)
    return float(2.0 * semi_minor_mm)


# ---------------------------------------------------------------------------
# ROW A: Direct Regression (Source: Cell 22)
# ---------------------------------------------------------------------------

class RowADataset(Dataset):
    """Dataset for Row A (Direct Regression to [HC, BPD] in mm). Source: Cell 22."""

    def __init__(self, data: pd.DataFrame, image_dir: str, transform=None):
        self.data = data.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        pixel_size = float(row["pixel size(mm)"])
        hc_val = float(row["head circumference (mm)"])
        mask_filename = row["filename"].replace(".png", "_Annotation.png")
        mask_path = os.path.join(self.image_dir, mask_filename)
        bpd_val = extract_bpd_from_mask(mask_path, pixel_size)

        targets = torch.tensor([hc_val, bpd_val], dtype=torch.float32)
        return image, targets


def build_model_row_a(pretrained: bool = True) -> nn.Module:
    """ConvNeXt-Tiny direct regression model (Source: Cell 22)."""
    return timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=2)


def get_transforms_row_a():
    """Augmentation and validation transforms for Row A (224x224 direct resize)."""
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ---------------------------------------------------------------------------
# ROW B: Geometry Approach v3 (Source: Cell 27)
# ---------------------------------------------------------------------------

class RowBDataset(Dataset):
    """Dataset for Row B (Geometry: 5 normalized ellipse parameters). Source: Cell 27."""

    def __init__(self, data: pd.DataFrame, image_dir: str, transform=None):
        self.data = data.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        pixel_size = float(row["pixel size(mm)"])
        mask_filename = row["filename"].replace(".png", "_Annotation.png")
        mask_path = os.path.join(self.image_dir, mask_filename)
        cx, cy, semi_major, semi_minor, angle = extract_full_ellipse_from_mask(mask_path, pixel_size)
        targets = torch.tensor([cx, cy, semi_major, semi_minor, angle], dtype=torch.float32)
        return image, targets


def build_model_row_b(pretrained: bool = True) -> nn.Module:
    """ConvNeXt-Tiny geometry regression model (Source: Cell 27)."""
    return timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=5)


def get_transforms_row_b():
    """Augmentation and validation transforms for Row B (224x224 Letterbox)."""
    train_tf = transforms.Compose([
        LetterboxResize(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        LetterboxResize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ---------------------------------------------------------------------------
# ROW C: Multi-Task U-Net (Source: Cell 25)
# ---------------------------------------------------------------------------

class MultiTaskUNet(nn.Module):
    """Multi-task U-Net with shared ResNet-34 encoder (Source: Cell 25)."""

    def __init__(self, encoder_name: str = "resnet34", pretrained: bool = True):
        super().__init__()
        weights = "imagenet" if pretrained else None
        self.unet = smp.Unet(encoder_name=encoder_name, encoder_weights=weights, in_channels=3, classes=1)
        in_feat = self.unet.encoder.out_channels[-1]  # 512 for resnet34
        self.reg_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_feat, 256),
            nn.ReLU(),
            nn.Linear(256, 5),  # [cx, cy, semi_major, semi_minor, angle]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.unet.encoder(x)
        decoder_out = self.unet.decoder(features)
        mask_logits = self.unet.segmentation_head(decoder_out)
        geo_preds = self.reg_head(features[-1])
        return mask_logits, geo_preds


class RowCDataset(Dataset):
    """Dataset for Row C (Returns image, binary mask, and 5 ellipse parameters). Source: Cell 25."""

    def __init__(self, data: pd.DataFrame, image_dir: str, transform=None, mask_size: int = 224):
        self.data = data.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.mask_size = mask_size

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")

        pixel_size = float(row["pixel size(mm)"])
        mask_filename = row["filename"].replace(".png", "_Annotation.png")
        mask_path = os.path.join(self.image_dir, mask_filename)

        cx, cy, semi_major, semi_minor, angle = extract_full_ellipse_from_mask(mask_path, pixel_size)
        geo_target = torch.tensor([cx, cy, semi_major, semi_minor, angle], dtype=torch.float32)

        raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if raw_mask is not None:
            contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                filled_mask = np.zeros_like(raw_mask)
                cv2.drawContours(filled_mask, [largest], -1, 255, thickness=cv2.FILLED)
                filled_mask = cv2.resize(filled_mask, (self.mask_size, self.mask_size), interpolation=cv2.INTER_NEAREST)
            else:
                filled_mask = np.zeros((self.mask_size, self.mask_size), dtype=np.uint8)
        else:
            filled_mask = np.zeros((self.mask_size, self.mask_size), dtype=np.uint8)

        mask_tensor = torch.tensor(filled_mask / 255.0, dtype=torch.float32).unsqueeze(0)
        if self.transform:
            image = self.transform(image)

        return image, mask_tensor, geo_target


def build_model_row_c(pretrained: bool = True) -> nn.Module:
    """Multi-task U-Net model (Source: Cell 25)."""
    return MultiTaskUNet(encoder_name="resnet34", pretrained=pretrained)


def get_transforms_row_c():
    """Augmentation and validation transforms for Row C (224x224 direct resize)."""
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ---------------------------------------------------------------------------
# ROW D: ResNet-34 U-Net (Source: Cell 31)
# ---------------------------------------------------------------------------

class RowDDataset(Dataset):
    """Dataset for Row D (Pure segmentation ResNet-34 U-Net, 224x224 Letterbox). Source: Cell 31."""

    def __init__(self, data: pd.DataFrame, image_dir: str, augment: bool = False, mask_size: int = 224):
        self.data = data.reset_index(drop=True)
        self.image_dir = image_dir
        self.augment = augment
        self.mask_size = mask_size
        self.letterbox = LetterboxWithCoords(mask_size)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")

        mask_filename = row["filename"].replace(".png", "_Annotation.png")
        mask_path = os.path.join(self.image_dir, mask_filename)
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if raw_mask is not None:
            contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                filled_mask = np.zeros_like(raw_mask)
                cv2.drawContours(filled_mask, [largest], -1, 255, thickness=cv2.FILLED)
            else:
                filled_mask = np.zeros_like(raw_mask)
        else:
            filled_mask = np.zeros((100, 100), dtype=np.uint8)

        mask_pil = Image.fromarray(filled_mask)
        canvas_img, scale, paste_x, paste_y = self.letterbox(image)
        canvas_mask = Image.new("L", (self.mask_size, self.mask_size), 0)
        w_m, h_m = mask_pil.size
        resized_m = mask_pil.resize((int(w_m * scale), int(h_m * scale)), Image.Resampling.NEAREST)
        canvas_mask.paste(resized_m, (paste_x, paste_y))

        if self.augment:
            if torch.rand(1).item() > 0.5:
                canvas_img = transforms.functional.hflip(canvas_img)
                canvas_mask = transforms.functional.hflip(canvas_mask)
            angle = float(torch.empty(1).uniform_(-15, 15).item())
            canvas_img = transforms.functional.rotate(canvas_img, angle)
            canvas_mask = transforms.functional.rotate(canvas_mask, angle)

        img_tensor = self.normalize(transforms.functional.to_tensor(canvas_img))
        mask_tensor = transforms.functional.to_tensor(canvas_mask)
        pixel_size = float(row["pixel size(mm)"])
        return img_tensor, mask_tensor, pixel_size, scale


def build_model_row_d(pretrained: bool = True) -> nn.Module:
    """ResNet-34 U-Net segmentation model (Source: Cell 31)."""
    weights = "imagenet" if pretrained else None
    return smp.Unet(encoder_name="resnet34", encoder_weights=weights, in_channels=3, classes=1)
