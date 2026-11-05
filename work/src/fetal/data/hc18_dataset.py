"""HC18 Dataset module with aspect-ratio letterboxing and augmentation.

Loads ultrasound head images and annotation masks from the HC18 challenge dataset.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 384


class LetterboxResize:
    """Resize image maintaining aspect ratio and padding to target square size."""

    def __init__(self, target_size: int = IMG_SIZE):
        self.target_size = target_size

    def __call__(self, image: Image.Image) -> tuple[Image.Image, float, int, int]:
        w, h = image.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = image.resize((new_w, new_h))
        canvas = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))
        return canvas, scale, paste_x, paste_y


class HC18SegDataset(Dataset):
    """HC18 Head Ultrasound Segmentation Dataset."""

    def __init__(
        self,
        csv_path: Union[str, Path, pd.DataFrame],
        image_dir: Union[str, Path],
        augment: bool = False,
        mask_size: int = IMG_SIZE,
    ):
        if isinstance(csv_path, pd.DataFrame):
            self.data = csv_path.reset_index(drop=True)
        else:
            self.data = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.augment = augment
        self.letterbox = LetterboxResize(mask_size)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        row = self.data.iloc[idx]
        img_path = self.image_dir / row["filename"]
        image = Image.open(img_path).convert("RGB")

        mask_filename = row["filename"].replace(".png", "_Annotation.png")
        mask_path = self.image_dir / mask_filename
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask is not None:
            contours, _ = cv2.findContours(
                raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if contours:
                largest = max(contours, key=cv2.contourArea)
                filled_mask = np.zeros_like(raw_mask)
                cv2.drawContours(filled_mask, [largest], -1, 255, thickness=cv2.FILLED)
            else:
                filled_mask = np.zeros_like(raw_mask)
        else:
            w, h = image.size
            filled_mask = np.zeros((h, w), dtype=np.uint8)

        mask_pil = Image.fromarray(filled_mask).convert("RGB")

        img_letterboxed, scale, pad_x, pad_y = self.letterbox(image)
        mask_letterboxed, _, _, _ = self.letterbox(mask_pil)

        if self.augment and np.random.rand() > 0.5:
            img_letterboxed = img_letterboxed.transpose(Image.FLIP_LEFT_RIGHT)
            mask_letterboxed = mask_letterboxed.transpose(Image.FLIP_LEFT_RIGHT)

        img_tensor = self.normalize(transforms.ToTensor()(img_letterboxed))
        mask_tensor = transforms.ToTensor()(mask_letterboxed)[0:1]

        pixel_size = float(row["pixel size(mm)"])
        return img_tensor, mask_tensor, pixel_size, scale
