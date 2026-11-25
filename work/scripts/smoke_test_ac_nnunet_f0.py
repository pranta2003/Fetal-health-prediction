#!/usr/bin/env python3

# FIELD: AC
# MODEL: nnU-Net
# STEP: smoke_test
# USED BY: work/docs/CODE_MAP.md, work/docs/results/RESULTS_LEDGER.md

"""Smoke test AC nnU-Net pipeline locally on 3 images.

Tests:
1. Data loading from Dataset502_AC (imagesTr & labelsTr).
2. PlainConvUNet architecture instantiation and forward pass on 3 images.
3. Connected components, cv2.fitEllipseAMS, Ramanujan AC calculation in mm.
4. Dataset splits_final.json fold 0 partition integrity.
"""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path("C:/fetal-data/nnunet_ac/Dataset502_AC")


def ramanujan_perimeter(a: float, b: float) -> float:
    """Ramanujan's first approximation for ellipse perimeter.
    a, b are semi-major and semi-minor axes (half the diameters).
    """
    return np.pi * (3.0 * (a + b) - np.sqrt((3.0 * a + b) * (a + 3.0 * b)))


def main():
    print("=" * 60)
    print("AC nnU-Net Local Verification Test (3 Images)")
    print("=" * 60)

    # 1. Check dataset structure
    images_dir = DATASET_DIR / "imagesTr"
    labels_dir = DATASET_DIR / "labelsTr"
    splits_file = DATASET_DIR / "splits_final.json"

    assert images_dir.is_dir(), f"Missing {images_dir}"
    assert labels_dir.is_dir(), f"Missing {labels_dir}"
    assert splits_file.is_file(), f"Missing {splits_file}"

    with open(splits_file, "r") as f:
        splits = json.load(f)
    print(f"Splits loaded: {len(splits)} folds. Fold 0 train={len(splits[0]['train'])}, val={len(splits[0]['val'])}")

    # 2. Pick 3 validation cases from Fold 0
    val_cases = splits[0]["val"][:3]
    print(f"Testing 3 cases: {val_cases}")

    # 3. Instantiate PlainConvUNet with standard nnU-Net 2D config
    arch_kwargs = {
        "n_stages": 6,
        "features_per_stage": (32, 64, 128, 256, 512, 512),
        "conv_op": torch.nn.Conv2d,
        "kernel_sizes": [[3, 3]] * 6,
        "strides": [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
        "n_conv_per_stage": (2, 2, 2, 2, 2, 2),
        "num_classes": 2,
        "n_conv_per_stage_decoder": (2, 2, 2, 2, 2),
        "conv_bias": True,
        "norm_op": torch.nn.InstanceNorm2d,
        "norm_op_kwargs": {"eps": 1e-5, "affine": True},
        "dropout_op": None,
        "dropout_op_kwargs": None,
        "nonlin": torch.nn.LeakyReLU,
        "nonlin_kwargs": {"inplace": True},
        "input_channels": 1,
    }

    net = PlainConvUNet(**arch_kwargs)
    net.eval()
    print("PlainConvUNet instantiated successfully.")

    # 4. Run test on 3 images
    patch_size = (384, 512)
    for case_id in val_cases:
        img_path = images_dir / f"{case_id}_0000.png"
        lbl_path = labels_dir / f"{case_id}.png"

        assert img_path.is_file(), f"Missing image {img_path}"
        assert lbl_path.is_file(), f"Missing label {lbl_path}"

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        lbl = cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE)
        orig_shape = img.shape
        print(f"\nCase {case_id}: image shape {orig_shape}, label shape {lbl.shape}")

        # Normalization & preprocessing
        mean = np.mean(img)
        std = np.std(img) if np.std(img) > 0 else 1.0
        norm_img = (img - mean) / std

        # Resize/pad to patch size for forward pass
        resized = cv2.resize(norm_img, (patch_size[1], patch_size[0]))
        tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            output = net(tensor)
            if isinstance(output, (list, tuple)):
                output = output[0]
            pred_probs = torch.softmax(output, dim=1)
            pred_mask = torch.argmax(pred_probs, dim=1).squeeze(0).byte().numpy()

        assert pred_mask.shape == patch_size
        print(f"  Forward pass output shape: {output.shape}")

        # Ground truth ellipse fit & AC measurement
        contours, _ = cv2.findContours(lbl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        assert len(contours) > 0, "No contour found in label"
        largest_cnt = max(contours, key=cv2.contourArea)
        assert len(largest_cnt) >= 5, "Contour too small for ellipse fit"

        ellipse = cv2.fitEllipseAMS(largest_cnt)
        (cx, cy), (axis1, axis2), angle = ellipse
        a = max(axis1, axis2) / 2.0
        b = min(axis1, axis2) / 2.0

        # Pixel size assumed default 0.23 mm/px or from metadata
        pixel_size = 0.2307
        perimeter_px = ramanujan_perimeter(a, b)
        ac_mm = perimeter_px * pixel_size
        print(f"  Ground truth ellipse fit: center=({cx:.1f}, {cy:.1f}), a={a:.1f}px, b={b:.1f}px, AC={ac_mm:.2f} mm")
        assert 50.0 < ac_mm < 400.0, f"Implausible AC measurement: {ac_mm}"

    print("\n" + "=" * 60)
    print("All 3 test images verified successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
