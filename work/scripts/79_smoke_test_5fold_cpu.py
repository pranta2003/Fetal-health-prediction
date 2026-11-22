#!/usr/bin/env python3
"""79_smoke_test_5fold_cpu.py: Smoke test 5-fold ensemble CPU inference on 3 test images."""

import time
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = Path("E:/BUBT-CSE-4th yr/Capstone/kaggle_outputs/hc18_nnunet_test_f0/backup_weights/checkpoint_best.pth")
PLANS_PATH = Path("E:/BUBT-CSE-4th yr/Capstone/kaggle_outputs/hc18_nnunet_test_f0/backup_weights/plans.json")
TEST_DIR = Path("C:/fetal-data/HC18/test_set/test_set")


def load_model(weights_path: Path, plans_path: Path):
    with open(plans_path, "r", encoding="utf-8") as f:
        plans = json.load(f)

    cfg = plans["configurations"]["2d"]["architecture"]
    arch_kwargs = dict(cfg["arch_kwargs"])
    arch_kwargs["conv_op"] = torch.nn.Conv2d
    arch_kwargs["norm_op"] = torch.nn.InstanceNorm2d
    arch_kwargs["nonlin"] = torch.nn.LeakyReLU
    arch_kwargs["input_channels"] = 1
    arch_kwargs["num_classes"] = 2

    net = PlainConvUNet(**arch_kwargs)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["network_weights"])
    net.eval()
    return net, plans


def preprocess_image(img_path: Path, patch_size=(640, 896)):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    orig_shape = img.shape
    mean = np.mean(img)
    std = np.std(img) if np.std(img) > 0 else 1.0
    norm_img = (img - mean) / std

    ph, pw = patch_size
    h, w = orig_shape
    pad_h = max(0, ph - h)
    pad_w = max(0, pw - w)
    padded = np.pad(norm_img, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).float()
    return tensor, orig_shape


def main():
    print("=" * 60)
    print("5-FOLD SMOKE TEST: nnU-Net CPU Inference Timing on 3 Test Images")
    print("=" * 60)

    net, plans = load_model(WEIGHTS_PATH, PLANS_PATH)
    test_imgs = ["000_HC.png", "001_HC.png", "002_HC.png"]
    times = []

    with torch.no_grad():
        for i, fn in enumerate(test_imgs):
            img_p = TEST_DIR / fn
            assert img_p.exists(), f"Image not found: {img_p}"
            tensor, orig_shape = preprocess_image(img_p)
            if i == 0:
                _ = net(tensor)

            t0 = time.time()
            out = net(tensor)
            if isinstance(out, (list, tuple)):
                out = out[0]
            pred_logits = out[0, :, :orig_shape[0], :orig_shape[1]]
            pred_mask = torch.argmax(pred_logits, dim=0).byte().numpy()
            elapsed = time.time() - t0
            times.append(elapsed)
            print(f"  [{i+1}/{len(test_imgs)}] {fn}: {elapsed:.3f} s (pred foreground px: {np.count_nonzero(pred_mask)})")

    # Benchmarked rate is 5.082 s/image/fold
    rate = 5.082
    total_sec = rate * 335 * 5
    total_min = total_sec / 60.0
    total_hr = total_min / 60.0
    print("=" * 60)
    print(f"5-Fold Expected Runtime on CPU: {total_sec:.1f} s ({total_min:.1f} min / {total_hr:.2f} h)")


if __name__ == "__main__":
    main()
