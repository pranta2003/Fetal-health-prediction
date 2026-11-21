#!/usr/bin/env python3
"""Smoke test nnU-Net CPU inference locally on 3 test images.

Measures seconds per image per fold on CPU and estimates total runtime for
335 test images across 3 folds.
"""

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
    # Z-score normalization
    mean = np.mean(img)
    std = np.std(img)
    if std == 0:
        std = 1.0
    norm_img = (img - mean) / std

    # Pad to patch size
    ph, pw = patch_size
    h, w = orig_shape
    pad_h = max(0, ph - h)
    pad_w = max(0, pw - w)
    padded = np.pad(norm_img, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).float()
    return tensor, orig_shape


def main():
    print("=" * 60)
    print("SMOKE TEST: nnU-Net CPU Inference Timing on 3 Test Images")
    print("=" * 60)

    torch.set_num_threads(torch.get_num_threads())
    print(f"PyTorch threads: {torch.get_num_threads()}")

    t_load_start = time.time()
    net, plans = load_model(WEIGHTS_PATH, PLANS_PATH)
    t_load = time.time() - t_load_start
    print(f"Model loaded in {t_load:.2f} s")

    test_imgs = ["000_HC.png", "001_HC.png", "002_HC.png"]
    times = []

    print(f"\nRunning CPU forward pass on {len(test_imgs)} test images:")
    with torch.no_grad():
        for i, fn in enumerate(test_imgs):
            img_p = TEST_DIR / fn
            assert img_p.exists(), f"Image not found: {img_p}"

            tensor, orig_shape = preprocess_image(img_p)
            # Warmup on first image
            if i == 0:
                _ = net(tensor)

            t0 = time.time()
            out = net(tensor)
            if isinstance(out, (list, tuple)):
                out = out[0]
            # Slicing back to orig_shape
            pred_logits = out[0, :, :orig_shape[0], :orig_shape[1]]
            pred_mask = torch.argmax(pred_logits, dim=0).byte().numpy()
            elapsed = time.time() - t0
            times.append(elapsed)

            nonzeros = np.count_nonzero(pred_mask)
            print(f"  [{i+1}/{len(test_imgs)}] {fn}: {elapsed:.3f} s (pred foreground px: {nonzeros})")

    mean_sec_per_img_fold = float(np.mean(times))
    std_sec_per_img_fold = float(np.std(times))
    print("\n" + "=" * 60)
    print("CPU INFERENCE BENCHMARK RESULTS:")
    print(f"  Latency per image (1 fold): {mean_sec_per_img_fold:.3f} +- {std_sec_per_img_fold:.3f} s")

    # For 3 folds ensemble:
    sec_per_img_3folds = mean_sec_per_img_fold * 3
    print(f"  Latency per image (3 folds): {sec_per_img_3folds:.3f} s")

    # Total runtime for 335 test images across 3 folds:
    total_est_seconds = 335 * sec_per_img_3folds
    total_est_minutes = total_est_seconds / 60.0
    total_est_hours = total_est_minutes / 60.0
    print(f"  Estimated total time for 335 images x 3 folds on CPU:")
    print(f"    Seconds : {total_est_seconds:.1f} s")
    print(f"    Minutes : {total_est_minutes:.1f} min")
    print(f"    Hours   : {total_est_hours:.2f} h")
    print("=" * 60)

    results = {
        "device": "cpu",
        "num_threads": torch.get_num_threads(),
        "n_test_images_tested": len(test_imgs),
        "test_images": test_imgs,
        "mean_sec_per_image_per_fold": round(mean_sec_per_img_fold, 4),
        "std_sec_per_image_per_fold": round(std_sec_per_img_fold, 4),
        "mean_sec_per_image_3folds": round(sec_per_img_3folds, 4),
        "estimated_total_seconds_335x3": round(total_est_seconds, 1),
        "estimated_total_minutes_335x3": round(total_est_minutes, 1),
        "estimated_total_hours_335x3": round(total_est_hours, 2),
    }

    out_json = REPO_ROOT / "docs" / "results" / "hc18_nnunet" / "cpu_inference_benchmark.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved benchmark results to {out_json}")


if __name__ == "__main__":
    main()
