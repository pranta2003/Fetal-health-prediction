# FASSD Training Run 1 Notes (Seed 42 RGB vs Gray)

## Experiment Configuration

- **Date**: 2026-11-16
- **Kaggle Kernel**: `prantakumerpandit390/fassd-train-seed42` (Version 2)
- **Code Snapshot Dataset**: `prantakumerpandit390/fetal-code-snapshot` (Private)
  - **Commit Hash**: `1218c9ed7498a65c8d6ec6d13f8a17004f22ce5a`
- **FASSD Dataset**: `prantakumerpandit390/fassd-fetal-abdomen-v2` (Private)
  - Split: `docs/results/fassd/split_v2.csv` (Union-find grouped split, seed 42, 0 cross-split hash/patient leakage)
  - Input resolution: $384 \times 512$ during training; evaluated at native resolution $768 \times 1024$
  - Model architecture: U-Net with `efficientnet-b4` encoder (ImageNet pretrained)
  - Loss: BCE + Soft-Dice Multilabel Loss
  - Optimizer: AdamW (`lr=3e-4`, `weight_decay=0.01`), cosine annealing, AMP, Model EMA (`decay=0.999`)
  - Epochs: 60 (patience 12, min delta 1e-4)

## Hardware & Environment

- **GPU**: NVIDIA Tesla T4 (15,360 MiB VRAM)
- **NVIDIA Driver**: 580.178.04
- **CUDA Version**: 13.0
- **PyTorch Device**: `cuda:0`

### `nvidia-smi` Output
```text
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.178.04             Driver Version: 580.178.04     CUDA Version: 13.0     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   53C    P8             10W /   70W |       3MiB /  15360MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

## Execution Timing & Wall-Clock

| Run | Input Mode | Seed | Start Offset | Finish Offset | Wall-Clock Duration | Epochs Trained | Best Epoch |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Run 1 | RGB (3-channel) | 42 | 34.0 s | 9561.6 s | 9527.6 s (~2h 38m 48s) | 60 | 25 |
| Run 2 | Grayscale (1-channel) | 42 | 9561.6 s | 19183.7 s | 9622.1 s (~2h 40m 22s) | 60 | 24 |
| **Total** | - | - | 0.0 s | 19185.4 s | 19185.4 s (~5h 19m 45s) | 120 | - |

## Results Summary

### Macro Test Metrics (271 test images across 25 patient groups)

| Mode | Seed | Best Val Epoch | Val Macro Dice | Test Macro Dice | Test Macro IoU | Test Macro HD95 (px) | Test Patient Bootstrap 95% CI |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **RGB** | 42 | 25 | 0.7802 | 0.7866 | 0.6758 | 22.01 | [0.7681, 0.8159] |
| **Gray** | 42 | 24 | 0.7806 | 0.7874 | 0.6772 | 20.21 | [0.7678, 0.8172] |

### Per-Structure Breakdown (Dice / HD95 in px)

| Structure | Metric | RGB (Seed 42) | Gray (Seed 42) | Delta (Gray - RGB) |
|:---|:---|:---:|:---:|:---:|
| **Artery** | Dice Mean ± SD | 0.7645 ± 0.1754 | 0.7592 ± 0.1807 | -0.0053 |
| | HD95 Mean ± SD (px) | 13.96 ± 77.63 | 10.41 ± 15.56 | -3.55 |
| **Liver** | Dice Mean ± SD | 0.8741 ± 0.0802 | 0.8777 ± 0.0702 | +0.0036 |
| | HD95 Mean ± SD (px) | 27.47 ± 25.36 | 24.59 ± 14.45 | -2.88 |
| **Stomach** | Dice Mean ± SD | 0.7360 ± 0.2486 | 0.7400 ± 0.2511 | +0.0040 |
| | HD95 Mean ± SD (px) | 26.59 ± 81.51 | 21.54 ± 24.33 | -5.05 |
| **Vein** | Dice Mean ± SD | 0.7717 ± 0.1417 | 0.7729 ± 0.1397 | +0.0012 |
| | HD95 Mean ± SD (px) | 20.04 ± 24.13 | 24.29 ± 79.85 | +4.25 |

### Observations

1. **Parity between Gray and RGB**: Both RGB and Grayscale models converge to almost identical test macro Dice (0.7866 for RGB vs 0.7874 for Gray). The 95% patient bootstrap confidence intervals heavily overlap ([0.7681, 0.8159] vs [0.7678, 0.8172]), indicating that converting ultrasound images to grayscale does not degrade anatomical segmentation performance.
2. **Boundary Accuracy (HD95)**: Grayscale achieved slightly improved average 95% Hausdorff Distance overall (20.21 px vs 22.01 px), particularly on artery and stomach structures.
3. **Empty Ground-Truth Handling**: On images with empty stomach annotations (8 cases in the test set), the model correctly predicted 0 positive pixels in 7 out of 8 cases for Gray and 8 out of 8 for RGB.
