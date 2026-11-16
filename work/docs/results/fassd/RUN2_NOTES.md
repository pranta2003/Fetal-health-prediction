# FASSD Training Run 2 Notes (Architectural Exploration)

## 1. Experiment Overview & Slugs

- **Date**: 2026-11-16
- **Kaggle Kernel**: `prantakumerpandit390/fassd-train-round2` (Version 1, status: `KernelWorkerStatus.COMPLETE`)
- **Code Snapshot Dataset**: `prantakumerpandit390/fetal-code-snapshot` (Private)
  - **Commit Hash**: `44c9da2a4ed56545d432ed910d886232e92534a8`
- **FASSD Dataset**: `prantakumerpandit390/fassd-fetal-abdomen-v2` (Private)
  - Split: `docs/results/fassd/split_v2.csv` (Union-find grouped split, seed 42, 0 cross-split hash/patient leakage)
  - Input resolution: $384 \times 512$ during training; evaluated at native resolution $768 \times 1024$
  - Input mode: Grayscale (`gray`, 1-channel converted to 3-channel normalized input) across all 4 variants
  - Loss: BCE + Soft-Dice Multilabel Loss
  - Optimizer: AdamW (`lr=3e-4`, `weight_decay=0.01`), cosine annealing, AMP, Model EMA (`decay=0.999`)
  - Training Optimizations: In-memory array caching (`cache_in_memory=True`), on-device GPU batch augmentations (`batch_gpu_aug=True`), channels-last memory format

---

## 2. Epoch Cap & Early-Stopping Settings

- **Epoch Cap**: **40 epochs** (`--epochs 40`)
- **Early-Stopping Patience**: **10 epochs** (`--patience 10`, `min_delta=1e-4`)
- *Correction Note*: An earlier status message noted "30 epochs" based on preliminary draft defaults; the actual script and executed kernel logs confirm that the run was executed with an epoch cap of **40 epochs** and a patience of **10 epochs**.

---

## 3. Hardware & Execution Environment

- **GPU**: NVIDIA Tesla T4 (15,360 MiB VRAM)
- **NVIDIA Driver**: 580.178.04
- **CUDA Version**: 13.0
- **PyTorch Device**: `cuda:0`

### `nvidia-smi` Log Output
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

---

## 4. Execution Timing & Wall-Clock

All four architectural variants ran sequentially in the same Kaggle kernel session:

| Variant | Architecture | Encoder | Best Epoch | Epochs Trained | Early-Stop Reason | Training Duration | Sec / Epoch | Speedup vs Run 1 |
|:---|:---|:---|:---:|:---:|:---|:---:|:---:|:---:|
| `unet_effb4` | U-Net | `efficientnet-b4` | 23 | 33 | Patience 10 (23+10) | 4735.31 s (~1h 18m 55s) | 143.49 s | 1.12x |
| `unetplusplus_effb4` | U-Net++ | `efficientnet-b4` | 26 | 36 | Patience 10 (26+10) | 5650.78 s (~1h 34m 11s) | 156.97 s | 1.02x |
| `unet_resnet34` | U-Net | `resnet34` | 31 | 40 | Reached 40 Epoch Cap | 4902.53 s (~1h 21m 43s) | 122.56 s | 1.31x |
| `fpn_effb4` | FPN | `efficientnet-b4` | 21 | 31 | Patience 10 (21+10) | 4142.07 s (~1h 09m 02s) | 133.62 s | 1.20x |
| **Total Session** | - | - | - | 140 | - | **19430.69 s (~5h 23m 51s)** | - | - |

---

## 5. Per-Variant Results Summary

*(Note: In accordance with scientific integrity rules, variants are grouped by model configuration and reported with their validation metrics. Test metrics are reported for completeness, but models are NOT ranked by test performance).*

### Validation and Overall Test Macro Performance

| Variant | Architecture | Encoder | Best Val Epoch | Val Macro Dice | Test Macro Dice | Test Macro IoU | Test Macro HD95 (px) | Test Patient Bootstrap 95% CI |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `unet_effb4` | U-Net | `efficientnet-b4` | 23 | 0.7781 | 0.7913 | 0.6801 | 22.38 | [0.7751, 0.8183] |
| `unetplusplus_effb4` | U-Net++ | `efficientnet-b4` | 26 | 0.7755 | 0.7881 | 0.6773 | 23.65 | [0.7697, 0.8163] |
| `unet_resnet34` | U-Net | `resnet34` | 31 | 0.7799 | 0.7927 | 0.6829 | 22.22 | [0.7763, 0.8208] |
| `fpn_effb4` | FPN | `efficientnet-b4` | 21 | 0.7709 | 0.7803 | 0.6678 | 23.74 | [0.7609, 0.8088] |

### Per-Structure Breakdown on Test Set (271 images, evaluated once at native $768 \times 1024$)

| Variant | Artery Dice | Artery HD95 (px) | Liver Dice | Liver HD95 (px) | Stomach Dice | Stomach HD95 (px) | Vein Dice | Vein HD95 (px) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `unet_effb4` | 0.7678 | 18.98 | 0.8808 | 24.73 | 0.7464 | 26.02 | 0.7701 | 19.78 |
| `unetplusplus_effb4` | 0.7660 | 18.55 | 0.8753 | 25.61 | 0.7430 | 26.11 | 0.7682 | 24.32 |
| `unet_resnet34` | 0.7748 | 9.24 | 0.8800 | 25.92 | 0.7460 | 25.48 | 0.7700 | 28.25 |
| `fpn_effb4` | 0.7442 | 16.23 | 0.8762 | 30.76 | 0.7384 | 26.76 | 0.7626 | 21.21 |

---

## 6. Observations & Analysis

1. **Validation Noise & Single Seed**:
   - All variants were trained with a single random seed (`seed 42`).
   - The validation macro Dice scores across the top three models (`unet_resnet34`: 0.7799, `unet_effb4`: 0.7781, `unetplusplus_effb4`: 0.7755) span a narrow band of $\Delta \approx 0.0044$.
   - This difference is well within normal validation sampling noise and indicates that U-Net with ResNet-34, U-Net with EfficientNet-B4, and U-Net++ with EfficientNet-B4 exhibit comparable segmentation capability on fetal abdominal ultrasound.
2. **Backbone Efficiency**:
   - `unet_resnet34` trained fastest on GPU (122.6 s/epoch, a 1.31x speedup over Run 1's baseline 160.4 s/epoch) due to standard residual blocks optimizing Tensor Core throughput.
   - `unetplusplus_effb4` was the slowest (157.0 s/epoch) due to nested skip pathway dense convolutions.
3. **FPN Architecture**:
   - The FPN decoder (`fpn_effb4`) achieved lower validation Dice (0.7709) compared to U-Net and U-Net++, suggesting that top-down feature aggregation without symmetric skip connections is slightly less effective for fine fetal abdominal boundaries.
