# HC18 nnU-Net v2 Execution Notes: Fold 0 (250 Epochs)

## 1. Overview & Execution Targets

This experiment evaluates the modern foundation framework **nnU-Net v2** on the HC18 grand-challenge benchmark under our strictly leak-free patient-grouped split protocol.

- **Kaggle Kernel**: `prantakumerpandit390/hc18-nnunet-f0-250ep`
- **Dataset**: `prantakumerpandit390/dataset501-hc18` (Private, verified ready)
- **Local Dataset Archive**: `C:\fetal-data\nnUNet_packaged\Dataset501_HC18.zip` (136,645,050 bytes)
- **Configuration**: 2D (`nnUNetPlans__2d`)
- **Trainer**: `nnUNetTrainer_250epochs` (250 epochs)
- **Target Fold**: Fold 0 (corresponding to Fold 1 in 1-indexed tables: 793 train, 206 validation cases)

---

## 2. Dataset Structure & Partition Integrity

The dataset follows the official nnU-Net v2 dataset convention (`Dataset501_HC18`):
- `dataset.json`: channel 0: `"US"`, labels: `{background: 0, head: 1}`, file ending: `".png"`.
- `imagesTr`: 999 2D grayscale ultrasound images.
- `labelsTr`: 999 rasterized binary solid-head ground truth masks (`{0, 1}`).
- `splits_final.json`: 5-fold patient-grouped split constructed from `oof_predictions.csv`:
  - Fold 0: 793 train cases, 206 validation cases.
  - Zero patient overlap between train and validation partitions.

### Kernel Diff Check
The kernel executes an automated integrity diff check before launching training:
```python
with open("nnUNet_preprocessed/Dataset501_HC18/splits_final.json") as f:
    splits = json.load(f)
assert len(splits[0]["train"]) == 793
assert len(splits[0]["val"]) == 206
assert len(set(splits[0]["train"]).intersection(set(splits[0]["val"]))) == 0
```

---

## 3. Pipeline Execution Steps in Kernel

1. **Environment Setup**:
   - `nnUNet_raw`: `/kaggle/working/nnUNet_raw`
   - `nnUNet_preprocessed`: `/kaggle/working/nnUNet_preprocessed`
   - `nnUNet_results`: `/kaggle/working/nnUNet_results`
2. **Dataset Unpacking**:
   - Extracts `Dataset501_HC18.zip` into `nnUNet_raw/`.
3. **Planning & Preprocessing**:
   - `nnUNetv2_plan_and_preprocess -d 501 -c 2d --verify_dataset_integrity`
4. **Splits Injection**:
   - Overwrites default random split with our patient-grouped `splits_final.json`.
5. **Training**:
   - `nnUNetv2_train 501 2d 0 -tr nnUNetTrainer_250epochs`
6. **Artifact Preservation**:
   - Copies validation masks, `summary.json`, `progress.png`, training logs, and manifest to `/kaggle/working/out/`.

---

## 4. Local Evaluation Procedure

When the kernel run completes, predictions are evaluated locally on laptop CPU via:
```bash
python scripts/46_eval_nnunet_hc18.py \
  --pred-dir "<path_to_downloaded_validation_fold0>" \
  --fold 1 \
  --n-bootstraps 5000 \
  --seed 42 \
  --out-json "docs/results/hc18_nnunet/eval_fold0_results.json"
```

The script:
1. Fits the submission ellipse via `fit_submission_ellipse()` (identical OpenCV fit pipeline).
2. Computes HC in mm (Ramanujan formula), absolute difference vs ground truth, binary Dice, and boundary Hausdorff distance (mm).
3. Performs patient-level paired bootstrap (5,000 resamples, seed 42) comparing nnU-Net vs our existing fold-1 U-Net baseline.

---

## 5. HC18 Official Corrected Metrics Record

The corrected HC18 official evaluation score sheet is committed in:
- `docs/results/hc18_official/metrics_ensemble_5fold_corrected.json` (SHA-256: `5bf81467e417b84343cb6f313845ee813a6c9f31cd41f8a9800f50b9c177f6cb`)
- Summary Metrics across 335 official test cases:
  - **Mean Absolute Difference**: **1.9472 $\pm$ 1.9114 mm**
  - **Mean Dice**: **97.977%** ($\pm$ 1.274%)
  - **Mean Hausdorff Distance**: **1.2144 mm** ($\pm$ 0.7521 mm)
  - **Mean Signed Difference**: **+0.7844 $\pm$ 2.6152 mm**
  - **Trimester 1**: MAD 1.60 mm, Dice 96.92% (55 cases)
  - **Trimester 2**: MAD 1.81 mm, Dice 98.21% (233 cases)
  - **Trimester 3**: MAD 3.02 mm, Dice 98.08% (47 cases)
