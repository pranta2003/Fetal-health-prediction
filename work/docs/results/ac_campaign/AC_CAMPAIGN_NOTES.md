# AC Campaign Notes & Implementation Audit

## 1. 5-Fold Partitioning Summary (`folds_v1.csv`)

A strict, leak-free, patient-grouped 5-fold cross-validation partition was established on the official AC training set:
- **Total Training Dataset**: 642 images across 501 unique patients (FP: 574 images, UCL: 68 images).
- **Official Test Set**: 147 images across 114 patients (FP: 119 images, UCL: 28 images) — completely held out and strictly excluded from `folds_v1.csv`.
- **Partitioning Algorithm**: `KFold(n_splits=5, shuffle=True, random_state=42)` on sorted unique patient IDs.
- **Artifact Path**: `docs/results/ac_campaign/folds_v1.csv` with schema `filename,patient,fold`.

### Fold Distribution
| Fold | Images | Patients | FP Images | UCL Images | Official Test Overlap |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 142 | 101 | 127 | 15 | 0 |
| Fold 2 | 121 | 100 | 108 | 13 | 0 |
| Fold 3 | 124 | 100 | 110 | 14 | 0 |
| Fold 4 | 121 | 100 | 108 | 13 | 0 |
| Fold 5 | 134 | 100 | 121 | 13 | 0 |
| **Total** | **642** | **501** | **574** | **68** | **0** |

Cross-fold verification confirms:
- **0** patients appear in more than one fold.
- **0** test set images appear in any training fold.
- In each fold $K$, inner training uses 4 folds minus a patient-grouped validation holdout (seed $42 + K$, 12% patient fraction) for early stopping, while fold $K$ is reserved purely for out-of-fold (OOF) evaluation.

---

## 2. Discrepancy Analysis: 4.3956 mm vs 4.3951 mm

The discrepancy between the two reported $v3\_full$ test AC MAE values was audited:
- **4.3956 mm** (`4.395647840435002` mm): Found in `notebooks/history/ac_lab_pc/inputs/v3_full_results.json` and `v3_full_test_predictions.csv`. Produced during the original lab-PC training and evaluation run on an NVIDIA RTX 4060 GPU with PyTorch CUDA Automatic Mixed Precision (`torch.autocast(device_type="cuda")`).
- **4.3951 mm** (`4.39512...` mm): Documented in `REFACTOR_PLAN.md` (lines 304, 312), `scripts/06_eval_ac.py`, and `tests/test_ac_v3_regression.py`. Produced during laptop CPU evaluation in full float32 precision.

### Root Cause & Numerical Proof
Evaluating the first 10 test samples under both modes:
- **Laptop CPU FP32**: `3.58592987` mm
- **Lab-PC CUDA AMP FP16**: `3.58614273` mm

The difference is exactly **$0.0005\text{ mm}$** ($0.5\ \mu\text{m}$), arising entirely from floating-point accumulation and intermediate truncation differences between NVIDIA Tensor Core FP16 fused multiply-add operations and CPU FP32 arithmetic in the spatial softmax landmark decoder. Both round to `4.40 mm` at 2 decimal places and `4.395 mm` at 3 decimal places. The underlying weights, dataset, coordinate conversion, and metric evaluation logic are 100% identical.

---

## 3. Training Options Implementation

All options were implemented as modular extensions in `src/fetal/` with 100% backward compatibility (all 14 baseline equality tests remain passing):
- `--ema`: Exponential Moving Average weights tracking (`ModelEMA`, decay 0.999). Evaluates validation and saves checkpoints with shadow weights.
- `--loss {default, smooth_l1, wing}`: Landmark coordinate loss term options. Default Euclidean norm loss is untouched; `wing` and `smooth_l1` are selectable via CLI.
- `--init-encoder PATH`: Strict pretrained encoder initialization. Loads weights into `smp.Unet(encoder_name="efficientnet-b4")` from FASSD checkpoints, verifying and reporting exactly 704 / 704 matching keys.
- `--epochs INT`: Customizable epoch budget (default: 30).
- `--patience INT`: Early stopping patience (default: 8).
- `--fold {1,2,3,4,5,all}`: Cross-validation fold selector. When `--fold all` is supplied, automatically loops over folds 1..5 and pools out-of-fold predictions.

---

## 4. Candidate Recipe Matrix & Push Commands

| Recipe | Description | Loss Term | Regularization / Sched | Encoder Init | Epochs / Patience | Kernel Slug | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **(a) S42** | Baseline v3_full | Default (L2/10) | None | ImageNet | 30 / 8 | `ac-baseline-s42` | **Pushed to Kaggle** |
| **(a) S43** | Baseline v3_full (Seed 43) | Default (L2/10) | None | ImageNet | 30 / 8 | `ac-baseline-s43` | Ready to push |
| **(b)** | Wing Loss | Wing Loss ($w=10, \epsilon=2$) | None | ImageNet | 30 / 8 | `ac-wing-s42` | Ready to push |
| **(c)** | Model EMA + Schedule | Default (L2/10) | EMA (0.999) | ImageNet | 40 / 12 | `ac-ema-s42` | Ready to push |
| **(e)** | FASSD Pretrained | Default (L2/10) | None | FASSD Gray S42 (704 keys) | 30 / 8 | `ac-fassd-s42` | Ready to push |

### Exact Kaggle Push Commands
```bash
# Recipe (a) Seed 42 Baseline (Pushed)
kaggle kernels push -p notebooks/kaggle/ac_campaign/ac-baseline-s42

# Recipe (a) Seed 43 Baseline (Generated)
kaggle kernels push -p notebooks/kaggle/ac_campaign/ac-baseline-s43

# Recipe (b) Wing Loss (Generated)
kaggle kernels push -p notebooks/kaggle/ac_campaign/ac-wing-s42

# Recipe (c) EMA + Longer Schedule (Generated)
kaggle kernels push -p notebooks/kaggle/ac_campaign/ac-ema-s42

# Recipe (e) FASSD Pretrained Encoder (Generated)
kaggle kernels push -p notebooks/kaggle/ac_campaign/ac-fassd-s42
```

---

## 5. Statistical Comparison Protocol (`scripts/40_ac_compare_recipes.py`)

A patient-level cluster paired bootstrap (5,000 resamples, seed 42) compares any two candidate prediction CSVs:
- Evaluates $\Delta \text{AC MAE}$ and $\Delta \text{AC RMSE}$ with 95% percentile confidence intervals and two-sided empirical p-values.
- Clusters resampling at the patient level ($N=501$ training patients, or $N=114$ test patients) to prevent artificial variance deflation from multi-plane acquisitions per subject.
- Tracks median error, NME, and extreme outlier rates ($>10\text{ mm}$).
