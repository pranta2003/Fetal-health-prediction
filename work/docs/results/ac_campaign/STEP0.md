# AC Campaign Step 0: Dataset Audit, Inventory & Experimental Plan

> [!IMPORTANT]
> **READ-ONLY Step 0**: No models trained, no Kaggle kernels executed, no data uploaded, and no challenge submissions made.
> All numerical values in this report are computed directly from committed repository files and local disk data by `scripts/38_ac_step0_audit.py`.

## 1. Inventory & Dataset Architecture

### 1.1 Data Locations & Availability

| Asset | Location / Path | Exists in Repo | Exists on Laptop | Exists on Lab PC |
|---|---|---|---|---|
| FP Abdomen Images | `C:/fetal-data/FP/images/Abdomen` (693 PNGs) | No (git ignored) | Yes | Yes (`D:/.../FP/images/Abdomen`) |
| UCL Abdomen Images | `C:/fetal-data/UCL/images/Abdomen` (130 JPGs) | No (git ignored) | Yes | Yes (`D:/.../UCL/images/Abdomen`) |
| FP Annotations | `C:/fetal-data/FP/annotations` (`Abdomen.csv`, `_Train`, `_Test`) | No (git ignored) | Yes | Yes (`D:/.../FP/annotations`) |
| UCL Annotations | `C:/fetal-data/UCL/annotations/px_to_mm` (`Abdomen.csv`, `_Train`, `_Test`) | No (git ignored) | Yes | Yes (`D:/.../UCL/annotations/px_to_mm`) |
| Original `ac_dsnt.py` | `notebooks/history/ac_lab_pc/inputs/scripts_original/ac_dsnt.py` | Yes (PR #34) | Yes | Yes |
| Original `ac_v3.py` | `notebooks/history/ac_lab_pc/inputs/scripts_original/ac_v3.py` | Yes (PR #34) | Yes | Yes |
| Original `ac_v3_ens.py` | `notebooks/history/ac_lab_pc/inputs/scripts_original/ac_v3_ens.py` | Yes (PR #34) | Yes | Yes |
| Committed Execution Logs | `notebooks/history/ac_lab_pc/inputs/{full_log.txt, v3_log.txt, ens_log.txt}` | Yes (PR #34) | Yes | Yes |
| Committed Lab-PC Predictions | `notebooks/history/ac_lab_pc/inputs/{dsnt_full_oof_results.csv, v3_full_test_predictions.csv, ens_test_predictions.csv}` | Yes (PR #34) | Yes | Yes |
| Refactored Dataset Loader | `src/fetal/data/ac_dataset.py` | Yes | Yes | N/A |
| Refactored Landmark Model | `src/fetal/models/ac_landmark.py` | Yes | Yes | N/A |
| Refactored Evaluator & Geometry | `src/fetal/eval/{ac_evaluator.py, ac_geometry.py, ensemble_evaluator.py}` | Yes | Yes | N/A |
| Refactored Training Scripts | `scripts/{04_train_ac.py, 05_train_ensemble.py, 06_eval_ac.py}` | Yes | Yes | N/A |
| Checkpoints (`.pt`) | `C:/fetal-models/ac_v3/{v3_full,base}/best.pt`, `ac_v3_ens/s{1..4}/best.pt` | No (git ignored) | Yes | Yes |

### 1.2 Subject Identification & Partitioning

- **Subject ID Extraction**:
  - **FP images**: `Patient00168_Plane2_1_of_2.png` $\rightarrow$ Regex `r'(Patient\d+)'` prefixed with `FP_` $\rightarrow$ `FP_Patient00168`.
  - **UCL images**: `001_2AC.jpg` $\rightarrow$ Regex `r'^(\d+)_'` prefixed with `UCL_` $\rightarrow$ `UCL_001`.
  - **UCL patient 125 exclusion**: Images prefixed with `125_` lack pixel-to-mm calibration metadata in `px_to_mm/Abdomen.csv` and are cleanly excluded from both training and evaluation (3 uncalibrated images, leaving 96 calibrated UCL images out of 99 in `Abdomen.csv`).

### 1.3 Dataset Splits: Development vs Official Test

- **Total Calibrated Dataset**: **789 images** across **615 unique patients**.
  - **FP**: 693 images, 586 patients.
  - **UCL (calibrated)**: 96 images, 29 patients.
- **Official Test Set**: **147 images / 114 patients**:
  - FP test: 125 images, 109 patients.
  - UCL test: 22 images, 5 patients.
- **Official Train Pool**: **642 images / 501 patients**:
  - FP train: 568 images, 477 patients.
  - UCL train: 74 images, 24 patients.
- **Development Set (5-Fold Out-of-Fold)**:
  - In `ac_dsnt.py` (v2), 5-fold GroupKFold by patient was run across all 789 images (615 patients), logged in `full_log.txt` and saved in `dsnt_full_oof_results.csv` (789 rows).
  - In `ac_v3.py` (v3), training used official-train patients (642 images / 501 patients) minus a 12% random patient hold-out (dev set), reporting once on the official test set (147 images).
- **Fold Assignment File Status**:
  - **No standalone `fold_assignment.csv` file was stored** on the lab PC or in the repo. Folds were generated deterministically at runtime via `sklearn.model_selection.GroupKFold(n_splits=5)` using patient IDs as grouping keys.

### 1.4 Hyperparameters & Training Configuration of `v3_full`

From committed `notebooks/history/ac_lab_pc/inputs/scripts_original/ac_v3.py` and `v3_log.txt`:
- **Architecture**: `smp.Unet(encoder_name='efficientnet-b4', encoder_weights='imagenet', in_channels=3, classes=4)`
- **Decoding**: DSNT-style differentiable spatial softmax over $H \times W$ heatmaps to produce $(x, y)$ coordinate expectations
- **Input Resolution**: $384 \times 384$ letterboxed with zero-padding (aspect ratio preserved)
- **Inpainting**: Burned-in yellow calipers and caliper measurement labels removed via HSV yellow thresholding and OpenCV Telea inpainting (`cv2.inpaint`)
- **Augmentations**: Strong augmentation pipeline: horizontal flip ($p=0.5$), random affine rotation $\pm 25^\circ$ ($p=0.7$), scaling $0.85$–$1.15$, translation $\pm 8\%$, photometric jitter ($p=0.8$), random erasing / cutout ($p=0.3$)
- **Loss Function**: $\mathcal{L} = \mathcal{L}_{\text{coord}} + \mathcal{L}_{\text{JS}} + \lambda_{\text{diam}} (\mathcal{L}_{\text{diam}} + \mathcal{L}_{\text{AC}})$
  - Landmark coordinate loss: $\mathcal{L}_{\text{coord}} = \frac{1}{10} \| \hat{P}_{\text{matched}} - P_{\text{gt}} \|_2$
  - Spatial regularizer: Jensen-Shannon divergence $\mathcal{L}_{\text{JS}}$ between predicted heatmap distribution and isotropic Gaussian ($\\sigma = 4.0 \times S / 384$)
  - Diameter & AC loss: $\lambda_{\text{diam}} = 1.0$, penalizing errors in TAD length, APAD length, and Ramanujan ellipse perimeter in mm
- **Optimizer**: AdamW, $\text{lr} = 2 \times 10^{-4}$, weight decay $0.01$
- **Scheduler**: `ReduceLROnPlateau` (factor $0.5$, patience $4$ epochs, monitoring dev score: $\text{AC MAE} + \text{Landmark Error}$)
- **Precision & Scaling**: PyTorch Automatic Mixed Precision (AMP `GradScaler`)
- **Batch Size**: 4 (with `drop_last=True`)
- **Early Stopping**: Patience 12 epochs on dev validation score
- **Seed Handling**: `random.seed(0)`, `np.random.seed(0)`, `torch.manual_seed(0)`

### 1.5 Metric Script & Biometry Calculation

- **Implementation**: `src/fetal/eval/ac_evaluator.py` and `src/fetal/eval/ac_geometry.py`
- **Landmark Permutation Matching**: 8 allowed relabelings (swapping endpoints of TAD/APAD pairs, swapping TAD and APAD axes) resolved by minimum Euclidean distance matching
- **Pixel-to-Millimeter Conversion**: $d_{\text{mm}} = d_{\text{px}} \times \text{px\_to\_mm\_rate} / \text{scale}$, where `scale` accounts for letterboxing
- **Perimeter Formula**: Ramanujan's first ellipse perimeter approximation:
  $$\text{AC} = \pi \left[ 3(a + b) - \sqrt{(3a + b)(a + 3b)} \right]$$
  where $a = \text{TAD}_{\text{mm}} / 2$ and $b = \text{APAD}_{\text{mm}} / 2$.
- **Metrics Produced**:
  - **AC MAE**: Mean absolute error in mm
  - **AC RMSE**: Root mean squared error in mm
  - **NME**: Normalized Mean Error normalized by mean ground truth diameter: $\frac{1}{2} \left( \frac{\bar{e}_{\text{TAD}}}{\text{TAD}_{\text{gt}}} + \frac{\bar{e}_{\text{APAD}}}{\text{APAD}_{\text{gt}}} \right)$
  - **>10 mm Outlier Rate**: Fraction of test samples with $|\text{AC}_{\text{pred}} - \text{AC}_{\text{gt}}| > 10\text{ mm}$

---

## 2. Leakage & Data Integrity Audit

Following the exact audit methodology applied to FASSD, we computed SHA-256 digests of all 789 images and cross-checked patient identifiers across splits.

### 2.1 SHA-256 Duplicate Image Analysis

- **Total Images Evaluated**: 789
- **Unique SHA-256 Hashes**: 777
- **Duplicate Hashes**: 11 hashes appear more than once (10 pairs, 1 triplet):

| SHA-256 Prefix | Image Name(s) | Source | Patient ID | Official Split | Status |
|---|---|---|---|---|---|
| `07366b89fab1...` | `Patient00906_Plane2_1_of_2.png, Patient00906_Plane2_2_of_2.png` | FP | `FP_Patient00906` | `train` | Same Patient, Same Split (Clean) |
| `7cfad1291ea2...` | `Patient00909_Plane2_1_of_2.png, Patient00909_Plane2_2_of_2.png` | FP | `FP_Patient00909` | `train` | Same Patient, Same Split (Clean) |
| `369d3fc0ca27...` | `Patient01151_Plane2_1_of_2.png, Patient01151_Plane2_2_of_2.png` | FP | `FP_Patient01151` | `train` | Same Patient, Same Split (Clean) |
| `f4bdd7f569cb...` | `Patient01363_Plane2_1_of_3.png, Patient01363_Plane2_2_of_3.png, Patient01363_Plane2_3_of_3.png` | FP | `FP_Patient01363` | `train` | Same Patient, Same Split (Clean) |
| `7efc4f006d63...` | `Patient01475_Plane2_1_of_2.png, Patient01475_Plane2_2_of_2.png` | FP | `FP_Patient01475` | `train` | Same Patient, Same Split (Clean) |
| `15bc130aa3ed...` | `Patient01573_Plane2_1_of_2.png, Patient01573_Plane2_2_of_2.png` | FP | `FP_Patient01573` | `train` | Same Patient, Same Split (Clean) |
| `c0cd8944a928...` | `Patient01620_Plane2_1_of_2.png, Patient01620_Plane2_2_of_2.png` | FP | `FP_Patient01620` | `train` | Same Patient, Same Split (Clean) |
| `169401402cd5...` | `Patient01647_Plane2_1_of_2.png, Patient01647_Plane2_2_of_2.png` | FP | `FP_Patient01647` | `train` | Same Patient, Same Split (Clean) |
| `70acab6ac589...` | `Patient01679_Plane2_1_of_2.png, Patient01679_Plane2_2_of_2.png` | FP | `FP_Patient01679` | `train` | Same Patient, Same Split (Clean) |
| `88276083098a...` | `Patient01678_Plane2_1_of_2.png, Patient01678_Plane2_2_of_2.png` | FP | `FP_Patient01678` | `test` | Same Patient, Same Split (Clean) |
| `a1a026b2efad...` | `024_2AC.jpeg, 024_3AC.jpeg` | UCL | `UCL_024` | `train` | Same Patient, Same Split (Clean) |

### 2.2 Leakage Findings Summary

- **Cross-Patient Duplicate Images**: **0** (ZERO cross-patient duplicates detected across all 789 images).
- **Cross-Split Duplicate Images**: **0** (ZERO identical images cross between train and test).
- **Patient Overlap between Official Train and Official Test**: **0** (ZERO patient overlap; all 501 train patients and 114 test patients are strictly disjoint).
- **Dev Folds Leakage Status**: In both `ac_dsnt.py` (GroupKFold by patient) and `ac_v3.py` (patient-level dev holdout), grouping is by patient. Because all identical images belong to the exact same patient, no identical image ever crosses fold boundaries.
- **Affected Numbers**: **None**. There is zero data leakage in the AC dataset.

---

## 3. Timing & Resource Profiling

### 3.1 Lab-PC Baseline GPU Timings (RTX 4060, 8 GB VRAM)

From committed execution logs in `notebooks/history/ac_lab_pc/inputs/`:
- **`v3_full` Training**:
  - Epochs executed: **51 epochs** (early stopped at patience 12)
  - Total training time: **4,498 seconds** (75.0 minutes / 1.25 hours)
  - Mean time per epoch: **88.2 seconds/epoch**
- **Ensemble Single-Seed Runs (`ens_log.txt`)**:
  - `s1`: 33 epochs, 2,887s (48.1m), 87.5s/epoch
  - `s2`: 53 epochs, 4,573s (76.2m), 86.3s/epoch
  - `s3`: 29 epochs, 2,530s (42.2m), 87.2s/epoch
  - `s4`: 86 epochs, 7,706s (128.4m), 89.6s/epoch
  - `res512`: 28 epochs, 4,389s (73.2m), 156.8s/epoch
- **5-Fold DSNT Training (`full_log.txt`)**:
  - Total 5-fold execution time across all 5 folds: **15,033 seconds** (4.18 hours)
  - Fold 1: 35 epochs, 2,719s (45.3m), 77.7s/epoch
  - Fold 2: 42 epochs, 3,154s (52.6m), 75.1s/epoch
  - Fold 3: 55 epochs, 5,163s (86.0m), 93.9s/epoch
  - Fold 4: 29 epochs, 1,899s (31.6m), 65.5s/epoch
  - Fold 5: 23 epochs, 2,098s (35.0m), 91.2s/epoch

### 3.2 Projected Kaggle T4 GPU Runtime Estimates

- **Hardware Assumptions**: Kaggle T4 has 16 GB VRAM with Turing architecture. While RTX 4060 has Ada Lovelace architecture with higher clocks, T4 provides double the VRAM. Accounting for memory bandwidth and clock frequencies, T4 throughput is estimated at ~1.3x RTX 4060 time (~115s/epoch at batch size 4; can be reduced to ~80s/epoch by scaling batch size to 8 or 16).
- **Projected Per-Fold Runtime (T4)**: ~45 epochs x 115 s ≈ 5,175 s (**~1.44 hours**).
- **Projected 5-Fold Protocol (T4, 1 Seed)**: 5 x 1.44 h ≈ **~7.2 GPU-hours**.

### 3.3 Candidate Recipes & Implementation Effort (5 Folds $\times$ 1 Seed)

| Recipe | Description | Files to Change / Implement | Rough GPU-Hours (T4, 5 folds $\times$ 1 seed) | Implementation Effort |
|---|---|---|---|---|
| **(a) Baseline `v3_full` Re-run** | U-Net effb4, spatial softmax, L1 coord + JS + diameter loss, strong aug | Already implemented in `src/fetal/train/recipes.py` | ~7.2 T4 GPU-h | **Low** (0 new files) |
| **(b) Robust Landmark Loss** | Smooth-L1 or Wing Loss on coordinate offsets instead of L2 norm | Add `wing_loss` / `smooth_l1` to `src/fetal/models/ac_landmark.py` | ~7.2 T4 GPU-h | **Medium** (~20 lines) |
| **(c) EMA + Extended Schedule** | Exponential Moving Average (decay 0.999), cosine schedule, 100 epochs | Add EMA update loop to `src/fetal/train/train_ac.py` | ~9.5 T4 GPU-h | **Medium** (~25 lines) |
| **(d) HRNet-W18 Backbone** | High-Resolution Network backbone preserving multiscale representations | Custom HRNet wrapper via `timm` (smp lacks HRNet) | ~8.0 T4 GPU-h | **High** (new model file) |
| **(e) FASSD-Pretrained Encoder** | Initialize U-Net effb4 encoder from FASSD Gray checkpoint (77.9 MB) | Checkpoint loader in `src/fetal/models/ac_landmark.py` | ~6.5 T4 GPU-h | **Low/Med** (~15 lines) |
| **(f) Scale / Flip TTA** | Multi-scale letterbox inference (0.9, 1.0, 1.1) combined with horizontal flip | Enhance `predict_batch` in `src/fetal/eval/ac_evaluator.py` | ~0.5 T4 GPU-h (eval only) | **Low** (~20 lines) |
| **(g) Heterogeneous Ensemble** | Blending predictions of (a), (c), (e) using coordinate median | Supported by `src/fetal/eval/ensemble_evaluator.py` | 0 h (reuses checkpoints) | **Low** (script config) |

#### Backbone Audit Note on Recipe (d):
- `segmentation_models_pytorch` (v0.5.0): HRNet encoders in registry = **`[]`** (None available directly in smp).
- `timm` (v1.0.30): HRNet models available = **`['hrnet_w18', 'hrnet_w18_small', 'hrnet_w18_small_v2', 'hrnet_w18_ssld']`**.
- Therefore, recipe (d) requires a custom timm-backed encoder or feature extractor rather than standard `smp.Unet(encoder_name='hrnet_w18')`.

#### Pretrained Weights Note on Recipe (e):
- FASSD Run 1 Gray Checkpoint path: `E:\BUBT-CSE-4th yr\Capstone\kaggle_outputs\fassd_run1\fassd_gray_seed42\best_model.pt`
- File exists: **True** (size: 77.91 MB)
- Encoder name: `efficientnet-b4`, validation macro Dice: `0.7806`.
- Provides genuine domain-specific ultrasound pretraining directly on fetal abdominal anatomy.

---

## 4. Proposed Campaign Protocol & Execution Plan

### 4.1 Development vs Test Discipline

1. **Strict Hold-out of Official Test Set**:
   - The official test set (**147 images / 114 patients**) is strictly sealed during exploratory modeling and hyperparameter tuning.
   - It will be evaluated **EXACTLY ONCE** on the final selected model / ensemble.
2. **Patient-Grouped 5-Fold Development Set**:
   - The official train set (**642 images / 501 patients**) serves as the development universe.
   - Split into 5 deterministic patient folds using `GroupKFold(n_splits=5)` keyed by patient ID.
   - All model selection decisions are made solely on **pooled out-of-fold (OOF) dev AC MAE**.
3. **Statistical Verification**:
   - Model comparisons evaluated using **paired patient-level bootstrap 95% confidence intervals** (5,000 resamples, seed 42) on dev OOF predictions.

### 4.2 Kaggle Packaging & Feasibility

- **Dataset Package Contents**:
  - `FP/images/Abdomen/` (693 PNGs, 118.8 MB)
  - `UCL/images/Abdomen/` (130 JPGs, 35.8 MB)
  - `FP/annotations/` (`Abdomen.csv`, `Abdomen_Train.csv`, `Abdomen_Test.csv`, 0.7 MB)
  - `UCL/annotations/px_to_mm/` (`Abdomen.csv`, `Abdomen_Train.csv`, `Abdomen_Test.csv`, 0.1 MB)
- **Total Dataset Size**: **155.41 MB** uncompressed (~140 MB compressed zip).
- **Feasibility**: High. Upload takes under 60 seconds. Uses <1% of Kaggle's 20 GB dataset quota.

### 4.3 Proposed Campaign Order & GPU Allocation

| Stage | Action / Experiment | Focus | Est. T4 GPU-Hours | Cumulative GPU-Hours |
|---|---|---|---|---|
| **Step 1** | Package private Kaggle dataset + generate `dev_split_5fold.csv` | Dataset preparation | 0.0 h | 0.0 h |
| **Step 2** | Recipe (a): Baseline `v3_full` 5-fold run | Benchmark establishment | ~7.2 h | 7.2 h |
| **Step 3** | Recipe (e): FASSD-pretrained encoder 5-fold run | In-domain transfer | ~6.5 h | 13.7 h |
| **Step 4** | Recipe (b)+(c): Robust loss + EMA 5-fold run | Optimization stability | ~8.0 h | 21.7 h |
| **Step 5** | Recipe (f): Scale/flip TTA evaluation on OOF | Inference enhancement | ~0.5 h | 22.2 h |
| **Step 6** | Recipe (g): Heterogeneous ensemble of top folds | Final candidate build | ~0.5 h | 22.7 h |
| **Step 7** | Single evaluation of winning candidate on Official Test | Final benchmark | ~0.5 h | **~23.2 h** |

- **Total Budget**: ~23–25 T4 GPU-hours, comfortably within Kaggle's 30 hours/week quota for a single weekly cycle.

---

## 5. Verification Checksums

- Generated by `scripts/38_ac_step0_audit.py`
- Working tree: `work/docs/results/ac_campaign`