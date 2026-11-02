# Multimodal Fetal Health Capstone - Repository Refactoring Plan
**Target Repository:** `C:\fetal-health-multimodal\`  
**Date:** 2026-10-25  
**Status:** Phase 1 (Plan Updated per Phase 2b-1 Review — Steps 2.1 & 2.2 Approved)  
**Safety Protocol:** Strict Read-Only on Existing Files | No Model/Weight Modification | No Pipeline Retraining | Zero Token/Credential Leakage

---

## 1. Executive Summary & Refactoring Philosophy

The goal of this refactoring is to consolidate, modularize, and productionize the multimodal fetal health assessment capstone project into a clean, reproducible, and verifiable engineering repository. 

### Core Refactoring Principles:
1. **MOVE, Do Not Rewrite Logic:** The existing numerical and architectural implementations have verified results comparable to the published baseline (AC MAE 4.40 mm single model, 4.34 mm 4-model ensemble; HC MAE 2.06 mm provisional; Tabular XGBoost AUC 0.9684). Logic, tensor calculations, loss formulations, and geometric algorithms will be extracted verbatim into clean modules without altering their mathematical behavior.
2. **Status Annotations & Verification Gates:**
   - **Stream 2 (AC Biometry):** VERIFIED via Phase 2a replication tests on official test set (`v3_full` AC MAE 4.3951 mm vs 4.3956 mm; 4-model ensemble median AC MAE 4.3426 mm vs 4.3430 mm).
   - **Stream 1 (HC18 Biometry):** Moved but NOT verified until it gets regression tests. Results are marked PROVISIONAL (ungrouped KFold; repeat scans of the same patient can leak across folds).
   - **Stream 3 (Clinical Tabular XGBoost):** Moved but NOT verified until it gets regression tests.
3. **Strict Configuration Decoupling:** Eliminate all 71 hardcoded paths (`/content/drive/...`, `D:\...`, `C:\...`). All dataset paths, checkpoint directories, output folders, and device selections will be managed via `configs/paths.yaml` and loaded via a central `src/fetal/config.py`.
   - `config.py` must work with `C:\fetal-data` (`FP\` and `UCL\` at top level) and `C:\fetal-models` via environment variables `FETAL_DATA_DIR` and `FETAL_MODEL_DIR`.
   - Retain `num_workers = 0` on Windows (original behaviour to avoid multiprocessing deadlocks).
4. **Preservation of Exploratory Work:** Archive dead-end experiments (HC18 ConvNeXt regression, multi-task U-Net, AC v1 heatmap regression) into dedicated archive modules (`src/fetal/archive/` and `notebooks/archive/`), maintaining full lineage and auditability without cluttering the production pipeline.
5. **Thin Runners & CLI Entrypoints:** Convert notebook monoliths into headless, numbered CLI scripts (`scripts/01_...` through `scripts/06_...`) and thin notebook runners that import clean functions from `src/fetal/`.

---

## 2. Comprehensive Inventory of Source Code

### 2.1 Stream 3: Tabular Clinical Fetal Risk Model (CDC Birth Data)
* **Status:** Moved but NOT verified until regression tests are executed.
* **Source Locations:** `Capstone_fetalDataset.ipynb` (Cells 2–11)
* **Description:** Preprocessing, cleaning, chunked sampling of multi-year CDC NATAL birth data, and training an XGBoost classifier for fetal mortality/risk prediction (AUC-ROC: 0.9684).

| Entity Name | Source Line/Cell | Type | Role & Description | Target Module |
|---|---|---|---|---|
| `get_available_cols` | Cell 2 | Function | Scans CSV headers to identify available CDC columns across years | `src/fetal/data/tabular_cdc.py` |
| `count_rows_chunked` | Cell 2 | Function | Computes line counts across large annual NATAL CSVs using chunks | `src/fetal/data/tabular_cdc.py` |
| `sample_csv_chunked` | Cell 2 | Function | Downsamples massive live-birth datasets uniformly into balanced subsets | `src/fetal/data/tabular_cdc.py` |
| `load_data` | Cell 8 | Function | Loads training CSV and validates outcome distributions | `src/fetal/data/tabular_cdc.py` |
| `clean_unknown_codes` | Cell 8 | Function | Replaces CDC-specific missing/unknown sentinel codes with `NaN` | `src/fetal/data/tabular_cdc.py` |
| `build_pipeline` | Cell 8 | Function | Constructs `sklearn` pipeline (imputers, one-hot encoders, XGBClassifier) | `src/fetal/models/clinical_xgb.py` |
| `train` | Cell 8 | Function | Executes StratifiedKFold cross-validation and exports trained JSON model | `src/fetal/train/train_clinical_xgb.py` |
| `NUMERIC_FEATURES` | Cells 6, 8 | Constant | List of numeric clinical variables (e.g., `MAGER`, `WTGAIN`, `COMBGEST`) | `src/fetal/models/clinical_xgb.py` |
| `CATEGORICAL_FEATURES`| Cells 6, 8 | Constant | List of categorical variables (e.g., `SEX`, `PAY_REC`, `PRECARE`) | `src/fetal/models/clinical_xgb.py` |
| `XGB_PARAMS` | Cells 6, 8 | Config | Hyperparameters (`max_depth=4`, `scale_pos_weight=9`, `eval_metric='auc'`) | `configs/clinical_xgb.yaml` |

---

### 2.2 Layer 3: Hadlock Growth & Estimated Fetal Weight (EFW) Formula
* **Status:** Verified against canonical notebook test vectors.
* **Source Locations:** `Capstone_fetalDataset.ipynb` (Cells 13–15)
* **Description:** Mathematical formulas based on Hadlock et al. (1984/1985) for computing Estimated Fetal Weight (EFW) from ultrasound biometry (HC, BPD, AC, FL) and evaluating gestational age growth percentiles (Hadlock et al. 1991).

| Entity Name | Source Line/Cell | Type | Role & Description | Target Module |
|---|---|---|---|---|
| `calculate_efw` | Cells 13, 15 | Function | Computes EFW in grams using 4-parameter Hadlock formula ($\log_{10} \text{EFW} = 1.3596 - 0.00386(\text{AC}\cdot\text{FL}) + 0.0064\text{HC} + 0.00061(\text{BPD}\cdot\text{AC}) + 0.0424\text{AC} + 0.174\text{FL}$) | `src/fetal/explain/hadlock.py` |
| `hadlock_reference_mean_sd` | Cell 15 | Function | Computes expected fetal weight mean and SD for a given gestational age (weeks) | `src/fetal/explain/hadlock.py` |
| `growth_percentile` | Cell 15 | Function | Computes gestational age-adjusted Z-score and percentile using cumulative normal dist | `src/fetal/explain/hadlock.py` |
| `flag_growth_restriction`| Cell 15 | Function | Flags Small for Gestational Age (SGA < 10th percentile) / Fetal Growth Restriction (FGR) | `src/fetal/explain/hadlock.py` |

---

### 2.3 Stream 1: Head Circumference (HC) & BPD Biometry (HC18 Challenge)
* **Status:** Moved but NOT verified until regression tests are executed. Results are PROVISIONAL (ungrouped KFold; repeat scans of the same patient can leak across folds).
* **Source Locations:** `Capstone_fetalDataset.ipynb` (Cells 16–36)
* **Description:** Segmentation and ellipse fitting on fetal head ultrasound images from the HC18 challenge. Includes exploratory ConvNeXt direct regression (Archived) and the 5-fold EfficientNet-B4 U-Net model with robust ellipse fitting. Uses 3-channel input (`in_channels=3`).

| Entity Name | Source Line/Cell | Type | Role & Description | Target Module / Status |
|---|---|---|---|---|
| `LetterboxResize` | Cells 23, 27, 29, 31, 32 | Class | Aspect-ratio preserving image and mask pad/resize transformation | `src/fetal/data/transforms.py` |
| `HC18SegDataset` | Cells 29, 31, 32 | Class | PyTorch Dataset loading HC18 images, pixel sizes, and ground-truth masks | `src/fetal/data/hc18_dataset.py` |
| `build_segmentation_model` | Cells 26, 29, 31, 32, 34 | Function | Instantiates `smp.Unet(encoder_name='efficientnet-b4', encoder_weights=None, in_channels=3, classes=1)` | `src/fetal/models/hc_segmentation.py` |
| `fit_ellipse_robust` | Cells 32, 34 | Function | Fits ellipse to segmented contour using `cv2.fitEllipseAMS` with fallback to `cv2.fitEllipse` | `src/fetal/eval/ellipse_fitting.py` |
| `hc_bpd_from_ellipse_params`| Cells 23, 25, 27, 31, 32, 34 | Function | Ramanujan formula on semi-axes for circumference (HC) and $2 \times \min(a, b)$ for BPD in mm | `src/fetal/eval/ellipse_fitting.py` |
| `predict_prob_with_tta` | Cell 32 | Function | Test-time augmentation (original + horizontal flip) for segmentation probability | `src/fetal/eval/hc_evaluator.py` |
| `measure_from_prob` | Cell 32 | Function | Thresholds probability mask, unpads to native resolution, and fits ellipse | `src/fetal/eval/hc_evaluator.py` |
| `evaluate_solo_with_tta` | Cell 32 | Function | Evaluates single fold model on validation fold with TTA | `src/fetal/eval/hc_evaluator.py` |
| `evaluate_true_oof` | Cell 32 | Function | Computes overall out-of-fold metrics across all 5 folds | `src/fetal/eval/hc_evaluator.py` |
| `train_one_fold` | Cells 31, 32 | Function | Trains single fold with BCE + Dice loss, AdamW, and CosineAnnealingLR | `src/fetal/train/train_hc.py` |
| `train_hc_model` | Cell 17 | Function | **ARCHIVE** (Approach A): Direct ConvNeXt regression baseline | `src/fetal/archive/hc18_approach_a.py` |
| `train_hc_model_normalized`| Cell 22 | Function | **ARCHIVE** (Approach B): Normalized ConvNeXt regression | `src/fetal/archive/hc18_approach_b.py` |
| `HC18GeometryDataset` | Cells 23, 27 | Class | **ARCHIVE** (Approach D): Direct ellipse parameter regression dataset | `src/fetal/archive/hc18_approach_d.py` |
| `HC18MultiTaskDataset` | Cell 25 | Class | **ARCHIVE** (Approach C): Multi-task segmentation + direct regression dataset | `src/fetal/archive/hc18_approach_c.py` |

---

### 2.4 Stream 2: Abdominal Circumference (AC) & Biometry (FP + UCL Pooled)
* **Status:** VERIFIED in Phase 2a against official test set checkpoints (`v3_full` and 4-model ensemble).
* **Source Locations:** `ac_v3.py`, `ac_v3_ens.py`, `ac_dsnt.py`, `Capstone_fetalDataset.ipynb` (Cells 40–60)
* **Description:** Abdominal landmark detection using Differentiable Spatial Numerical Transform (DSNT) with soft-argmax, Jensen-Shannon divergence loss, and diameter supervision on pooled Fetch-Project (FP) and UCL datasets. Calculates AC in mm via Ramanujan's ellipse approximation formula on semi-axes ($a = \text{TAD}/2, b = \text{APAD}/2$). Split uses a random 12% patient hold-out (`DEV_FRACTION = 0.12`) ensuring zero patient overlap across splits.

| Entity Name | Source Line/File | Type | Role & Description | Target Module |
|---|---|---|---|---|
| `AbdomenDS` | `ac_v3.py` (L108-164) | Class | PyTorch Dataset loading FP/UCL images, metadata, and normalized coordinates | `src/fetal/data/ac_dataset.py` |
| `read_names` | `ac_v3.py` (L51-57) | Function | Extracts image filenames from directory or annotation file | `src/fetal/data/ac_dataset.py` |
| `load_source` | `ac_v3.py` (L59-86) | Function | Parses FP or UCL CSV annotation tables and normalizes landmark coordinates | `src/fetal/data/ac_dataset.py` |
| `build_data` | `ac_v3.py` (L88-92) | Function | Merges FP and UCL datasets, resolving relative paths and px-to-mm scales | `src/fetal/data/ac_dataset.py` |
| `split_data` | `ac_v3.py` (L94-106) | Function | Random 12% patient hold-out dev split with zero patient overlap across train/dev/test | `src/fetal/data/ac_dataset.py` |
| `remove_yellow` | `ac_v3.py` (L166-172) | Function | Inpainting filter to suppress yellow sonographer measurement calipers/text | `src/fetal/data/transforms.py` |
| `read_rgb` | `ac_v3.py` (L174-177) | Function | Reads grayscale or BGR ultrasound frame into RGB uint8 format | `src/fetal/data/transforms.py` |
| `letterbox` | `ac_v3.py` (L179-195) | Function | Pads and scales image to target square resolution preserving aspect ratio | `src/fetal/data/transforms.py` |
| `random_affine` | `ac_v3.py` (L197-227) | Function | Spatial data augmentation: rotation, scale, translation, horizontal flip | `src/fetal/data/transforms.py` |
| `photometric` | `ac_v3.py` (L229-234) | Function | Color jitter and contrast/brightness pertubations | `src/fetal/data/transforms.py` |
| `random_erase` | `ac_v3.py` (L236-241) | Function | Random cutout augmentation for occlusion robustness | `src/fetal/data/transforms.py` |
| `build_model` | `ac_v3.py` (L243-248) | Function | Builds `smp.Unet(encoder_name='efficientnet-b4', in_channels=3, classes=4)` | `src/fetal/models/ac_landmark.py` |
| `decode` | `ac_v3.py` (L250-258) | Function | Soft-argmax (DSNT) spatial coordinate decoding from heatmaps | `src/fetal/models/ac_landmark.py` |
| `PERMS` / `PERMS_T` | `ac_v3.py` (L52-59) | Tensor | 8 allowed landmark relabelings (swapping endpoint pairs and symmetric axes) | `src/fetal/eval/ac_geometry.py` |
| `match` | `ac_v3.py` (L260-267) | Function | Landmark permutation matching (handles symmetric endpoint ordering) | `src/fetal/eval/ac_geometry.py` |
| `ac_t` | `ac_v3.py` (L269-272) | Function | Calculates AC in mm from semi-axes ($a=\text{TAD}/2, b=\text{APAD}/2$) via Ramanujan formula | `src/fetal/eval/ac_geometry.py` |
| `lengths_mm` | `ac_v3.py` (L274-278) | Function | Computes TAD and APAD diameter lengths in millimeters | `src/fetal/eval/ac_geometry.py` |
| `js_loss` | `ac_v3.py` (L280-288) | Function | Jensen-Shannon divergence loss over 2D spatial heatmap distributions | `src/fetal/models/losses.py` |
| `compute_loss` | `ac_v3.py` (L290-305) | Function | Composite loss: Smooth L1 landmark loss + JS divergence + Diameter loss | `src/fetal/models/losses.py` |
| `predict_batch` | `ac_v3.py` (L307-321) | Function | Inference with optional horizontal flip Test-Time Augmentation (TTA) | `src/fetal/eval/ac_evaluator.py` |
| `metrics_rows` | `ac_v3.py` (L323-345) | Function | Generates per-sample prediction error metrics table | `src/fetal/eval/ac_evaluator.py` |
| `run_eval` | `ac_v3.py` (L347-360) | Function | Evaluates model on validation/test DataLoader and returns aggregate stats | `src/fetal/eval/ac_evaluator.py` |
| `summarize` | `ac_v3.py` (L362-378) | Function | Computes MAE, RMSE, Median, Bias, TAD/APAD MAE, NME, and >10mm error rate | `src/fetal/eval/ac_evaluator.py` |
| `atomic_save` | `ac_v3.py` (L380-384) | Function | Atomic checkpoint writer using temporary file swap | `src/fetal/train/checkpoint.py` |
| `run_one` | `ac_v3.py` (L386-407) | Function | Full single-recipe training loop with early stopping and best checkpointing | `src/fetal/train/train_ac.py` |
| `split_seed` | `ac_v3_ens.py` (L33-43)| Function | Generates seed-specific dev holdout split while maintaining official test set | `src/fetal/train/train_ensemble.py` |
| `eval_models` | `ac_v3_ens.py` (L59-99)| Function | Evaluates multi-checkpoint ensemble (Mean and Median pooling) | `src/fetal/eval/ensemble_evaluator.py` |
| `AbdomenLandmarkDataset`| `ac_dsnt.py` | Class | **ARCHIVE** (AC v2): 5-fold cross-validation DSNT dataset | `src/fetal/archive/ac_v2_dsnt.py` |
| `score_image` / `yellow_fraction` | Cell 58 | Function | Diagnostic failure analysis and artifact detection routines | `src/fetal/explain/diagnostics.py` |

---

## 3. Target Repository Architecture & Layout

```
C:\fetal-health-multimodal\
│
├── .gitignore                      <- Anchored /data/, /models/, /logs/; no *.txt; tokens & secrets excluded
├── README.md                       <- Project architecture, reproduction commands, benchmark results
├── requirements.txt                <- Pinned dependencies
├── pyproject.toml                  <- Package metadata & build definition
│
├── configs/                        <- Centralized YAML configuration files
│   ├── paths.yaml                  <- Decoupled environment paths (data, checkpoints, outputs)
│   ├── clinical_xgb.yaml           <- Hyperparameters for Stream 3 clinical XGBoost
│   ├── hc_segmentation.yaml        <- Hyperparameters for Stream 1 HC18 EfficientNet-B4 U-Net
│   ├── ac_v3_single.yaml           <- Winning v3_full recipe hyperparameters
│   └── ac_v3_ensemble.yaml         <- Seeds s1-s4 ensemble configuration
│
├── src/
│   └── fetal/                      <- Core Python Package
│       ├── __init__.py
│       ├── config.py               <- YAML loader with env var override and path validation
│       │
│       ├── data/                   <- Datasets, Loaders, and Preprocessing
│       │   ├── __init__.py
│       │   ├── transforms.py       <- LetterboxResize, remove_yellow, photometric, random_affine
│       │   ├── tabular_cdc.py      <- CDC NATAL data loaders, chunked samplers, code cleaning
│       │   ├── hc18_dataset.py     <- HC18 dataset & letterbox mask loader
│       │   └── ac_dataset.py       <- FP & UCL pooled landmark dataset & metadata parser
│       │
│       ├── models/                 <- Model Architectures & Loss Functions
│       │   ├── __init__.py
│       │   ├── clinical_xgb.py     <- Scikit-learn Pipeline with XGBClassifier
│       │   ├── hc_segmentation.py  <- EfficientNet-B4 U-Net for head biometry (in_channels=3)
│       │   ├── ac_landmark.py      <- EfficientNet-B4 U-Net + Soft-Argmax (DSNT)
│       │   └── losses.py           <- JS-Divergence, Smooth L1, Diameter supervision loss
│       │
│       ├── train/                  <- Training Pipelines & Checkpoint Utilities
│       │   ├── __init__.py
│       │   ├── checkpoint.py       <- Atomic weight serialization
│       │   ├── train_clinical_xgb.py
│       │   ├── train_hc.py
│       │   ├── train_ac.py
│       │   └── train_ensemble.py
│       │
│       ├── eval/                   <- Evaluation, Geometry & Metrics
│       │   ├── __init__.py
│       │   ├── ellipse_fitting.py  <- cv2.fitEllipseAMS, Ramanujan HC/BPD circumference
│       │   ├── ac_geometry.py      <- PERMS, match, lengths_mm, Ramanujan ac_t formula
│       │   ├── hc_evaluator.py     <- HC18 evaluation routines (MAE, RMSE, TTA)
│       │   ├── ac_evaluator.py     <- AC evaluation routines (NME, outlier rates, TTA)
│       │   └── ensemble_evaluator.py <- Multi-model ensemble aggregator (Mean & Median)
│       │
│       ├── explain/                <- Interpretability & Clinical Formulas
│       │   ├── __init__.py
│       │   ├── hadlock.py          <- Hadlock (1984/1985) EFW, percentile & SGA/FGR flags
│       │   ├── diagnostics.py      <- AC failure mode detection, yellow callout detector
│       │   └── visualization.py    <- Diagnostic sheet generators, landmark overlays
│       │
│       └── archive/                <- Archived Prior Iterations (Not Deleted, Isolated)
│           ├── __init__.py
│           ├── hc18_approach_a.py  <- Baseline ConvNeXt direct regression
│           ├── hc18_approach_b.py  <- Normalized ConvNeXt direct regression
│           ├── hc18_approach_c.py  <- Multi-task U-Net (Segmentation + Ellipse)
│           ├── hc18_approach_d.py  <- Huber Loss Geometry model
│           ├── ac_v1_heatmap.py    <- Sigmoid heatmap regression (Cell 51)
│           └── ac_v2_dsnt.py       <- 5-Fold cross-validation DSNT (`ac_dsnt.py`)
│
├── scripts/                        <- Numbered Executable CLI Entrypoints
│   ├── 01_prepare_data.py          <- Prepares / checks raw directories and annotations
│   ├── 02_train_stream3_xgb.py     <- Trains Stream 3 clinical XGBoost
│   ├── 03_train_stream1_hc18.py    <- Trains Stream 1 HC18 EfficientNet-B4 K-Fold
│   ├── 04_train_stream2_ac_v3.py   <- Trains Stream 2 AC v3 model
│   ├── 05_train_stream2_ensemble.py<- Runs 4-seed AC ensemble training
│   ├── 06_evaluate_all.py          <- Evaluates all saved models and generates summary report
│   ├── 07_run_diagnostics.py       <- Runs failure analysis & exports worst-20 image sheets
│   └── run_all.ps1                 <- PowerShell master automation script
│
├── notebooks/                      <- Thin Interactive Runners
│   ├── 01_clinical_risk_demo.ipynb <- Thin runner calling `src.fetal.models.clinical_xgb`
│   ├── 02_hc18_biometry_demo.ipynb <- Thin runner calling `src.fetal.eval.hc_evaluator`
│   ├── 03_ac_biometry_demo.ipynb   <- Thin runner evaluating `v3_full` and ensemble
│   ├── 04_hadlock_efw_demo.ipynb   <- Thin runner computing multimodal EFW & growth flags
│   └── archive/                    <- Original legacy notebook preserved
│       └── Capstone_fetalDataset_legacy.ipynb
│
├── tests/                          <- Automated Verification & Regression Tests
│   ├── conftest.py
│   ├── test_imports.py             <- Verifies all package modules compile and import
│   ├── test_hadlock.py             <- Verifies Hadlock math against canonical clinical test cases
│   ├── test_ac_v3_regression.py    <- Regression test: v3_full on test set == 4.40 mm MAE
│   └── test_ac_ensemble_regression.py <- Regression test: s1-s4 median ensemble == 4.34 mm MAE
│
└── reports/                        <- Markdown and JSON Evaluation Summaries
    ├── baseline_vs_v3_metrics.json
    └── capstone_final_summary.md
```

---

## 4. Hard-Coded Path Decoupling & Configuration Design

### 4.1 All Discovered Hard-Coded Paths

All 71 discovered hard-coded paths fall into distinct categories, parameterized in `configs/paths.yaml`:

| Hard-Coded Source Path | Source Location | Proposed Parameter in `configs/paths.yaml` |
|---|---|---|
| `/content/drive/MyDrive/fetal-health-project/data/raw/{yr}_NATAL.csv` | Notebook Cell 2 | `data.tabular_raw_dir` |
| `/content/drive/MyDrive/fetal-health-project/data/processed/livebirth_sampled_all_years.csv` | Notebook Cells 2, 4 | `data.tabular_processed_dir` |
| `/content/drive/MyDrive/fetal-health-project/data/processed/clinical_training_data.csv` | Notebook Cells 4, 5, 6, 8 | `data.tabular_clinical_csv` |
| `/content/drive/MyDrive/fetal-health-project/models/stream3_clinical_xgb.json` | Notebook Cells 6, 8 | `models.clinical_xgb_checkpoint` |
| `/content/drive/MyDrive/fetal-health-project/data/raw/HC18` | Notebook Cells 23, 25, 27, 29, 31, 32, 34 | `data.hc18_raw_dir` |
| `/content/drive/MyDrive/fetal-health-project/models/hc18_seg_fold*.pt` | Notebook Cells 31, 32, 34 | `models.hc18_checkpoint_dir` |
| `/content/drive/MyDrive/fetal-health-project/data/raw/FETAL_BIOMETRY_ABDOMEN` | Notebook Cells 41, 44, 51, 53, 58; `ac_v3.py` | `data.ac_raw_dir` |
| `/content/drive/MyDrive/fetal-health-project/data/raw/FETAL_BIOMETRY_ABDOMEN/FP/annotations/Abdomen.csv` | Notebook Cell 41 | `data.fp_annotation_csv` |
| `/content/drive/MyDrive/fetal-health-project/data/raw/FETAL_BIOMETRY_ABDOMEN/UCL/annotations/Abdomen.csv` | Notebook Cell 41 | `data.ucl_annotation_csv` |
| `/content/drive/MyDrive/fetal-health-project/models/ac_v3` | `ac_v3.py` | `models.ac_v3_dir` |
| `/content/drive/MyDrive/fetal-health-project/models/ac_v3_ens` | Notebook Cells 75, 76; `ac_v3_ens.py` | `models.ac_ens_dir` |
| `/content/drive/MyDrive/fetal-health-project/models/ac_dsnt_full` | Notebook Cells 55, 58 | `models.ac_dsnt_dir` |
| `/content/stage/{slug}`, `/content/data` | Notebook Cells 65, 73 | `paths.temp_staging_dir` |

### 4.2 Proposed `configs/paths.yaml` Schema

```yaml
# configs/paths.yaml
# Environment-agnostic path configuration
# Override via environment variables: FETAL_DATA_DIR, FETAL_MODEL_DIR

environment:
  default_device: "auto"    # "cuda", "cpu", or "auto"
  num_workers: 0            # Kept at 0 on Windows (original behaviour to prevent multiprocessing issues)

data:
  root_dir: "${FETAL_DATA_DIR:-C:/fetal-data}"
  tabular:
    raw_dir: "${FETAL_DATA_DIR:-C:/fetal-data}/raw/cdc"
    processed_dir: "${FETAL_DATA_DIR:-C:/fetal-data}/processed/cdc"
    training_csv: "${FETAL_DATA_DIR:-C:/fetal-data}/processed/cdc/clinical_training_data.csv"
  hc18:
    root_dir: "${FETAL_DATA_DIR:-C:/fetal-data}/raw/HC18"
    training_images: "${FETAL_DATA_DIR:-C:/fetal-data}/raw/HC18/training_set"
    training_csv: "${FETAL_DATA_DIR:-C:/fetal-data}/raw/HC18/training_set_pixel_size_and_HC.csv"
    test_images: "${FETAL_DATA_DIR:-C:/fetal-data}/raw/HC18/test_set"
    test_csv: "${FETAL_DATA_DIR:-C:/fetal-data}/raw/HC18/test_set_pixel_size.csv"
  abdomen:
    # Direct top-level FP/ and UCL/ support under FETAL_DATA_DIR
    root_dir: "${FETAL_DATA_DIR:-C:/fetal-data}"
    fp_images: "${FETAL_DATA_DIR:-C:/fetal-data}/FP/images/Abdomen"
    fp_annotations: "${FETAL_DATA_DIR:-C:/fetal-data}/FP/annotations"
    ucl_images: "${FETAL_DATA_DIR:-C:/fetal-data}/UCL/images/Abdomen"
    ucl_annotations: "${FETAL_DATA_DIR:-C:/fetal-data}/UCL/annotations/px_to_mm"

models:
  checkpoint_root: "${FETAL_MODEL_DIR:-C:/fetal-models}"
  clinical_xgb: "${FETAL_MODEL_DIR:-C:/fetal-models}/clinical_xgb/stream3_clinical_xgb.json"
  hc18_checkpoints: "${FETAL_MODEL_DIR:-C:/fetal-models}/hc18"
  ac_v3_full: "${FETAL_MODEL_DIR:-C:/fetal-models}/ac_v3/ac_v3/v3_full/best.pt"
  ac_v3_base: "${FETAL_MODEL_DIR:-C:/fetal-models}/ac_v3/ac_v3/base/best.pt"
  ac_ensemble_seeds:
    s1: "${FETAL_MODEL_DIR:-C:/fetal-models}/ac_v3_ens/ac_v3_ens/s1/best.pt"
    s2: "${FETAL_MODEL_DIR:-C:/fetal-models}/ac_v3_ens/ac_v3_ens/s2/best.pt"
    s3: "${FETAL_MODEL_DIR:-C:/fetal-models}/ac_v3_ens/ac_v3_ens/s3/best.pt"
    s4: "${FETAL_MODEL_DIR:-C:/fetal-models}/ac_v3_ens/ac_v3_ens/s4/best.pt"

outputs:
  reports_dir: "./reports"
  diagnostics_dir: "./reports/diagnostics"
```

---

## 5. Regression Testing & Verification Protocol

Before declaring any refactored module production-ready, it must pass the following regression tests:

### Regression Test A: Single-Model `v3_full` on Official Test Set (VERIFIED)
* **Target Metric:** AC MAE **$4.40 \pm 0.01$ mm** (with flip-TTA, 147 test samples).
* **Reference Output (`ac_v3/v3_full/results.json`):**
  * `test_tta.ac_mae`: **4.3956 mm** (rounds to **4.40 mm**)
  * `test_tta.ac_rmse`: **6.1573 mm**
  * `test_tta.ac_median`: **2.9078 mm**
  * `test_tta.tad_mae`: **1.5038 mm**
  * `test_tta.apad_mae`: **1.7626 mm**
  * `test_tta.nme`: **0.0731 (7.31%)**
  * `test_tta.over10`: **0.0816 (8.16%)**
  * Sub-cohorts: FP only MAE **4.02 mm**; UCL only MAE **6.53 mm**.
* **Phase 2a Replicated Metric:** **4.3951 mm** (Discrepancy: $0.0005\text{ mm} \le 0.02\text{ mm}$).
* **Verification Command:**
  ```powershell
  python -m pytest tests/test_ac_v3_regression.py -v
  ```

### Regression Test B: 4-Model Ensemble (`s1` through `s4`) Median Evaluation (VERIFIED)
* **Target Metric:** AC MAE **$4.34 \pm 0.01$ mm** (Median pooling, flip-TTA, 147 test samples).
* **Reference Output (`ac_v3_ens/results.json`):**
  * `ENS_mean.ac_mae`: **4.4693 mm**
  * `ENS_median.ac_mae`: **4.3430 mm** (rounds to **4.34 mm**)
  * `ENS_median.ac_rmse`: **6.0343 mm**
  * `ENS_median.ac_median`: **3.1235 mm**
  * `ENS_median.tad_mae`: **1.6001 mm**
  * `ENS_median.apad_mae`: **1.7034 mm**
  * `ENS_median.over10`: **0.0680 (6.80%)**
* **Phase 2a Replicated Metric:** **4.3426 mm** (Discrepancy: $0.0004\text{ mm} \le 0.02\text{ mm}$).
* **Verification Command:**
  ```powershell
  python -m pytest tests/test_ac_ensemble_regression.py -v
  ```

### Regression Test C: Compilation & Import Verification
* Verifies syntax, imports, and interface compatibility for every single module in `src/fetal/`:
  ```powershell
  python -m pytest tests/test_imports.py -v
  ```

### Regression Test D: Hadlock Math Canonical Verification
* Evaluates `calculate_efw(hc_mm=280.0, bpd_mm=80.0, ac_mm=250.0, fl_mm=55.0)` against the benchmark output ($1403.0\text{ g}$) matching Notebook Cell 13 & 14.

---

## 6. Security, Environment, and Dependency Specifications

### 6.1 Pinned `requirements.txt`
```
torch>=2.0.0
torchvision>=0.15.0
segmentation-models-pytorch==0.5.0
xgboost>=1.7.0
scikit-learn>=1.2.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
opencv-python>=4.8.0
pyyaml>=6.0
pytest>=7.4.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

### 6.2 Strict `.gitignore` Policy (Anchored paths, no *.txt wildcard)
```gitignore
# Python bytecode & cache
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/

# Virtual Environments
.venv/
venv/
ENV/

# Anchored dataset and model directories (Never commit data or weights)
/data/
/models/
/logs/

# Weights & Binary Checkpoints (globally excluded wherever placed)
*.pt
*.pth
*.onnx
*.bin
*.json.model
*.zip
*.tar.gz

# Diagnostics & Reports output artifacts
/reports/diagnostics/

# Credentials, Tokens, Keys, Environment secrets
.env
*.token
kaggle.json
credentials.json
*token*
*secret*
*password*
.DS_Store
Thumbs.db
```

---

## 7. Execution Roadmap

1. **Step 2.1 — Repo Scaffolding (Approved):** Create `C:\fetal-health-multimodal\` directory tree, `.gitignore`, `requirements.txt`, `README.md`, `configs/paths.yaml`, and `src/fetal/config.py`.
2. **Step 2.2 — Core Extraction (Geometry & Clinical Math, Approved):** Extract `hadlock.py`, `ellipse_fitting.py`, and `ac_geometry.py`. Run unit tests locking identical outputs to the original code on random inputs and notebook test vectors.
3. **Step 2.3 — Transforms & Datasets (Awaiting Approval):** Extract `transforms.py`, `tabular_cdc.py`, `hc18_dataset.py`, and `ac_dataset.py`.
4. **Step 2.4 — Models & Losses (Awaiting Approval):** Extract `clinical_xgb.py`, `hc_segmentation.py` (`in_channels=3`), `ac_landmark.py`, and `losses.py`.
5. **Step 2.5 — Evaluators & Regression Execution (Awaiting Approval):** Extract `ac_evaluator.py`, `ensemble_evaluator.py`, `hc_evaluator.py`, and verify regression tests.
6. **Step 2.6 — Thin CLI Runners & Notebooks (Awaiting Approval):** Create numbered scripts `01` through `07`, PowerShell automation `run_all.ps1`, and thin runner notebooks.
7. **Step 2.7 — Archiving (Awaiting Approval):** Safely place legacy code in `src/fetal/archive/` and archive notebook in `notebooks/archive/`.
