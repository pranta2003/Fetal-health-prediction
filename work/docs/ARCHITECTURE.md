# Multimodal Fetal Health Assessment: Architecture Guide

This document provides a high-level overview of the repository architecture and data flow, prepared for project supervisors and stakeholders. It explains what each module does in clear, accessible terms, how data flows through the system, and the current verification status of each component.

---

## 1. Repository Layout & Organization

The repository is modularized into distinct, decoupled components with single responsibilities:

| Folder / Component | Purpose (Supervisor Summary) | Key Files |
|---|---|---|
| `src/fetal/data/` | **Loads and prepares images**: Reads ultrasound scans, resizes with aspect-ratio letterboxing, formats landmark annotations, handles data augmentations, and provides patient-grouped cross-validation splitting. | `ac_dataset.py`, `transforms.py`, `hc18_split.py` |
| `src/fetal/models/` | **Neural network architectures and losses**: Defines deep learning models (e.g. UNet landmark heatmaps) and training loss functions (Jensen-Shannon divergence + cross-entropy). | `ac_landmark.py`, `losses.py` |
| `src/fetal/train/` | **Training loop and recipes**: Manages training epochs, learning rate schedules, seed configs, validation checkpoints, and experiment hyperparameter recipes. | `train_ac.py`, `train_ensemble.py` (+ `checkpoint.py`, `recipes.py`, `constants.py`) |
| `src/fetal/eval/` | **Accuracy measuring, ellipse fitting, & ensemble**: Evaluates trained models, fits ellipses to predicted landmarks using Ramanujan circumference equations, and ensembles multi-seed models via median aggregation. | `ac_evaluator.py`, `ensemble_evaluator.py`, `ellipse_fitting.py`, `ac_geometry.py` |
| `src/fetal/infer/` | **Inference and prediction pipeline**: Single-image biometry and landmark prediction interface (`predict_ac`). | `ac_infer.py` |
| `src/fetal/explain/` | **Clinical interpretation**: Calculates estimated fetal weight (EFW) using Hadlock's formula and computes gestational growth percentiles for fetal growth restriction (FGR) screening. SHAP and Grad-CAM explanations will be integrated here. | `hadlock.py` |
| `src/fetal/clinical/` | **Stream 3 tabular risk module**: Leakage-audited maternal clinical risk model using CDC birth cohorts with temporal train/test splits, prior-shift population calibration, and feature grouping. | `constants.py`, `data.py`, `model.py`, `eval.py`, `audit.py` |
| `src/fetal/config.py` + `configs/` | **Single source of truth for paths**: Resolves dataset folders (`C:\fetal-data`) and model checkpoint directories (`C:\fetal-models`) without hardcoding them in pipeline scripts. | `configs/paths.yaml`, `src/fetal/config.py` |
| `docs/` | **Project documentation**: Architecture guide and system overview for supervisors and developers. | `ARCHITECTURE.md` |
| `archive/` | **Legacy & exploratory archives**: Empty package; original scripts live in `tests/reference/`. | `tests/reference/` |
| `scripts/` | **Numbered buttons (CLI runners)**: Simple, push-button scripts for end-to-end execution: <br> • `04_train_ac.py`: Train single AC landmark models <br> • `05_train_ensemble.py`: Train 4-seed ensemble <br> • `06_eval_ac.py`: Evaluate checkpoints on official test set <br> • `07_audit_clinical.py`: Run clinical risk audit & temporal training <br> • `08_make_manifest.py`: writes/verifies MANIFEST.sha256 <br> • `09_hc18_leakage_report.py`: Audit HC18 patient distribution and old KFold leakage <br> • `run_all.ps1`: Automated test & evaluation orchestrator | `04_*.py`, `05_*.py`, `06_*.py`, `07_*.py`, `08_make_manifest.py`, `09_hc18_leakage_report.py`, `run_all.ps1` |
| `tests/` | **Proof that moved code reproduces original numbers**: Unit tests, equality assertions against backup scripts, and automated regression suites against published baselines. | `test_ac_v3_regression.py`, `test_ac_ensemble_regression.py`, `test_ac_infer.py`, `test_manifest.py`, `test_clinical_regression.py`, `test_hc18_split.py` |
| `notebooks/` | **Exploratory & audit notebooks**: Standalone Jupyter notebooks (e.g., CDC data leakage audit). | `S3_clinical_audit.ipynb` |

---

## 2. Multimodal Data Flow

The pipeline combines imaging streams (ultrasound biometry) with maternal tabular risk data to predict fetal growth restriction and adverse perinatal outcomes:

```
[ Ultrasound Scans ]
         │
         ├──► [ Stream 1: Head Circumference (HC18) ]  ──► BPD, HC (mm)
         │
         ├──► [ Stream 2: Abdominal Circumference ]     ──► TAD, APAD, AC (mm)
         │
         └──► [ Femur Length (FL) ]                   ──► FL (mm)
                                                               │
                                                               ▼
                                               [ Hadlock Formula (Layer 3) ]
                                                               │
                                                               ├─► Estimated Fetal Weight (EFW in grams)
                                                               └─► Gestational Age Growth Percentile (<10% = FGR)
                                                               │
[ Maternal Risk Cohort ]                                       │
(Demographics, Parity, BP, BMI)                                │
         │                                                     │
         ▼                                                     │
[ Stream 3: Tabular Risk (XGBoost) ]                           │
(Leakage-Audited, Prior-Shift Calibrated)                      │
         │                                                     │
         └───────────────────────────┬─────────────────────────┘
                                     │
                                     ▼
                      [ Multimodal Fusion Model ]
                      (Combines Biometry + Maternal Risk)
                                     │
                                     ▼
                        [ Clinical Predictions ]
                        • Adverse Fetal Outcome Risk
                        • Fetal Growth Restriction (FGR)
                        • SHAP / Grad-CAM Explainability
```

---

## 3. Project Status & Verification Table

Every component has a strictly tracked verification status based on automated regression testing:

| Component / Stream | Clinical Task | Verification Status | Benchmark / Reproduction Results |
|---|---|---|---|
| **Stream 2 (AC Biometry)** | Abdominal circumference estimation from ultrasound | **VERIFIED** | Single model: **4.395 mm** AC MAE (official test set). <br> 4-seed ensemble: **4.343 mm** AC MAE. <br> Matches original within regression tolerance. |
| **Stream 3 (Clinical Tabular Risk)** | Maternal stillbirth risk assessment (CDC cohort) | Reproduced on CDC data (retrospective, population-level; not a clinical tool) | All 14 audit rows reproduced on real CSV in Colab ($\Delta = 0.0000$): <br> • Primary model (E1, $\ge 20$ wk): **0.738 AUC** (temporal split) <br> • Clinical-only model (E4, $\ge 20$ wk): **0.707 AUC** <br> • Clinical-only term band (E5, $\ge 37$ wk): **0.692 AUC** <br> • All-features upper bound: **0.787 AUC** (audit only) |
| **Layer 3 (Hadlock EFW)** | Fetal weight and growth percentile calculation | **VERIFIED** | Reproduces the published EFW test vector (1403.0 g). |
| **Stream 1 (HC18 Biometry)** | Head circumference segmentation | **NOT YET EXTRACTED** | Exploration code exists in notebook; must be refactored with patient-grouped folds to avoid data leakage before verification. |
| **Femur Length (FL)** | Femur biometry | **NOT STARTED** | Future work. |
| **FASSD Dataset** | Standard plane detection | **NOT STARTED** | Future work. |
| **Multimodal Fusion** | Joint tabular + imaging risk prediction | **NOT STARTED** | Planned for subsequent phase. |
| **Explainability (SHAP / Grad-CAM)** | Visual and feature attribution explanations | **NOT STARTED** | Initial tabular SHAP group analysis prototyped; full integration planned. |
