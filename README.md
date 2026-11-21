# Growth-Bridged Integration of Ultrasound Biometry and Maternal Clinical Data for Explainable Fetal Health and Stillbirth Risk Prediction

[![CI Tests](https://github.com/pranta/fetal-health-multimodal/actions/workflows/check.yml/badge.svg)](https://github.com/pranta/fetal-health-multimodal/actions)
[![Verification](https://img.shields.io/badge/Report%20Numbers-158%20Verified-brightgreen)](work/docs/report/REPORT_NUMBERS.md)
[![Proof Map](https://img.shields.io/badge/Proof%20Claims-0%20Missing-success)](work/docs/report/PROOF_LOCATIONS.md)

An end-to-end multimodal clinical AI framework bridging 2D fetal ultrasound biometry, fetal growth centiles via Hadlock's formula, and maternal clinical risk data for explainable intrauterine fetal health and stillbirth risk assessment.

---

## 1. Project Overview & Architecture

Accurate assessment of fetal growth and intra-uterine risk requires synthesizing high-resolution morphological biometry from standard ultrasound planes with longitudinal maternal clinical and socio-demographic indicators. 

This repository implements a **six-layer, three-stream multimodal architecture**:
- **Stream 1 (Ultrasound Biometry — Head & Femur):** Deep convolutional biometry extraction:
  - *Head Circumference (HC):* nnU-Net v2 2D architecture and EfficientNet-B4 U-Net ensembles on the HC18 Grand Challenge dataset with ellipse geometry fitting and ring-thickness annotation offset correction.
  - *Femur Length (FL):* ConvNeXt-Tiny with Differentiable Spatial to Numerical Transform (DSNT) landmark detection.
- **Stream 2 (Ultrasound Biometry — Abdomen):**
  - *Abdominal Circumference (AC):* Multi-plane keypoint and contour regression on the FASSD and UCL ultrasound cohorts.
- **Layer 3 (Growth-Bridged Feature Engineering):**
  - Clinical Estimated Fetal Weight (EFW) calculation via Hadlock's multi-biometric formula (`EFW(HC, AC, FL)`), mapped against INTERGROWTH-21st and Hadlock gestational-age centiles to compute growth velocity and percentiles.
- **Stream 3 (Maternal Clinical Risk Modeling):**
  - Population-scale stillbirth risk modeling on the CDC National Vital Statistics System (NVSS) birth/death cohorts (530k+ cases). Includes a complete leakage audit removing gestational delivery confounders, evaluated across LightGBM, XGBoost, and CatBoost with isotonic probability calibration.
- **Layer 4 (Multimodal Cross-Attention Fusion):**
  - Multi-stream transformer fusing visual biometry representations, Hadlock growth centiles, and maternal clinical risk embeddings. *(In progress)*
- **Layer 5 (Explainability & Uncertainty):**
  - Post-hoc model interpretability via TreeSHAP and Grad-CAM, accompanied by biometry confidence intervals. *(In progress)*
- **Layer 6 (Clinical Decision-Support Interface):**
  - Explainable fetal risk scoring and structured triage reporting. *(In progress)*

---

## 2. Honest Component Status Table

All reported metrics are tracked in [`work/docs/results/RESULTS_LEDGER.md`](work/docs/results/RESULTS_LEDGER.md) and automatically verified by [`work/scripts/57_verify_report_numbers.py`](work/scripts/57_verify_report_numbers.py).

| Component / Sub-Model | Target Task | Dataset / Cohort | Status | Primary Metric (Provisional: 2026-11-22) | Automated Test / Verification |
|---|---|---|:---:|:---:|---|
| **Stream 1: Head Biometry (HC)** | Head Circumference (HC) & BPD | HC18 Grand Challenge (335 test images) | **DONE** | **MAE 1.8275 mm** (nnU-Net f0 measfix)<br>Ensemble 5-fold: **1.8350 mm**<br>Dice: **97.99%**, Signed bias: **+0.038 mm** | `tests/test_hc18_submission.py`<br>`tests/test_hc18_nnunet.py` |
| **Stream 1: Femur Biometry (FL)** | Femur Length (FL) | BabyNet++ / FPUS23 / UCL | **DONE** | **MAE 1.1179 mm** (Round 4 ensemble)<br>Length NME: **0.0330** | `tests/test_femur_models.py`<br>`tests/test_femur_round2_checks.py` |
| **Stream 2: Abdomen Biometry (AC)** | Abdominal Circumference (AC) | FASSD & UCL Cohort | **AUDITED / BENCHMARKED** | Baseline ensemble: **4.343 mm**<br>Single model: **4.395 mm**<br>FASSD s42 OOF: **4.946 mm** | `tests/test_ac_v3_regression.py`<br>`tests/test_ac_ensemble_regression.py` |
| **Layer 3: Hadlock Growth Bridging** | EFW & Growth Centiles | Canonical Test Vectors | **DONE** | Vector test EFW: **1403.0 g** (exact)<br>HC/AC/FL Hadlock formula verified | `tests/test_hadlock.py`<br>`tests/test_hadlock_step2.py` |
| **Stream 3: Clinical Risk (CDC)** | Stillbirth Risk Prediction | CDC NVSS 2022+ (532,392 cohort) | **AUDITED & MODERNIZED** | **Test AUC 0.7383** (LightGBM)<br>XGBoost: **0.7382**, CatBoost: **0.7383**<br>Calibrated Brier: **0.17095** | `tests/test_clinical_regression.py`<br>`tests/test_stream3_bench.py` |
| **Layer 4: Cross-Attention Fusion** | Multimodal Fusion | Integrated Biometry + Clinical | **IN PROGRESS** | Scaffolding complete; fusion training pending | In progress |
| **Layer 5: Explainability (XAI)** | SHAP + Grad-CAM | Visual & Tabular Streams | **IN PROGRESS** | Tabular SHAP operational; visual CAM pending | In progress |
| **Layer 6: Clinical Decision Support** | Risk Triage Dashboard | End-to-End System | **IN PROGRESS** | Prototype reporting under development | In progress |

---

## 3. Key Research Findings & Evidence

### 3.1 HC18 Head Biometry: Solving Ellipse Geometry & Measurement Bias
1. **Ellipse Orientation Bug Resolution:** The original pipeline exhibited an angular sign discrepancy in `cv2.fitEllipseAMS`. Enforcing strict quadrant mapping $\theta \in [-\pi/2, \pi/2)$ reduced mean absolute difference on the official Grand Challenge test set from 7.02 mm to 1.95 mm.
2. **Annotation Offset Correction (`measfix`):** High-resolution audit revealed that ground-truth ellipse fitting on training masks leaves a signed bias of $+0.7134 \pm 0.2768$ mm relative to clinical ultrasound calipers (due to manual trace outer border thickness). Applying an empirical radius adjustment ($\Delta r = 0.1134$ mm) on predicted masks shifted the official test score to:
   - **nnU-Net v2 (Fold 0 + measfix):** Abs Diff **1.8275 mm** ($\pm 1.7280$ mm), Signed Bias **+0.0382 mm**, Dice **97.988%**, HD **1.2039 mm** (Rank 107).
   - **5-Model EfficientNet-B4 Ensemble:** Abs Diff **1.8350 mm** ($\pm 1.7456$ mm), Signed Bias **+0.0319 mm**, Dice **97.970%**.
   - **Heterogeneous Blend Candidate:** Combines nnU-Net and 5-model ensemble ellipse parameters via circular modulo averaging, achieving balanced test set distribution (mean HC 175.2869 mm).

### 3.2 Stream 3 Clinical Model: Audit of Target Leakage
1. **Historical Leakage Unmasked:** The original exploratory Colab notebook reported an unrealistic AUC of **0.9684**. An exhaustive 14-row audit confirmed this was caused by direct target leakage from `COMBGEST` (gestational age at time of delivery, which trivially separates fetal deaths) and premature inclusion of losses $<20$ weeks.
2. **Rigorous Temporal Evaluation:**
   - Removing `COMBGEST` and evaluating on clinically available features ($\ge 20$ weeks, temporal split) yields a realistic **Row E1 AUC of 0.7381**.
   - Clinical-only features (Row E4, strictly maternal health history): **AUC 0.7069**.
   - Modernized 2022+ cohort (532,392 total records; 130,110 held-out temporal test set): LightGBM **0.7383**, CatBoost **0.7383**, XGBoost **0.7382**.
   - Isotonic probability calibration successfully reduced test Brier score from **0.19641** to **0.17095**.

---

## 4. Repository Structure

```
.
├── README.md                      <- Root documentation & project roadmap
├── requirements.txt               <- Top-level Python dependencies
├── archive/                       <- Preserved historical artifacts & legacy scripts
│   ├── legacy_notebooks/          <- Capstone_fetalDataset.ipynb (original notebook)
│   ├── legacy_models/             <- Early exploratory weights & abdomen logs
│   └── legacy_src/                <- Early standalone scripts
└── work/                          <- Active production codebase
    ├── configs/                   <- Path and hyperparameter configuration YAMLs
    ├── docs/
    │   ├── CLEANUP_PLAN.md        <- Audit trail for repo reorganization
    │   ├── report/
    │   │   ├── REPORT_NUMBERS.json<- 158 machine-verified metrics
    │   │   ├── REPORT_NUMBERS.md  <- Markdown table of verified numbers
    │   │   └── PROOF_LOCATIONS.md <- Source-to-claim proof location index
    │   └── results/
    │       ├── RESULTS_LEDGER.md  <- Comprehensive ledger of all experimental runs
    │       ├── hc18_official/     <- Official HC18 Grand Challenge evaluation logs
    │       ├── hc18_nnunet/       <- nnU-Net training logs & CPU inference benchmarks
    │       └── stream3_modern/    <- Modernized CDC clinical model metrics
    ├── candidates/                <- Sealed submission candidates (CSVs & SHA256)
    ├── notebooks/                 <- Active analysis & Kaggle automation notebooks
    ├── scripts/                   <- Automated execution, evaluation & verification CLI
    ├── src/
    │   └── fetal/                 <- Production Python package
    │       ├── clinical/          <- Stream 3 data loading, modeling & calibration
    │       ├── eval/              <- Robust ellipse fitting & biometry evaluators
    │       ├── geom/              <- Geometry, transforms & Hadlock calculations
    │       ├── models/            <- Model architectures (UNet, DSNT, ConvNeXt)
    │       └── train/             <- Training recipes, loss functions & EMA
    └── tests/                     <- Test suite (196 passing automated tests)
```

---

## 5. Getting Started & Reproduction Guide

### 5.1 Environment Setup
Install dependencies into a Python 3.10–3.13 environment:
```bash
# Clone the repository
git clone https://github.com/pranta/fetal-health-multimodal.git
cd fetal-health-multimodal/work

# Install dependencies and editable fetal package
pip install -r requirements.txt
pip install -e .
```

### 5.2 Running Automated Tests
Run the comprehensive test suite (196 automated tests):
```bash
# From within the work/ directory
pytest -v
```

### 5.3 Automated Verification of All Report Numbers
Run the independent verification pipeline to validate every claim and number against raw source files:
```bash
# Build the verified report numbers ledger (158 numbers)
python scripts/56_build_report_numbers.py

# Verify all numbers against disk files (asserts 0 errors)
python scripts/57_verify_report_numbers.py

# Build and verify the proof map index (asserts 0 missing claims)
python scripts/65_build_proof_map.py
```

### 5.4 Full Suite Runner (`run_all.ps1`)
On Windows PowerShell:
```powershell
# Fast unit & equivalence tests (default)
.\scripts\run_all.ps1

# Include slow regression tests (evaluates real model checkpoints)
.\scripts\run_all.ps1 -Slow

# Verify weights manifest
.\scripts\run_all.ps1 -Manifest
```

---

## 6. Audit & Evidence Trails

- **Report Numbers Table:** [`work/docs/report/REPORT_NUMBERS.md`](work/docs/report/REPORT_NUMBERS.md)
- **Proof Locations Index:** [`work/docs/report/PROOF_LOCATIONS.md`](work/docs/report/PROOF_LOCATIONS.md)
- **Results Ledger:** [`work/docs/results/RESULTS_LEDGER.md`](work/docs/results/RESULTS_LEDGER.md)
- **Repository Cleanup Plan:** [`work/docs/CLEANUP_PLAN.md`](work/docs/CLEANUP_PLAN.md)
