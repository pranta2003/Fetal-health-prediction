# Multi-Modal Fusion System Design: Clinical Biometric-to-Risk Bridge

**Document Status**: Architectural Design Specification (Pre-Implementation)  
**Scope**: Methodological bridge uniting Ultrasound Biometrics (HC, AC, FL) and CDC NVSS Tabular Maternal Risk  
**Target Length**: < 2 pages  

---

## 1. Executive Summary & Clinical Motivation
In antenatal care, fetal ultrasound and maternal demographic/clinical history are collected separately but inform a unified clinical decision: identifying fetuses at elevated risk of intrauterine growth restriction (IUGR) and stillbirth. Because available public datasets come from separate cohorts (HC18, UCL/FP AC, Femur ultrasound, and CDC NVSS birth/fetal death records), an end-to-end deep multimodal network trained on identical patients is precluded.

With supervisor approval to leverage separate datasets, this specification establishes two distinct, principled fusion strategies:
1. **Primary Strategy (Plan A — Growth Bridge)**: Maps automated ultrasound biometrics into standard clinical Estimated Fetal Weight (EFW), establishes gestational-age-specific growth percentiles against historical CDC reference distributions, and provides Small-for-Gestational-Age (SGA) priors into the tabular risk model.
2. **Secondary Strategy (Plan B — Calibrated Late Fusion)**: Evaluates late probabilistic fusion combining independent, well-calibrated risk estimates from the image and tabular streams via a constrained meta-learner.

---

## 2. Code Provenance & Feature Specifications

### 2.1 Existing Hadlock Implementation
- **Full Path**: `src/fetal/explain/hadlock.py`
- **Key Primitives**:
  - `calculate_efw(hc_mm, bpd_mm, ac_mm, fl_mm)`: Hadlock 1985 4-parameter log-linear equation yielding EFW in grams.
  - `hadlock_reference_mean_sd(ga_weeks)`: Hadlock 1991 regression formula for mean EFW and 12.7% CV standard deviation across $GA \in [20, 42]$.
  - `growth_percentile(efw_grams, ga_weeks)`: Normal cumulative distribution function mapping EFW and GA to a percentile ($0\text{--}100$).
  - `flag_growth_restriction(percentile, threshold=10)`: Standard clinical cutoff ($\mathbf{1}[\text{percentile} < 10]$).

### 2.2 CDC NVSS Target Columns
From the CDC NVSS vital statistics database (`src/fetal/clinical/constants.py` and `data.py`):
- **Gestational Age**:
  - `COMBGEST`: Combined gestational age in completed weeks (valid clinical range: 20 to 42; 99 = unknown).
  - `OEGest_Comb`: Obstetric estimate of gestational age in completed weeks (fallback/audit reference).
- **Fetal Birth Weight**:
  - `DBWT`: Delivered birth weight in grams (recorded as 4-digit numeric grams: 0227--8165 g; 9999 = unknown/not stated).
  - `BWTR4`: Recoded birth weight category ($<1500$ g, 1500--2499 g, 2500--3999 g, $\ge 4000$ g).

---

## 3. Primary Architecture: The Biometric Growth Bridge (Plan A)

### 3.1 Pipeline Flow
```
[Head Ultrasound]  --> nnU-Net / AMS Ellipse --> HC (mm), BPD (mm)
[Abdomen Image]    --> Regression / U-Net     --> AC (mm)           --> Hadlock EFW (g)
[Femur Ultrasound] --> Landmark / Heatmap    --> FL (mm)
                                                          |
                                                          v
[Gestational Age (weeks)] ---------------------> CDC NVSS Percentile Curve (2021)
                                                          |
                                                          v
                                                  Percentile / SGA Flag
                                                          |
                                                          v
[Maternal Clinical Features] --------------> Tabular GBDT Risk Model --> Calibrated P(Risk)
```

### 3.2 Step-by-Step Mechanism
1. **Biometric Extraction**: HC, AC, and FL are inferred using our verified stage-1 models (e.g. 5-fold nnU-Net for HC, verified AC landmark model, verified Femur model).
2. **EFW Computation**: Parameters are converted to centimeters and passed into `calculate_efw`.
3. **Reference Distribution Construction**:
   - CDC NVSS training records ($\le 2021$, $N = 402,282$) are filtered to singleton deliveries with valid `COMBGEST` ($20 \le \text{GA} \le 42$) and `DBWT` ($227 \le \text{DBWT} \le 8165$).
   - For each gestational week $w \in \{20, \dots, 42\}$, compute empirical percentiles: $p_{10}(w), p_{50}(w), p_{90}(w)$, plus mean and standard deviation.
   - Save table to `docs/results/fusion/gestational_percentiles.csv`.
4. **Bridge Scoring**:
   - Fetal biometric percentile: $Z_{\text{growth}} = (\text{EFW} - \mu_{\text{CDC}}(GA)) / \sigma_{\text{CDC}}(GA)$.
   - SGA indicator: $\text{SGA} = \mathbf{1}[\text{EFW} < p_{10}(GA)]$.
5. **Tabular Conditioning**: Feed $\text{SGA}$ or $Z_{\text{growth}}$ as an antepartum maternal-fetal risk modifier into the tuned LightGBM/CatBoost tabular classifier.

---

## 4. Secondary Architecture: Calibrated Late Fusion (Plan B)

### 4.1 Concept
When full 3-plane biometrics are unavailable or noisy, each stream independently outputs a calibrated posterior:
1. **$P_{\text{tab}}$**: Antenatal maternal baseline probability $P(\text{Stillbirth} \mid X_{\text{maternal}})$, calibrated via isotonic regression on the training cohort ($Brier = 0.17095$).
2. **$P_{\text{bio}}$**: Biometric anomaly probability $P(\text{Fetal Compromise} \mid X_{\text{biometrics}})$, calibrated on biometric CV holdouts.

### 4.2 Fusion Aggregator
- **Log-Odds Ensemble**:
  $$\text{logit}(P_{\text{fused}}) = w_0 + w_{\text{tab}} \cdot \text{logit}(P_{\text{tab}}) + w_{\text{bio}} \cdot \text{logit}(P_{\text{bio}})$$
  constrained such that $w_{\text{tab}}, w_{\text{bio}} \ge 0$, regularized to preserve maternal baseline risk when biometrics are unmeasured.

---

## 5. Implementation Requirements

| Requirement | Plan A (Growth Bridge) | Plan B (Late Fusion) |
|---|---|---|
| **Input Data** | HC, AC, FL (mm) + Gestational Age (wk) | Calibrated probabilities $P_{\text{tab}}, P_{\text{bio}}$ |
| **Artifacts Needed** | `docs/results/fusion/gestational_percentiles.csv` | Isotonic calibration models for both streams |
| **Dependencies** | `src/fetal/explain/hadlock.py` | `sklearn.isotonic.IsotonicRegression` |
| **Leakage Guard** | Percentile curves fit strictly on CDC data $\le 2021$ | Calibration fits strictly on pre-2022 splits |
| **Validation Cohort** | 3-plane UCL/FP test set ($N=22$) + NVSS test ($\ge 2022$) | Held-out multi-modal benchmark cases |

---

## 6. Scientific Integrity & Boundary of Claims

### 6.1 What We MAY Honestly Claim
1. **Population-Level Clinical Link**: Hadlock's formula is standard obstetric practice; linking automated biometrics to CDC population birth-weight distributions produces an interpretable, standardized fetal growth metric.
2. **Strict Antepartum Feature Legality**: Using ultrasound EFW avoids post-delivery leakage; delivery-time birth weight (`DBWT`) is strictly used as the reference standard to calibrate growth distributions, never as an inference-time input.
3. **Rigorous Temporal Split**: All growth percentiles and tabular risk weights are derived strictly from training years $\le 2021$, evaluated on held-out 2022+ records.

### 6.2 What We CANNOT Claim
1. **No Identical-Patient Multimodal Dataset**: We CANNOT claim to have trained or tested an end-to-end model on patients who simultaneously contributed ultrasound raw scans and CDC vital statistics records.
2. **No Direct Image-to-Death Prediction**: Ultrasound pixels do not predict stillbirth directly; they predict geometric biometrics, which inform fetal weight percentiles.
3. **No Clinical Diagnostic Efficacy**: This bridge demonstrates computational feasibility and population-level risk calibration; prospective clinical utility requires randomized obstetric trial validation.
