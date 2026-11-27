# Growth-Bridge Fusion Status

**Last Updated**: 2026-11-27  
**Design Reference**: [docs/FUSION_DESIGN.md](file:///C:/fetal-github-clone/work/docs/FUSION_DESIGN.md)

---

## 1. What Is Built

1. **Stage-1 Ultrasound Biometry Extractors**:
   - **HC (and BPD)**: 5-fold nnU-Net ensemble on HC18 (1.7832 mm official test MAE) and Head Landmark caliper model.
   - **AC**: 4-seed `efficientnet-b4` Landmark U-Net ensemble (4.4028 mm 5-fold OOF MAE on 642 images).
   - **FL**: Femur diaphysis landmark model (`scripts/27_femur_round2.py`).
   - **3-Plane EFW Feasibility**: Validated on held-out 3-plane cohort (`scripts/26_hadlock_demo_step2.py`).
2. **Stage-2 CDC Tabular Risk Model (Stream 3)**:
   - Calibrated LightGBM/CatBoost/Blend models trained on CDC NVSS $\le 2021$ ($N=402,282$) and evaluated on $\ge 2022$ ($N=130,110$), achieving 0.7383 Test AUC.
3. **CDC Live-Birth Gestational Growth Reference Table**:
   - [docs/results/fusion/gestational_percentiles.csv](file:///C:/fetal-github-clone/work/docs/results/fusion/gestational_percentiles.csv): Empirical live-birth (`OUTCOME == 0`) birth weight `p10`, `p50`, `p90`, `mean`, and `std` for completed gestational weeks `20..42` constructed strictly from CDC training years $\le 2021$ (`scripts/81_build_gestational_percentiles.py`).
4. **Growth-Bridge Scoring Function**:
   - [src/fetal/fusion/growth_bridge.py](file:///C:/fetal-github-clone/work/src/fetal/fusion/growth_bridge.py): `assess_fetal_growth(hc_mm, ac_mm, fl_mm, ga_weeks, bpd_mm=None)` computes Hadlock 4-parameter EFW (g) via `fetal.explain.hadlock.calculate_efw` and classifies it into CDC live-birth percentile bands (`below_p10`, `p10_to_p90`, `above_p90`) with `sga` flag and $Z$-score.
   - Verified in [tests/test_growth_bridge.py](file:///C:/fetal-github-clone/work/tests/test_growth_bridge.py) (including the 1403.0 g Hadlock vector and boundary weeks 20 and 42).

---

## 2. What Is Still Missing

1. **Joined Biometric-to-Risk Inference Pipeline**: End-to-end CLI/function taking raw 3-plane ultrasound images + gestational age + maternal clinical covariates to emit fused stillbirth risk $P(\text{Stillbirth} \mid X_{\text{maternal}}, \text{EFW/SGA})$.
2. **Biometric Uncertainty / Calibration Propagation**: Propagating ultrasound EFW measurement error ($\sigma_{\text{EFW}}$) into the SGA boundary classification near $p_{10}$.
3. **Prospective / Linked Validation Cohort**: No dataset on disk contains both raw fetal ultrasound scans and CDC vital-statistics stillbirth outcomes for the same patients.

---

## 3. Open Design Question: Joining the Growth Feature to the Stillbirth Risk Model

Because the ultrasound cohorts (HC18, FP, UCL) and the CDC NVSS vital-statistics cohort ($N=532,392$) come from **disjoint patient populations**, we cannot train an end-to-end multimodal model on paired `(ultrasound, stillbirth)` rows. Two options exist to bridge the growth feature (`SGA` / $Z_{\text{growth}}$) to the tabular stillbirth risk model:

### Option 1: Train Tabular Risk Model with CDC Birth-Weight Proxy (`DBWT` -> $Z_{\text{growth}}$ / `SGA`) During Training, Swap in Ultrasound Hadlock EFW at Inference
- **Mechanism**: During training on CDC $\le 2021$, compute `Z_growth` and `SGA` from delivered birth weight `DBWT` vs live-birth `gestational_percentiles.csv` (or train a weight-conditioned risk model); at inference, compute `EFW` from ultrasound biometrics and feed it into the same `Z_growth` / `SGA` feature slot.
- **Pros**: Directly learns the empirical log-odds relationship between fetal growth restriction (`SGA`) and stillbirth across 402,282 real deliveries with zero synthetic assumptions.
- **Cons**: **Domain / measurement shift** — `DBWT` is post-delivery scale weight (available only after birth, and for stillbirths may reflect post-mortem maceration/weight loss or delivery timing), whereas `EFW` is antepartum ultrasound estimation with ~8–10% measurement error; also risks **label leakage** if `DBWT` reflects delivery-event pathology rather than antepartum growth status.

### Option 2: Two-Stage Decoupled Prior / Likelihood Ratio Calibration (Plan B / Bayesian Log-Odds Update)
- **Mechanism**: Keep the tabular maternal risk model $P_{\text{tab}}(\text{Stillbirth} \mid X_{\text{maternal}})$ strictly free of `DBWT`, and apply a calibrated external obstetric likelihood ratio (or log-odds offset $\Delta_{\text{SGA}}$ derived from published SGA stillbirth relative risk adjusted for ultrasound EFW error) when `assess_fetal_growth` flags `below_p10` / $Z_{\text{growth}} < -1.28$.
- **Pros**: Zero post-delivery `DBWT` leakage in the tabular model; modular, transparent, and explicitly accounts for ultrasound EFW uncertainty without conflating post-mortem `DBWT` with antepartum `EFW`.
- **Cons**: Assumes conditional independence between maternal covariates in `P_tab` and the `SGA` likelihood ratio (may double-count risk factors like hypertension/diabetes that already drive both `P_tab` and `SGA`), and lacks paired validation data to tune the exact log-odds weight.
