# Report Proof Gaps and Resolution Plan

This document audits all planned or anticipated numbers in the final report that currently lack complete end-to-end notebook or script artifact proof on disk, detailing why the gap exists and the exact operational steps to close each gap.

---

## 1. Summary of Identified Gaps

| ID | Domain / Field | Planned Number / Metric | Current Status on Disk | Root Cause of Gap | Closing Action & Target Artifact |
|---|---|---|---|---|---|
| GAP-01 | HC18 Head Circumference | nnU-Net Fold 0 Test Set HC MAE & Signed Bias vs Ensemble | Only validation fold 0 mask metrics exist (`hc18_nnunet_val_f0.json`) | Test inference kernel `hc18-nnunet-test-f0` dataset discovery path needed correction | Run corrected `hc18-nnunet-test-f0` on Kaggle GPU to produce `submission_nnunet_f0_raw.csv` and `submission_nnunet_f0_measfix.csv`. |
| GAP-02 | HC18 Head Circumference | Full 5-Fold nnU-Net Test & Validation Performance | Only Fold 0 trained and validated | Kaggle GPU weekly quota management (~2.4h remaining; quota resets Sat Nov 21 06:00 Dhaka) | Train folds 1–4 (`hc18-nnunet-f1f2-100ep`, `hc18-nnunet-f3f4-100ep`) after quota refresh. |
| GAP-03 | AC Abdominal Circumference | AC + FASSD Multi-task / Pretraining (Seed 42) | Single-task baseline (`ac_baseline_s42.json`) and EMA v2 (`ac_ema_s42.json`) exist | `ac-fassd-s42` kernel queued behind test inference | Run `ac-fassd-s42` on Kaggle GPU once `hc18-nnunet-test-f0` finishes; save to `docs/results/ac_campaign/ac_fassd_s42.json`. |
| GAP-04 | Tabular Stillbirth Prediction | Modern GBDT Campaign (Optuna CatBoost + LightGBM + XGBoost + Ensembling + Calibration) | Legacy E1–E5 baselines, XGBoost/CatBoost untuned, temporal split exist in `docs/results/tabular/` | Modern hyperparameter tuning and calibrated ensembling pending CPU run | Run Kaggle CPU kernel `stream3-modern-s1` using Optuna (40 trials), evaluate with 5000-sample bootstrap AUC CI, save to `docs/results/stream3_modern/`. |
| GAP-05 | Multi-centre Landmarks | Normalized Mean Error (NME) Formula Verification vs arXiv 2512.16710 Table 2 | Local NME implementation yields 0.0808 (TAD/APAD), whereas paper Table 2 reports 0.0163 / 0.0242 | Normalization basis divergence: inter-landmark Euclidean distance vs bounding box / caliper length | Execute `scripts/59_audit_nme_formulas.py` to compute both definitions across official test (147 images) and OOF (642 images). |
| GAP-06 | Multi-centre Dataset | Dataset Image Count Discrepancy (Abdomen 823 vs 789) | Repo uses 642 train + 147 test = 789 images; arXiv 2512.16710 lists FP 693 + UCL 130 = 823 images (34 missing) | Exclusion of missing/corrupt annotations or quality filters | Execute `scripts/60_audit_dataset_counts.py` to identify exact filenames of 34 excluded images and verify femur (895) and head counts. |
| GAP-07 | HC18 Ladder | Reference Definition Audit for Grand Challenge Ladder Rows A–E | Ladder rows historically mixed ground-truth mask-fitted ellipse HC with official CSV HC (+0.7134 mm bias) | Calibration audit discovered mask-ellipse vs CSV reference offset | Execute `scripts/61_audit_hc18_ladder_refs.py` to recompute available predictions against CSV HC directly. |

---

## 2. Detailed Closure Steps

### GAP-01 & GAP-02: HC18 nnU-Net Pipeline
1. **Inference on Test Set (Fold 0)**:
   - Script: `scripts/52_make_hc18_nnunet_test_kernel.py` generates `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f0/`.
   - Update input dataset path to exact mount: `/kaggle/input/datasets/prantakumerpandit390/hc18-ultrasound/test_set/test_set`.
   - Execute on Kaggle GPU (~15 min runtime).
   - Download predictions, apply measurement-fix shrink ($\Delta = -0.7134$ mm) to generate `submission_nnunet_f0_measfix.csv`.
2. **Remaining Folds (Folds 1–4)**:
   - Do NOT run before Saturday 2026-11-21 06:00 Dhaka.
   - Upon quota refresh, execute 100-epoch kernels `hc18-nnunet-f1f2-100ep` and `hc18-nnunet-f3f4-100ep`.

### GAP-03: AC + FASSD Multi-Task
1. Push `ac-fassd-s42` kernel only after `hc18-nnunet-test-f0` has completed.
2. Confirm encoder weight initialization from FASSD pretraining artifact.
3. Evaluate pooled 5-fold OOF AC MAE, TAD, APAD, and save result ledger to `docs/results/ac_campaign/ac_fassd_s42.json`.

### GAP-04: Tabular Modern Campaign
1. Does not consume GPU quota (runs on Kaggle CPU or local workstation).
2. Protocol strictly replicates temporal split of E1 (no future leakage).
3. Tune CatBoost, LightGBM, and XGBoost with Optuna (40 trials each).
4. Form calibrated probability blend (Isotonic / Platt) and output test ROC-AUC with 95% bootstrap CI.

### GAP-05, GAP-06, GAP-07: Dataset, Metric, and Reference Audits
1. Resolved in Part 2 verification scripts (`scripts/59_audit_nme_formulas.py`, `scripts/60_audit_dataset_counts.py`, `scripts/61_audit_hc18_ladder_refs.py`).
2. All outputs recorded deterministically in JSON ledgers and hashed into `SHA256SUMS`.
