# Dissertation Evidence & Provenance Map

This document establishes the end-to-end evidence chain for all experimental lines explored across the multimodal fetal assessment system, including successful models, ablations, and failed exploratory runs.

---

## 1. Experimental Approaches Evidence Map

| Field | Approach / Run | Status / Outcome | Notebook Path | Script Path | Result File | Tracked in SHA256SUMS? | Execution Target | Exact Re-Run Command |
|:---|:---|:---|:---|:---|:---|:---:|:---|:---|
| **HC18** | Row A: Direct Regression | Baseline (Cell 22) | `notebooks/benchmarks/hc18_ladder.ipynb` | `scripts/14_verify_hc18_ladder.py` | `docs/results/hc18_ladder/row_A_metrics.json` | Yes | Laptop CPU | `python scripts/14_verify_hc18_ladder.py` |
| **HC18** | Row B: Geometry Approach v3 | Parameter Regression | `notebooks/benchmarks/hc18_ladder.ipynb` | `scripts/14_verify_hc18_ladder.py` | `docs/results/hc18_ladder/row_B_metrics.json` | Yes | Laptop CPU | `python scripts/14_verify_hc18_ladder.py` |
| **HC18** | Row C: Multi-Task U-Net | Multi-Task Shared Encoder | `notebooks/benchmarks/hc18_ladder.ipynb` | `scripts/14_verify_hc18_ladder.py` | `docs/results/hc18_ladder/row_C_metrics.json` | Yes | Laptop CPU | `python scripts/14_verify_hc18_ladder.py` |
| **HC18** | Row D: ResNet-34 U-Net | Segmentation Mask | `notebooks/benchmarks/hc18_ladder.ipynb` | `scripts/15_verify_row_d_5fold.py` | `docs/results/hc18_ladder/d_5fold_results.json` | Yes | Laptop CPU | `python scripts/15_verify_row_d_5fold.py` |
| **HC18** | Row E: EfficientNet-B4 U-Net 384 | Stream 1 Final (OOF) | `notebooks/benchmarks/hc18_ladder.ipynb` | `scripts/10_train_hc18.py` | `docs/results/hc18/metrics.json` | Yes | Laptop CPU | `python scripts/10_train_hc18.py --evaluate` |
| **HC18** | Submission Ensemble 5-Fold Corrected (c1) | Official Test Baseline (MAE 1.95 mm) | `notebooks/benchmarks/hc18_official.ipynb` | `scripts/34_eval_hc18_submission.py` | `docs/results/hc18_official/metrics_ensemble_5fold_corrected.json` | Yes | Grand Challenge Platform | `python scripts/34_eval_hc18_submission.py` |
| **HC18** | Calibration Scale Probe `mult050` | **FAILED / NEGATIVE**: Scaled +0.716 mm, MAE worsened 1.95 -> 2.24 mm | `notebooks/benchmarks/hc18_oof_calibration.ipynb` | `scripts/43_probe_hc18_calibration.py` | `docs/results/hc18_calibration/metrics_probe_mult050.json` | Yes | Grand Challenge Platform | `python scripts/43_probe_hc18_calibration.py` |
| **HC18** | nnU-Net v2 Fold 0 (Raw vs Mask) | Validation (MAE 1.93 mm vs inflated mask) | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f0-250ep/hc18-nnunet-f0-250ep.ipynb` | `scripts/46_eval_nnunet_hc18.py` | `docs/results/hc18_nnunet/f0_eval.json` | Yes | Kaggle GPU (`hc18-nnunet-f0-250ep` v1) | `python scripts/46_eval_nnunet_hc18.py` |
| **HC18** | Measurement Fix Study & nnU-Net vs CSV | **Measurement Fix Verified**: Bias +0.43 -> -0.28 mm, delta_r=0.1134 mm | `notebooks/benchmarks/hc18_official.ipynb` | `scripts/47_fit_measfix_transform.py` | `docs/results/hc18_official/measfix_study.json` | Yes | Laptop CPU | `python scripts/47_fit_measfix_transform.py` |
| **AC** | v1 Heatmap Regression | **FAILED / COLLAPSE**: MAE 182.60 mm vs mean-AC baseline | `notebooks/history/Capstone_fetalDataset_history.ipynb` | `scripts/16_verify_ac_lab_pc.py` | `docs/results/ac_lab_pc/history_table.json` | Yes | Google Colab GPU | Historical notebook cell 51 |
| **AC** | DSNT 5-Fold OOF (789 images) | Intermediate OOF (MAE 5.18 mm) | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `scripts/16_verify_ac_lab_pc.py` | `docs/results/ac_lab_pc/history_table.json` | Yes | Lab PC GPU | `python scripts/16_verify_ac_lab_pc.py` |
| **AC** | v3 Base (Official Test 147) | Ablation Baseline (MAE 5.81 mm) | `notebooks/benchmarks/ac_results.ipynb` | `scripts/06_eval_ac.py` | `docs/results/ac/ac_v3/base/results.json` | Yes | Lab PC GPU | `python scripts/06_eval_ac.py --base` |
| **AC** | v3 Full (Official Test 147) | Single Best Model (MAE 4.40 mm) | `notebooks/benchmarks/ac_results.ipynb` | `scripts/06_eval_ac.py` | `docs/results/ac/ac_v3/v3_full/results.json` | Yes | Lab PC GPU | `python scripts/06_eval_ac.py` |
| **AC** | v3 Ensemble (4-seed median) | Selected Test Model (MAE 4.34 mm) | `notebooks/benchmarks/ac_results.ipynb` | `scripts/06_eval_ac.py` | `docs/results/ac/ac_v3_ens/results.json` | Yes | Lab PC GPU | `python scripts/06_eval_ac.py --ensemble` |
| **AC** | `ac-baseline-s42` (Kaggle Campaign) | Verified 5-Fold OOF (MAE 4.7744 mm, 642 imgs) | `notebooks/kaggle/ac_campaign/ac-baseline-s42/ac-baseline-s42.ipynb` | `scripts/04_train_ac.py` / `scripts/40_ac_compare_recipes.py` | `docs/results/ac_campaign/ac_baseline_s42.json` | Yes | Kaggle GPU (`ac-baseline-s42` v2) | `python scripts/40_ac_compare_recipes.py` |
| **AC** | `ac-ema-s42` v1 | **FAILED / CANCELLED**: EMA decay too slow (steps/ep ~30), dev MAE 53 mm | `notebooks/kaggle/ac_campaign/ac-ema-s42/ac-ema-s42.ipynb` | `scripts/04_train_ac.py` | `docs/results/ac_campaign/AC_CAMPAIGN_NOTES.md` | Yes | Kaggle GPU (`ac-ema-s42` v1) | Cancelled at Epoch 18 |
| **AC** | `ac-ema-s42` v2 | **Fixed EMA (decay 0.99)**: MAE 4.6430 mm (-0.131 mm vs base) | `notebooks/kaggle/ac_campaign/ac-ema-s42/ac-ema-s42.ipynb` | `scripts/40_ac_compare_recipes.py` | `docs/results/ac_campaign/ac_ema_s42.json` | Yes | Kaggle GPU (`ac-ema-s42` v2) | `python scripts/40_ac_compare_recipes.py` |
| **Femur** | Femur Round 1 Ablation | High Outlier Sensitivity / Baseline Drift | `notebooks/benchmarks/femur_ablation.ipynb` | `scripts/17_eval_femur_ablation.py` | `docs/results/femur_ablation/femur_ablation_results.json` | Yes | Lab PC GPU | `python scripts/17_eval_femur_ablation.py` |
| **Femur** | Femur Round 2 Rows R1–R3 | Inferior Ablations (val MAE > 1.15 mm) | `notebooks/benchmarks/femur_round2.ipynb` | `scripts/20_eval_femur_round2.py` | `docs/results/femur_round2/femur_round2_results.json` | Yes | Lab PC GPU | `python scripts/20_eval_femur_round2.py` |
| **Femur** | Femur Round 2 Row R4 | Selected Model (3-seed median, test MAE 1.12 mm) | `notebooks/benchmarks/femur_round2.ipynb` | `scripts/20_eval_femur_round2.py` | `docs/results/femur_round2/femur_round2_results.json` | Yes | Lab PC GPU | `python scripts/20_eval_femur_round2.py` |
| **FASSD** | FASSD Post-Processing Tuning | **NEGATIVE RESULT on Val**: Morphology/connected components degraded artery/stomach | `notebooks/benchmarks/fassd_postproc.ipynb` | `scripts/36_fassd_postproc.py` | `docs/results/fassd/postproc/fassd_postproc_results.json` | Yes | Laptop CPU | `python scripts/36_fassd_postproc.py` |
| **FASSD** | FASSD 4-Structure Ensemble | Selected Part A Candidate (Test Macro Dice 0.7914) | `notebooks/benchmarks/fassd_ensemble.ipynb` | `scripts/37_fassd_ensemble.py` | `docs/results/fassd/ensemble/fassd_ensemble_results.json` | Yes | Laptop CPU | `python scripts/37_fassd_ensemble.py` |
| **Tabular** | Row A: Leakage Baseline | **LEAKAGE DETECTED**: COMBGEST in input gives artificial AUC 0.9684 | `notebooks/history/Capstone_fetalDataset_history.ipynb` | `tests/test_clinical_regression.py` | `tests/test_clinical_regression.py` | Yes | Google Colab | Historical notebook cell 8 |
| **Tabular** | Rows B–D5: Specification Audit | Progressive ablation of leaked features and complete cases | `notebooks/S3_clinical_audit.ipynb` | `scripts/07_audit_clinical.py` | `docs/results/stream3_bench/benchmark_results.json` | Yes | Laptop CPU | `python scripts/07_audit_clinical.py` |
| **Tabular** | E1–E5 Temporal Benchmark | Temporal protocol (Train <=2021, Test >=2022), LR/RF/LGBM/XGB | `notebooks/benchmarks/stream3_bench.ipynb` | `scripts/18_bench_stream3.py` | `docs/results/stream3_bench/benchmark_results.json` | Yes | Laptop CPU | `python scripts/18_bench_stream3.py` |

---

## 2. Kaggle Kernels & Datasets Status Ledger

| Name / Slug | Type | Status | Target / Hardware | Execution Notes / Artifact Provenance |
|:---|:---:|:---:|:---:|:---|
| `prantakumerpandit390/ac-fetal-v1` | Dataset | Active | Kaggle Storage | Packaged via `scripts/41_prepare_ac_kaggle_dataset.py`, contains official train (642) and test (147) images + `folds_v1.csv`. |
| `prantakumerpandit390/hc18-ultrasound` | Dataset | Active | Kaggle Storage | Official Grand Challenge HC18 dataset (999 train + 335 test PNGs + CSVs). |
| `prantakumerpandit390/ac-baseline-s42` | GPU Kernel | **Finished (Succeeded)** | Kaggle T4 GPU | 5-fold CV baseline. Downloaded to `kaggle_outputs/ac_baseline_s42`, verified MAE 4.7744 mm (`8ca2aed7...`). |
| `prantakumerpandit390/ac-ema-s42` (v1) | GPU Kernel | **Cancelled (Failure)** | Kaggle T4 GPU | Cancelled at Epoch 18 due to slow EMA decay warmup failure. |
| `prantakumerpandit390/ac-ema-s42` (v2) | GPU Kernel | **Finished (Succeeded)** | Kaggle T4 GPU | Fixed decay 0.99, downloaded to `kaggle_outputs/ac_ema_s42`, verified MAE 4.6430 mm (`ff8ca055...`). |
| `prantakumerpandit390/ac-baseline-s43` | GPU Kernel | Ready | Kaggle T4 GPU | Seed 43 replication kernel generated via `scripts/42_make_ac_kaggle_kernels.py`. |
| `prantakumerpandit390/ac-wing-s42` | GPU Kernel | Ready | Kaggle T4 GPU | Wing loss coordinate term kernel generated via `scripts/42_make_ac_kaggle_kernels.py`. |
| `prantakumerpandit390/ac-fassd-s42` | GPU Kernel | Ready | Kaggle T4 GPU | FASSD pretrained encoder initialisation (`best_model.pt` mounted and verified intact). |
| `prantakumerpandit390/hc18-nnunet-f0-250ep` | GPU Kernel | **Finished (Succeeded)** | Kaggle T4 GPU | Fold 0 250ep training. Checkpoint intact in Kaggle storage (`b62968b6...`), partial download locally. |
| `prantakumerpandit390/hc18-nnunet-test-f0` | GPU Kernel | In-Fix | Kaggle T4 GPU | Inference on official 335 test set using `hc18-nnunet-f0-250ep` weights via `kernel_sources`. |
| `prantakumerpandit390/hc18-nnunet-f1f2-100ep` | GPU Kernel | Held (Quota Guard) | Kaggle T4 GPU | DO NOT PUSH before Saturday 2026-11-21 06:00 Dhaka (quota reset). |
| `prantakumerpandit390/hc18-nnunet-f3f4-100ep` | GPU Kernel | Held (Quota Guard) | Kaggle T4 GPU | DO NOT PUSH before Saturday 2026-11-21 06:00 Dhaka (quota reset). |
| `prantakumerpandit390/stream3-modern-s1` | CPU Kernel | Ready | Kaggle CPU (No GPU Quota) | Temporal protocol Optuna hyperparameter tuning for CatBoost, LightGBM, and XGBoost. |
