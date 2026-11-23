# Proof Locations Map

This document maps every empirical claim and metric across the thesis/report to its exact on-disk artifact, producing script or notebook, and git commit provenance.

**Total Numbers Mapped**: 194 | **Verified Files on Disk**: 194 | **Missing Files**: 0

---

## HC18 Domain Claims (95 metrics)

| ID | Report Claim | Value | Result File Path | Producing Script / Notebook | Git Commit | File Status |
|---|---|:---:|---|---|:---:|:---:|
| `hc18_submission_c1_mae` | 5-fold ensemble corrected (c1) - HC MAE [Abs Diff] | **1.9472** | `docs/results/hc18_official/metrics_ensemble_5fold_corrected.json` | `notebooks/benchmarks/hc18_official.ipynb` | `768e1a6` | EXISTS |
| `hc18_submission_c1_bias_pred_true` | 5-fold ensemble corrected (c1) - Signed Bias [pred-true] | **0.7844** | `docs/results/hc18_official/metrics_ensemble_5fold_corrected.json` | `notebooks/benchmarks/hc18_official.ipynb` | `768e1a6` | EXISTS |
| `hc18_submission_c1_bias_true_pred` | 5-fold ensemble corrected (c1) - Signed Bias [true-pred] | **-0.7844** | `docs/results/hc18_official/metrics_ensemble_5fold_corrected.json` | `notebooks/benchmarks/hc18_official.ipynb` | `768e1a6` | EXISTS |
| `hc18_submission_c1_dice` | 5-fold ensemble corrected (c1) - Dice (%) | **97.98** | `docs/results/hc18_official/metrics_ensemble_5fold_corrected.json` | `notebooks/benchmarks/hc18_official.ipynb` | `768e1a6` | EXISTS |
| `hc18_submission_c1_hd` | 5-fold ensemble corrected (c1) - HD | **1.2144** | `docs/results/hc18_official/metrics_ensemble_5fold_corrected.json` | `notebooks/benchmarks/hc18_official.ipynb` | `768e1a6` | EXISTS |
| `hc18_probe_mult050_mae` | 5-fold ensemble + scale 1.004065 probe (mult050) - HC MAE [Abs Diff] | **2.2417** | `docs/results/hc18_calibration/metrics_probe_mult050.json` | `notebooks/benchmarks/hc18_oof_calibration.ipynb` | `01879db` | EXISTS |
| `hc18_probe_mult050_bias_pred_true` | 5-fold ensemble + scale 1.004065 probe (mult050) - Signed Bias [pred-true] | **1.4999** | `docs/results/hc18_calibration/metrics_probe_mult050.json` | `notebooks/benchmarks/hc18_oof_calibration.ipynb` | `01879db` | EXISTS |
| `hc18_probe_mult050_bias_true_pred` | 5-fold ensemble + scale 1.004065 probe (mult050) - Signed Bias [true-pred] | **-1.4999** | `docs/results/hc18_calibration/metrics_probe_mult050.json` | `notebooks/benchmarks/hc18_oof_calibration.ipynb` | `01879db` | EXISTS |
| `hc18_probe_mult050_dice` | 5-fold ensemble + scale 1.004065 probe (mult050) - Dice (%) | **97.88** | `docs/results/hc18_calibration/metrics_probe_mult050.json` | `notebooks/benchmarks/hc18_oof_calibration.ipynb` | `01879db` | EXISTS |
| `hc18_probe_mult050_hd` | 5-fold ensemble + scale 1.004065 probe (mult050) - HD | **1.258** | `docs/results/hc18_calibration/metrics_probe_mult050.json` | `notebooks/benchmarks/hc18_oof_calibration.ipynb` | `01879db` | EXISTS |
| `hc18_submission_measfix_mae` | 5-fold ensemble + measfix (delta_r=0.1134mm) - HC MAE [Abs Diff] | **1.835** | `docs/results/hc18_official/metrics_ensemble_5fold_measfix.json` | `notebooks/benchmarks/hc18_official.ipynb` | `7c75db9` | EXISTS |
| `hc18_submission_measfix_bias_pred_true` | 5-fold ensemble + measfix (delta_r=0.1134mm) - Signed Bias [pred-true] | **0.0743** | `docs/results/hc18_official/metrics_ensemble_5fold_measfix.json` | `notebooks/benchmarks/hc18_official.ipynb` | `7c75db9` | EXISTS |
| `hc18_submission_measfix_bias_true_pred` | 5-fold ensemble + measfix (delta_r=0.1134mm) - Signed Bias [true-pred] | **-0.0743** | `docs/results/hc18_official/metrics_ensemble_5fold_measfix.json` | `notebooks/benchmarks/hc18_official.ipynb` | `7c75db9` | EXISTS |
| `hc18_submission_measfix_dice` | 5-fold ensemble + measfix (delta_r=0.1134mm) - Dice (%) | **98.02** | `docs/results/hc18_official/metrics_ensemble_5fold_measfix.json` | `notebooks/benchmarks/hc18_official.ipynb` | `7c75db9` | EXISTS |
| `hc18_submission_measfix_hd` | 5-fold ensemble + measfix (delta_r=0.1134mm) - HD | **1.199** | `docs/results/hc18_official/metrics_ensemble_5fold_measfix.json` | `notebooks/benchmarks/hc18_official.ipynb` | `7c75db9` | EXISTS |
| `hc18_submission_nnunet_f0_measfix_mae` | nnU-Net fold 0 + measfix (delta_r=0.1134mm) - HC MAE [Abs Diff] | **1.8275** | `docs/results/hc18_official/metrics_nnunet_f0_measfix.json` | `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f0/hc18-nnunet-test-f0.ipynb` | `01f8849` | EXISTS |
| `hc18_submission_nnunet_f0_measfix_bias_pred_true` | nnU-Net fold 0 + measfix (delta_r=0.1134mm) - Signed Bias [pred-true] | **0.0382** | `docs/results/hc18_official/metrics_nnunet_f0_measfix.json` | `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f0/hc18-nnunet-test-f0.ipynb` | `01f8849` | EXISTS |
| `hc18_submission_nnunet_f0_measfix_bias_true_pred` | nnU-Net fold 0 + measfix (delta_r=0.1134mm) - Signed Bias [true-pred] | **-0.0382** | `docs/results/hc18_official/metrics_nnunet_f0_measfix.json` | `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f0/hc18-nnunet-test-f0.ipynb` | `01f8849` | EXISTS |
| `hc18_submission_nnunet_f0_measfix_dice` | nnU-Net fold 0 + measfix (delta_r=0.1134mm) - Dice (%) | **97.99** | `docs/results/hc18_official/metrics_nnunet_f0_measfix.json` | `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f0/hc18-nnunet-test-f0.ipynb` | `01f8849` | EXISTS |
| `hc18_submission_nnunet_f0_measfix_hd` | nnU-Net fold 0 + measfix (delta_r=0.1134mm) - HD | **1.2039** | `docs/results/hc18_official/metrics_nnunet_f0_measfix.json` | `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f0/hc18-nnunet-test-f0.ipynb` | `01f8849` | EXISTS |
| `hc18_submission_nnunet_f01234_measfix_mae` | nnU-Net 5-fold ensemble + measfix (delta_r=0.1134mm) - HC MAE [Abs Diff] | **1.7832** | `docs/results/hc18_official/metrics_nnunet_f01234_measfix.json` | `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f01234/hc18-nnunet-test-f01234.ipynb` | `2b473b7` | EXISTS |
| `hc18_submission_nnunet_f01234_measfix_bias_pred_true` | nnU-Net 5-fold ensemble + measfix (delta_r=0.1134mm) - Signed Bias [pred-true] | **0.0684** | `docs/results/hc18_official/metrics_nnunet_f01234_measfix.json` | `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f01234/hc18-nnunet-test-f01234.ipynb` | `2b473b7` | EXISTS |
| `hc18_submission_nnunet_f01234_measfix_bias_true_pred` | nnU-Net 5-fold ensemble + measfix (delta_r=0.1134mm) - Signed Bias [true-pred] | **-0.0684** | `docs/results/hc18_official/metrics_nnunet_f01234_measfix.json` | `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f01234/hc18-nnunet-test-f01234.ipynb` | `2b473b7` | EXISTS |
| `hc18_submission_nnunet_f01234_measfix_dice` | nnU-Net 5-fold ensemble + measfix (delta_r=0.1134mm) - Dice (%) | **98.08** | `docs/results/hc18_official/metrics_nnunet_f01234_measfix.json` | `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f01234/hc18-nnunet-test-f01234.ipynb` | `2b473b7` | EXISTS |
| `hc18_submission_nnunet_f01234_measfix_hd` | nnU-Net 5-fold ensemble + measfix (delta_r=0.1134mm) - HD | **1.1623** | `docs/results/hc18_official/metrics_nnunet_f01234_measfix.json` | `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f01234/hc18-nnunet-test-f01234.ipynb` | `2b473b7` | EXISTS |
| `hc18_nnunet_f0_raw_vs_mask_mae` | nnU-Net 250ep fold 0 (raw) - HC MAE [Abs Diff] | **1.9267** | `docs/results/hc18_nnunet/f0_eval.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f0-250ep/hc18-nnunet-f0-250ep.ipynb` | `110fdd0` | EXISTS |
| `hc18_nnunet_f0_raw_vs_csv_mae` | nnU-Net 250ep fold 0 (raw) - HC MAE [Abs Diff] | **1.8879** | `docs/results/hc18_official/measfix_study.json` | `notebooks/benchmarks/hc18_official.ipynb` | `01879db` | EXISTS |
| `hc18_nnunet_f0_raw_vs_csv_bias_pred_true` | nnU-Net 250ep fold 0 (raw) - Signed Bias [pred-true] | **0.4256** | `docs/results/hc18_official/measfix_study.json` | `notebooks/benchmarks/hc18_official.ipynb` | `01879db` | EXISTS |
| `hc18_nnunet_f0_measfix_vs_csv_mae` | nnU-Net 250ep fold 0 (measfix: delta_r=0.1134mm) - HC MAE [Abs Diff] | **1.8686** | `docs/results/hc18_official/measfix_study.json` | `notebooks/benchmarks/hc18_official.ipynb` | `01879db` | EXISTS |
| `hc18_nnunet_f0_measfix_vs_csv_bias_pred_true` | nnU-Net 250ep fold 0 (measfix: delta_r=0.1134mm) - Signed Bias [pred-true] | **-0.2846** | `docs/results/hc18_official/measfix_study.json` | `notebooks/benchmarks/hc18_official.ipynb` | `01879db` | EXISTS |
| `hc18_nnunet_f1_raw_vs_csv_mae` | nnU-Net 100ep fold 1 (raw) - HC MAE [Abs Diff] | **1.9582** | `docs/results/hc18_nnunet/f1_eval.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f1f2-100ep/hc18-nnunet-f1f2-100ep.ipynb` | `67bfbf6` | EXISTS |
| `hc18_nnunet_f1_raw_vs_csv_bias_pred_true` | nnU-Net 100ep fold 1 (raw) - Signed Bias [pred-true] | **0.791** | `docs/results/hc18_nnunet/f1_eval.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f1f2-100ep/hc18-nnunet-f1f2-100ep.ipynb` | `67bfbf6` | EXISTS |
| `hc18_nnunet_f1_measfix_vs_csv_mae` | nnU-Net 100ep fold 1 (measfix: delta_r=0.1134mm) - HC MAE [Abs Diff] | **1.8767** | `docs/results/hc18_nnunet/f1_eval.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f1f2-100ep/hc18-nnunet-f1f2-100ep.ipynb` | `67bfbf6` | EXISTS |
| `hc18_nnunet_f1_measfix_vs_csv_bias_pred_true` | nnU-Net 100ep fold 1 (measfix: delta_r=0.1134mm) - Signed Bias [pred-true] | **0.0806** | `docs/results/hc18_nnunet/f1_eval.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f1f2-100ep/hc18-nnunet-f1f2-100ep.ipynb` | `67bfbf6` | EXISTS |
| `hc18_nnunet_f2_raw_vs_csv_mae` | nnU-Net 100ep fold 2 (raw) - HC MAE [Abs Diff] | **2.1076** | `docs/results/hc18_nnunet/f2_eval.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f1f2-100ep/hc18-nnunet-f1f2-100ep.ipynb` | `67bfbf6` | EXISTS |
| `hc18_nnunet_f2_raw_vs_csv_bias_pred_true` | nnU-Net 100ep fold 2 (raw) - Signed Bias [pred-true] | **0.447** | `docs/results/hc18_nnunet/f2_eval.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f1f2-100ep/hc18-nnunet-f1f2-100ep.ipynb` | `67bfbf6` | EXISTS |
| `hc18_nnunet_f2_measfix_vs_csv_mae` | nnU-Net 100ep fold 2 (measfix: delta_r=0.1134mm) - HC MAE [Abs Diff] | **2.0993** | `docs/results/hc18_nnunet/f2_eval.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f1f2-100ep/hc18-nnunet-f1f2-100ep.ipynb` | `67bfbf6` | EXISTS |
| `hc18_nnunet_f2_measfix_vs_csv_bias_pred_true` | nnU-Net 100ep fold 2 (measfix: delta_r=0.1134mm) - Signed Bias [pred-true] | **-0.2633** | `docs/results/hc18_nnunet/f2_eval.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f1f2-100ep/hc18-nnunet-f1f2-100ep.ipynb` | `67bfbf6` | EXISTS |
| `hc18_nnunet_f3_raw_vs_csv_mae` | nnU-Net 100ep fold 3 (raw) - HC MAE [Abs Diff] | **2.2217** | `docs/results/hc18_nnunet/f3_eval.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f3f4-100ep/hc18-nnunet-f3f4-100ep.ipynb` | `46c6af8` | EXISTS |
| `hc18_nnunet_f3_raw_vs_csv_bias_pred_true` | nnU-Net 100ep fold 3 (raw) - Signed Bias [pred-true] | **0.728** | `docs/results/hc18_nnunet/f3_eval.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f3f4-100ep/hc18-nnunet-f3f4-100ep.ipynb` | `46c6af8` | EXISTS |
| `hc18_nnunet_f3_measfix_vs_csv_mae` | nnU-Net 100ep fold 3 (measfix: delta_r=0.1134mm) - HC MAE [Abs Diff] | **2.1408** | `docs/results/hc18_nnunet/f3_eval.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f3f4-100ep/hc18-nnunet-f3f4-100ep.ipynb` | `46c6af8` | EXISTS |
| `hc18_nnunet_f3_measfix_vs_csv_bias_pred_true` | nnU-Net 100ep fold 3 (measfix: delta_r=0.1134mm) - Signed Bias [pred-true] | **0.0178** | `docs/results/hc18_nnunet/f3_eval.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f3f4-100ep/hc18-nnunet-f3f4-100ep.ipynb` | `46c6af8` | EXISTS |
| `hc18_nnunet_f4_raw_vs_csv_mae` | nnU-Net 100ep fold 4 (raw) - HC MAE [Abs Diff] | **1.976** | `docs/results/hc18_nnunet/f4_eval.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f3f4-100ep/hc18-nnunet-f3f4-100ep.ipynb` | `46c6af8` | EXISTS |
| `hc18_nnunet_f4_raw_vs_csv_bias_pred_true` | nnU-Net 100ep fold 4 (raw) - Signed Bias [pred-true] | **0.7405** | `docs/results/hc18_nnunet/f4_eval.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f3f4-100ep/hc18-nnunet-f3f4-100ep.ipynb` | `46c6af8` | EXISTS |
| `hc18_nnunet_f4_measfix_vs_csv_mae` | nnU-Net 100ep fold 4 (measfix: delta_r=0.1134mm) - HC MAE [Abs Diff] | **1.8798** | `docs/results/hc18_nnunet/f4_eval.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f3f4-100ep/hc18-nnunet-f3f4-100ep.ipynb` | `46c6af8` | EXISTS |
| `hc18_nnunet_f4_measfix_vs_csv_bias_pred_true` | nnU-Net 100ep fold 4 (measfix: delta_r=0.1134mm) - Signed Bias [pred-true] | **0.0303** | `docs/results/hc18_nnunet/f4_eval.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f3f4-100ep/hc18-nnunet-f3f4-100ep.ipynb` | `46c6af8` | EXISTS |
| `hc18_nnunet_pooled_raw_vs_csv_mae` | nnU-Net pooled folds 0-2 (raw) - HC MAE [Abs Diff] | **1.9853** | `docs/results/hc18_nnunet/pooled_eval.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f1f2-100ep/hc18-nnunet-f1f2-100ep.ipynb` | `fcabfdf` | EXISTS |
| `hc18_nnunet_pooled_raw_vs_csv_bias_pred_true` | nnU-Net pooled folds 0-2 (raw) - Signed Bias [pred-true] | **0.5515** | `docs/results/hc18_nnunet/pooled_eval.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f1f2-100ep/hc18-nnunet-f1f2-100ep.ipynb` | `fcabfdf` | EXISTS |
| `hc18_nnunet_pooled_measfix_vs_csv_mae` | nnU-Net pooled folds 0-2 (measfix: delta_r=0.1134mm) - HC MAE [Abs Diff] | **1.9495** | `docs/results/hc18_nnunet/pooled_eval.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f1f2-100ep/hc18-nnunet-f1f2-100ep.ipynb` | `fcabfdf` | EXISTS |
| `hc18_nnunet_pooled_measfix_vs_csv_bias_pred_true` | nnU-Net pooled folds 0-2 (measfix: delta_r=0.1134mm) - Signed Bias [pred-true] | **-0.1588** | `docs/results/hc18_nnunet/pooled_eval.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f1f2-100ep/hc18-nnunet-f1f2-100ep.ipynb` | `fcabfdf` | EXISTS |
| `hc18_nnunet_pooled_dice` | nnU-Net pooled folds 0-2 - Dice (%) | **97.57** | `docs/results/hc18_nnunet/pooled_eval.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f1f2-100ep/hc18-nnunet-f1f2-100ep.ipynb` | `fcabfdf` | EXISTS |
| `hc18_nnunet_pooled_hd` | nnU-Net pooled folds 0-2 - HD | **1.3964** | `docs/results/hc18_nnunet/pooled_eval.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f1f2-100ep/hc18-nnunet-f1f2-100ep.ipynb` | `fcabfdf` | EXISTS |
| `hc18_nnunet_pooled_5fold_raw_vs_csv_mae` | nnU-Net pooled 5-fold CV (raw) - HC MAE [Abs Diff] | **2.0294** | `docs/results/hc18_nnunet/pooled_eval_5fold.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f3f4-100ep/hc18-nnunet-f3f4-100ep.ipynb` | `46c6af8` | EXISTS |
| `hc18_nnunet_pooled_5fold_raw_vs_csv_bias_pred_true` | nnU-Net pooled 5-fold CV (raw) - Signed Bias [pred-true] | **0.6221** | `docs/results/hc18_nnunet/pooled_eval_5fold.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f3f4-100ep/hc18-nnunet-f3f4-100ep.ipynb` | `46c6af8` | EXISTS |
| `hc18_nnunet_pooled_5fold_measfix_vs_csv_mae` | nnU-Net pooled 5-fold CV (measfix: delta_r=0.1134mm) - HC MAE [Abs Diff] | **1.9733** | `docs/results/hc18_nnunet/pooled_eval_5fold.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f3f4-100ep/hc18-nnunet-f3f4-100ep.ipynb` | `46c6af8` | EXISTS |
| `hc18_nnunet_pooled_5fold_measfix_vs_csv_bias_pred_true` | nnU-Net pooled 5-fold CV (measfix: delta_r=0.1134mm) - Signed Bias [pred-true] | **-0.0882** | `docs/results/hc18_nnunet/pooled_eval_5fold.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f3f4-100ep/hc18-nnunet-f3f4-100ep.ipynb` | `46c6af8` | EXISTS |
| `hc18_nnunet_pooled_5fold_dice` | nnU-Net pooled 5-fold CV - Dice (%) | **97.49** | `docs/results/hc18_nnunet/pooled_eval_5fold.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f3f4-100ep/hc18-nnunet-f3f4-100ep.ipynb` | `46c6af8` | EXISTS |
| `hc18_nnunet_pooled_5fold_hd` | nnU-Net pooled 5-fold CV - HD | **1.4203** | `docs/results/hc18_nnunet/pooled_eval_5fold.json` | `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f3f4-100ep/hc18-nnunet-f3f4-100ep.ipynb` | `46c6af8` | EXISTS |
| `hc18_blend_nnunet_eff_measfix_raw_mae` | Heterogeneous Blend: nnU-Net (measfix) + EffNet (raw) - HC MAE [Abs Diff] | **1.854** | `docs/results/hc18_nnunet/blend_study.json` | `notebooks/benchmarks/hc18_official.ipynb` | `fcabfdf` | EXISTS |
| `hc18_blend_nnunet_eff_measfix_raw_bias` | Heterogeneous Blend: nnU-Net (measfix) + EffNet (raw) - Signed Bias [pred-true] | **-0.0016** | `docs/results/hc18_nnunet/blend_study.json` | `notebooks/benchmarks/hc18_official.ipynb` | `fcabfdf` | EXISTS |
| `hc18_blend_nnunet_eff_measfix_raw_delta` | Heterogeneous Blend: nnU-Net (measfix) + EffNet (raw) vs nnU-Net alone - Delta MAE | **-0.0955 [-0.1869, 0.0178]** | `docs/results/hc18_nnunet/blend_study.json` | `notebooks/benchmarks/hc18_official.ipynb` | `fcabfdf` | EXISTS |
| `hc18_test_f012_gpu_cpu_max_diff` | nnU-Net test 3-fold ensemble GPU vs CPU inference - Max Abs Diff (GPU vs CPU) | **0.0048** | `docs/results/hc18_nnunet/test_f012_candidates_summary.json` | `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f012/hc18-nnunet-test-f012.ipynb` | `d25dfb9` | EXISTS |
| `hc18_test_f012_raw_mean_hc` | nnU-Net test 3-fold ensemble (raw) - Mean HC (mm) | **175.9986** | `docs/results/hc18_nnunet/test_f012_candidates_summary.json` | `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f012/hc18-nnunet-test-f012.ipynb` | `d25dfb9` | EXISTS |
| `hc18_test_f012_measfix_mean_hc` | nnU-Net test 3-fold ensemble + measfix (delta_r=0.1134mm) - Mean HC (mm) | **175.2884** | `docs/results/hc18_nnunet/test_f012_candidates_summary.json` | `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f012/hc18-nnunet-test-f012.ipynb` | `d25dfb9` | EXISTS |
| `hc18_test_f012_measfix_diff_vs_f0` | nnU-Net test 3-fold ensemble + measfix vs fold 0 measfix - Mean HC diff vs f0 (mm) | **0.0189** | `docs/results/hc18_nnunet/test_f012_candidates_summary.json` | `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f012/hc18-nnunet-test-f012.ipynb` | `d25dfb9` | EXISTS |
| `hc18_test_f012_measfix_diff_vs_ens` | nnU-Net test 3-fold ensemble + measfix vs 5-fold ensemble measfix - Mean HC diff vs 5-fold ens (mm) | **-0.0172** | `docs/results/hc18_nnunet/test_f012_candidates_summary.json` | `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f012/hc18-nnunet-test-f012.ipynb` | `d25dfb9` | EXISTS |
| `hc18_test_f01234_raw_mean_hc` | nnU-Net test 5-fold ensemble (raw) - Mean HC (mm) | **176.01** | `docs/results/hc18_nnunet/test_f01234_candidates_summary.json` | `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f01234/hc18-nnunet-test-f01234.ipynb` | `3de958b` | EXISTS |
| `hc18_test_f01234_measfix_mean_hc` | nnU-Net test 5-fold ensemble + measfix (delta_r=0.1134mm) - Mean HC (mm) | **175.2998** | `docs/results/hc18_nnunet/test_f01234_candidates_summary.json` | `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f01234/hc18-nnunet-test-f01234.ipynb` | `3de958b` | EXISTS |
| `hc18_test_f01234_measfix_diff_vs_f012` | nnU-Net test 5-fold ensemble + measfix vs 3-fold ensemble measfix - Mean HC diff vs f012 (mm) | **0.0113** | `docs/results/hc18_nnunet/test_f01234_candidates_summary.json` | `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f01234/hc18-nnunet-test-f01234.ipynb` | `3de958b` | EXISTS |
| `hc18_test_f01234_measfix_abs_diff_vs_f012` | nnU-Net test 5-fold ensemble + measfix vs 3-fold ensemble measfix - Mean Absolute HC diff vs f012 (mm) | **0.1728** | `docs/results/hc18_nnunet/test_f01234_candidates_summary.json` | `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f01234/hc18-nnunet-test-f01234.ipynb` | `3de958b` | EXISTS |
| `hc18_test_f01234_measfix_diff_vs_ens` | nnU-Net test 5-fold ensemble + measfix vs 5-fold ensemble measfix - Mean HC diff vs 5-fold ens (mm) | **-0.0059** | `docs/results/hc18_nnunet/test_f01234_candidates_summary.json` | `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f01234/hc18-nnunet-test-f01234.ipynb` | `3de958b` | EXISTS |
| `hc18_blend_nnunet_f01234_ens5_mean_hc` | Heterogeneous Blend (5-fold nnU-Net measfix + 5-fold ens measfix) - Mean HC (mm) | **175.3021** | `docs/results/hc18_nnunet/test_f01234_candidates_summary.json` | `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f01234/hc18-nnunet-test-f01234.ipynb` | `3de958b` | EXISTS |
| `hc18_single_oof_999_raw_vs_csv_mae` | EfficientNet-B4 U-Net 384 (single model OOF) - HC MAE [Abs Diff] | **2.3019** | `docs/results/hc18_official/measfix_study.json` | `notebooks/benchmarks/hc18_official.ipynb` | `01879db` | EXISTS |
| `hc18_single_oof_999_raw_vs_csv_bias_pred_true` | EfficientNet-B4 U-Net 384 (single model OOF) - Signed Bias [pred-true] | **-1.2576** | `docs/results/hc18_official/measfix_study.json` | `notebooks/benchmarks/hc18_official.ipynb` | `01879db` | EXISTS |
| `hc18_annotation_mask_vs_csv_diff_mean` | OpenCV fitEllipse on Ground Truth Training Annotations - Signed Bias [pred-true] | **0.7126** | `docs/results/hc18_official/measfix_study.json` | `notebooks/benchmarks/hc18_official.ipynb` | `01879db` | EXISTS |
| `hc18_annotation_mask_vs_csv_delta_r` | Radius shrink offset (delta_r) - Delta Radius | **0.1134** | `docs/results/hc18_official/measfix_study.json` | `notebooks/benchmarks/hc18_official.ipynb` | `01879db` | EXISTS |
| `hc18_ladder_row_a_hc_mae` | Row A: Direct Regression (Cell 22): Direct continuous scalar regression - HC MAE | **10.6728** | `docs/results/hc18_ladder/ladder_results.json` | `notebooks/benchmarks/hc18_ladder.ipynb` | `52119c8` | EXISTS |
| `hc18_ladder_row_a_bpd_mae` | Row A: Direct Regression (Cell 22): Direct continuous scalar regression - BPD MAE | **3.2616** | `docs/results/hc18_ladder/ladder_results.json` | `notebooks/benchmarks/hc18_ladder.ipynb` | `52119c8` | EXISTS |
| `hc18_ladder_row_b_hc_mae` | Row B: Geometry Approach v3 (Cell 27): Ellipse parameter regression (Huber loss) - HC MAE | **11.5964** | `docs/results/hc18_ladder/ladder_results.json` | `notebooks/benchmarks/hc18_ladder.ipynb` | `52119c8` | EXISTS |
| `hc18_ladder_row_b_bpd_mae` | Row B: Geometry Approach v3 (Cell 27): Ellipse parameter regression (Huber loss) - BPD MAE | **3.2198** | `docs/results/hc18_ladder/ladder_results.json` | `notebooks/benchmarks/hc18_ladder.ipynb` | `52119c8` | EXISTS |
| `hc18_ladder_row_c_hc_mae` | Row C: Multi-Task U-Net (Cell 25): Segmentation + shared-encoder ellipse regression - HC MAE | **10.5104** | `docs/results/hc18_ladder/ladder_results.json` | `notebooks/benchmarks/hc18_ladder.ipynb` | `52119c8` | EXISTS |
| `hc18_ladder_row_c_bpd_mae` | Row C: Multi-Task U-Net (Cell 25): Segmentation + shared-encoder ellipse regression - BPD MAE | **2.9474** | `docs/results/hc18_ladder/ladder_results.json` | `notebooks/benchmarks/hc18_ladder.ipynb` | `52119c8` | EXISTS |
| `hc18_ladder_row_d_hc_mae` | Row D: ResNet-34 U-Net (Cell 31 recipe): Segmentation mask + robust native ellipse fit - HC MAE | **1.9246** | `docs/results/hc18_ladder/ladder_results.json` | `notebooks/benchmarks/hc18_ladder.ipynb` | `52119c8` | EXISTS |
| `hc18_ladder_row_d_bpd_mae` | Row D: ResNet-34 U-Net (Cell 31 recipe): Segmentation mask + robust native ellipse fit - BPD MAE | **0.6003** | `docs/results/hc18_ladder/ladder_results.json` | `notebooks/benchmarks/hc18_ladder.ipynb` | `52119c8` | EXISTS |
| `hc18_ladder_row_e_hc_mae` | Row E: EfficientNet-B4 U-Net 384 (Stream 1 Final): High-res segmentation + TTA + fitEllipseAMS - HC MAE | **2.4817** | `docs/results/hc18_ladder/ladder_results.json` | `notebooks/benchmarks/hc18_ladder.ipynb` | `52119c8` | EXISTS |
| `hc18_ladder_row_e_bpd_mae` | Row E: EfficientNet-B4 U-Net 384 (Stream 1 Final): High-res segmentation + TTA + fitEllipseAMS - BPD MAE | **0.6982** | `docs/results/hc18_ladder/ladder_results.json` | `notebooks/benchmarks/hc18_ladder.ipynb` | `52119c8` | EXISTS |
| `hc18_ladder_row_d_5fold_no_tta_hc_mae` | Row D: ResNet-34 U-Net 5-fold pooled (no TTA) - HC MAE | **2.2688** | `docs/results/hc18_ladder/d_5fold_results.json` | `notebooks/benchmarks/hc18_ladder.ipynb` | `52119c8` | EXISTS |
| `hc18_ladder_row_d_5fold_tta_hc_mae` | Row D: ResNet-34 U-Net 5-fold pooled (with TTA) - HC MAE | **2.2323** | `docs/results/hc18_ladder/d_5fold_results.json` | `notebooks/benchmarks/hc18_ladder.ipynb` | `52119c8` | EXISTS |
| `hc18_ladder_row_e_5fold_oof_hc_mae` | Row E: EfficientNet-B4 U-Net 384 5-fold pooled OOF - HC MAE | **2.3019** | `docs/results/hc18_ladder/d_5fold_results.json` | `notebooks/benchmarks/hc18_ladder.ipynb` | `52119c8` | EXISTS |
| `pub_hc18_fiorentino2020_with_post_proc_mad` | Published: Fiorentino2020 (with post-proc) (via Mask-R2CNN (PMC8580944)) - HC MAE [MAD] | **1.9** | `docs/results/published_refs.json` | `docs/results/published_refs.json` | `a6d1a58` | EXISTS |
| `pub_hc18_mask_r2cnn_mad` | Published: Mask-R2CNN (via Mask-R2CNN (PMC8580944)) - HC MAE [MAD] | **1.95** | `docs/results/published_refs.json` | `docs/results/published_refs.json` | `a6d1a58` | EXISTS |
| `pub_hc18_sobhaninia2019_mad` | Published: Sobhaninia2019 (via Mask-R2CNN (PMC8580944)) - HC MAE [MAD] | **2.12** | `docs/results/published_refs.json` | `docs/results/published_refs.json` | `a6d1a58` | EXISTS |
| `pub_hc18_al_bander2019_mad` | Published: Al-Bander2019 (via Mask-R2CNN (PMC8580944)) - HC MAE [MAD] | **2.33** | `docs/results/published_refs.json` | `docs/results/published_refs.json` | `a6d1a58` | EXISTS |
| `pub_hc18_rong2019_mad` | Published: Rong2019 (via Mask-R2CNN (PMC8580944)) - HC MAE [MAD] | **2.45** | `docs/results/published_refs.json` | `docs/results/published_refs.json` | `a6d1a58` | EXISTS |
| `pub_hc18_van_den_heuvel2018_mad` | Published: van den Heuvel2018 (via Mask-R2CNN (PMC8580944)) - HC MAE [MAD] | **2.8** | `docs/results/published_refs.json` | `docs/results/published_refs.json` | `a6d1a58` | EXISTS |

---

## AC Domain Claims (52 metrics)

| ID | Report Claim | Value | Result File Path | Producing Script / Notebook | Git Commit | File Status |
|---|---|:---:|---|---|:---:|:---:|
| `ac_baseline_s42_ac_mae` | v3_full baseline (seed 42) - AC MAE | **4.7744** | `docs/results/ac_campaign/ac_baseline_s42.json` | `notebooks/kaggle/ac_campaign/ac-baseline-s42/ac-baseline-s42.ipynb` | `2e42c9d` | EXISTS |
| `ac_baseline_s42_ac_rmse` | v3_full baseline (seed 42) - AC RMSE | **7.6074** | `docs/results/ac_campaign/ac_baseline_s42.json` | `notebooks/kaggle/ac_campaign/ac-baseline-s42/ac-baseline-s42.ipynb` | `2e42c9d` | EXISTS |
| `ac_baseline_s42_ac_median` | v3_full baseline (seed 42) - AC Median | **2.8474** | `docs/results/ac_campaign/ac_baseline_s42.json` | `notebooks/kaggle/ac_campaign/ac-baseline-s42/ac-baseline-s42.ipynb` | `2e42c9d` | EXISTS |
| `ac_baseline_s42_ac_bias` | v3_full baseline (seed 42) - AC Bias [pred-true] | **-0.0826** | `docs/results/ac_campaign/ac_baseline_s42.json` | `notebooks/kaggle/ac_campaign/ac-baseline-s42/ac-baseline-s42.ipynb` | `2e42c9d` | EXISTS |
| `ac_baseline_s42_tad_mae` | v3_full baseline (seed 42) - TAD MAE | **1.7331** | `docs/results/ac_campaign/ac_baseline_s42.json` | `notebooks/kaggle/ac_campaign/ac-baseline-s42/ac-baseline-s42.ipynb` | `2e42c9d` | EXISTS |
| `ac_baseline_s42_apad_mae` | v3_full baseline (seed 42) - APAD MAE | **2.0793** | `docs/results/ac_campaign/ac_baseline_s42.json` | `notebooks/kaggle/ac_campaign/ac-baseline-s42/ac-baseline-s42.ipynb` | `2e42c9d` | EXISTS |
| `ac_baseline_s42_nme` | v3_full baseline (seed 42) - NME | **0.0808** | `docs/results/ac_campaign/ac_baseline_s42.json` | `notebooks/kaggle/ac_campaign/ac-baseline-s42/ac-baseline-s42.ipynb` | `2e42c9d` | EXISTS |
| `ac_baseline_s42_over10_pct` | v3_full baseline (seed 42) - >10mm (%) | **12.15** | `docs/results/ac_campaign/ac_baseline_s42.json` | `notebooks/kaggle/ac_campaign/ac-baseline-s42/ac-baseline-s42.ipynb` | `2e42c9d` | EXISTS |
| `ac_ema_s42_ac_mae` | v3_full + EMA decay 0.99 (seed 42) - AC MAE | **4.643** | `docs/results/ac_campaign/ac_ema_s42.json` | `notebooks/kaggle/ac_campaign/ac-ema-s42/ac-ema-s42.ipynb` | `e913ba7` | EXISTS |
| `ac_ema_s42_ac_rmse` | v3_full + EMA decay 0.99 (seed 42) - AC RMSE | **7.2886** | `docs/results/ac_campaign/ac_ema_s42.json` | `notebooks/kaggle/ac_campaign/ac-ema-s42/ac-ema-s42.ipynb` | `e913ba7` | EXISTS |
| `ac_ema_s42_ac_median` | v3_full + EMA decay 0.99 (seed 42) - AC Median | **2.8375** | `docs/results/ac_campaign/ac_ema_s42.json` | `notebooks/kaggle/ac_campaign/ac-ema-s42/ac-ema-s42.ipynb` | `e913ba7` | EXISTS |
| `ac_ema_s42_ac_bias` | v3_full + EMA decay 0.99 (seed 42) - AC Bias [pred-true] | **-0.791** | `docs/results/ac_campaign/ac_ema_s42.json` | `notebooks/kaggle/ac_campaign/ac-ema-s42/ac-ema-s42.ipynb` | `e913ba7` | EXISTS |
| `ac_ema_s42_tad_mae` | v3_full + EMA decay 0.99 (seed 42) - TAD MAE | **1.6868** | `docs/results/ac_campaign/ac_ema_s42.json` | `notebooks/kaggle/ac_campaign/ac-ema-s42/ac-ema-s42.ipynb` | `e913ba7` | EXISTS |
| `ac_ema_s42_apad_mae` | v3_full + EMA decay 0.99 (seed 42) - APAD MAE | **2.0389** | `docs/results/ac_campaign/ac_ema_s42.json` | `notebooks/kaggle/ac_campaign/ac-ema-s42/ac-ema-s42.ipynb` | `e913ba7` | EXISTS |
| `ac_ema_s42_nme` | v3_full + EMA decay 0.99 (seed 42) - NME | **0.0793** | `docs/results/ac_campaign/ac_ema_s42.json` | `notebooks/kaggle/ac_campaign/ac-ema-s42/ac-ema-s42.ipynb` | `e913ba7` | EXISTS |
| `ac_ema_s42_over10_pct` | v3_full + EMA decay 0.99 (seed 42) - >10mm (%) | **11.84** | `docs/results/ac_campaign/ac_ema_s42.json` | `notebooks/kaggle/ac_campaign/ac-ema-s42/ac-ema-s42.ipynb` | `e913ba7` | EXISTS |
| `ac_fassd_s42_ac_mae` | v3_full + FASSD pretraining (seed 42) - AC MAE | **4.9459** | `docs/results/ac_campaign/ac_fassd_s42.json` | `notebooks/kaggle/ac_campaign/ac-fassd-s42/ac-fassd-s42.ipynb` | `67bfbf6` | EXISTS |
| `ac_fassd_s42_ac_rmse` | v3_full + FASSD pretraining (seed 42) - AC RMSE | **8.3335** | `docs/results/ac_campaign/ac_fassd_s42.json` | `notebooks/kaggle/ac_campaign/ac-fassd-s42/ac-fassd-s42.ipynb` | `67bfbf6` | EXISTS |
| `ac_fassd_s42_ac_median` | v3_full + FASSD pretraining (seed 42) - AC Median | **2.9702** | `docs/results/ac_campaign/ac_fassd_s42.json` | `notebooks/kaggle/ac_campaign/ac-fassd-s42/ac-fassd-s42.ipynb` | `67bfbf6` | EXISTS |
| `ac_fassd_s42_ac_bias` | v3_full + FASSD pretraining (seed 42) - AC Bias [pred-true] | **-0.4653** | `docs/results/ac_campaign/ac_fassd_s42.json` | `notebooks/kaggle/ac_campaign/ac-fassd-s42/ac-fassd-s42.ipynb` | `67bfbf6` | EXISTS |
| `ac_fassd_s42_tad_mae` | v3_full + FASSD pretraining (seed 42) - TAD MAE | **1.7911** | `docs/results/ac_campaign/ac_fassd_s42.json` | `notebooks/kaggle/ac_campaign/ac-fassd-s42/ac-fassd-s42.ipynb` | `67bfbf6` | EXISTS |
| `ac_fassd_s42_apad_mae` | v3_full + FASSD pretraining (seed 42) - APAD MAE | **2.1315** | `docs/results/ac_campaign/ac_fassd_s42.json` | `notebooks/kaggle/ac_campaign/ac-fassd-s42/ac-fassd-s42.ipynb` | `67bfbf6` | EXISTS |
| `ac_fassd_s42_nme` | v3_full + FASSD pretraining (seed 42) - NME | **0.082** | `docs/results/ac_campaign/ac_fassd_s42.json` | `notebooks/kaggle/ac_campaign/ac-fassd-s42/ac-fassd-s42.ipynb` | `67bfbf6` | EXISTS |
| `ac_fassd_s42_over10_pct` | v3_full + FASSD pretraining (seed 42) - >10mm (%) | **12.62** | `docs/results/ac_campaign/ac_fassd_s42.json` | `notebooks/kaggle/ac_campaign/ac-fassd-s42/ac-fassd-s42.ipynb` | `67bfbf6` | EXISTS |
| `ac_hist_v1_heatmap_regression_colab_mae` | Historical AC: v1 heatmap regression (Colab) - AC MAE | **182.6** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `ac_hist_v1_heatmap_regression_colab_rmse` | Historical AC: v1 heatmap regression (Colab) - AC RMSE | **195.13** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `ac_hist_dsnt_5_fold_oof_mae` | Historical AC: DSNT 5-fold OOF - AC MAE | **5.18** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `ac_hist_dsnt_5_fold_oof_rmse` | Historical AC: DSNT 5-fold OOF - AC RMSE | **9.07** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `ac_hist_v3_base_mae` | Historical AC: v3 base - AC MAE | **5.81** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `ac_hist_v3_base_rmse` | Historical AC: v3 base - AC RMSE | **8.87** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `ac_hist_v3_full_mae` | Historical AC: v3_full - AC MAE | **4.4** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `ac_hist_v3_full_rmse` | Historical AC: v3_full - AC RMSE | **6.16** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `ac_hist_v3_ens_s1_mae` | Historical AC: v3_ens (s1) - AC MAE | **4.96** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `ac_hist_v3_ens_s1_rmse` | Historical AC: v3_ens (s1) - AC RMSE | **7.33** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `ac_hist_v3_ens_s2_mae` | Historical AC: v3_ens (s2) - AC MAE | **4.32** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `ac_hist_v3_ens_s2_rmse` | Historical AC: v3_ens (s2) - AC RMSE | **6.05** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `ac_hist_v3_ens_s3_mae` | Historical AC: v3_ens (s3) - AC MAE | **4.54** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `ac_hist_v3_ens_s3_rmse` | Historical AC: v3_ens (s3) - AC RMSE | **6.59** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `ac_hist_v3_ens_s4_mae` | Historical AC: v3_ens (s4) - AC MAE | **4.49** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `ac_hist_v3_ens_s4_rmse` | Historical AC: v3_ens (s4) - AC RMSE | **6.27** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `ac_hist_v3_ens_res512_mae` | Historical AC: v3_ens (res512) - AC MAE | **5.02** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `ac_hist_v3_ens_res512_rmse` | Historical AC: v3_ens (res512) - AC RMSE | **8.01** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `ac_hist_v3_ens_4_seed_mean_mae` | Historical AC: v3_ens (4-seed mean) - AC MAE | **4.47** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `ac_hist_v3_ens_4_seed_mean_rmse` | Historical AC: v3_ens (4-seed mean) - AC RMSE | **6.31** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `ac_hist_v3_ens_4_seed_median_mae` | Historical AC: v3_ens (4-seed median) - AC MAE | **4.34** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `ac_hist_v3_ens_4_seed_median_rmse` | Historical AC: v3_ens (4-seed median) - AC RMSE | **6.03** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `ac_hist_smoke_log_code_check_mae` | Historical AC: smoke_log (code check) - AC MAE | **111.65** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `ac_hist_smoke_log_code_check_rmse` | Historical AC: smoke_log (code check) - AC RMSE | **117.88** | `docs/results/ac_lab_pc/history_table.json` | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | `52119c8` | EXISTS |
| `pub_ac_combined_to_combined_apad_nme` | Published: BiometryNet Combined -> Combined (arXiv 2512.16710) - APAD NME | **0.07** | `docs/results/published_refs.json` | `docs/results/published_refs.json` | `a6d1a58` | EXISTS |
| `pub_ac_combined_to_combined_tad_nme` | Published: BiometryNet Combined -> Combined (arXiv 2512.16710) - TAD NME | **0.08** | `docs/results/published_refs.json` | `docs/results/published_refs.json` | `a6d1a58` | EXISTS |
| `pub_ac_fp_to_fp_apad_nme` | Published: BiometryNet FP -> FP (arXiv 2512.16710) - APAD NME | **0.08** | `docs/results/published_refs.json` | `docs/results/published_refs.json` | `a6d1a58` | EXISTS |
| `pub_ac_fp_to_fp_tad_nme` | Published: BiometryNet FP -> FP (arXiv 2512.16710) - TAD NME | **0.08** | `docs/results/published_refs.json` | `docs/results/published_refs.json` | `a6d1a58` | EXISTS |

---

## FEMUR Domain Claims (3 metrics)

| ID | Report Claim | Value | Result File Path | Producing Script / Notebook | Git Commit | File Status |
|---|---|:---:|---|---|:---:|:---:|
| `pub_femur_fl_nme` | Published: BiometryNet FL (arXiv 2512.16710) - FL NME | **0.03** | `docs/results/published_refs.json` | `docs/results/published_refs.json` | `a6d1a58` | EXISTS |
| `femur_r4_ensemble_fl_mae` | Row R4: 3-seed median ensemble (Round 2 best) - FL MAE | **1.1179 [0.856, 1.4588]** | `docs/results/femur_round2/femur_round2_results.json` | `notebooks/benchmarks/femur_round2.ipynb` | `a6d1a58` | EXISTS |
| `femur_r4_ensemble_fl_nme` | Row R4: 3-seed median ensemble (Round 2 best) - FL NME | **0.033** | `docs/results/femur_round2/femur_round2_results.json` | `notebooks/benchmarks/femur_round2.ipynb` | `a6d1a58` | EXISTS |

---

## TABULAR Domain Claims (32 metrics)

| ID | Report Claim | Value | Result File Path | Producing Script / Notebook | Git Commit | File Status |
|---|---|:---:|---|---|:---:|:---:|
| `stream3_e1_logistic_regression_auc` | Stream 3: Logistic Regression (E1 (Primary, >=20 wk)) - Test AUC | **0.6659 [0.6629, 0.6691]** | `docs/results/stream3_bench/benchmark_results.json` | `notebooks/benchmarks/stream3_bench.ipynb` | `a6d1a58` | EXISTS |
| `stream3_e1_random_forest_auc` | Stream 3: Random Forest (E1 (Primary, >=20 wk)) - Test AUC | **0.7322 [0.7297, 0.7354]** | `docs/results/stream3_bench/benchmark_results.json` | `notebooks/benchmarks/stream3_bench.ipynb` | `a6d1a58` | EXISTS |
| `stream3_e1_lightgbm_auc` | Stream 3: LightGBM (E1 (Primary, >=20 wk)) - Test AUC | **0.738 [0.7357, 0.7413]** | `docs/results/stream3_bench/benchmark_results.json` | `notebooks/benchmarks/stream3_bench.ipynb` | `a6d1a58` | EXISTS |
| `stream3_e1_our_xgboost_auc` | Stream 3: Our XGBoost (E1 (Primary, >=20 wk)) - Test AUC | **0.7381 [0.7358, 0.7413]** | `docs/results/stream3_bench/benchmark_results.json` | `notebooks/benchmarks/stream3_bench.ipynb` | `a6d1a58` | EXISTS |
| `stream3_e4_logistic_regression_auc` | Stream 3: Logistic Regression (E4 (Clinical-Only, >=20 wk)) - Test AUC | **0.6585 [0.6557, 0.6617]** | `docs/results/stream3_bench/benchmark_results.json` | `notebooks/benchmarks/stream3_bench.ipynb` | `a6d1a58` | EXISTS |
| `stream3_e4_random_forest_auc` | Stream 3: Random Forest (E4 (Clinical-Only, >=20 wk)) - Test AUC | **0.7043 [0.7013, 0.7075]** | `docs/results/stream3_bench/benchmark_results.json` | `notebooks/benchmarks/stream3_bench.ipynb` | `a6d1a58` | EXISTS |
| `stream3_e4_lightgbm_auc` | Stream 3: LightGBM (E4 (Clinical-Only, >=20 wk)) - Test AUC | **0.707 [0.7041, 0.7101]** | `docs/results/stream3_bench/benchmark_results.json` | `notebooks/benchmarks/stream3_bench.ipynb` | `a6d1a58` | EXISTS |
| `stream3_e4_our_xgboost_auc` | Stream 3: Our XGBoost (E4 (Clinical-Only, >=20 wk)) - Test AUC | **0.707 [0.704, 0.7101]** | `docs/results/stream3_bench/benchmark_results.json` | `notebooks/benchmarks/stream3_bench.ipynb` | `a6d1a58` | EXISTS |
| `stream3_modern_tuned_xgboost_auc` | Stream 3 Modern: Tuned XGBoost - Test AUC | **0.7382 [0.7349, 0.7412]** | `docs/results/stream3_modern/stream3_modern_results.json` | `notebooks/kaggle/stream3_modern/stream3-modern-s1/stream3-modern-s1.ipynb` | `7c75db9` | EXISTS |
| `stream3_modern_tuned_lightgbm_auc` | Stream 3 Modern: Tuned LightGBM - Test AUC | **0.7383 [0.7351, 0.7412]** | `docs/results/stream3_modern/stream3_modern_results.json` | `notebooks/kaggle/stream3_modern/stream3-modern-s1/stream3-modern-s1.ipynb` | `7c75db9` | EXISTS |
| `stream3_modern_tuned_catboost_auc` | Stream 3 Modern: Tuned CatBoost - Test AUC | **0.7383 [0.735, 0.7414]** | `docs/results/stream3_modern/stream3_modern_results.json` | `notebooks/kaggle/stream3_modern/stream3-modern-s1/stream3-modern-s1.ipynb` | `7c75db9` | EXISTS |
| `stream3_modern_tuned_lr_auc` | Stream 3 Modern: Tuned Logistic Regression - Test AUC | **0.6659 [0.6624, 0.6691]** | `docs/results/stream3_modern/stream3_modern_results.json` | `notebooks/kaggle/stream3_modern/stream3-modern-s1/stream3-modern-s1.ipynb` | `7c75db9` | EXISTS |
| `stream3_modern_calibrated_blend_auc` | Stream 3 Modern: Calibrated GBM+LR Blend - Test AUC | **0.7382 [0.735, 0.7412]** | `docs/results/stream3_modern/stream3_modern_results.json` | `notebooks/kaggle/stream3_modern/stream3-modern-s1/stream3-modern-s1.ipynb` | `7c75db9` | EXISTS |
| `stream3_modern_calibrated_blend_raw_auc` | Stream 3 Modern: Raw GBM+LR Blend - Test AUC | **0.7383 [0.735, 0.7412]** | `docs/results/stream3_modern/stream3_modern_results.json` | `notebooks/kaggle/stream3_modern/stream3-modern-s1/stream3-modern-s1.ipynb` | `7c75db9` | EXISTS |
| `stream3_audit_row_a_auc` | Row A: all rows, WITH COMBGEST (leaked gestational age) - Test AUC | **0.9684** | `tests/test_clinical_regression.py` | `notebooks/history/Capstone_fetalDataset_history.ipynb` | `a6d1a58` | EXISTS |
| `stream3_audit_row_b_auc` | Row B: all rows, no COMBGEST - Test AUC | **0.8832** | `tests/test_clinical_regression.py` | `notebooks/history/Capstone_fetalDataset_history.ipynb` | `a6d1a58` | EXISTS |
| `stream3_audit_row_c_auc` | Row C: >=20 wk, WITH COMBGEST (leaked gestational age) - Test AUC | **0.9382** | `tests/test_clinical_regression.py` | `notebooks/history/Capstone_fetalDataset_history.ipynb` | `a6d1a58` | EXISTS |
| `stream3_audit_row_d_auc` | Row D: >=20 wk, no COMBGEST - Test AUC | **0.7862** | `tests/test_clinical_regression.py` | `notebooks/history/Capstone_fetalDataset_history.ipynb` | `a6d1a58` | EXISTS |
| `stream3_audit_row_d2_auc` | Row D2: >=20 wk dedup, minus PRECARE - Test AUC | **0.7703** | `tests/test_clinical_regression.py` | `notebooks/history/Capstone_fetalDataset_history.ipynb` | `a6d1a58` | EXISTS |
| `stream3_audit_row_d3_auc` | Row D3: >=20 wk, complete cases only - Test AUC | **0.7464** | `tests/test_clinical_regression.py` | `notebooks/history/Capstone_fetalDataset_history.ipynb` | `a6d1a58` | EXISTS |
| `stream3_audit_row_d4_auc` | Row D4: >=37 wk, complete cases only - Test AUC | **0.7131** | `tests/test_clinical_regression.py` | `notebooks/history/Capstone_fetalDataset_history.ipynb` | `a6d1a58` | EXISTS |
| `stream3_audit_row_d5_auc` | Row D5: >=20 wk, numeric features only - Test AUC | **0.7434** | `tests/test_clinical_regression.py` | `notebooks/history/Capstone_fetalDataset_history.ipynb` | `a6d1a58` | EXISTS |
| `stream3_audit_final_temporal_all_auc` | Temporal Reference: >=20 wk, ALL features (temporal) - Test AUC | **0.7865** | `tests/test_clinical_regression.py` | `notebooks/history/Capstone_fetalDataset_history.ipynb` | `a6d1a58` | EXISTS |
| `stream3_audit_row_e1_auc` | Row E1: >=20 wk primary features (temporal) - Test AUC | **0.7381** | `tests/test_clinical_regression.py` | `notebooks/history/Capstone_fetalDataset_history.ipynb` | `a6d1a58` | EXISTS |
| `stream3_audit_row_e2_auc` | Row E2: >=37 wk primary features (temporal) - Test AUC | **0.7117** | `tests/test_clinical_regression.py` | `notebooks/history/Capstone_fetalDataset_history.ipynb` | `a6d1a58` | EXISTS |
| `stream3_audit_row_e3_auc` | Row E3: >=37 wk ALL features, reference (temporal) - Test AUC | **0.753** | `tests/test_clinical_regression.py` | `notebooks/history/Capstone_fetalDataset_history.ipynb` | `a6d1a58` | EXISTS |
| `stream3_audit_row_e4_auc` | Row E4: >=20 wk clinical-only features (temporal) - Test AUC | **0.7069** | `tests/test_clinical_regression.py` | `notebooks/history/Capstone_fetalDataset_history.ipynb` | `a6d1a58` | EXISTS |
| `stream3_audit_row_e5_auc` | Row E5: >=37 wk clinical-only features (temporal) - Test AUC | **0.6917** | `tests/test_clinical_regression.py` | `notebooks/history/Capstone_fetalDataset_history.ipynb` | `a6d1a58` | EXISTS |
| `stream3_true_prevalence_assumed` | Population Stillbirth Prevalence Assumption (CDC NVSS national rate) - Population Prevalence | **0.0055** | `src/fetal/clinical/constants.py` | `notebooks/history/Capstone_fetalDataset_history.ipynb` | `a6d1a58` | EXISTS |
| `pub_stream3_early_lr_auc` | Published: LR Early Stillbirth (Koivu & Sairanen 2020) - Test AUC | **0.73** | `docs/results/published_refs.json` | `docs/results/published_refs.json` | `a6d1a58` | EXISTS |
| `pub_stream3_early_selu_auc` | Published: SELU Early Stillbirth (Koivu & Sairanen 2020) - Test AUC | **0.75** | `docs/results/published_refs.json` | `docs/results/published_refs.json` | `a6d1a58` | EXISTS |
| `pub_stream3_early_lightgbm_auc` | Published: LightGBM Early Stillbirth (Koivu & Sairanen 2020) - Test AUC | **0.75** | `docs/results/published_refs.json` | `docs/results/published_refs.json` | `a6d1a58` | EXISTS |

---

## HEAD Domain Claims (4 metrics)

| ID | Report Claim | Value | Result File Path | Producing Script / Notebook | Git Commit | File Status |
|---|---|:---:|---|---|:---:|:---:|
| `pub_head_bpd_nme` | Published: BiometryNet BPD (arXiv 2512.16710) - BPD NME | **0.04** | `docs/results/published_refs.json` | `docs/results/published_refs.json` | `a6d1a58` | EXISTS |
| `pub_head_ofd_nme` | Published: BiometryNet OFD (arXiv 2512.16710) - OFD NME | **0.03** | `docs/results/published_refs.json` | `docs/results/published_refs.json` | `a6d1a58` | EXISTS |
| `head_landmark_ensemble_bpd_nme` | Head Landmark: 3-seed median ensemble - BPD NME | **0.029** | `docs/results/head_landmark/head_landmark_results.json` | `notebooks/benchmarks/head_landmark.ipynb` | `a6d1a58` | EXISTS |
| `head_landmark_ensemble_ofd_nme` | Head Landmark: 3-seed median ensemble - OFD NME | **0.0274** | `docs/results/head_landmark/head_landmark_results.json` | `notebooks/benchmarks/head_landmark.ipynb` | `a6d1a58` | EXISTS |

---

## FUSION Domain Claims (6 metrics)

| ID | Report Claim | Value | Result File Path | Producing Script / Notebook | Git Commit | File Status |
|---|---|:---:|---|---|:---:|:---:|
| `fusion_cdc_reference_total_live_births` | CDC NVSS Gestational Growth Reference (2016-2021) - Cohort Size (N) | **262542** | `docs/results/fusion/gestational_percentiles_summary.json` | `scripts/81_build_gestational_percentiles.py` | `master` | EXISTS |
| `fusion_cdc_p50_week_20` | CDC NVSS Gestational Growth Reference: Week 20 - Median Birth Weight (p50) | **336.0** | `docs/results/fusion/gestational_percentiles_summary.json` | `scripts/81_build_gestational_percentiles.py` | `master` | EXISTS |
| `fusion_cdc_p50_week_28` | CDC NVSS Gestational Growth Reference: Week 28 - Median Birth Weight (p50) | **1191.0** | `docs/results/fusion/gestational_percentiles_summary.json` | `scripts/81_build_gestational_percentiles.py` | `master` | EXISTS |
| `fusion_cdc_p50_week_34` | CDC NVSS Gestational Growth Reference: Week 34 - Median Birth Weight (p50) | **2410.0** | `docs/results/fusion/gestational_percentiles_summary.json` | `scripts/81_build_gestational_percentiles.py` | `master` | EXISTS |
| `fusion_cdc_p50_week_37` | CDC NVSS Gestational Growth Reference: Week 37 - Median Birth Weight (p50) | **3033.0** | `docs/results/fusion/gestational_percentiles_summary.json` | `scripts/81_build_gestational_percentiles.py` | `master` | EXISTS |
| `fusion_cdc_p50_week_40` | CDC NVSS Gestational Growth Reference: Week 40 - Median Birth Weight (p50) | **3459.0** | `docs/results/fusion/gestational_percentiles_summary.json` | `scripts/81_build_gestational_percentiles.py` | `master` | EXISTS |

---

## FASSD Domain Claims (2 metrics)

| ID | Report Claim | Value | Result File Path | Producing Script / Notebook | Git Commit | File Status |
|---|---|:---:|---|---|:---:|:---:|
| `fassd_ensemble_macro_dice` | FASSD: 4-structure segmentation ensemble (part A settings) - Macro Dice | **0.7914** | `docs/results/fassd/ensemble/fassd_ensemble_results.json` | `notebooks/benchmarks/fassd_ensemble.ipynb` | `a6d1a58` | EXISTS |
| `fassd_ensemble_macro_hd95` | FASSD: 4-structure segmentation ensemble (part A settings) - Macro HD95 | **22.3118** | `docs/results/fassd/ensemble/fassd_ensemble_results.json` | `notebooks/benchmarks/fassd_ensemble.ipynb` | `a6d1a58` | EXISTS |

---
