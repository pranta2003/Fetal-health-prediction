# Master File Guide: Comprehensive Repository Architecture

This guide provides an exhaustive inventory of every script, source module, unit test, configuration file, and notebook folder across the repository. All classifications and descriptions reflect verified source code analysis.

## Master File Table

| Path | Field | Plain-English Summary | Model or Method | Type |
| :--- | :--- | :--- | :--- | :--- |
| `scripts/04_train_ac.py` | AC | Trains AC landmark model: Check if all n_folds out-of-fold prediction CSVs exist, pool them, | EfficientNet-B4 Landmark U-Net | training |
| `scripts/05_train_ensemble.py` | General | Project utility or CI check: checks repo state | None | check tool |
| `scripts/06_eval_ac.py` | AC | Evaluates AC landmark models: OOF and test metrics | EfficientNet-B4 Landmark U-Net | evaluation |
| `scripts/07_audit_clinical.py` | Tabular | Runs 14-row clinical audit reproducing historical leakage vs clean models | XGBoost / LightGBM / CatBoost / LR | check tool |
| `scripts/08_make_manifest.py` | Report tools | Report generation, verification, or audit tool: 08_make_manifest.py: Build or verify SHA-256 manifest of model checkpoints directory. | None (Audit & Verification Harness) | check tool |
| `scripts/09_hc18_leakage_report.py` | HC18 | Audits patient leakage in historical random split vs grouped split | EfficientNet-B4 U-Net | check tool |
| `scripts/10_train_hc18.py` | HC18 | Trains our 5-fold EfficientNet-B4 U-Net head circumference model | EfficientNet-B4 U-Net | training |
| `scripts/11_hc18_oof_report.py` | HC18 | Evaluates HC18 model: Out-Of-Fold (OOF) evaluation report for HC18 5-fold segmentation pipeline. | EfficientNet-B4 U-Net | evaluation |
| `scripts/12_hc18_test_inference.py` | HC18 | Runs inference to produce official HC18 challenge test submission CSVs | EfficientNet-B4 U-Net | evaluation |
| `scripts/14_hash_results.py` | Report tools | Report generation, verification, or audit tool: 14_hash_results.py: Generate or verify SHA256SUMS for results directory. | None (Audit & Verification Harness) | check tool |
| `scripts/15_femur_data_report.py` | Femur | Audits femur length dataset, outliers, and pixel calibration: Femur (FL) biometry dataset inspection and validation report. | EfficientNet-B4 / ConvNeXt Landmark | check tool |
| `scripts/16_extract_notebook_results.py` | AC | AC landmark utility: 16_extract_notebook_results.py: Extract historical metrics from notebook. | EfficientNet-B4 Landmark U-Net | check tool |
| `scripts/18_bench_stream3.py` | Tabular | Trains and benchmarks GBDT models on CDC vital statistics cohort | XGBoost / LightGBM / CatBoost / LR | training |
| `scripts/19_hc18_ladder.py` | HC18 | Trains or benchmarks historical approach ladder (Rows A-D vs E) | Historical (ResNet-34 / Multi-Task) | training |
| `scripts/20_benchmark_summary.py` | General | Project utility or CI check: 20_benchmark_summary.py: Synthesize multi-modal benchmark results and published baselines. | None | check tool |
| `scripts/21_femur_data_followup.py` | Femur | Audits femur length dataset, outliers, and pixel calibration: 21_femur_data_followup.py: Femur (FL) dataset inspection, outlier analysis, and split verification. | EfficientNet-B4 / ConvNeXt Landmark | check tool |
| `scripts/22_femur_ablation.py` | Femur | Trains or evaluates femur diaphysis landmark models: 22_femur_ablation.py: Femur landmark ablation study runner and report generator. | EfficientNet-B4 / ConvNeXt Landmark | evaluation |
| `scripts/23_three_plane_check.py` | Fusion | Audits multi-plane patient overlap across FP and UCL cohorts | Hadlock Formula / CDC Growth Charts | check tool |
| `scripts/24_hadlock_demo_step1.py` | Fusion | Demonstrates 3-plane Hadlock EFW estimation on held-out cohort: Hadlock Demo Step 1: Count usable subjects and run external validation on FP/UCL head images. | Hadlock Formula / CDC Growth Charts | evaluation |
| `scripts/25_head_landmark.py` | Head landmarks | Evaluates head caliper endpoints (BPD, OFD) on external cohort: Runner and evaluation script for Head Landmark model (Hadlock demo + dataset paper benchmark). | EfficientNet-B4 Caliper U-Net | evaluation |
| `scripts/26_hadlock_demo_step2.py` | Fusion | Demonstrates 3-plane Hadlock EFW estimation on held-out cohort: 26_hadlock_demo_step2.py: Hadlock demo step 2 - multi-plane EFW evaluation on strictly held-out 3-plane test cohort. | Hadlock Formula / CDC Growth Charts | evaluation |
| `scripts/27_femur_round2.py` | Femur | Trains or evaluates femur diaphysis landmark models: 27_femur_round2.py: Femur Round 2 runner, report generator, and artifact validator. | EfficientNet-B4 / ConvNeXt Landmark | evaluation |
| `scripts/28_ac_history_table.py` | AC | Audits AC annotations, duplicate images, or lab-PC history: 28_ac_history_table.py: Synthesize AC lab-PC history proof tables and fact checks. | EfficientNet-B4 Landmark U-Net | check tool |
| `scripts/29_hc18_official_summary.py` | HC18 | Evaluates HC18 model: Generate official HC18 Grand Challenge test set evaluation summary. | EfficientNet-B4 U-Net | evaluation |
| `scripts/30_hc18_ellipse_roundtrip.py` | HC18 | Evaluates HC18 model: HC18 Ellipse Roundtrip Dice Analysis. | EfficientNet-B4 U-Net | evaluation |
| `scripts/31_femur_round2_checks.py` | Femur | Trains or evaluates femur diaphysis landmark models: 31_femur_round2_checks.py: Sensitivity of baseline, run integrity checks, and worst-10 audit. | EfficientNet-B4 / ConvNeXt Landmark | evaluation |
| `scripts/32_femur_outlier_impact.py` | AC | AC landmark utility: 32_femur_outlier_impact.py: FL MAE impact of outlier 005_3FL.jpeg with patient-level bootstrap CIs. | EfficientNet-B4 Landmark U-Net | check tool |
| `scripts/33_fassd_audit.py` | FASSD | Audits FASSD images/masks or builds leak-free split_v2.csv via Union-Find | EfficientNet-B4 Multilabel U-Net / U-Net++ | data prep |
| `scripts/34_hc18_oof_calibration.py` | HC18 | Evaluates measurement-fix or calibration transformations: 34_hc18_oof_calibration.py: HC18 out-of-fold post-hoc calibration study. | EfficientNet-B4 U-Net / Post-hoc Transform | evaluation |
| `scripts/35_fassd_train.py` | FASSD | Trains multilabel 4-channel segmentation model for fetal abdominal organs | EfficientNet-B4 Multilabel U-Net / U-Net++ | training |
| `scripts/36_fassd_postproc.py` | FASSD | Tunes and evaluates morphology and connected component post-processing | EfficientNet-B4 Multilabel U-Net / U-Net++ | evaluation |
| `scripts/37_fassd_ensemble.py` | FASSD | Ensembles multi-architecture probability maps and evaluates macro Dice | EfficientNet-B4 Multilabel U-Net / U-Net++ | evaluation |
| `scripts/38_ac_step0_audit.py` | AC | Audits AC annotations, duplicate images, or lab-PC history: scripts/38_ac_step0_audit.py | EfficientNet-B4 Landmark U-Net | check tool |
| `scripts/39_make_ac_folds.py` | AC | Prepares AC folds, dataset, or Kaggle kernels: scripts/39_make_ac_folds.py | EfficientNet-B4 Landmark U-Net | data prep |
| `scripts/40_ac_compare_recipes.py` | AC | Evaluates AC landmark models: scripts/40_ac_compare_recipes.py | EfficientNet-B4 Landmark U-Net | evaluation |
| `scripts/41_prepare_ac_kaggle_dataset.py` | AC | Prepares AC folds, dataset, or Kaggle kernels: scripts/41_prepare_ac_kaggle_dataset.py | EfficientNet-B4 Landmark U-Net | data prep |
| `scripts/42_make_ac_kaggle_kernels.py` | AC | Prepares AC folds, dataset, or Kaggle kernels: scripts/42_make_ac_kaggle_kernels.py | EfficientNet-B4 Landmark U-Net | Kaggle kernel builder |
| `scripts/43_hc18_calibration_candidates.py` | HC18 | Evaluates measurement-fix or calibration transformations: scripts/43_hc18_calibration_candidates.py | EfficientNet-B4 U-Net / Post-hoc Transform | evaluation |
| `scripts/44_generate_best_published.py` | Report tools | Report generation, verification, or audit tool: scripts/44_generate_best_published.py | None (Audit & Verification Harness) | check tool |
| `scripts/45_hc18_to_nnunet.py` | HC18 | Converts HC18 training data to nnU-Net v2 Dataset501_HC18 format | nnU-Net | data prep |
| `scripts/46_eval_nnunet_hc18.py` | HC18 | Evaluates HC18 nnU-Net masks on validation folds: 46_eval_nnunet_hc18.py: Evaluate nnU-Net predicted masks for HC18 fold 1. | nnU-Net | evaluation |
| `scripts/47_audit_bias_direction.py` | General | Project utility or CI check: Audit bias direction discrepancy between validation and test sets for HC18. | None | check tool |
| `scripts/47_make_nnunet_kaggle_kernel.py` | HC18 | Builds Kaggle kernel for HC18 nnU-Net: scripts/47_make_nnunet_kaggle_kernel.py | nnU-Net | Kaggle kernel builder |
| `scripts/48_make_worst8_contact_sheet.py` | AC | Diagnostic inspection of AC landmark outliers: Create a 2x4 contact sheet PNG of the worst 8 AC OOF images. | EfficientNet-B4 Landmark U-Net | evaluation |
| `scripts/49_update_results_ledger.py` | Report tools | Report generation, verification, or audit tool: Generate docs/results/RESULTS_LEDGER.md strictly from JSON files on disk. | None (Audit & Verification Harness) | check tool |
| `scripts/50_measurement_fix_study.py` | General | Project utility or CI check: HC18 Measurement-Fix Study. | None | check tool |
| `scripts/51_test_nnunet_test_script.py` | HC18 | Tests or verifies HC18 nnU-Net inference locally: Test the submission ellipse extraction pipeline on 3 validation masks locally. | nnU-Net | check tool |
| `scripts/52_make_hc18_nnunet_test_kernel.py` | HC18 | Builds Kaggle kernel for HC18 nnU-Net: Generate Kaggle kernel for HC18 nnU-Net fold 0 test set inference. | nnU-Net | Kaggle kernel builder |
| `scripts/53_make_stream3_modern_kernel.py` | Tabular | Generates Kaggle Optuna tuning kernel for tabular models | XGBoost / LightGBM / CatBoost / LR | Kaggle kernel builder |
| `scripts/53_test_discovery_mock.py` | General | Project utility or CI check: Test the dataset discovery code on a mock Kaggle input layout. | None | check tool |
| `scripts/54_eval_linear_fit_followup.py` | General | Project utility or CI check: Compute linear fit comparison on nnU-Net 206 and OOF 999 against CSV HC. | None | check tool |
| `scripts/55_make_hc18_nnunet_parallel_kernels.py` | HC18 | Builds Kaggle kernel for HC18 nnU-Net: Generate Kaggle kernels for parallel 100-epoch nnU-Net training (Folds 1-2 and 3-4). | nnU-Net | Kaggle kernel builder |
| `scripts/56_build_report_numbers.py` | Report tools | Report generation, verification, or audit tool: scripts/56_build_report_numbers.py | None (Audit & Verification Harness) | check tool |
| `scripts/57_verify_report_numbers.py` | Report tools | Report generation, verification, or audit tool: scripts/57_verify_report_numbers.py | None (Audit & Verification Harness) | check tool |
| `scripts/58_audit_notebooks.py` | Report tools | Report generation, verification, or audit tool: scripts/58_audit_notebooks.py | None (Audit & Verification Harness) | check tool |
| `scripts/59_audit_nme_formulas.py` | Report tools | Report generation, verification, or audit tool: scripts/59_audit_nme_formulas.py | None (Audit & Verification Harness) | check tool |
| `scripts/60_audit_dataset_counts.py` | Report tools | Report generation, verification, or audit tool: scripts/60_audit_dataset_counts.py | None (Audit & Verification Harness) | check tool |
| `scripts/61_audit_hc18_ladder_refs.py` | HC18 | Trains or benchmarks historical approach ladder (Rows A-D vs E) | Historical (ResNet-34 / Multi-Task) | training |
| `scripts/62_train_stream3_modern.py` | Tabular | Trains and benchmarks GBDT models on CDC vital statistics cohort | XGBoost / LightGBM / CatBoost / LR | training |
| `scripts/63_verify_nnunet_candidates.py` | HC18 | HC18 nnU-Net helper:  | nnU-Net | check tool |
| `scripts/64_format_hc18_readme.py` | HC18 | HC18 utility or check: Format and verify HC18 official metrics table for README.md. | EfficientNet-B4 U-Net | check tool |
| `scripts/65_build_proof_map.py` | Report tools | Report generation, verification, or audit tool: Build docs/report/PROOF_LOCATIONS.md mapping every report number to file, script/notebook, and commit. | None (Audit & Verification Harness) | check tool |
| `scripts/66_make_hc18_nnunet_test_f012_kernel.py` | HC18 | Builds Kaggle kernel for HC18 nnU-Net: Generate Kaggle kernel for HC18 nnU-Net 3-fold ensemble (folds 0, 1, 2) test set inference. | nnU-Net | Kaggle kernel builder |
| `scripts/67_download_f1f2_and_ac_fassd.py` | AC | AC landmark utility: scripts/67_download_f1f2_and_ac_fassd.py | EfficientNet-B4 Landmark U-Net | check tool |
| `scripts/68_eval_nnunet_f1_f2.py` | HC18 | Evaluates HC18 nnU-Net masks on validation folds: 68_eval_nnunet_f1_f2.py: Comprehensive evaluation of nnU-Net Fold 1 and Fold 2 predictions. | nnU-Net | evaluation |
| `scripts/69_hc18_nnunet_blend_and_pooled.py` | HC18 | Processes predictions into submission CSVs or blends: 69_hc18_nnunet_blend_and_pooled.py: Pooled nnU-Net evaluation and heterogeneous blend study. | nnU-Net | evaluation |
| `scripts/70_build_blend_candidate.py` | General | Project utility or CI check: 70_build_blend_candidate.py: Build and verify the heterogeneous blend candidate. | None | check tool |
| `scripts/71_make_hc18_nnunet_test_f012_cpu_kernel.py` | HC18 | Builds Kaggle kernel for HC18 nnU-Net: Generate Kaggle kernel for HC18 nnU-Net 3-fold ensemble CPU inference. | nnU-Net | Kaggle kernel builder |
| `scripts/72_smoke_test_nnunet_cpu.py` | HC18 | Tests or verifies HC18 nnU-Net inference locally: Smoke test nnU-Net CPU inference locally on 3 test images. | nnU-Net | check tool |
| `scripts/73_check_pr54_premerge.py` | General | Project utility or CI check: Check PR #54 pre-merge conditions: | None | check tool |
| `scripts/74_check_master_premerge.py` | General | Project utility or CI check: checks repo state | None | check tool |
| `scripts/75_process_nnunet_f012_test.py` | HC18 | Tests or verifies HC18 nnU-Net inference locally:  | nnU-Net | check tool |
| `scripts/76_eval_nnunet_f3_f4_and_pooled5.py` | HC18 | Evaluates HC18 nnU-Net masks on validation folds: 76_eval_nnunet_f3_f4_and_pooled5.py: Evaluation of nnU-Net Folds 3, 4 and 5-Fold Pooled Benchmark. | nnU-Net | evaluation |
| `scripts/77_make_hc18_nnunet_test_f01234_kernels.py` | HC18 | Builds Kaggle kernel for HC18 nnU-Net: 77_make_hc18_nnunet_test_f01234_kernels.py: Generate 5-fold ensemble test kernels (GPU & CPU). | nnU-Net | Kaggle kernel builder |
| `scripts/78_build_blend_f012_ens5.py` | General | Project utility or CI check: 78_build_blend_f012_ens5.py: Build and verify the heterogeneous blend candidate. | None | check tool |
| `scripts/79_smoke_test_5fold_cpu.py` | General | Project utility or CI check: 79_smoke_test_5fold_cpu.py: Smoke test 5-fold ensemble CPU inference on 3 test images. | None | check tool |
| `scripts/80_process_nnunet_f01234_test.py` | HC18 | Tests or verifies HC18 nnU-Net inference locally: 80_process_nnunet_f01234_test.py: Process 5-fold nnU-Net test outputs, candidates, and blend. | nnU-Net | check tool |
| `scripts/81_build_gestational_percentiles.py` | Fusion | Extracts CDC birth weight gestational age percentiles (weeks 20-42) for live births | Hadlock Formula / CDC Growth Charts | data prep |
| `scripts/82_eval_ac_s43_and_wing_comparison.py` | AC | Evaluates AC landmark models: scripts/82_eval_ac_s43_and_wing_comparison.py | EfficientNet-B4 Landmark U-Net | evaluation |
| `scripts/83_ac_to_nnunet.py` | AC | Converts AC landmark images to nnU-Net solid ellipse masks | nnU-Net | data prep |
| `scripts/84_make_ac_nnunet_kaggle_kernel.py` | AC | Builds Kaggle training kernel for AC nnU-Net | nnU-Net | Kaggle kernel builder |
| `scripts/85_eval_ac_nnunet_f0.py` | AC | Evaluates AC nnU-Net predicted masks on validation images | nnU-Net | evaluation |
| `scripts/86_eval_ac_seeds_and_4seed_ensemble.py` | AC | Evaluates AC landmark models: scripts/86_eval_ac_seeds_and_4seed_ensemble.py | EfficientNet-B4 Landmark U-Net | evaluation |
| `scripts/analyze_worst_and_calipers.py` | General | Project utility or CI check: # Experimental Results Ledger | None | check tool |
| `scripts/build_data_inventory.py` | Report tools | Report generation, verification, or audit tool: scripts/build_data_inventory.py | None (Audit & Verification Harness) | check tool |
| `scripts/check_secrets.py` | General | Project utility or CI check: check_secrets.py: Scan workspace for leaked tokens, keys, and private credentials. | None | check tool |
| `scripts/diagnose_worst8.py` | General | Project utility or CI check: checks repo state | None | check tool |
| `scripts/download_hc18_nnunet_artifacts.py` | HC18 | HC18 nnU-Net helper:  | nnU-Net | check tool |
| `scripts/kaggle/run_femur_ablation.py` | Femur | Trains or evaluates femur diaphysis landmark models: Self-contained Kaggle runner for Femur Model Ablation Study (Rows A, B, C; seeds 42, 43, 44). | EfficientNet-B4 / ConvNeXt Landmark | training |
| `scripts/kaggle/run_femur_round2.py` | Femur | Trains or evaluates femur diaphysis landmark models: Self-contained Kaggle runner for Femur Round 2: Modern Techniques. | EfficientNet-B4 / ConvNeXt Landmark | training |
| `scripts/kaggle/run_hc18_close.py` | HC18 | Runs inference to produce official HC18 challenge test submission CSVs | EfficientNet-B4 U-Net | evaluation |
| `scripts/kaggle/run_hc18_ladder.py` | HC18 | Trains or benchmarks historical approach ladder (Rows A-D vs E) | Historical (ResNet-34 / Multi-Task) | training |
| `scripts/kaggle/run_head_landmark.py` | Head landmarks | Evaluates head caliper endpoints (BPD, OFD) on external cohort: Self-contained Kaggle runner for Head Landmark regression model (seeds 42, 43, 44). | EfficientNet-B4 Caliper U-Net | evaluation |
| `scripts/make_bench_notebooks.py` | Report tools | Report generation, verification, or audit tool: Helper script to construct benchmark notebooks for Stream 3 and AC. | None (Audit & Verification Harness) | check tool |
| `scripts/make_fassd_audit_notebook.py` | FASSD | Audits FASSD images/masks or builds leak-free split_v2.csv via Union-Find | EfficientNet-B4 Multilabel U-Net / U-Net++ | data prep |
| `scripts/make_fassd_ensemble_notebook.py` | FASSD | Ensembles multi-architecture probability maps and evaluates macro Dice | EfficientNet-B4 Multilabel U-Net / U-Net++ | evaluation |
| `scripts/make_fassd_kaggle_notebook.py` | FASSD | Builds Kaggle training or benchmark notebook for FASSD models | EfficientNet-B4 Multilabel U-Net / U-Net++ | Kaggle kernel builder |
| `scripts/make_fassd_kaggle_notebook_round2.py` | FASSD | Builds Kaggle training or benchmark notebook for FASSD models | EfficientNet-B4 Multilabel U-Net / U-Net++ | Kaggle kernel builder |
| `scripts/make_fassd_postproc_notebook.py` | FASSD | Tunes and evaluates morphology and connected component post-processing | EfficientNet-B4 Multilabel U-Net / U-Net++ | evaluation |
| `scripts/make_fassd_split_v2.py` | FASSD | Audits FASSD images/masks or builds leak-free split_v2.csv via Union-Find | EfficientNet-B4 Multilabel U-Net / U-Net++ | data prep |
| `scripts/make_femur_ablation_notebook.py` | Femur | Trains or evaluates femur diaphysis landmark models: make_femur_ablation_notebook.py: Generate notebooks/benchmarks/femur_ablation.ipynb. | EfficientNet-B4 / ConvNeXt Landmark | evaluation |
| `scripts/make_hadlock_step2_notebook.py` | Fusion | Generates benchmark notebook for Hadlock multimodal fusion | Hadlock Formula / CDC Growth Charts | Kaggle kernel builder |
| `scripts/make_hc18_calibration_notebook.py` | HC18 | Evaluates measurement-fix or calibration transformations: make_hc18_calibration_notebook.py: Generate notebooks/benchmarks/hc18_oof_calibration.ipynb. | EfficientNet-B4 U-Net / Post-hoc Transform | evaluation |
| `scripts/make_head_external_notebook.py` | Head landmarks | Builds notebook for head landmark caliper evaluation | EfficientNet-B4 Caliper U-Net | Kaggle kernel builder |
| `scripts/make_head_landmark_notebook.py` | Head landmarks | Builds notebook for head landmark caliper evaluation | EfficientNet-B4 Caliper U-Net | Kaggle kernel builder |
| `scripts/smoke_test_ac_nnunet_f0.py` | AC | Smoke tests AC nnU-Net training pipeline on 3 images | nnU-Net | check tool |
| `src/fetal/__init__.py` | General | Fetal Health Multimodal Assessment Package. | None | check tool |
| `src/fetal/archive/__init__.py` | General | Archived prior experimental models and pipelines. | None | check tool |
| `src/fetal/clinical/__init__.py` | Tabular | Stream 3: Clinical tabular risk assessment module. | XGBoost / GBDT | data prep |
| `src/fetal/clinical/audit.py` | Tabular | Evaluates tabular ROC-AUC, Brier score, calibration, or runs 14-row leakage audit | XGBoost / GBDT | evaluation |
| `src/fetal/clinical/constants.py` | Tabular | Data loading, sentinel cleaning, and feature constants for CDC tabular records | XGBoost / GBDT | data prep |
| `src/fetal/clinical/data.py` | Tabular | Data loading, sentinel cleaning, and feature constants for CDC tabular records | XGBoost / GBDT | data prep |
| `src/fetal/clinical/eval.py` | Tabular | Evaluates tabular ROC-AUC, Brier score, calibration, or runs 14-row leakage audit | XGBoost / GBDT | evaluation |
| `src/fetal/clinical/model.py` | Tabular | XGBoost classifier training with prevalence adjustment | XGBoost / GBDT | training |
| `src/fetal/config.py` | General | Central project configuration loading paths from paths.yaml | None | config |
| `src/fetal/data/__init__.py` | General | Data loaders, transformations, and preprocessing. | None | data prep |
| `src/fetal/data/ac_dataset.py` | AC | AC dataset loader, image augmentations, and yellow caliper marker removal | EfficientNet-B4 Landmark U-Net | data prep |
| `src/fetal/data/fassd.py` | FASSD | FASSD array loader, structure masks, and Union-Find patient hash grouping | EfficientNet-B4 Multilabel U-Net | data prep |
| `src/fetal/data/femur_dataset.py` | Femur | Femur diaphysis dataset loader, calibration parsing, and patient-grouped split | EfficientNet-B4 / ConvNeXt Landmark | data prep |
| `src/fetal/data/hc18_dataset.py` | HC18 | HC18 dataset loader with aspect-preserving resize and patient-grouped 5-fold split | EfficientNet-B4 U-Net | data prep |
| `src/fetal/data/hc18_split.py` | HC18 | HC18 dataset loader with aspect-preserving resize and patient-grouped 5-fold split | EfficientNet-B4 U-Net | data prep |
| `src/fetal/data/head_dataset.py` | Head landmarks | Head caliper landmark dataset loader with BPD/OFD points and patient splits | EfficientNet-B4 Caliper U-Net | data prep |
| `src/fetal/data/transforms.py` | AC | AC dataset loader, image augmentations, and yellow caliper marker removal | EfficientNet-B4 Landmark U-Net | data prep |
| `src/fetal/eval/__init__.py` | General | Evaluation, geometry, and metric calculation modules. | None | evaluation |
| `src/fetal/eval/ac_evaluator.py` | AC | Landmark matching, Ramanujan ellipse circumference in mm, and paired bootstrap CIs | EfficientNet-B4 Landmark U-Net | evaluation |
| `src/fetal/eval/ac_geometry.py` | AC | Landmark matching, Ramanujan ellipse circumference in mm, and paired bootstrap CIs | EfficientNet-B4 Landmark U-Net | evaluation |
| `src/fetal/eval/ellipse_fitting.py` | HC18 | Mask-to-ellipse fitting, Ramanujan HC/BPD measurement, and test metrics | EfficientNet-B4 U-Net | evaluation |
| `src/fetal/eval/ensemble_evaluator.py` | General | Evaluation metrics module | None | evaluation |
| `src/fetal/eval/femur_evaluator.py` | Femur | Femur diaphysis length calculation, DARK decoding, and bootstrap CIs | EfficientNet-B4 / ConvNeXt Landmark | evaluation |
| `src/fetal/eval/hc18_eval.py` | HC18 | Mask-to-ellipse fitting, Ramanujan HC/BPD measurement, and test metrics | EfficientNet-B4 U-Net | evaluation |
| `src/fetal/eval/head_evaluator.py` | Head landmarks | Head caliper evaluation, BPD/OFD lengths, Ramanujan HC, and NME metrics | EfficientNet-B4 Caliper U-Net | evaluation |
| `src/fetal/explain/__init__.py` | Fusion | Hadlock 1985 EFW formulas and CDC birth weight gestational age percentiles | Hadlock Formula / Growth Percentiles | evaluation |
| `src/fetal/explain/hadlock.py` | Fusion | Hadlock 1985 EFW formulas and CDC birth weight gestational age percentiles | Hadlock Formula / Growth Percentiles | evaluation |
| `src/fetal/infer/__init__.py` | AC | Single-image end-to-end inference returning AC measurement in mm | EfficientNet-B4 Landmark U-Net | evaluation |
| `src/fetal/infer/ac_infer.py` | AC | Single-image end-to-end inference returning AC measurement in mm | EfficientNet-B4 Landmark U-Net | evaluation |
| `src/fetal/models/__init__.py` | General | Model architectures and loss functions. | None | training |
| `src/fetal/models/ac_landmark.py` | AC | Landmark coordinate heatmap regression network, wing loss, and coordinate loss | EfficientNet-B4 Landmark U-Net | training |
| `src/fetal/models/femur_landmark.py` | Femur | Dual-peak heatmap U-Net and ConvNeXt-Tiny landmark regression models | EfficientNet-B4 / ConvNeXt Landmark | training |
| `src/fetal/models/hc18_old_approaches.py` | HC18 | U-Net architecture with EfficientNet-B4 encoder and historical Row A-D models | EfficientNet-B4 U-Net / Historical | training |
| `src/fetal/models/hc18_unet.py` | HC18 | U-Net architecture with EfficientNet-B4 encoder and historical Row A-D models | EfficientNet-B4 U-Net / Historical | training |
| `src/fetal/models/head_landmark.py` | Head landmarks | 4-endpoint head caliper landmark regression model with soft-argmax | EfficientNet-B4 Caliper U-Net | training |
| `src/fetal/models/losses.py` | AC | Landmark coordinate heatmap regression network, wing loss, and coordinate loss | EfficientNet-B4 Landmark U-Net | training |
| `src/fetal/seg/__init__.py` | FASSD | FASSD fetal abdominal structures segmentation module. | EfficientNet-B4 Multilabel U-Net | training |
| `src/fetal/seg/fassd_ensemble.py` | FASSD | Multi-model probability map ensembling and validation candidate selection | EfficientNet-B4 Multilabel U-Net / U-Net++ | evaluation |
| `src/fetal/seg/fassd_postproc.py` | FASSD | Morphological cleaning, connected components, and threshold tuning | EfficientNet-B4 Multilabel U-Net / U-Net++ | evaluation |
| `src/fetal/seg/fassd_train.py` | FASSD | 4-channel multilabel segmentation training loop with BCE+Dice loss and AMP | EfficientNet-B4 Multilabel U-Net / U-Net++ | training |
| `src/fetal/train/__init__.py` | General | Training utilities module | None | training |
| `src/fetal/train/checkpoint.py` | General | Saves and restores PyTorch model checkpoints and optimizer states | None | check tool |
| `src/fetal/train/constants.py` | General | Training utilities module | None | training |
| `src/fetal/train/ema.py` | AC | Training loops, multi-seed ensembling, training recipes, and Model EMA for AC | EfficientNet-B4 Landmark U-Net | training |
| `src/fetal/train/recipes.py` | AC | Training loops, multi-seed ensembling, training recipes, and Model EMA for AC | EfficientNet-B4 Landmark U-Net | training |
| `src/fetal/train/train_ac.py` | AC | Training loops, multi-seed ensembling, training recipes, and Model EMA for AC | EfficientNet-B4 Landmark U-Net | training |
| `src/fetal/train/train_ensemble.py` | AC | Training loops, multi-seed ensembling, training recipes, and Model EMA for AC | EfficientNet-B4 Landmark U-Net | training |
| `src/fetal/train/train_hc18.py` | HC18 | 5-fold training loop with early stopping for HC18 segmentation | EfficientNet-B4 U-Net | training |
| `tests/conftest.py` | General | General project unit test: verifies package integrity | None | check tool |
| `tests/reference/ac_v3.py` | AC | Unit test for AC: burned-in yellow calipers / labels -> inpaint (applied identically in train and test) | EfficientNet-B4 Landmark U-Net | check tool |
| `tests/reference/ac_v3_ens_original.py` | AC | Unit test for AC: verifies AC models, geometry, and recipes | EfficientNet-B4 Landmark U-Net | check tool |
| `tests/reference/ac_v3_original.py` | AC | Unit test for AC: burned-in yellow calipers / labels -> inpaint (applied identically in train and test) | EfficientNet-B4 Landmark U-Net | check tool |
| `tests/reference/hc18_original.py` | HC18 | Unit test for HC18: Prefer fitEllipseAMS (more robust to noisy mask boundaries); fall back to fitEllipse. | EfficientNet-B4 U-Net / nnU-Net | check tool |
| `tests/test_ac_campaign_options.py` | AC | Unit test for AC: tests/test_ac_campaign_options.py | EfficientNet-B4 Landmark U-Net | check tool |
| `tests/test_ac_ensemble_regression.py` | AC | Unit test for AC: REGRESSION TEST B: | EfficientNet-B4 Landmark U-Net | check tool |
| `tests/test_ac_equality.py` | AC | Unit test for AC: Verify build_data produces identical DataFrame to original. | EfficientNet-B4 Landmark U-Net | check tool |
| `tests/test_ac_infer.py` | AC | Unit test for AC: Tests for AC single-image inference API (fetal.infer.ac_infer). | EfficientNet-B4 Landmark U-Net | check tool |
| `tests/test_ac_lab_history.py` | AC | Unit test for AC: test_ac_lab_history.py: Unit tests verifying AC lab-PC history proof data and facts. | EfficientNet-B4 Landmark U-Net | check tool |
| `tests/test_ac_step0.py` | AC | Unit test for AC: tests/test_ac_step0.py | EfficientNet-B4 Landmark U-Net | check tool |
| `tests/test_ac_train_smoke.py` | AC | Unit test for AC: Smoke-level test: 2 tiny batches on CPU. | EfficientNet-B4 Landmark U-Net | check tool |
| `tests/test_ac_v3_regression.py` | AC | Unit test for AC: REGRESSION TEST A: | EfficientNet-B4 Landmark U-Net | check tool |
| `tests/test_benchmark_summary.py` | Report tools | Unit test for Report tools: Unit tests for multi-modal benchmark summary (scripts/20_benchmark_summary.py). | None (Verification Harness) | check tool |
| `tests/test_clinical_data.py` | Tabular | Unit test for Tabular: Test that CAT_FEATURES sentinel values and COMBGEST 99 are converted to NaN, and deduplication works. | XGBoost / GBDT | check tool |
| `tests/test_clinical_eval.py` | Tabular | Unit test for Tabular: Verify bootstrap CI generates valid lower and upper percentiles containing the point estimate. | XGBoost / GBDT | check tool |
| `tests/test_clinical_guard.py` | Tabular | Unit test for Tabular: Verify that none of the predefined feature sets contain outcome-time variables. | XGBoost / GBDT | check tool |
| `tests/test_clinical_model.py` | Tabular | Unit test for Tabular: Create a minimal synthetic dataset with YEAR, OUTCOME, and feature columns. | XGBoost / GBDT | check tool |
| `tests/test_clinical_regression.py` | Tabular | Unit test for Tabular: Automated regression test suite for Stream 3 clinical tabular risk models. | XGBoost / GBDT | check tool |
| `tests/test_code_map.py` | General | General project unit test: test_code_map.py: Verify that all code files are mapped in docs/CODE_MAP.md. | None | check tool |
| `tests/test_fassd_audit.py` | FASSD | Unit test for FASSD: test_fassd_audit.py: Unit tests for FASSD data loader and audit tools using synthetic data. | EfficientNet-B4 Multilabel U-Net | check tool |
| `tests/test_fassd_ensemble.py` | FASSD | Unit test for FASSD: Unit tests for FASSD Multilabel Segmentation Ensembling module. | EfficientNet-B4 Multilabel U-Net | check tool |
| `tests/test_fassd_postproc.py` | FASSD | Unit test for FASSD: Tests for FASSD post-processing module. | EfficientNet-B4 Multilabel U-Net | check tool |
| `tests/test_fassd_train.py` | FASSD | Unit test for FASSD: Unit tests for FASSD multilabel segmentation training and metrics. | EfficientNet-B4 Multilabel U-Net | check tool |
| `tests/test_femur_data.py` | Femur | Unit test for Femur: Fast unit tests for femur dataset parsing and length calculation. | EfficientNet-B4 / ConvNeXt Landmark | check tool |
| `tests/test_femur_models.py` | Femur | Unit test for Femur: Tests for Femur (FL) landmark models, decoders, losses, and splits. | EfficientNet-B4 / ConvNeXt Landmark | check tool |
| `tests/test_femur_round2_checks.py` | Femur | Unit test for Femur: Automated regression and verification tests for Femur Round 2 baseline sensitivity and run integrity. | EfficientNet-B4 / ConvNeXt Landmark | check tool |
| `tests/test_geometry.py` | AC | Unit test for AC: Verify permutation tensor is strictly identical to original. | EfficientNet-B4 Landmark U-Net | check tool |
| `tests/test_hadlock.py` | Fusion | Unit test for Fusion: Test vector quoted verbatim from Capstone_fetalDataset.ipynb Cell 13: | Hadlock Formula / Growth Percentiles | check tool |
| `tests/test_hadlock_demo.py` | Fusion | Unit test for Fusion: Tests for Hadlock Demo Step 1: Usable subjects audit and head external validation. | Hadlock Formula / Growth Percentiles | check tool |
| `tests/test_hadlock_step2.py` | Fusion | Unit test for Fusion: test_hadlock_step2.py: Unit tests for Hadlock demo step 2. | Hadlock Formula / Growth Percentiles | check tool |
| `tests/test_hc18_biometrics.py` | HC18 | Unit test for HC18: Unit tests for HC18 biometric evaluation, submission formatting, tercile bins, and bootstrap CI. | EfficientNet-B4 U-Net / nnU-Net | check tool |
| `tests/test_hc18_calibration.py` | HC18 | Unit test for HC18: test_hc18_calibration.py: Unit tests for HC18 OOF calibration study using synthetic data. | EfficientNet-B4 U-Net / nnU-Net | check tool |
| `tests/test_hc18_constants.py` | HC18 | Unit test for HC18: Verify training constants in train_hc18.py match the original notebook reference. | EfficientNet-B4 U-Net / nnU-Net | check tool |
| `tests/test_hc18_equivalence.py` | HC18 | Unit test for HC18: Equivalence and isolation tests for HC18 extraction against reference implementation. | EfficientNet-B4 U-Net / nnU-Net | check tool |
| `tests/test_hc18_ladder.py` | HC18 | Unit test for HC18: test_hc18_ladder.py: Tests for HC18 historical approaches and ladder fold-1 setup. | EfficientNet-B4 U-Net / nnU-Net | check tool |
| `tests/test_hc18_nnunet.py` | HC18 | Unit test for HC18: tests/test_hc18_nnunet.py | EfficientNet-B4 U-Net / nnU-Net | check tool |
| `tests/test_hc18_official_summary.py` | HC18 | Unit test for HC18: Tests for HC18 official test set evaluation summary integrity. | EfficientNet-B4 U-Net / nnU-Net | check tool |
| `tests/test_hc18_roundtrip.py` | HC18 | Unit test for HC18: Unit tests for HC18 ellipse roundtrip decoding and rasterization conventions. | EfficientNet-B4 U-Net / nnU-Net | check tool |
| `tests/test_hc18_split.py` | HC18 | Unit test for HC18: Tests for HC18 patient-grouped cross-validation splitting. | EfficientNet-B4 U-Net / nnU-Net | check tool |
| `tests/test_hc18_submission.py` | HC18 | Unit test for HC18: Fast unit tests for HC18 challenge submission ellipse formatting and pipeline. | EfficientNet-B4 U-Net / nnU-Net | check tool |
| `tests/test_hc18_train_smoke.py` | HC18 | Unit test for HC18: Smoke test for HC18 training pipeline on CPU. | EfficientNet-B4 U-Net / nnU-Net | check tool |
| `tests/test_head_landmark.py` | Head landmarks | Unit test for Head landmarks: Automated unit and regression tests for Head landmark pipeline. | EfficientNet-B4 Caliper U-Net | check tool |
| `tests/test_lit_benchmark.py` | Report tools | Unit test for Report tools: tests/test_lit_benchmark.py | None (Verification Harness) | check tool |
| `tests/test_manifest.py` | Report tools | Unit test for Report tools: Tests for weights manifest generation and verification tool (scripts/08_make_manifest.py). | None (Verification Harness) | check tool |
| `tests/test_recipes_equality.py` | AC | Unit test for AC: verifies AC models, geometry, and recipes | EfficientNet-B4 Landmark U-Net | check tool |
| `tests/test_report_numbers.py` | Report tools | Unit test for Report tools: tests/test_report_numbers.py | None (Verification Harness) | check tool |
| `tests/test_split_equality.py` | AC | Unit test for AC: Verify split_data produces identical tr, dv, te frames with exact ordering and patients. | EfficientNet-B4 Landmark U-Net | check tool |
| `tests/test_stream3_bench.py` | Tabular | Unit test for Tabular: Unit tests for Stream 3 clinical model benchmark (scripts/18_bench_stream3.py). | XGBoost / GBDT | check tool |
| `tests/test_three_plane.py` | Fusion | Unit test for Fusion: tests/test_three_plane.py: Unit tests for three-plane subject and leakage check. | Hadlock Formula / Growth Percentiles | check tool |
| `tests/test_train_equivalence.py` | General | General project unit test: Test 2: AC_SMOKE=1 training equivalence between reference ac_v3_original.py and new fetal.train.train_ac. | None | check tool |
| `configs/paths.yaml` | General | Single source of truth for all data, model, and output paths | None | config |
| `notebooks` | General | Notebook directory containing: S3_clinical_audit.ipynb | None | evaluation |
| `notebooks/benchmarks` | Report tools | Executed benchmark notebooks verifying reproducible claims: 15 notebooks | Multi-Model Benchmarks | evaluation |
| `notebooks/history` | General | Historical pipeline record notebook: Capstone_fetalDataset_history.ipynb | Historical Pipelines | evaluation |
| `notebooks/history/ac_lab_pc` | AC | Historical lab-PC proof notebook for AC models: ac_lab_pc_history.ipynb | EfficientNet-B4 Landmark U-Net | evaluation |
| `notebooks/kaggle` | General | Notebook directory containing: fassd_train_kernel.ipynb, fassd_train_kernel_round2.ipynb | None | evaluation |
| `notebooks/kaggle/ac_campaign/ac-baseline-s42` | AC | Kaggle 5-fold cross-validation notebooks for AC landmark recipes: ac-baseline-s42.ipynb | EfficientNet-B4 Landmark U-Net | training |
| `notebooks/kaggle/ac_campaign/ac-baseline-s43` | AC | Kaggle 5-fold cross-validation notebooks for AC landmark recipes: ac-baseline-s43.ipynb | EfficientNet-B4 Landmark U-Net | training |
| `notebooks/kaggle/ac_campaign/ac-baseline-s44` | AC | Kaggle 5-fold cross-validation notebooks for AC landmark recipes: ac-baseline-s44.ipynb | EfficientNet-B4 Landmark U-Net | training |
| `notebooks/kaggle/ac_campaign/ac-baseline-s45` | AC | Kaggle 5-fold cross-validation notebooks for AC landmark recipes: ac-baseline-s45.ipynb | EfficientNet-B4 Landmark U-Net | training |
| `notebooks/kaggle/ac_campaign/ac-ema-s42` | AC | Kaggle 5-fold cross-validation notebooks for AC landmark recipes: ac-ema-s42.ipynb | EfficientNet-B4 Landmark U-Net | training |
| `notebooks/kaggle/ac_campaign/ac-fassd-s42` | AC | Kaggle 5-fold cross-validation notebooks for AC landmark recipes: ac-fassd-s42.ipynb | EfficientNet-B4 Landmark U-Net | training |
| `notebooks/kaggle/ac_campaign/ac-wing-s42` | AC | Kaggle 5-fold cross-validation notebooks for AC landmark recipes: ac-wing-s42.ipynb | EfficientNet-B4 Landmark U-Net | training |
| `notebooks/kaggle/ac_nnunet/ac-nnunet-f0` | AC | Kaggle training notebook for AC nnU-Net: ac-nnunet-f0.ipynb | nnU-Net | training |
| `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f0-250ep` | HC18 | Kaggle notebooks for HC18 nnU-Net: hc18-nnunet-f0-250ep.ipynb | nnU-Net | training |
| `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f1f2-100ep` | HC18 | Kaggle notebooks for HC18 nnU-Net: hc18-nnunet-f1f2-100ep.ipynb | nnU-Net | training |
| `notebooks/kaggle/hc18_nnunet/hc18-nnunet-f3f4-100ep` | HC18 | Kaggle notebooks for HC18 nnU-Net: hc18-nnunet-f3f4-100ep.ipynb | nnU-Net | training |
| `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f0` | HC18 | Kaggle notebooks for HC18 nnU-Net: hc18-nnunet-test-f0.ipynb | nnU-Net | evaluation |
| `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f012` | HC18 | Kaggle notebooks for HC18 nnU-Net: hc18-nnunet-test-f012.ipynb | nnU-Net | evaluation |
| `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f012-cpu` | HC18 | Kaggle notebooks for HC18 nnU-Net: hc18-nnunet-test-f012-cpu.ipynb | nnU-Net | evaluation |
| `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f01234` | HC18 | Kaggle notebooks for HC18 nnU-Net: hc18-nnunet-test-f01234.ipynb | nnU-Net | evaluation |
| `notebooks/kaggle/hc18_nnunet_test/hc18-nnunet-test-f01234-cpu` | HC18 | Kaggle notebooks for HC18 nnU-Net: hc18-nnunet-test-f01234-cpu.ipynb | nnU-Net | evaluation |
| `notebooks/kaggle/stream3_modern/stream3-modern-s1` | Tabular | Kaggle tuning or audit notebook for tabular Stream 3: stream3-modern-s1.ipynb | XGBoost / LightGBM / CatBoost | training |

---

## Workflow Subsystem Mappings: Train, Predict, and Evaluate

### (a) HC18 Head Circumference with nnU-Net
- **Train (Data Prep & Kernel Generation)**:
  - `scripts/45_hc18_to_nnunet.py`: Converts raw HC18 data to nnU-Net v2 `Dataset501_HC18` with patient-grouped folds in `splits_final.json`.
  - `scripts/47_make_nnunet_kaggle_kernel.py`: Builds 250-epoch Fold-0 Kaggle training kernel (`hc18-nnunet-f0-250ep`).
  - `scripts/55_make_hc18_nnunet_parallel_kernels.py`: Builds parallel 100-epoch Kaggle training kernels for folds 1-2 and 3-4.
  - `notebooks/kaggle/hc18_nnunet/`: Executed Kaggle GPU training orchestration notebooks.
- **Predict (Test Inference & Candidate Post-Processing)**:
  - `scripts/52_make_hc18_nnunet_test_kernel.py`: Builds single-fold (fold 0) test inference Kaggle kernel.
  - `scripts/66_make_hc18_nnunet_test_f012_kernel.py`: Builds 3-fold (folds 0, 1, 2) GPU test inference kernel.
  - `scripts/71_make_hc18_nnunet_test_f012_cpu_kernel.py`: Builds 3-fold CPU test inference kernel with automated discovery.
  - `scripts/77_make_hc18_nnunet_test_f01234_kernels.py`: Builds 5-fold (folds 0-4) GPU and CPU test inference kernels.
  - `scripts/51_test_nnunet_test_script.py`, `scripts/72_smoke_test_nnunet_cpu.py`, `scripts/79_smoke_test_5fold_cpu.py`: Local test inference verification and CPU smoke test runners.
  - `scripts/75_process_nnunet_f012_test.py`: Transforms 3-fold test predictions into raw and measfix (`delta_r = 0.1134` mm) submission CSVs.
  - `scripts/80_process_nnunet_f01234_test.py`: Transforms 5-fold test predictions into raw, measfix, and heterogeneous blend submission CSVs.
- **Evaluate (Validation Folds & Challenge Test Analysis)**:
  - `scripts/46_eval_nnunet_hc18.py`: Evaluates fold 1 validation predictions against official ground truth ellipse reference.
  - `scripts/68_eval_nnunet_f1_f2.py`: Evaluates folds 1 and 2 validation predictions with paired patient-level bootstrap vs EfficientNet-B4.
  - `scripts/76_eval_nnunet_f3_f4_and_pooled5.py`: Evaluates folds 3 and 4, and computes pooled 5-fold cross-validation metrics (MAE, bias, Dice, HD).
  - `scripts/69_hc18_nnunet_blend_and_pooled.py`: Evaluates heterogeneous ellipse blending between nnU-Net and our EfficientNet-B4 ensemble.
  - `scripts/29_hc18_official_summary.py`: Evaluates official Grand Challenge test set results on grand-challenge.org.

### (b) HC18 Head Circumference with our EfficientNet-B4 5-Model Ensemble
- **Train**:
  - `src/fetal/train/train_hc18.py`: 5-fold cross-validation training loop with early stopping, mixed precision, and BCE+Dice loss.
  - `scripts/10_train_hc18.py`: CLI training runner for the 5-fold EfficientNet-B4 U-Net models.
  - `scripts/19_hc18_ladder.py`, `scripts/kaggle/run_hc18_ladder.py`: Historical approach ladder runner (Rows A-D vs Row E).
  - `notebooks/benchmarks/hc18_ladder.ipynb`: Executed benchmark training notebook.
- **Predict**:
  - `src/fetal/eval/hc18_eval.py`: Predicts probability masks, fits Ramanujan ellipses, and extracts HC/BPD in mm.
  - `scripts/12_hc18_test_inference.py`: Runs 5-model ensemble test inference to build `submission_ensemble_5fold.csv` and `submission_single_fold1.csv`.
  - `scripts/50_measurement_fix_study.py`: Applies $\delta_r = 0.1134$ mm semi-axis adjustment (`submission_ensemble_5fold_corrected_measfix.csv`).
  - `scripts/70_build_blend_candidate.py`, `scripts/78_build_blend_f012_ens5.py`: Builds heterogeneous ellipse blend candidates with nnU-Net.
- **Evaluate**:
  - `scripts/11_hc18_oof_report.py`: Evaluates honest out-of-fold predictions on all 999 training images across 5 folds.
  - `scripts/09_hc18_leakage_report.py`: Quantifies patient leakage impact between random splits and patient-grouped splits.
  - `scripts/30_hc18_ellipse_roundtrip.py`: Evaluates ellipse round-trip rasterization conventions.
  - `scripts/34_hc18_oof_calibration.py`, `scripts/43_hc18_calibration_candidates.py`: Evaluates post-hoc multiplicative and linear calibrations.
  - `scripts/47_audit_bias_direction.py`: Audits signed bias directionality convention (`pred - true` vs `true - pred`).
  - `notebooks/benchmarks/hc18_official.ipynb`, `notebooks/benchmarks/hc18_oof_calibration.ipynb`: Executed benchmark evaluation notebooks.

### (c) AC with our Landmark Model
- **Train**:
  - `src/fetal/train/train_ac.py`: Training loop for landmark coordinate heatmap regression model.
  - `src/fetal/train/train_ensemble.py`: 4-seed ensemble training orchestration.
  - `scripts/04_train_ac.py`, `scripts/05_train_ensemble.py`: CLI runners for single-seed and multi-seed training.
  - `scripts/42_make_ac_kaggle_kernels.py`: Generates Kaggle cross-validation kernels (`ac-baseline-s42` through `s45`, `ac-wing-s42`, `ac-ema-s42`, `ac-fassd-s42`).
  - `notebooks/kaggle/ac_campaign/`: Executed 5-fold cross-validation notebooks on Kaggle.
- **Predict**:
  - `src/fetal/infer/ac_infer.py`: Single-image inference returning AC in mm.
  - `src/fetal/eval/ac_evaluator.py`, `src/fetal/eval/ensemble_evaluator.py`: Batched landmark prediction, test-time augmentation (TTA), and multi-seed averaging.
  - `src/fetal/eval/ac_geometry.py`: Geometric landmark matching and Ramanujan ellipse circumference calculation.
- **Evaluate**:
  - `scripts/06_eval_ac.py`: Evaluates official held-out test set reproducing 4.395 mm (v3_full) and 4.343 mm (ensemble) metrics.
  - `scripts/38_ac_step0_audit.py`: Read-only dataset audit and duplicate image SHA-256 verification.
  - `scripts/40_ac_compare_recipes.py`: Paired cluster bootstrap (5,000 resamples, seed 42) comparing candidate recipes.
  - `scripts/82_eval_ac_s43_and_wing_comparison.py`: Evaluates seed 43 baseline, wing loss, and 2-seed ensemble.
  - `scripts/86_eval_ac_seeds_and_4seed_ensemble.py`: Evaluates seeds 44 and 45, 2-seed ensemble, and 4-seed ensemble with paired bootstraps.
  - `scripts/48_make_worst8_contact_sheet.py`, `scripts/diagnose_worst8.py`, `scripts/analyze_worst_and_calipers.py`: Diagnostic outlier inspection.
  - `notebooks/benchmarks/ac_results.ipynb`: Executed benchmark notebook reproducing test set results.

### (d) AC with nnU-Net
- **Train**:
  - `scripts/83_ac_to_nnunet.py`: Converts AC dataset into nnU-Net v2 `Dataset502_AC` format with filled solid ellipse masks.
  - `scripts/84_make_ac_nnunet_kaggle_kernel.py`: Builds `ac-nnunet-f0` 100-epoch Fold-0 Kaggle training kernel.
  - `scripts/smoke_test_ac_nnunet_f0.py`: Smoke tests local training pipeline forward pass and loss computation.
  - `notebooks/kaggle/ac_nnunet/ac-nnunet-f0/`: Kaggle training kernel for AC nnU-Net.
- **Predict**:
  - `scripts/85_eval_ac_nnunet_f0.py`: Predicts 142 fold-0 validation cases, extracts largest connected component, fits ellipse with `cv2.fitEllipseAMS`, and measures AC in mm.
- **Evaluate**:
  - `scripts/85_eval_ac_nnunet_f0.py`: Evaluates fold-0 MAE, RMSE, median, bias, >10mm rate, and paired bootstrap vs landmark models `ac-baseline-s42`, `s43`, and 2-seed average.

### (e) The Tabular Models (Stream 3)
- **Train**:
  - `src/fetal/clinical/model.py`: Prevalence-adjusted XGBoost model training.
  - `scripts/18_bench_stream3.py`: Trains and benchmarks Logistic Regression, Random Forest, LightGBM, and XGBoost.
  - `scripts/53_make_stream3_modern_kernel.py`: Builds Kaggle Optuna tuning kernel `stream3-modern-s1`.
  - `scripts/62_train_stream3_modern.py`: Trains Optuna-tuned LightGBM, XGBoost, CatBoost, and Calibrated Blend on <=2021 cohort.
  - `notebooks/kaggle/stream3_modern/stream3-modern-s1/`: Kaggle tuning kernel.
- **Predict**:
  - `src/fetal/clinical/eval.py`: Predicts risk probabilities, prevalence-corrected scores, and isotonic calibrations.
  - `scripts/62_train_stream3_modern.py`: Generates test set predictions on >=2022 temporal hold-out cohort.
- **Evaluate**:
  - `src/fetal/clinical/eval.py`: Calculates ROC-AUC, 95% bootstrap confidence intervals, Brier score, and calibration curves.
  - `src/fetal/clinical/audit.py`, `scripts/07_audit_clinical.py`: 14-row clinical audit explaining historical data leakage vs honest models.
  - `notebooks/S3_clinical_audit.ipynb`, `notebooks/benchmarks/stream3_bench.ipynb`: Executed audit and benchmark notebooks.

---

## Storage Layout & NTFS Directory Junctions

Large project data, checkpoints, results, and caches have been relocated to secondary volume `E:\fetal_moved\` and `E:\fetal_cache\`, linked back transparently to `C:\` via NTFS directory junctions (`mklink /J`). All code and scripts continue referencing default `C:\...` paths with zero configuration changes:
- `C:\fetal-data` &harr; `E:\fetal_moved\fetal-data` (23.54 GB)
- `C:\fetal-models` &harr; `E:\fetal_moved\fetal-models` (2.67 GB)
- `C:\fetal-results` &harr; `E:\fetal_moved\fetal-results` (1.95 GB)
- `C:\fetal-health-multimodal` &harr; `E:\fetal_moved\fetal-health-multimodal` (1.27 GB)
- `C:\fetal-stage` &harr; `E:\fetal_moved\fetal-stage` (0.64 GB)
- Full audit and verification ledger is documented in [MOVED.md](file:///E:/fetal_moved/MOVED.md).
