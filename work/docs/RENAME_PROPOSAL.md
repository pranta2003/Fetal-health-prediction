# Script Renaming Proposal (Design Only - No Files Renamed)

This proposal defines a standardized naming convention for all scripts in `work/scripts/` following the pattern:

```text
<field>_<model>_<step>.py
```

Where:
- `<field>`: `hc18`, `ac`, `femur`, `head_landmark`, `fassd`, `tabular`, `fusion`, `report_tools`, `general`
- `<model>`: `nnunet`, `effb4`, `convnext`, `xgb`, `hadlock`, `ladder`, `multi`, `none`
- `<step>`: `train`, `eval`, `dataprep`, `kernel_builder`, `test`, `audit`, `postproc`, `ensemble`, `diag`, `util`

### Risk Classification:
- **RISKY**: Referenced by an executed notebook (`notebooks/`), Kaggle kernel, or GitHub Actions workflow.
- **SAFE**: Standalone development script, test verification utility, or referenced strictly by local documentation.

## Renaming Table

| Old Name | Proposed New Name | Risk Status | Referencing Files |
| :--- | :--- | :---: | :--- |
| `04_train_ac.py` | `ac_efficientnetb4_training_04_train_ac.py` | **RISKY** | `work/docs/ARCHITECTURE.md`<br>`work/docs/CODE_MAP.md`<br>`work/docs/report/EVIDENCE_MAP.md`<br>`work/docs/results/ac_campaign/STEP0.md`<br>*(+11 more)* |
| `05_train_ensemble.py` | `general_none_utility_05_train_ensemble.py` | **SAFE** | `work/docs/ARCHITECTURE.md`<br>`work/docs/CODE_MAP.md`<br>`work/docs/results/ac_campaign/STEP0.md`<br>`work/scripts/38_ac_step0_audit.py`<br>*(+1 more)* |
| `06_eval_ac.py` | `ac_efficientnetb4_evaluation_06_eval_ac.py` | **RISKY** | `work/docs/ARCHITECTURE.md`<br>`work/docs/CODE_MAP.md`<br>`work/docs/report/EVIDENCE_MAP.md`<br>`work/docs/results/ac_campaign/AC_CAMPAIGN_NOTES.md`<br>*(+5 more)* |
| `07_audit_clinical.py` | `tabular_xgboost_audit_07_audit_clinical.py` | **SAFE** | `work/docs/ARCHITECTURE.md`<br>`work/docs/CODE_MAP.md`<br>`work/docs/report/EVIDENCE_MAP.md`<br>`work/scripts/run_all.ps1` |
| `08_make_manifest.py` | `report_tools_none_report_tools_08_make_manifest.py` | **SAFE** | `work/docs/ARCHITECTURE.md`<br>`work/docs/CODE_MAP.md`<br>`work/scripts/run_all.ps1`<br>`work/tests/test_code_map.py`<br>*(+1 more)* |
| `09_hc18_leakage_report.py` | `hc18_efficientnetb4_leakage_audit_09_hc18_leakage_report.py` | **SAFE** | `work/docs/ARCHITECTURE.md`<br>`work/docs/CODE_MAP.md`<br>`work/docs/PROGRESS.md` |
| `10_train_hc18.py` | `hc18_efficientnetb4_training_10_train_hc18.py` | **SAFE** | `work/docs/ARCHITECTURE.md`<br>`work/docs/CODE_MAP.md`<br>`work/docs/PROGRESS.md`<br>`work/docs/report/EVIDENCE_MAP.md`<br>*(+1 more)* |
| `11_hc18_oof_report.py` | `hc18_efficientnetb4_evaluation_11_hc18_oof_report.py` | **RISKY** | `work/docs/ARCHITECTURE.md`<br>`work/docs/CODE_MAP.md`<br>`work/scripts/kaggle/run_hc18_close.py` |
| `12_hc18_test_inference.py` | `hc18_efficientnetb4_inference_12_hc18_test_inference.py` | **RISKY** | `work/docs/ARCHITECTURE.md`<br>`work/docs/CODE_MAP.md`<br>`work/docs/PROGRESS.md`<br>`work/scripts/kaggle/run_hc18_close.py` |
| `14_hash_results.py` | `report_tools_none_report_tools_14_hash_results.py` | **SAFE** | `work/docs/ARCHITECTURE.md`<br>`work/docs/CODE_MAP.md`<br>`work/docs/PROGRESS.md` |
| `15_femur_data_report.py` | `femur_efficientnetb4_audit_15_femur_data_report.py` | **RISKY** | `work/docs/CODE_MAP.md`<br>`work/notebooks/benchmarks/benchmark_summary.ipynb`<br>`work/scripts/20_benchmark_summary.py` |
| `16_extract_notebook_results.py` | `ac_efficientnetb4_utility_16_extract_notebook_results.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `18_bench_stream3.py` | `tabular_xgboost_training_18_bench_stream3.py` | **RISKY** | `work/docs/CODE_MAP.md`<br>`work/docs/report/EVIDENCE_MAP.md`<br>`work/docs/report/REPORT_NUMBERS.json`<br>`work/docs/report/REPORT_NUMBERS.md`<br>*(+4 more)* |
| `19_hc18_ladder.py` | `hc18_historical_training_19_hc18_ladder.py` | **RISKY** | `work/docs/CODE_MAP.md`<br>`work/scripts/kaggle/run_hc18_ladder.py` |
| `20_benchmark_summary.py` | `general_none_utility_20_benchmark_summary.py` | **RISKY** | `work/docs/CODE_MAP.md`<br>`work/notebooks/benchmarks/benchmark_summary.ipynb`<br>`work/scripts/make_bench_notebooks.py`<br>`work/tests/test_benchmark_summary.py` |
| `21_femur_data_followup.py` | `femur_efficientnetb4_audit_21_femur_data_followup.py` | **RISKY** | `work/docs/CODE_MAP.md`<br>`work/notebooks/benchmarks/femur_data.ipynb`<br>`work/scripts/make_bench_notebooks.py` |
| `22_femur_ablation.py` | `femur_efficientnetb4_evaluation_22_femur_ablation.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `23_three_plane_check.py` | `fusion_hadlock_audit_23_three_plane_check.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/tests/test_hadlock_step2.py`<br>`work/tests/test_three_plane.py` |
| `24_hadlock_demo_step1.py` | `fusion_hadlock_evaluation_24_hadlock_demo_step1.py` | **RISKY** | `work/docs/CODE_MAP.md`<br>`work/notebooks/benchmarks/head_external.ipynb`<br>`work/scripts/make_head_external_notebook.py` |
| `25_head_landmark.py` | `head_landmarks_efficientnetb4_evaluation_25_head_landmark.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `26_hadlock_demo_step2.py` | `fusion_hadlock_evaluation_26_hadlock_demo_step2.py` | **RISKY** | `work/docs/CODE_MAP.md`<br>`work/notebooks/benchmarks/hadlock_step2.ipynb`<br>`work/scripts/make_hadlock_step2_notebook.py` |
| `27_femur_round2.py` | `femur_efficientnetb4_evaluation_27_femur_round2.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `28_ac_history_table.py` | `ac_efficientnetb4_audit_28_ac_history_table.py` | **RISKY** | `work/docs/CODE_MAP.md`<br>`work/notebooks/history/ac_lab_pc/README.md`<br>`work/notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb`<br>`work/tests/test_ac_lab_history.py` |
| `29_hc18_official_summary.py` | `hc18_efficientnetb4_evaluation_29_hc18_official_summary.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `30_hc18_ellipse_roundtrip.py` | `hc18_efficientnetb4_evaluation_30_hc18_ellipse_roundtrip.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `31_femur_round2_checks.py` | `femur_efficientnetb4_evaluation_31_femur_round2_checks.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/results/femur_round2/baseline_sensitivity.json` |
| `32_femur_outlier_impact.py` | `ac_efficientnetb4_utility_32_femur_outlier_impact.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `33_fassd_audit.py` | `fassd_efficientnetb4_data_prep_33_fassd_audit.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `34_hc18_oof_calibration.py` | `hc18_efficientnetb4_calibration_34_hc18_oof_calibration.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/tests/test_hc18_calibration.py` |
| `35_fassd_train.py` | `fassd_efficientnetb4_training_35_fassd_train.py` | **RISKY** | `work/docs/CODE_MAP.md`<br>`work/docs/results/fassd/RUN1_HD95_VERIFICATION.md`<br>`work/notebooks/kaggle/fassd_train_kernel.ipynb`<br>`work/notebooks/kaggle/fassd_train_kernel_round2.ipynb`<br>*(+2 more)* |
| `36_fassd_postproc.py` | `fassd_efficientnetb4_postproc_36_fassd_postproc.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/report/EVIDENCE_MAP.md` |
| `37_fassd_ensemble.py` | `fassd_efficientnetb4_ensemble_37_fassd_ensemble.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/report/EVIDENCE_MAP.md` |
| `38_ac_step0_audit.py` | `ac_efficientnetb4_audit_38_ac_step0_audit.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/results/ac_campaign/STEP0.md`<br>`work/tests/test_ac_step0.py` |
| `39_make_ac_folds.py` | `ac_efficientnetb4_data_prep_39_make_ac_folds.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `40_ac_compare_recipes.py` | `ac_efficientnetb4_evaluation_40_ac_compare_recipes.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/report/EVIDENCE_MAP.md`<br>`work/docs/report/REPORT_NUMBERS.json`<br>`work/docs/report/REPORT_NUMBERS.md`<br>*(+2 more)* |
| `41_prepare_ac_kaggle_dataset.py` | `ac_efficientnetb4_data_prep_41_prepare_ac_kaggle_dataset.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/report/EVIDENCE_MAP.md` |
| `42_make_ac_kaggle_kernels.py` | `ac_efficientnetb4_data_prep_42_make_ac_kaggle_kernels.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/report/EVIDENCE_MAP.md` |
| `43_hc18_calibration_candidates.py` | `hc18_efficientnetb4_calibration_43_hc18_calibration_candidates.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `44_generate_best_published.py` | `report_tools_none_report_tools_44_generate_best_published.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/results/literature/BEST_PUBLISHED.md` |
| `45_hc18_to_nnunet.py` | `hc18_nnunet_data_prep_45_hc18_to_nnunet.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `46_eval_nnunet_hc18.py` | `hc18_nnunet_evaluation_46_eval_nnunet_hc18.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/report/EVIDENCE_MAP.md`<br>`work/docs/report/REPORT_NUMBERS.json`<br>`work/docs/report/REPORT_NUMBERS.md`<br>*(+2 more)* |
| `47_audit_bias_direction.py` | `general_none_utility_47_audit_bias_direction.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `47_make_nnunet_kaggle_kernel.py` | `hc18_nnunet_kernel_builder_47_make_nnunet_kaggle_kernel.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/tests/test_hc18_nnunet.py` |
| `48_make_worst8_contact_sheet.py` | `ac_efficientnetb4_diagnostic_48_make_worst8_contact_sheet.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `49_update_results_ledger.py` | `report_tools_none_report_tools_49_update_results_ledger.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `50_measurement_fix_study.py` | `general_none_utility_50_measurement_fix_study.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `51_test_nnunet_test_script.py` | `hc18_nnunet_smoke_test_51_test_nnunet_test_script.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `52_make_hc18_nnunet_test_kernel.py` | `hc18_nnunet_kernel_builder_52_make_hc18_nnunet_test_kernel.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/report/GAPS.md`<br>`work/docs/report/REPORT_NUMBERS.json`<br>`work/docs/report/REPORT_NUMBERS.md`<br>*(+1 more)* |
| `53_make_stream3_modern_kernel.py` | `tabular_xgboost_kernel_builder_53_make_stream3_modern_kernel.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `53_test_discovery_mock.py` | `general_none_utility_53_test_discovery_mock.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `54_eval_linear_fit_followup.py` | `general_none_utility_54_eval_linear_fit_followup.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `55_make_hc18_nnunet_parallel_kernels.py` | `hc18_nnunet_kernel_builder_55_make_hc18_nnunet_parallel_kernels.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `56_build_report_numbers.py` | `report_tools_none_report_tools_56_build_report_numbers.py` | **SAFE** | `README.md`<br>`work/docs/CLEANUP_PLAN.md`<br>`work/docs/CODE_MAP.md`<br>`work/scripts/57_verify_report_numbers.py`<br>*(+1 more)* |
| `57_verify_report_numbers.py` | `report_tools_none_report_tools_57_verify_report_numbers.py` | **SAFE** | `README.md`<br>`work/docs/CLEANUP_PLAN.md`<br>`work/docs/CODE_MAP.md`<br>`work/docs/report/AUDIT_SUMMARY.md`<br>*(+2 more)* |
| `58_audit_notebooks.py` | `report_tools_none_report_tools_58_audit_notebooks.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `59_audit_nme_formulas.py` | `report_tools_none_report_tools_59_audit_nme_formulas.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/report/GAPS.md` |
| `60_audit_dataset_counts.py` | `report_tools_none_report_tools_60_audit_dataset_counts.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/report/GAPS.md` |
| `61_audit_hc18_ladder_refs.py` | `hc18_historical_training_61_audit_hc18_ladder_refs.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/report/GAPS.md` |
| `62_train_stream3_modern.py` | `tabular_xgboost_training_62_train_stream3_modern.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/report/REPORT_NUMBERS.json`<br>`work/docs/report/REPORT_NUMBERS.md`<br>`work/docs/results/stream3_modern/train_modern.py`<br>*(+2 more)* |
| `63_verify_nnunet_candidates.py` | `hc18_nnunet_utility_63_verify_nnunet_candidates.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `64_format_hc18_readme.py` | `hc18_efficientnetb4_utility_64_format_hc18_readme.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `65_build_proof_map.py` | `report_tools_none_report_tools_65_build_proof_map.py` | **SAFE** | `README.md`<br>`work/docs/CLEANUP_PLAN.md`<br>`work/docs/CODE_MAP.md`<br>`work/scripts/74_check_master_premerge.py` |
| `66_make_hc18_nnunet_test_f012_kernel.py` | `hc18_nnunet_kernel_builder_66_make_hc18_nnunet_test_f012_kernel.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `67_download_f1f2_and_ac_fassd.py` | `ac_efficientnetb4_utility_67_download_f1f2_and_ac_fassd.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `68_eval_nnunet_f1_f2.py` | `hc18_nnunet_evaluation_68_eval_nnunet_f1_f2.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/report/REPORT_NUMBERS.json`<br>`work/docs/report/REPORT_NUMBERS.md`<br>`work/scripts/56_build_report_numbers.py` |
| `69_hc18_nnunet_blend_and_pooled.py` | `hc18_nnunet_inference_69_hc18_nnunet_blend_and_pooled.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/report/REPORT_NUMBERS.json`<br>`work/docs/report/REPORT_NUMBERS.md`<br>`work/scripts/56_build_report_numbers.py` |
| `70_build_blend_candidate.py` | `general_none_utility_70_build_blend_candidate.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `71_make_hc18_nnunet_test_f012_cpu_kernel.py` | `hc18_nnunet_kernel_builder_71_make_hc18_nnunet_test_f012_cpu_kernel.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `72_smoke_test_nnunet_cpu.py` | `hc18_nnunet_smoke_test_72_smoke_test_nnunet_cpu.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `73_check_pr54_premerge.py` | `general_none_utility_73_check_pr54_premerge.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `74_check_master_premerge.py` | `general_none_utility_74_check_master_premerge.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `75_process_nnunet_f012_test.py` | `hc18_nnunet_smoke_test_75_process_nnunet_f012_test.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/report/REPORT_NUMBERS.json`<br>`work/docs/report/REPORT_NUMBERS.md`<br>`work/scripts/56_build_report_numbers.py` |
| `76_eval_nnunet_f3_f4_and_pooled5.py` | `hc18_nnunet_evaluation_76_eval_nnunet_f3_f4_and_pooled5.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/report/REPORT_NUMBERS.json`<br>`work/docs/report/REPORT_NUMBERS.md`<br>`work/scripts/56_build_report_numbers.py` |
| `77_make_hc18_nnunet_test_f01234_kernels.py` | `hc18_nnunet_kernel_builder_77_make_hc18_nnunet_test_f01234_kernels.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/report/REPORT_NUMBERS.json`<br>`work/docs/report/REPORT_NUMBERS.md`<br>`work/scripts/56_build_report_numbers.py` |
| `78_build_blend_f012_ens5.py` | `general_none_utility_78_build_blend_f012_ens5.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `79_smoke_test_5fold_cpu.py` | `general_none_utility_79_smoke_test_5fold_cpu.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `80_process_nnunet_f01234_test.py` | `hc18_nnunet_smoke_test_80_process_nnunet_f01234_test.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/report/REPORT_NUMBERS.json`<br>`work/docs/report/REPORT_NUMBERS.md`<br>`work/scripts/56_build_report_numbers.py` |
| `81_build_gestational_percentiles.py` | `fusion_hadlock_data_prep_81_build_gestational_percentiles.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/report/PROOF_LOCATIONS.md`<br>`work/docs/report/REPORT_NUMBERS.json`<br>`work/docs/report/REPORT_NUMBERS.md`<br>*(+2 more)* |
| `82_eval_ac_s43_and_wing_comparison.py` | `ac_efficientnetb4_evaluation_82_eval_ac_s43_and_wing_comparison.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/report/PROOF_LOCATIONS.md`<br>`work/docs/report/REPORT_NUMBERS.json`<br>`work/docs/report/REPORT_NUMBERS.md`<br>*(+1 more)* |
| `83_ac_to_nnunet.py` | `ac_nnunet_data_prep_83_ac_to_nnunet.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/report/PROOF_LOCATIONS.md`<br>`work/docs/report/REPORT_NUMBERS.json`<br>`work/docs/report/REPORT_NUMBERS.md`<br>*(+2 more)* |
| `84_make_ac_nnunet_kaggle_kernel.py` | `ac_nnunet_kernel_builder_84_make_ac_nnunet_kaggle_kernel.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/results/RESULTS_LEDGER.md` |
| `85_eval_ac_nnunet_f0.py` | `ac_nnunet_evaluation_85_eval_ac_nnunet_f0.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/results/RESULTS_LEDGER.md` |
| `86_eval_ac_seeds_and_4seed_ensemble.py` | `ac_efficientnetb4_evaluation_86_eval_ac_seeds_and_4seed_ensemble.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/report/PROOF_LOCATIONS.md`<br>`work/docs/report/REPORT_NUMBERS.json`<br>`work/docs/report/REPORT_NUMBERS.md`<br>*(+1 more)* |
| `analyze_worst_and_calipers.py` | `general_none_utility_analyze_worst_and_calipers.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `build_data_inventory.py` | `report_tools_none_report_tools_build_data_inventory.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/results/data_inventory.json` |
| `check_secrets.py` | `general_none_utility_check_secrets.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `diagnose_worst8.py` | `general_none_utility_diagnose_worst8.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `download_hc18_nnunet_artifacts.py` | `hc18_nnunet_utility_download_hc18_nnunet_artifacts.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `run_femur_ablation.py` | `femur_efficientnetb4_training_run_femur_ablation.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `run_femur_round2.py` | `femur_efficientnetb4_training_run_femur_round2.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `run_hc18_close.py` | `hc18_efficientnetb4_inference_run_hc18_close.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/PROGRESS.md` |
| `run_hc18_ladder.py` | `hc18_historical_training_run_hc18_ladder.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `run_head_landmark.py` | `head_landmarks_efficientnetb4_evaluation_run_head_landmark.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `make_bench_notebooks.py` | `report_tools_none_report_tools_make_bench_notebooks.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `make_fassd_audit_notebook.py` | `fassd_efficientnetb4_data_prep_make_fassd_audit_notebook.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `make_fassd_ensemble_notebook.py` | `fassd_efficientnetb4_ensemble_make_fassd_ensemble_notebook.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `make_fassd_kaggle_notebook.py` | `fassd_efficientnetb4_kernel_builder_make_fassd_kaggle_notebook.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `make_fassd_kaggle_notebook_round2.py` | `fassd_efficientnetb4_kernel_builder_make_fassd_kaggle_notebook_round2.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `make_fassd_postproc_notebook.py` | `fassd_efficientnetb4_postproc_make_fassd_postproc_notebook.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `make_fassd_split_v2.py` | `fassd_efficientnetb4_data_prep_make_fassd_split_v2.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `make_femur_ablation_notebook.py` | `femur_efficientnetb4_evaluation_make_femur_ablation_notebook.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `make_hadlock_step2_notebook.py` | `fusion_hadlock_kernel_builder_make_hadlock_step2_notebook.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `make_hc18_calibration_notebook.py` | `hc18_efficientnetb4_calibration_make_hc18_calibration_notebook.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `make_head_external_notebook.py` | `head_landmarks_efficientnetb4_kernel_builder_make_head_external_notebook.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `make_head_landmark_notebook.py` | `head_landmarks_efficientnetb4_kernel_builder_make_head_landmark_notebook.py` | **SAFE** | `work/docs/CODE_MAP.md` |
| `smoke_test_ac_nnunet_f0.py` | `ac_nnunet_smoke_test_smoke_test_ac_nnunet_f0.py` | **SAFE** | `work/docs/CODE_MAP.md`<br>`work/docs/results/RESULTS_LEDGER.md` |
