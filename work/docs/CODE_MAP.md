# CODE MAP: which file belongs to which part

Everything current lives inside the folder `work/`. The old folders at the top level (`src/`, `models/`) are the older version; ignore them.

## The one rule
File names tell you the part. The prefix is the part:

| Prefix / name | Part | Dataset |
|---|---|---|
| `hc18_...` | **Head** (head circumference HC, BPD) | HC18 |
| `ac_...` | **Abdomen** (abdominal circumference AC) | FP + UCL abdomen images |
| `femur_...` | **Femur** (femur length FL) | femur images inside FP + UCL |
| `fassd_...` | **Organs** (explanations) | FASSD |
| inside `clinical/` | **Maternal risk** (stillbirth risk, tables) | CDC natality + fetal death files |
| `hadlock.py` | **Weight formula** (EFW from HC, AC, FL) | no dataset, a formula |
| no prefix (`config.py`, `transforms.py`, `checkpoint.py` ...) | **Shared helpers** | none |

## HEAD (HC18): all files
| Step | File | What it does (plain words) |
|---|---|---|
| 1 Prepare data | `src/fetal/data/hc18_dataset.py` | Loads head images and masks, resizes without distortion, flips for training |
| 1 Prepare data | `src/fetal/data/hc18_split.py` | Splits patients into 5 folds so no patient is in both train and test |
| 1 Prepare data | `scripts/45_hc18_to_nnunet.py` | Converts HC18 training data to nnU-Net v2 Dataset501_HC18 format with splits_final.json matching patient-grouped folds |
| 2 Model | `src/fetal/models/hc18_unet.py` | The neural network (U-Net, EfficientNet-B4) and its loss |
| 2 Model | `src/fetal/models/hc18_old_approaches.py` | Historical baseline models (Rows A, B, C, D) |
| 3 Train | `src/fetal/train/train_hc18.py` | The training loop, 5 folds, early stopping |
| 3 Train | `scripts/10_train_hc18.py` | The button that starts training |
| 4 Measure | `src/fetal/eval/hc18_eval.py` | Turns the mask into an ellipse, then HC and BPD in mm |
| 4 Measure | `src/fetal/eval/ellipse_fitting.py` | Fits the ellipse (shared with abdomen tests) |
| 5 Report | `scripts/11_hc18_oof_report.py` | Computes the honest score on held-out images |
| 5 Report | `scripts/12_hc18_test_inference.py` | Makes the challenge submission files |
| 5 Report | `scripts/09_hc18_leakage_report.py` | Shows how much the old method leaked |
| 5 Report | `scripts/19_hc18_ladder.py` | Trains the historical approach ladder on grouped fold 1 |
| 5 Report | `scripts/29_hc18_official_summary.py` | Generates summary tables and claims for official HC18 challenge test evaluations |
| 5 Report | `scripts/30_hc18_ellipse_roundtrip.py` | Evaluates ellipse round-trip reconstruction and rasterization conventions on training masks |
| 5 Report | `scripts/34_hc18_oof_calibration.py` | Evaluates nested cross-validation post-hoc calibration on single-fold out-of-fold predictions |
| 5 Report | `scripts/43_hc18_calibration_candidates.py` | Evaluates nested damped multiplicative post-hoc calibration and generates exploratory submission CSVs |
| 5 Report | `scripts/46_eval_nnunet_hc18.py` | Evaluates nnU-Net predicted masks on fold 1 with submission ellipse fitting, Dice, Hausdorff, and paired bootstrap vs existing model |
| 5 Report | `scripts/47_audit_bias_direction.py` | Audits signed bias convention (PRED-TRUE vs TRUE-PRED) and dual references |
| 5 Report | `scripts/50_measurement_fix_study.py`, `scripts/54_eval_linear_fit_followup.py` | Evaluates measurement-fix transformation on GT annotation masks, validation folds, and linear fit followup |
| 5 Inference | `scripts/51_test_nnunet_test_script.py`, `scripts/52_make_hc18_nnunet_test_kernel.py`, `scripts/53_test_discovery_mock.py`, `scripts/55_make_hc18_nnunet_parallel_kernels.py`, `scripts/download_hc18_nnunet_artifacts.py`, `scripts/63_verify_nnunet_candidates.py`, `scripts/64_format_hc18_readme.py`, `scripts/66_make_hc18_nnunet_test_f012_kernel.py`, `scripts/67_download_f1f2_and_ac_fassd.py`, `scripts/68_eval_nnunet_f1_f2.py`, `scripts/69_hc18_nnunet_blend_and_pooled.py`, `scripts/70_build_blend_candidate.py`, `scripts/71_make_hc18_nnunet_test_f012_cpu_kernel.py`, `scripts/72_smoke_test_nnunet_cpu.py` | Generates Kaggle kernels for nnU-Net test inference (GPU & CPU) and parallel training, tests inference and discovery mock locally, evaluates folds and heterogeneous ellipse blend, and builds submission candidates |
| 5 Report | `scripts/47_make_nnunet_kaggle_kernel.py` | Generates Kaggle training notebook and metadata for HC18 nnU-Net v2 (2D, Fold 0, 250 epochs) |
| 5 Report | `scripts/make_hc18_calibration_notebook.py` | Generates benchmark notebook for HC18 OOF calibration study |
| 5 Report | `scripts/kaggle/run_hc18_close.py` | Runs steps 11 and 12 on a Kaggle GPU |
| 5 Report | `scripts/kaggle/run_hc18_ladder.py` | Runs the historical approach ladder on a Kaggle GPU |
| Proof | `notebooks/benchmarks/hc18_ladder.ipynb` | Executed notebook comparing historical ladder and 5-fold Row D vs E |
| Proof | `notebooks/benchmarks/hc18_official.ipynb` | Executed notebook presenting official HC18 challenge test evaluation and roundtrip Dice |
| Proof | `notebooks/benchmarks/hc18_oof_calibration.ipynb` | Executed notebook presenting calibration comparison, bias analysis, and patient bootstrap CIs |
| Proof | `tests/test_hc18_*.py`, `tests/test_hc18_ladder.py`, `tests/test_hc18_official_summary.py`, `tests/test_hc18_roundtrip.py`, `tests/test_hc18_calibration.py`, `tests/test_lit_benchmark.py`, `tests/test_hc18_nnunet.py`, `tests/reference/hc18_original.py` | Tests that the moved code equals the original notebook and tests fold-1 integrity |
| Results | `docs/results/hc18/`, `docs/results/hc18_ladder/`, `docs/results/hc18_official/`, `docs/results/hc18_calibration/`, `docs/results/hc18_calibration/candidates/`, `docs/results/hc18_nnunet/` | The real numbers, official challenge evaluation JSONs, submission files, calibration results, exploratory candidates, and summary tables |

## HEAD LANDMARK (FP+UCL): all files
| Step | File | What it does |
|---|---|---|
| 1 Prepare data | `src/fetal/data/head_dataset.py` | Reads FP and UCL head images and OFD/BPD landmark annotations, patient-grouped split |
| 2 Model | `src/fetal/models/head_landmark.py` | U-Net EfficientNet-B4 landmark model predicting 4 head caliper endpoints |
| 3 Measure | `src/fetal/eval/head_evaluator.py` | Calculates BPD, OFD, Ramanujan HC, NME metrics, and 3-seed median ensembling |
| 4 Report | `scripts/25_head_landmark.py` | Evaluates head landmark models, generates outlier overlays and benchmark tables |
| 4 Report | `scripts/kaggle/run_head_landmark.py` | Runs 3-seed head landmark training and evaluation on Kaggle GPU |
| Proof | `notebooks/benchmarks/head_landmark.ipynb` | Executed notebook with benchmark metrics and outlier visualizations (generated by `scripts/make_head_landmark_notebook.py`) |
| Proof | `tests/test_head_landmark.py` | Tests round-trip soft-argmax, hflip symmetry, true point metrics, and zero patient overlap |
| Results | `docs/results/head_landmark/` | Head landmark evaluation metrics, markdown table, outlier overlays, and Kaggle log |

## ABDOMEN (AC): all files
| Step | File | What it does |
|---|---|---|
| 1 Prepare data | `src/fetal/data/ac_dataset.py` | Reads the FP and UCL abdomen images and landmark annotations |
| 1 Prepare data | `src/fetal/data/transforms.py` | Resizing, cleaning the yellow markers, augmentation |
| 2 Model | `src/fetal/models/ac_landmark.py` | The network that predicts 4 landmark points |
| 2 Model | `src/fetal/models/losses.py` | The loss used to train it |
| 3 Train | `src/fetal/train/train_ac.py`, `train_ensemble.py`, `recipes.py`, `constants.py` | Training loops, the settings, and the 4-seed ensemble |
| 3 Train | `scripts/04_train_ac.py`, `scripts/05_train_ensemble.py` | The buttons that start training |
| 4 Measure | `src/fetal/eval/ac_geometry.py` | From landmarks to AC length in mm |
| 4 Measure | `src/fetal/eval/ac_evaluator.py`, `ensemble_evaluator.py` | Scores on the official test images |
| 5 Report | `scripts/06_eval_ac.py` | The button that reproduces 4.395 mm and 4.343 mm |
| Use | `src/fetal/infer/ac_infer.py` | One function: give an image, get the AC in mm |
| Proof | `notebooks/benchmarks/ac_results.ipynb` | Executed notebook reproducing official test set results for AC v3_full and seed ensemble |
| Proof | `tests/test_ac_*.py`, `test_geometry.py`, `test_recipes_equality.py`, `test_split_equality.py`, `test_train_equivalence.py`, `tests/reference/ac_*.py` | Tests that the moved code equals the original |
| History Proof | `scripts/28_ac_history_table.py` | Synthesizes AC lab-PC historical proof benchmark tables, Colab history evidence (v1/v2), and audits logs |
| History Proof | `notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb` | Executed notebook displaying AC lab-PC history proof tables, log excerpts, and assertions |
| History Proof | `tests/test_ac_lab_history.py` | Unit tests checking all AC lab-PC history proof assertions pass on committed inputs |
| History Results | `docs/results/ac_lab_pc/history_table.json`, `docs/results/ac_lab_pc/history_table.md` | AC lab-PC history tables and fact check audit |
| Campaign Step 0 | `scripts/38_ac_step0_audit.py` | Read-only dataset audit, SHA-256 duplicate image verification, timing parser, and campaign planning |
| Campaign Proof | `tests/test_ac_step0.py` | Unit tests for AC Step 0 audit integrity, timing logs, and duplicate hash verification |
| Campaign Results | `docs/results/ac_campaign/STEP0.md`, `docs/results/ac_campaign/SHA256SUMS` | Comprehensive AC Step 0 audit report, candidate recipes, and checksums |
| Campaign Folds | `scripts/39_make_ac_folds.py`, `docs/results/ac_campaign/folds_v1.csv` | Fixed leak-free patient-grouped 5-fold cross-validation partition on official train set (642 images, 501 patients) |
| Campaign Options | `src/fetal/train/ema.py`, `src/fetal/models/losses.py`, `src/fetal/models/ac_landmark.py` | Model EMA, wing loss coordinate term, and strict FASSD pretrained encoder loading (704 keys) |
| Campaign Compare | `scripts/40_ac_compare_recipes.py` | Patient-level cluster paired bootstrap (5,000 resamples, seed 42) comparing candidate recipes |
| Campaign Diagnostics | `scripts/48_make_worst8_contact_sheet.py`, `scripts/diagnose_worst8.py`, `scripts/analyze_worst_and_calipers.py` | Visual contact sheets and deep-dive error diagnostic reports for top worst outliers |
| Campaign Kaggle | `scripts/41_prepare_ac_kaggle_dataset.py`, `scripts/42_make_ac_kaggle_kernels.py`, `notebooks/kaggle/ac_campaign/` | Packages `prantakumerpandit390/ac-fetal-v1` dataset and generates 5 Kaggle cross-validation kernels |
| Campaign Proof | `tests/test_ac_campaign_options.py` | Unit tests for 5-fold integrity, coordinate loss variants, ModelEMA, pretrained encoder loading, and paired bootstrap |
| Campaign Results | `docs/results/ac_campaign/AC_CAMPAIGN_NOTES.md` | Full documentation of fold distribution, numerical discrepancy analysis (4.3956 vs 4.3951 mm), candidate recipe matrix, and push commands |

## FEMUR (FL)
| Step | File | What it does |
|---|---|---|
| 1 Prepare data | `src/fetal/data/femur_dataset.py` | Reads FP and UCL femur images and landmark annotations, computes FL in mm, excludes uncalibrated images, patient-grouped splits |
| 2 Model | `src/fetal/models/femur_landmark.py` | U-Net EfficientNet-B4 dual-peak heatmap model and ConvNeXt-Tiny coordinate regression model with order-invariant loss |
| 3 Measure | `src/fetal/eval/femur_evaluator.py` | Calculates FL mm, NME, bootstrap 95% CIs, and 3-seed median ensembling |
| 4 Train / Report | `scripts/22_femur_ablation.py` | Evaluates femur models (Rows A, B, C) across 3 seeds, generates comparison table and metrics |
| 4 Train / Report | `scripts/kaggle/run_femur_ablation.py` | Runs 3-seed femur model ablation training and evaluation on Kaggle GPU |
| 4 Train / Report | `scripts/27_femur_round2.py` | Validates and reports Femur Round 2 modern techniques (HRNet DARK, Integral soft-argmax, bone line, strong aug, ensemble) |
| 4 Train / Report | `scripts/kaggle/run_femur_round2.py` | Runs 3-seed Femur Round 2 training, validation selection, and paired bootstrap difference on Kaggle GPU |
| 4 Check / Audit | `scripts/31_femur_round2_checks.py` | Sensitivity of baseline, paired bootstrap differences, run integrity audit, and worst-10 squared error share |
| 4 Check / Audit | `scripts/32_femur_outlier_impact.py` | Evaluates FL MAE impact across all 348 images vs excluding outlier 005_3FL.jpeg with patient-level bootstrap CIs |
| 5 Report | `scripts/15_femur_data_report.py` | Data check, landmark validation, patient leakage check, and FL distribution |
| 5 Report | `scripts/21_femur_data_followup.py` | Outlier inspection (FL > 85 mm and 10 shortest), uncalibrated exclusions, UCL 125 audit, Hadlock check, and split verification |
| Proof | `notebooks/benchmarks/femur_data.ipynb` | Executed benchmark notebook for femur data follow-up, outlier overlays, and split integrity |
| Proof | `notebooks/benchmarks/femur_ablation.ipynb` | Executed benchmark notebook with femur ablation metrics and tables (generated by `scripts/make_femur_ablation_notebook.py`) |
| Proof | `notebooks/benchmarks/femur_round2.ipynb` | Executed benchmark notebook for Femur Round 2 modern techniques and paired bootstrap differences |
| Proof | `tests/test_femur_*.py` | Fast tests for coordinate parsing, endpoint math, heatmap round-trip, order-invariance, DARK decoding, soft-argmax, patient isolation, and baseline sensitivity checks |
| Results | `docs/results/femur/`, `docs/results/femur_ablation/`, `docs/results/femur_round2/` | Results JSON (`femur_data_followup.json`, `femur_ablation_results.json`, `femur_round2_results.json`, `baseline_sensitivity.json`), tables, predictions, and `SHA256SUMS` |

## FASSD
Fetal Abdominal Structures Segmentation Dataset (artery, liver, stomach, vein). See [ORGANS (FASSD)](#organs-fassd-all-files) below.

## MATERNAL RISK (Stream 3): all files
| File | What it does |
|---|---|
| `src/fetal/clinical/data.py` | Loads and cleans the CDC table, removes duplicates |
| `src/fetal/clinical/constants.py` | The list of allowed features and the forbidden ones (like gestational age at birth) |
| `src/fetal/clinical/model.py` | The XGBoost model |
| `src/fetal/clinical/eval.py` | AUC, confidence interval, prevalence correction, feature importance |
| `src/fetal/clinical/audit.py` | The 14 audit rows that prove where the old 0.97 came from |
| `scripts/07_audit_clinical.py` | The button that runs the audit |
| `scripts/18_bench_stream3.py` | Benchmarks XGBoost against Logistic Regression, Random Forest, and LightGBM |
| `scripts/53_make_stream3_modern_kernel.py` | Generates modern Optuna tuning kernel for CatBoost, LightGBM, and XGBoost on Kaggle CPU |
| `scripts/62_train_stream3_modern.py` | Trains and evaluates modern tuned clinical models (LightGBM, XGBoost, CatBoost, Calibrated Blend) on 2022+ cohort |
| `notebooks/S3_clinical_audit.ipynb` | The same audit as a notebook |
| `notebooks/benchmarks/stream3_bench.ipynb` | Executed notebook comparing Stream 3 tabular risk models and reproducing XGBoost |
| `tests/test_clinical_*.py`, `tests/test_stream3_*.py` | Tests, including the guard against forbidden features and row consistency |
| Results | `docs/results/stream3_bench/` | Stream 3 benchmark comparison metrics and markdown table |

## WEIGHT FORMULA (Hadlock)
`src/fetal/explain/hadlock.py` turns HC, AC and FL into an estimated weight and a growth percentile. Tests: `tests/test_hadlock.py`.
`scripts/23_three_plane_check.py` audits multi-plane subject overlap, metadata columns, and cross-plane split leakage between FP and UCL cohorts. Results: `docs/results/three_plane/three_plane_check.json`. Tests: `tests/test_three_plane.py`.
`scripts/24_hadlock_demo_step1.py` audits usable 3-plane subjects and runs HC18 5-fold ensemble external validation on FP and UCL head ultrasound images. Results: `docs/results/hadlock_demo/hadlock_demo_step1.json`, `docs/results/hadlock_demo/head_external_table.md`, `docs/results/hadlock_demo/head_predictions.csv`. Tests: `tests/test_hadlock_demo.py`.
`scripts/54_reconstruct_head_landmarks.py` reconstructs landmark coordinates and evaluates NME metrics for head ultrasound.
`scripts/26_hadlock_demo_step2.py` evaluates 4-parameter EFW estimation on the strictly held-out 3-plane test cohort (22 subjects), computes bootstrap CIs, one-at-a-time replacement, outlier audit, and sensitivity. Results: `docs/results/hadlock_demo/step2_per_subject.csv`, `docs/results/hadlock_demo/step2_summary.json`, `docs/results/hadlock_demo/step2_summary.md`. Tests: `tests/test_hadlock_step2.py`.
`notebooks/benchmarks/head_external.ipynb` displays external validation metrics and worst outlier predictions (generated by `scripts/make_head_external_notebook.py`).
`notebooks/benchmarks/hadlock_step2.ipynb` displays multi-plane EFW predictions, replacement analysis, and predicted-vs-annotated scatter plot (generated by `scripts/make_hadlock_step2_notebook.py`).

## ORGANS (FASSD): all files
| Step | File | What it does |
|---|---|---|
| 1 Prepare data | `src/fetal/data/fassd.py` | Loads FASSD array data and images, isolates allow_pickle to trusted loader, deterministic patient-grouped split, hash-based Union-Find patient grouping |
| 1 Split leakage fix & Kaggle dataset | `scripts/make_fassd_split_v2.py` | Resolves cross-patient hash duplicates, generates group split_v2.csv, encodes compact bit-flag Kaggle dataset, and verifies 100% mask round trip |
| 2 Audit & Report | `scripts/33_fassd_audit.py` | Audits patient ID widths, pairwise PNG/NPY agreement, structure statistics, duplicate hashes, color Doppler content, and generates 12 overlays |
| 2 Proof | `scripts/make_fassd_audit_notebook.py` | Generates benchmark notebook for FASSD audit |
| 3 Train / Model | `src/fetal/seg/fassd_train.py` | 4-channel sigmoid multilabel segmentation dataset, U-Net architecture, combined BCE+Dice loss, Model EMA, surface HD95, and patient bootstrap |
| 3 Training Script | `scripts/35_fassd_train.py` | CLI training runner on Kaggle T4 GPU or CPU smoke mode with cosine annealing, AMP, and JSON/CSV logging |
| 4 Kaggle Kernel | `scripts/make_fassd_kaggle_notebook.py`, `scripts/make_fassd_kaggle_notebook_round2.py` | Generates `notebooks/kaggle/fassd_train_kernel.ipynb` and Round 2 training orchestration |
| 5 Post-processing | `src/fetal/seg/fassd_postproc.py` | Connected components, area filtering, threshold tuning, and post-processing evaluation |
| 5 Post-processing Runner | `scripts/36_fassd_postproc.py` | Executes validation hyperparameter tuning, test set evaluation, and paired bootstrap CIs |
| 5 Proof | `scripts/make_fassd_postproc_notebook.py`, `scripts/make_fassd_ensemble_notebook.py` | Generates benchmark notebooks for FASSD post-processing and ensembling |
| 6 Ensembling | `src/fetal/seg/fassd_ensemble.py` | Probability map averaging, validation ensemble candidate selection, and test bootstrap evaluation |
| 6 Ensembling Runner | `scripts/37_fassd_ensemble.py` | Executes multi-model probability ensembling study, candidate selection, test evaluation, and CIs |
| Proof | `notebooks/benchmarks/fassd_audit.ipynb`, `notebooks/benchmarks/fassd_postproc.ipynb`, `notebooks/benchmarks/fassd_ensemble.ipynb` | Executed notebooks rendering FASSD audit tables, split integrity, overlays, post-processing, and ensembling |
| Proof | `notebooks/kaggle/fassd_train_kernel.ipynb` | Kaggle GPU training orchestration notebook |
| Proof | `tests/test_fassd_audit.py`, `tests/test_fassd_train.py`, `tests/test_fassd_postproc.py`, `tests/test_fassd_ensemble.py` | Unit tests for patient ID parsing, array loading, synthetic segmentation dataset, loss functions, EMA, metric edge cases, morphology, post-processing, and ensembling |
| Results | `docs/results/fassd/` | Audit JSON, markdown report, patient-grouped split_v1.csv, group-partitioned split_v2.csv, 12 sample overlays, and SHA256SUMS |

## SHARED HELPERS
| File | What it does |
|---|---|
| `src/fetal/config.py` + `configs/paths.yaml` | Where the data and weights live (one place) |
| `src/fetal/train/checkpoint.py` | Saves and loads model weights |
| `scripts/08_make_manifest.py`, `scripts/14_hash_results.py`, `scripts/16_extract_notebook_results.py`, `scripts/make_bench_notebooks.py`, `scripts/20_benchmark_summary.py`, `scripts/44_generate_best_published.py`, `scripts/check_secrets.py`, `scripts/49_update_results_ledger.py`, `scripts/56_build_report_numbers.py`, `scripts/57_verify_report_numbers.py`, `scripts/58_audit_notebooks.py`, `scripts/59_audit_nme_formulas.py`, `scripts/60_audit_dataset_counts.py`, `scripts/61_audit_hc18_ladder_refs.py`, `scripts/65_build_proof_map.py`, `scripts/73_check_pr54_premerge.py`, `scripts/74_check_master_premerge.py` | Make and check file fingerprints, extract historical metrics from notebooks, construct benchmark notebooks, generate supervisor summary, synthesize published benchmarks (BEST_PUBLISHED.md), pre-commit secret scans, update results ledger, build report numbers ledger, cryptographically verify numbers, audit notebooks, audit NME formulas, audit multi-centre cohort counts, audit HC18 ladder references, generate verified proof map, and check PR pre-merge assertions |
| `notebooks/benchmarks/benchmark_summary.ipynb` | Executed notebook synthesizing multi-modal benchmark summary |
| `docs/results/published_refs.json`, `docs/results/benchmark_summary.md`, `docs/results/literature/BEST_PUBLISHED.md`, `docs/report/REPORT_NUMBERS.json`, `docs/report/REPORT_NUMBERS.md` | Published reference numbers with URLs/protocols, synthesized executive summary table, detailed best published literature benchmarks & modern methods scan, and verified report numbers ledger |
| `tests/test_benchmark_summary.py`, `tests/test_lit_benchmark.py`, `tests/test_report_numbers.py` | Unit tests ensuring AC Ours/Claim consistency, published references, dynamic ladder notes, literature benchmark generation, calibration candidate integrity, and report numbers verification |
| `scripts/run_all.ps1` | One command to run groups of the buttons above |

## How the whole system fits (the flow)
Head image -> HC18 model -> HC and BPD.
Abdomen image -> AC model -> AC.
Femur image -> femur model -> FL (coming).
HC + AC + FL -> Hadlock formula -> estimated weight and growth percentile.
Mother's table data -> clinical model -> stillbirth risk.
Both streams -> fusion model (coming) -> final prediction, with explanations (SHAP, organs).

## How to read any part
Open files in step order (1 data, 2 model, 3 train, 4 measure, 5 report). For each file, open the test with the same name in `tests/`. The test shows, in a few lines, how the file is used.
