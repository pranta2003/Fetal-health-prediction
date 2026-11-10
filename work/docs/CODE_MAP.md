# CODE MAP: which file belongs to which part

Everything current lives inside the folder `work/`. The old folders at the top level (`src/`, `models/`) are the older version; ignore them.

## The one rule
File names tell you the part. The prefix is the part:

| Prefix / name | Part | Dataset |
|---|---|---|
| `hc18_...` | **Head** (head circumference HC, BPD) | HC18 |
| `ac_...` | **Abdomen** (abdominal circumference AC) | FP + UCL abdomen images |
| `femur_...` | **Femur** (femur length FL) | femur images inside FP + UCL |
| `fassd_...` (coming) | **Organs** (explanations) | FASSD |
| inside `clinical/` | **Maternal risk** (stillbirth risk, tables) | CDC natality + fetal death files |
| `hadlock.py` | **Weight formula** (EFW from HC, AC, FL) | no dataset, a formula |
| no prefix (`config.py`, `transforms.py`, `checkpoint.py` ...) | **Shared helpers** | none |

## HEAD (HC18): all files
| Step | File | What it does (plain words) |
|---|---|---|
| 1 Prepare data | `src/fetal/data/hc18_dataset.py` | Loads head images and masks, resizes without distortion, flips for training |
| 1 Prepare data | `src/fetal/data/hc18_split.py` | Splits patients into 5 folds so no patient is in both train and test |
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
| 5 Report | `scripts/kaggle/run_hc18_close.py` | Runs steps 11 and 12 on a Kaggle GPU |
| 5 Report | `scripts/kaggle/run_hc18_ladder.py` | Runs the historical approach ladder on a Kaggle GPU |
| Proof | `notebooks/benchmarks/hc18_ladder.ipynb` | Executed notebook comparing historical ladder and 5-fold Row D vs E |
| Proof | `tests/test_hc18_*.py`, `tests/test_hc18_ladder.py`, `tests/reference/hc18_original.py` | Tests that the moved code equals the original notebook and tests fold-1 integrity |
| Results | `docs/results/hc18/`, `docs/results/hc18_ladder/` | The real numbers and the submission files |

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

## FEMUR (FL)
| Step | File | What it does |
|---|---|---|
| 1 Prepare data | `src/fetal/data/femur_dataset.py` | Reads FP and UCL femur images and landmark annotations, computes FL in mm, excludes uncalibrated images, patient-grouped splits |
| 5 Report | `scripts/15_femur_data_report.py` | Data check, landmark validation, patient leakage check, and FL distribution |
| 5 Report | `scripts/21_femur_data_followup.py` | Outlier inspection (FL > 85 mm and 10 shortest), uncalibrated exclusions, UCL 125 audit, Hadlock check, and split verification |
| Proof | `notebooks/benchmarks/femur_data.ipynb` | Executed benchmark notebook for femur data follow-up, outlier overlays, and split integrity |
| Proof | `tests/test_femur_*.py` | Fast tests for coordinate parsing, endpoint math, uncalibrated exclusion, and patient isolation |
| Results | `docs/results/femur/` | Results JSON (`femur_data_followup.json`), outlier overlay images, and `SHA256SUMS` |

## FASSD
Not built yet. When it arrives, it follows the same pattern: `fassd_dataset.py`, `fassd_model.py`, `train_fassd.py`, `fassd_eval.py`.

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
| `notebooks/S3_clinical_audit.ipynb` | The same audit as a notebook |
| `notebooks/benchmarks/stream3_bench.ipynb` | Executed notebook comparing Stream 3 tabular risk models and reproducing XGBoost |
| `tests/test_clinical_*.py`, `tests/test_stream3_*.py` | Tests, including the guard against forbidden features and row consistency |
| Results | `docs/results/stream3_bench/` | Stream 3 benchmark comparison metrics and markdown table |

## WEIGHT FORMULA (Hadlock)
`src/fetal/explain/hadlock.py` turns HC, AC and FL into an estimated weight and a growth percentile. Tests: `tests/test_hadlock.py`.
`scripts/23_three_plane_check.py` audits multi-plane subject overlap, metadata columns, and cross-plane split leakage between FP and UCL cohorts. Results: `docs/results/three_plane/three_plane_check.json`. Tests: `tests/test_three_plane.py`.

## SHARED HELPERS
| File | What it does |
|---|---|
| `src/fetal/config.py` + `configs/paths.yaml` | Where the data and weights live (one place) |
| `src/fetal/train/checkpoint.py` | Saves and loads model weights |
| `scripts/08_make_manifest.py`, `scripts/14_hash_results.py`, `scripts/16_extract_notebook_results.py`, `scripts/make_bench_notebooks.py`, `scripts/20_benchmark_summary.py` | Make and check file fingerprints, extract historical metrics from notebooks, construct benchmark notebooks, and generate supervisor summary |
| `notebooks/benchmarks/benchmark_summary.ipynb` | Executed notebook synthesizing multi-modal benchmark summary |
| `docs/results/published_refs.json`, `docs/results/benchmark_summary.md` | Published reference numbers with URLs/protocols, and synthesized executive summary table |
| `tests/test_benchmark_summary.py` | Unit tests ensuring AC Ours/Claim consistency, published references, and dynamic ladder notes |
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
