# Project Progress Log

## Activity Log

- **Phase 1 & 2a-2c**: Modular refactor of Stream 2 (AC Biometry), training recipes, evaluation suite, Hadlock EFW layer, and baseline reproduction on official test set (4.395 mm single, 4.343 mm ensemble).
- **Phase 3a-3b**: Stream 3 (CDC clinical tabular risk) extraction with leakage audit, temporal split validation, prior-shift population calibration, and reproduction of all 14 audit benchmarks.
- **Phase 4a-4b**: Architecture documentation corrections, Hadlock EFW test vector alignment, manifest checksum verification, and constant updates (NCHS Data Brief 530).
- **Phase 5a**: HC18 preparation: patient distribution audit, leak-free GroupKFold patient-isolation splitting (`make_grouped_folds`), and validation script `09_hc18_leakage_report.py`.
- **Phase 6a**: Stream 1 (HC18 Head Circumference Biometry) modular extraction into `src/fetal/` (`data/hc18_dataset.py`, `models/hc18_unet.py`, `eval/hc18_eval.py`, `train/train_hc18.py`), equivalence verification vs reference notebook (`test_hc18_equivalence.py`), 2-epoch CPU smoke verification (`test_hc18_train_smoke.py`), and CLI runner `scripts/10_train_hc18.py`. Status: extracted; smoke-tested; full run pending.
- **Phase 6b**: HC18 training constants verification (`test_hc18_constants.py`) asserting all 11 hyperparameters match reference notebook; added patient-grouped `inner_holdout` option (`split_patient_inner_holdout`) to isolate early stopping/checkpoint selection from outer validation scoring with zero patient leakage; added `--inner-holdout` CLI flag to `10_train_hc18.py`.
- **Phase 6c**: HC18 leak-free out-of-fold (OOF) evaluation report and official test inference pipeline. Verified pooled OOF metrics target across 5 folds: HC MAE 2.30 mm (RMSE 3.58 mm), BPD MAE 0.75 mm (RMSE 1.09 mm) within 0.02 mm tolerance of metrics.json. Added error analysis across true-HC terciles ('head-size proxy for gestational age') and patient-level bootstrap 95% CI. Created Grand Challenge test inference pipeline (`scripts/12_hc18_test_inference.py`) outputting single-model (Fold 1) and 5-model ensemble ellipse parameters in official submission CSV format (center x/y, semi-axes in mm, angle in radians). Added fast unit tests for biometrics, submission format, tercile bins, and bootstrap CI.
- **Phase 10 (Kaggle GPU Execution & Artifact Provenance)**: Executed real full-dataset OOF evaluation and official test inference on Kaggle GPU via `scripts/kaggle/run_hc18_close.py`.
  - Pooled OOF biometrics on 999 images: HC MAE **2.3019 mm** (RMSE **3.5797 mm**), BPD MAE **0.7535 mm** (RMSE **1.0938 mm**), within tolerance of Run B baseline (HC MAE diff 0.0019, RMSE diff 0.0003, BPD MAE diff 0.0035, RMSE diff 0.0038).
  - Patient-level bootstrap 95% CI for pooled HC MAE: [**2.1327**, **2.4793**] mm (1,000 resamples).
  - Gestational age proxy tercile analysis: Small (< T1, 45.1–167.2 mm): HC MAE 1.92 mm, Medium (T1–T2, 167.5–183.0 mm): HC MAE 2.02 mm, Large (> T2, 183.0–349.3 mm): HC MAE 2.97 mm.
  - Test set inference: generated valid submission CSVs for 335 scans (single fold 1: `submission_single_fold1.csv`, 5-fold ensemble: `submission_ensemble_5fold.csv`).
  - Results stored verbatim in `docs/results/hc18/` with full checksum verification via `scripts/14_hash_results.py` (`SHA256SUMS`).


