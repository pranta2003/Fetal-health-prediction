# Project Progress Log

## Activity Log

- **Phase 1 & 2a-2c**: Modular refactor of Stream 2 (AC Biometry), training recipes, evaluation suite, Hadlock EFW layer, and baseline reproduction on official test set (4.395 mm single, 4.343 mm ensemble).
- **Phase 3a-3b**: Stream 3 (CDC clinical tabular risk) extraction with leakage audit, temporal split validation, prior-shift population calibration, and reproduction of all 14 audit benchmarks.
- **Phase 4a-4b**: Architecture documentation corrections, Hadlock EFW test vector alignment, manifest checksum verification, and constant updates (NCHS Data Brief 530).
- **Phase 5a**: HC18 preparation: patient distribution audit, leak-free GroupKFold patient-isolation splitting (`make_grouped_folds`), and validation script `09_hc18_leakage_report.py`.
- **Phase 6a**: Stream 1 (HC18 Head Circumference Biometry) modular extraction into `src/fetal/` (`data/hc18_dataset.py`, `models/hc18_unet.py`, `eval/hc18_eval.py`, `train/train_hc18.py`), equivalence verification vs reference notebook (`test_hc18_equivalence.py`), 2-epoch CPU smoke verification (`test_hc18_train_smoke.py`), and CLI runner `scripts/10_train_hc18.py`. Status: extracted; smoke-tested; full run pending.
