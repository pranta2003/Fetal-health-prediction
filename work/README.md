# Multimodal Fetal Health Assessment Pipeline

This repository contains the modularized codebase for multimodal fetal health assessment, combining clinical tabular risk modeling (Stream 3), fetal head circumference biometry (Stream 1), abdominal circumference biometry (Stream 2), and clinical fetal weight estimation via Hadlock's formula.

## Verification Status & Stream Status
- **Stream 2 (AC Landmark & Biometry):** **VERIFIED** on official test set (`v3_full` AC MAE 4.395 mm, 4-seed median ensemble 4.343 mm) via automated regression tests (`tests/test_ac_v3_regression.py` and `tests/test_ac_ensemble_regression.py`). Baseline reproduction script confirms exact reproduction within 0.002 mm.
- **Layer 3 (Hadlock EFW & Growth):** **VERIFIED** against canonical notebook test vectors (1403.0 g).
- **Stream 1 (HC18 Biometry Segmentation):** **NOT yet extracted / NOT verified**. Remains in exploration notebook; results in preliminary experiments were provisional (ungrouped KFold).
- **Stream 3 (Clinical Tabular Risk XGBoost):** **NOT yet extracted / NOT verified**. Remains in exploration notebook.

## Repository Layout
- `configs/`: YAML configuration files (paths, model hyperparameters).
- `src/fetal/`: Modular Python package containing data loaders, model architectures, geometry utilities, evaluators, and clinical calculations.
- `scripts/`: Numbered CLI runner scripts for pipeline execution (`04_train_ac.py`, `05_train_ensemble.py`, `06_eval_ac.py`, and `run_all.ps1`).
- `tests/`: Automated unit, equality, equivalence, and regression test suites.
- `reports/`: Metric summaries and evaluation artifacts.

## Environment Setup
Model weights and dataset files live strictly **outside** this repository:
- Weights directory: `C:\fetal-models` (or configurable via `FETAL_MODEL_DIR`).
- Data directory: `C:\fetal-data` (or configurable via `FETAL_DATA_DIR`).

Install requirements into a virtual environment (Python 3.11 - 3.13):
```bash
pip install -r requirements.txt
pip install -e .
```

## How to Run

### Automated Test Runner (`run_all.ps1`)
From PowerShell inside `work/`:
```powershell
# Run fast test suite (default):
.\scripts\run_all.ps1

# Run fast suite plus slow regression tests (evaluates real checkpoints):
.\scripts\run_all.ps1 -Slow

# Run fast suite and evaluate existing checkpoints on official test set:
.\scripts\run_all.ps1 -Eval

# Run fast suite and execute full training pipeline:
.\scripts\run_all.ps1 -Train
```

### Manual Commands

#### 1. Running Unit and Equality Tests
```bash
pytest -v
```

#### 2. Running Full Regression Tests
Requires checkpoints in `FETAL_MODEL_DIR` and data in `FETAL_DATA_DIR`:
```bash
pytest -m slow -v -s
```

#### 3. Evaluating Checkpoints
```bash
python scripts/06_eval_ac.py
```

#### 4. Training AC Models
```bash
# Smoke test run (2 epochs on tiny subset):
python scripts/04_train_ac.py --smoke
python scripts/05_train_ensemble.py --smoke --skip-512

# Full training run:
python scripts/04_train_ac.py
python scripts/05_train_ensemble.py
```
