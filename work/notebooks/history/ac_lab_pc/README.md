# AC Lab-PC History Proof

## Overview

This directory preserves the historical proof artifacts and logs recovered from the lab-PC training runs for the Fetal Abdominal Circumference (AC) task.

These files serve as **historic-proof evidence** demonstrating the developmental evolution and verification of models trained on the lab PC.

> [!WARNING]
> **Ablation / Historical Proof Only**: The test set was evaluated multiple times during lab-PC experimentation (flip-TTA ablations across various architectures and seeds). Consequently, the official test numbers reported here represent historical proof points and ablation studies rather than untouched, hold-out benchmark evaluations. Furthermore, the DSNT 5-fold cross-validation results utilized a pooled 789-image dataset (including official-test images out-of-fold) under a fundamentally different protocol, and are therefore not directly comparable with official-test holdout benchmarks.

---

## Directory Contents

### 1. `inputs/scripts_original/`
Original Python scripts executed on the lab PC:
- `ac_dsnt.py`: DSNT (Differentiable Spatial Numerical Transform) 5-fold cross-validation pipeline.
- `ac_v3.py`: Base AC landmark detection training script introducing diameter-based loss and inpainting.
- `ac_v3_ens.py`: Multi-seed ensemble script orchestrating runs for seeds 1–4, 512px resolution evaluation, and test prediction ensembling.

### 2. `inputs/` (Execution Logs, UTF-16 LE)
- `full_log.txt`: Complete training log for the DSNT 5-fold cross-validation experiment.
- `v3_log.txt`: Training and official test evaluation log for `base` and `v3_full` models.
- `ens_log.txt`: Training log for seeds 1–4, 512px resolution run, and test ensembling.
- `smoke_log.txt`: Execution log for a 160-image smoke test run (code verification only).

> [!IMPORTANT]
> **UTF-16 LE Encoding**: All `.txt` log files were written on Windows using PowerShell redirection / `Tee-Object` and are encoded strictly as **UTF-16 LE with BOM (CRLF line endings)**. When reading these files programmatically in Python, always open them with `encoding="utf-16"`.

### 3. `inputs/` (Predictions & Result Summaries)
- `dsnt_full_oof_results.csv`: Out-of-fold predictions on 789 pooled images from 5-fold cross-validation.
- `v3_base_results.json` & `v3_base_test_predictions.csv`: Evaluation metrics and test predictions for `v3 base`.
- `v3_full_results.json` & `v3_full_test_predictions.csv`: Evaluation metrics and test predictions for `v3_full`.
- `ens_results.json` & `ens_test_predictions.csv`: Evaluation metrics across seeds 1–4, res512, ensemble mean, and ensemble median, along with test predictions.

---

## Historical Context & Missing Logs

During active development on the lab PC:
- Earlier iterative logs for **v1**, **v2**, and preliminary direct-regression experiments were not retained or were overwritten during disk cleanups.
- Only the **v3** series and the **DSNT 5-fold cross-validation** logs and checkpoints survived on the lab PC filesystem.
- In `ens_log.txt`, the initial invocation specified `seeds=[1, 2, 3, 4, 5]`. Due to a process interrupt and corrupt checkpoint recovery, the script was restarted with `seeds=[1, 2, 3, 4]`. Consequently, **Seed 5 was planned but never trained** (confirmed by zero `[s5]` log lines).

---

## Verification & Reproducibility

- **Table Generation Script**: [`scripts/28_ac_history_table.py`](file:///C:/fetal-github-clone/work/scripts/28_ac_history_table.py) parses the inputs, checks assertions against logs and results JSONs, and generates:
  - [`docs/results/ac_lab_pc/history_table.json`](file:///C:/fetal-github-clone/work/docs/results/ac_lab_pc/history_table.json)
  - [`docs/results/ac_lab_pc/history_table.md`](file:///C:/fetal-github-clone/work/docs/results/ac_lab_pc/history_table.md)
- **Automated Tests**: [`tests/test_ac_lab_history.py`](file:///C:/fetal-github-clone/work/tests/test_ac_lab_history.py) asserts all equality checks and fact audits.
- **Interactive Notebook**: [`notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb`](file:///C:/fetal-github-clone/work/notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb) provides visual inspection of the tables and logs.
