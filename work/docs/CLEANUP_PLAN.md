# Repository Cleanup Plan

**Branch:** `repo-cleanup` (branched from `part3-experiments`)  
**Date:** 2026-11-22  
**Governing Principle:** Strictly adhere to the deletion whitelist/blacklist. Never delete items in proof map, results, notebooks, scripts, src, tests, configs, requirements, or `Capstone_fetalDataset.ipynb`—move to `archive/` if possibly useful; delete only provably unreferenced junk (specifically accidentally committed venv binaries). Consolidate repository documentation into a single root `README.md`.

---

## 1. Inventory & Disposition Table

| Path | Size (bytes) | Description | Grep References in `work/` | Decision | Reason |
|---|---|---|---|---|---|
| `.github/workflows/check.yml` | 375 | GitHub Actions CI check | None | **KEEP** | Standard CI configuration for repository. |
| `.gitignore` | 112 | Root git ignore rules | None | **KEEP** | Root gitignore, updated to cover virtualenvs, build artifacts, and caches. |
| `README.md` (root) | 2,643 | Old root project overview | Mentioned in `REFACTOR_PLAN.md` | **KEEP & REWRITE** | Primary repository README to be updated with canonical title, unified architecture, honest status table, verified metrics, folder map, and reproduction instructions. |
| `requirements.txt` (root) | 396 | Root requirements file | None | **KEEP & SYNC** | Root pip requirements file, synchronized with `work/requirements.txt`. |
| `Capstone_fetalDataset.ipynb` | 3,903,118 | Original Colab exploratory notebook | Historical mentions in `REFACTOR_PLAN.md` and reference test headers | **MOVE** -> `archive/legacy_notebooks/Capstone_fetalDataset.ipynb` | Preserved for historical provenance; active modular code lives in `work/src/fetal/` and clean history notebook in `work/notebooks/history/`. |
| `models/stream3_clinical_xgb.json` | 595,677 | Original uncalibrated XGBoost clinical model | Mentioned in `REFACTOR_PLAN.md` and `configs/paths.yaml` (optional fallback) | **MOVE** -> `archive/legacy_models/stream3_clinical_xgb.json` | Replaced by calibrated `stream3-modern-s1` pipeline. Preserved in archive. |
| `src/stream3_clinical/train_clinical_model.py` | 5,138 | Early standalone script for clinical model | None | **MOVE** -> `archive/legacy_src/stream3_clinical/train_clinical_model.py` | Superseded by modular package `work/src/fetal/clinical/`. Preserved in archive. |
| `src/stream3_clinical/support.txt` | 805 | Notes on clinical model feature columns | None | **MOVE** -> `archive/legacy_src/stream3_clinical/support.txt` | Preserved in archive. |
| `models/abdomen_dataset/model_B4u/ac_dsnt.py` | 18,729 | Early exploratory AC DSNT model script | None | **MOVE** -> `archive/legacy_models/abdomen_dataset/ac_dsnt.py` | Preserved in archive. Modular code lives in `work/src/fetal/models/ac_dsnt.py`. |
| `models/abdomen_dataset/model_B4u/ac_v3.py` | 19,281 | Early exploratory AC v3 model script | None | **MOVE** -> `archive/legacy_models/abdomen_dataset/ac_v3.py` | Preserved in archive. Modular code lives in `work/src/fetal/models/ac_v3.py`. |
| `models/abdomen_dataset/model_B4u/ac_v3_ens.py` | 8,689 | Early exploratory AC v3 ensemble script | None | **MOVE** -> `archive/legacy_models/abdomen_dataset/ac_v3_ens.py` | Preserved in archive. Modular code lives in `work/src/fetal/models/ac_v3_ensemble.py`. |
| `models/abdomen_dataset/model_B4u/ens_log.txt` | 31,270 | Early training log for AC ensemble | None | **MOVE** -> `archive/legacy_models/abdomen_dataset/ens_log.txt` | Preserved in archive. |
| `models/abdomen_dataset/model_B4u/full_log.txt` | 62,530 | Early training log for AC full | None | **MOVE** -> `archive/legacy_models/abdomen_dataset/full_log.txt` | Preserved in archive. |
| `models/abdomen_dataset/model_B4u/smoke_log.txt` | 6,990 | Early smoke test log | None | **MOVE** -> `archive/legacy_models/abdomen_dataset/smoke_log.txt` | Preserved in archive. |
| `models/abdomen_dataset/model_B4u/v3_log.txt` | 17,780 | Early training log for AC v3 | None | **MOVE** -> `archive/legacy_models/abdomen_dataset/v3_log.txt` | Preserved in archive. |
| `models/abdomen_dataset/model_B4u/ac_dsnt_full/ac_dsnt.py` | 18,729 | Duplicate script copy | None | **MOVE** -> `archive/legacy_models/abdomen_dataset/ac_dsnt_full/ac_dsnt.py` | Preserved in archive. |
| `models/abdomen_dataset/model_B4u/ac_dsnt_full/full_log.txt` | 62,530 | Duplicate log copy | None | **MOVE** -> `archive/legacy_models/abdomen_dataset/ac_dsnt_full/full_log.txt` | Preserved in archive. |
| `models/abdomen_dataset/model_B4u/ac_dsnt_full/n` | 3 | Marker file | None | **MOVE** -> `archive/legacy_models/abdomen_dataset/ac_dsnt_full/n` | Preserved in archive. |
| `models/abdomen_dataset/model_B4u/ac_dsnt_full/smoke_log.txt` | 6,990 | Duplicate log copy | None | **MOVE** -> `archive/legacy_models/abdomen_dataset/ac_dsnt_full/smoke_log.txt` | Preserved in archive. |
| `models/abdomen_dataset/model_B4u/ac_dsnt_smoke/v` | 2 | Marker file | None | **MOVE** -> `archive/legacy_models/abdomen_dataset/ac_dsnt_smoke/v` | Preserved in archive. |
| `models/abdomen_dataset/model_B4u/ac_v3_ens/s1/c` | 2 | Marker file | None | **MOVE** -> `archive/legacy_models/abdomen_dataset/ac_v3_ens/s1/c` | Preserved in archive. |
| `models/abdomen_dataset/model_B4u/ac_v3/base/a` | 2 | Marker file | None | **MOVE** -> `archive/legacy_models/abdomen_dataset/ac_v3/base/a` | Preserved in archive. |
| `models/abdomen_dataset/model_B4u/ac_v3/base/results.json` | 1,764 | Early baseline evaluation JSON | None | **MOVE** -> `archive/legacy_models/abdomen_dataset/ac_v3/base/results.json` | Preserved in archive. |
| `models/abdomen_dataset/model_B4u/ac_v3/base/test_predictions.csv` | 26,587 | Early baseline test predictions | None | **MOVE** -> `archive/legacy_models/abdomen_dataset/ac_v3/base/test_predictions.csv` | Preserved in archive. |
| `models/abdomen_dataset/model_B4u/ac_v3/v3_full/results.json` | 1,764 | Early v3_full evaluation JSON | None | **MOVE** -> `archive/legacy_models/abdomen_dataset/ac_v3/v3_full/results.json` | Preserved in archive. |
| `models/abdomen_dataset/model_B4u/ac_v3/v3_full/test_predictions.csv` | 26,587 | Early v3_full test predictions | None | **MOVE** -> `archive/legacy_models/abdomen_dataset/ac_v3/v3_full/test_predictions.csv` | Preserved in archive. |
| `models/abdomen_dataset/model_B4u/ac_v3/v3_full/v` | 2 | Marker file | None | **MOVE** -> `archive/legacy_models/abdomen_dataset/ac_v3/v3_full/v` | Preserved in archive. |
| `models/abdomen_dataset/model_B4u/script/*` (19 files: `pip.exe`, `python.exe`, `f2py.exe`, `hf.exe`, `httpx2.exe`, `idna.exe`, `isympy.exe`, `numpy-config.exe`, `pip3.12.exe`, `pip3.exe`, `pythonw.exe`, `tiny-agents.exe`, `torchfrtrace.exe`, `torchrun.exe`, `tqdm.exe`, `activate`, `activate.bat`, `Activate.ps1`, `deactivate.bat`, `n`) | ~1,850,000 | Venv executable binaries and activation scripts accidentally committed in initial repo creation | None | **DELETE** | Provably unreferenced junk. Binary executables from a local virtual environment do not belong in source control. |
| `work/README.md` | 3,558 | Previous subfolder README | None | **DELETE** (after merge) | Merged into root `README.md` to ensure a single authoritative README at the repository root. |
| `work/docs/results/*/*.log` (7 log files: `06_eval_run.log`, `femur_ablation.log`, `femur_round2.log`, `hc18-close.log`, `hc18-ladder.log`, `head_landmark.log`, `stream3-modern-s1.log`) | Various | Execution and audit log files | Tracked in `SHA256SUMS` and `PROOF_LOCATIONS.md` | **KEEP** | Whitelist: essential proof and audit evidence. |

---

## 2. Whitelist / Blacklist Policy
- **Blacklist (NEVER delete):**
  - Files recorded in `docs/report/PROOF_LOCATIONS.md`
  - Files recorded in `docs/results/RESULTS_LEDGER.md`
  - Files recorded in any `SHA256SUMS` or `SHA256SUMS.txt`
  - Code under `work/src/`, `work/scripts/`, `work/tests/`, `work/configs/`, `work/notebooks/`
  - `Capstone_fetalDataset.ipynb` (must be MOVED to `archive/`, never deleted)
- **Whitelist for deletion:**
  - Provably unreferenced junk:
    1. Venv binaries inside `models/abdomen_dataset/model_B4u/script/` (`*.exe`, activation scripts).
    2. Redundant duplicate `work/README.md` after its entire contents are verified merged into root `README.md`.

---

## 3. Verification Steps
After executing this cleanup plan:
1. `pytest tests/` inside `work/` must pass all 196 tests.
2. `python scripts/56_build_report_numbers.py` must build all 158 verified report numbers without errors.
3. `python scripts/57_verify_report_numbers.py` must verify all 158 claims with 0 errors.
4. `python scripts/65_build_proof_map.py` must report 0 missing claims.
