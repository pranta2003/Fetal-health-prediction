# Experimental Results Ledger

| Date | Run ID / Recipe | Split / Protocol | N (Pts) | AC MAE (mm) | AC RMSE (mm) | AC Median (mm) | AC Bias (mm) | TAD MAE (mm) | APAD MAE (mm) | NME | >10mm (%) | Artifact SHA-256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-19 | `ac-baseline-s42` (v3_full baseline, seed 42) | 5-fold CV (`folds_v1.csv`) | 642 (501) | 4.7744 | 7.6074 | 2.8474 | -0.0826 | 1.7331 | 2.0793 | 0.0808 | 12.15% | `8ca2aed7810fcb0b4a84ca5ba57e53f101d50f1349ac6484a351a070ec779f52` |

### Detailed Notes & Checkpoint Provenance
- **`ac-baseline-s42`**:
  - Kaggle Kernel: `prantakumerpandit390/ac-baseline-s42`
  - Output Archive: `E:\BUBT-CSE-4th yr\Capstone\kaggle_outputs\ac_baseline_s42`
  - Summary JSON: `docs/results/ac_campaign/ac_baseline_s42.json` (`8ca2aed7810fcb0b4a84ca5ba57e53f101d50f1349ac6484a351a070ec779f52`)
  - Protocol: Patient-grouped 5-fold cross-validation on 642 images across 501 patients, evaluated strictly out-of-fold.
