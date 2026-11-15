# AC Lab-PC History Proof Table

Historical proof benchmarks extracted strictly from the committed lab-PC artifacts (`notebooks/history/ac_lab_pc/inputs/`).

## Performance Table

| Model / Run | Res | N | AC MAE (mm) | RMSE (mm) | Median (mm) | Bias (mm) | AC % | >10mm (%) | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **v1 heatmap regression (Colab)** | 384 | 789 | 182.60 | 195.13 | - | - | - | - | v1 heatmap regression, Colab, 5-fold out-of-fold (789 images): FAILED/COLLAPSED run (AC MAE 182.60 mm vs mean-AC baseline), not comparable with official-test rows |
| **DSNT 5-fold OOF** | 384 | 789 | 5.18 | 9.07 | - | - | 2.60% | - | history row, DIFFERENT protocol (789 images incl. the official-test images, out-of-fold), not comparable with the official-test rows |
| **v3 base** | 384 | 147 | 5.81 | 8.87 | 3.65 | -3.26 | 2.80% | 16.3% | ablation-study rows on the official test (test set was looked at many times) |
| **v3_full** | 384 | 147 | 4.40 | 6.16 | 2.91 | -1.55 | 2.14% | 8.2% | ablation-study rows on the official test (test set was looked at many times) |
| **v3_ens (s1)** | 384 | 147 | 4.96 | 7.33 | 3.14 | +0.96 | 2.39% | 12.2% | ablation-study rows on the official test (test set was looked at many times) |
| **v3_ens (s2)** | 384 | 147 | 4.32 | 6.05 | 2.73 | -0.99 | 2.17% | 8.8% | ablation-study rows on the official test (test set was looked at many times) |
| **v3_ens (s3)** | 384 | 147 | 4.54 | 6.59 | 3.37 | -0.74 | 2.20% | 10.2% | ablation-study rows on the official test (test set was looked at many times) |
| **v3_ens (s4)** | 384 | 147 | 4.49 | 6.27 | 3.64 | -0.49 | 2.17% | 8.2% | ablation-study rows on the official test (test set was looked at many times) |
| **v3_ens (res512)** | 512 | 147 | 5.02 | 8.01 | 3.32 | -1.53 | 2.48% | 11.6% | ablation-study rows on the official test (test set was looked at many times) |
| **v3_ens (4-seed mean)** | 384 | 147 | 4.47 | 6.31 | 3.01 | -0.53 | 2.17% | 9.5% | ablation-study rows on the official test (test set was looked at many times) |
| **v3_ens (4-seed median)** | 384 | 147 | 4.34 | 6.03 | 3.12 | -0.60 | 2.14% | 6.8% | ablation-study rows on the official test (test set was looked at many times) |
| **smoke_log (code check)** | - | 32 | 111.65 | 117.88 | - | - | 57.72% | - | code check only, not a result |
| **v2 DSNT smoke (Colab CPU)** | 384 | 160 | - | - | - | - | - | - | DSNT smoke on Colab CPU, 1 fold, 5 epochs: code check only; the real v2 run is the lab-PC full_log row |

> **Single 384-px seeds summary**: AC MAE 4.58 ± 0.27 mm (min 4.32, max 4.96) across seeds [1, 2, 3, 4].

## Log Fact-Check Audit

Automated scan of the 4 committed UTF-16 LE execution logs:

| Log File | Lines | Invocations (`device=`) | Resumed Lines | Tracebacks | Seeds in Header | Status / Notes |
|---|:---:|:---:|:---:|:---:|---|---|
| `v3_log.txt` | 82 | 1 | 0 | 0 | - | 1 clean invocation: base + v3_full ablation. |
| `ens_log.txt` | 297 | 4 | 0 | 1 | [[1, 2, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]] | Seeds [1,2,3,4,5] requested in 1st invocation; restart switched to [1,2,3,4]; s5 was never trained. |
| `smoke_log.txt` | 49 | 1 | 0 | 0 | - | code check only, not a result |
| `full_log.txt` | 301 | 5 | 1 | 5 | - | 5 invocations (`Using device:`), 1 fold resumed. 5 crashes (RuntimeError: ios_base::badbit set: iostream stream error; RuntimeError: [enforce fail at inline_container.cc:672] . unexpected pos 92174400 vs 92174288; ModuleNotFoundError: No module named 'numpy'; FileNotFoundError: [Errno 2] No such file or directory: 'C:/Users/BITC MEDIA/Documents/nor/fetal-project/data/raw/FETAL_BIOMETRY_ABDOMEN/FP/annotations/Abdomen.csv'; FileNotFoundError: [Errno 2] No such file or directory: 'PASTE_THE_PATH_HERE/FP/annotations/Abdomen.csv'), run resumed; the script refuses to resume with a different CONFIG. |

### Key Verification Assertions
1. **Seed 5 Status**: `Seed 5 was planned but never trained (no [s5] lines)`. Confirmed: exactly 0 `[s5]` lines across all logs.
2. **DSNT OOF Equality**: Computed metrics on `dsnt_full_oof_results.csv` ($N=789$, MAE=5.18 mm, RMSE=9.07 mm, % err=2.60%, FP MAE=5.08 mm, UCL MAE=5.91 mm) match `full_log.txt` final block exact to printed precision.
3. **v3 Official Test Equality**: Values in `v3_base_results.json` and `v3_full_results.json` match `v3_log.txt` test table exact to printed precision.
4. **Ensemble Consistency**: Values in `ens_results.json` match `ens_log.txt` table exact to printed precision. Recomputed MAE from `ens_test_predictions.csv` matches `ENS_mean` (4.4693 mm) exactly.

## Colab Historical Development Evidence (v1 & v2)

Preserved in committed Colab notebook (`notebooks/history/Capstone_fetalDataset_history.ipynb`):
- **v1 Heatmap Regression**: Pooled dataset: 789 images from 615 unique patients (586 FP + 29 UCL). Folds 2–3 completed 100 epochs; Fold 4 early-stopped at epoch 97 (val_loss 0.000344); Fold 5 early-stopped at epoch 89 (val_loss 0.000351). Out-of-fold metrics: TAD MAE 57.98 mm, APAD MAE 57.46 mm, AC MAE 182.60 mm (collapsed run).
- **v2 DSNT Smoke**: DSNT smoke on Colab CPU, 1 fold, 5 epochs: code check only; the real v2 run is the lab-PC full_log row.

