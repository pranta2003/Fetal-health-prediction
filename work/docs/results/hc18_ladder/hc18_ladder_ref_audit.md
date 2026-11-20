# HC18 Ladder Ground-Truth Reference Audit

Evaluation of each ladder row against BOTH the stored file reference and official CSV ground truth.

| Row ID | Approach / Model | Split | N | Reference in Original Run | Orig Ref MAE (mm) | CSV Ref MAE (mm) | Orig Signed Bias (mm) | CSV Signed Bias (mm) | Status |
|---|---|---|:---:|---|:---:|:---:|:---:|:---:|---|
| `row_A` | Row A: Direct Regression | Fold 1 (N=206) | 206 | **Official CSV HC** | 10.67 | **10.67** | +5.31 | **+5.31** | Recomputed |
| `row_B` | Row B: Geometry Approach v3 | Fold 1 (N=206) | 206 | **Official CSV HC** | 11.60 | **11.60** | +5.61 | **+5.61** | Recomputed |
| `row_C` | Row C: Multi-Task U-Net | Fold 1 (N=206) | 206 | **Official CSV HC** | 10.51 | **10.51** | +3.53 | **+3.53** | Recomputed |
| `row_D_fold1` | Row D: ResNet-34 U-Net (Fold 1) | Fold 1 (N=206) | 206 | **Official CSV HC** | 1.92 | **1.92** | -0.26 | **-0.26** | Recomputed |
| `row_D_5fold` | Row D: ResNet-34 U-Net (5-Fold Pooled) | 5-Fold CV (N=999) | 999 | **Official CSV HC** | 2.27 | **2.27** | +0.54 | **+0.54** | Recomputed |
| `row_D_tta_5fold` | Row D: ResNet-34 U-Net + TTA (5-Fold Pooled) | 5-Fold CV with TTA (N=999) | 999 | **Official CSV HC** | 2.23 | **2.23** | +0.54 | **+0.54** | Recomputed |
| `row_E_oof` | Row E: EfficientNet-B4 U-Net 384 (Stream 1 Final) | 999 Pooled OOF | 999 | **Mask-Ellipse HC** | 2.30 | **1.99** | -1.26 | **+0.22** | Recomputed |
| `historical_row_d_run1` | Row D Historical Single Run (Cell 31 original) | Fold 1 (N=206) | — | — | — | — | — | — | `cannot recompute against CSV HC (prediction CSV not on disk)` |

## Key Audit Insight

- **Rows A, B, C, D**: Trained and scored directly against official CSV ground truth.
- **Row E (Stream 1 Final OOF)**: Originally scored against the mask-fitted ellipse (+1.48 mm offset), yielding 2.30 mm MAE and -1.26 mm bias. When re-evaluated against the true clinical CSV ground truth, Row E achieves **1.99 mm MAE** and **+0.22 mm signed bias**, outperforming its previously documented score.
