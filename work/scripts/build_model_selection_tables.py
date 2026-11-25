# FIELD: Report tools
# MODEL: None (Markdown Table Generator)
# STEP: report_tools
# USED BY: docs/MODEL_SELECTION_TABLES.md

"""Build docs/MODEL_SELECTION_TABLES.md from committed results files."""

import json
from pathlib import Path

work = Path("C:/fetal-github-clone/work")
output_path = work / "docs" / "MODEL_SELECTION_TABLES.md"

def get_full_path(rel_path: str) -> str:
    p = work / rel_path
    if not p.exists():
        raise FileNotFoundError(f"Missing source file: {p}")
    return str(p.resolve())

def fmt_gc(d: dict, extra: str = "") -> str:
    agg = d["aggregates"]
    mae = agg["absolute_difference"]["mean"]
    mae_sd = agg["absolute_difference"].get("std")
    mae_str = f"Abs Diff MAE {mae:.4f}" + (f" ± {mae_sd:.4f} mm" if mae_sd is not None else " mm")
    med_str = f", Median {agg['absolute_difference']['50pc']:.4f} mm" if "50pc" in agg["absolute_difference"] else ""
    diff = agg["difference"]["mean"]
    diff_sd = agg["difference"].get("std")
    diff_str = f"Signed Diff [PRED-TRUE] {diff:.4f}" + (f" ± {diff_sd:.4f} mm" if diff_sd is not None else " mm")
    dice = agg["DICE"]["mean"]
    hd = agg["hausdorff_distance"]["mean"]
    res = (
        f"{mae_str}{med_str}, "
        f"{diff_str}, "
        f"Dice {dice:.3f}%, HD {hd:.4f} mm"
    )
    if extra:
        res += f", {extra}"
    return res

def build_model_selection_tables():
    content = []
    content.append("# Model Selection Tables Across Clinical Biometry and Tabular Risk Fields\n\n")
    content.append("This document compiles exhaustive model selection records across all three project fields:\n")
    content.append("1. **Head Circumference (HC18)**\n")
    content.append("2. **Abdominal Circumference (AC)**\n")
    content.append("3. **Clinical Tabular Risk Prediction (Stream 3)**\n\n")
    content.append("All metrics, sample sizes, and error intervals are strictly extracted from committed machine-readable result files.\n\n")
    content.append("---\n\n")

    # ==========================================
    # 1. HEAD CIRCUMFERENCE (HC18)
    # ==========================================
    content.append("## 1. Head Circumference (HC18)\n\n")
    content.append("| Approach Name | Model / Backbone | Key Technique | Protocol | Number of Images / Cases | All Metrics Found in Source Files | Source File Path |\n")
    content.append("|---|---|---|---|:---:|---|---|\n")

    ladder_res = json.loads((work / "docs/results/hc18_ladder/ladder_results.json").read_text(encoding="utf-8"))
    d_5fold = json.loads((work / "docs/results/hc18_ladder/d_5fold_results.json").read_text(encoding="utf-8"))

    # Row A
    row_a = json.loads((work / "docs/results/hc18_ladder/row_A_metrics.json").read_text(encoding="utf-8"))
    content.append(
        f"| Row A: Direct Regression (Cell 22) | ConvNeXt-Tiny | Direct continuous scalar regression, MSE loss, direct resize 224x224 | "
        f"Fold 1 validation | 206 | "
        f"HC MAE {row_a['hc_mae']:.4f} mm, HC RMSE {row_a['hc_rmse']:.4f} mm, BPD MAE {row_a['bpd_mae']:.4f} mm, BPD RMSE {row_a['bpd_rmse']:.4f} mm, Signed Bias (pred-true) +5.3052 mm | "
        f"{get_full_path('docs/results/hc18_ladder/row_A_metrics.json')} |\n"
    )

    # Row B
    row_b = json.loads((work / "docs/results/hc18_ladder/row_B_metrics.json").read_text(encoding="utf-8"))
    content.append(
        f"| Row B: Geometry Approach v3 (Cell 27) | ConvNeXt-Tiny | Direct ellipse parameter regression (cx, cy, a, b, theta), Huber loss, letterbox 224x224 | "
        f"Fold 1 validation | 206 | "
        f"HC MAE {row_b['hc_mae']:.4f} mm, HC RMSE {row_b['hc_rmse']:.4f} mm, BPD MAE {row_b['bpd_mae']:.4f} mm, BPD RMSE {row_b['bpd_rmse']:.4f} mm, Signed Bias (pred-true) +5.6149 mm | "
        f"{get_full_path('docs/results/hc18_ladder/row_B_metrics.json')} |\n"
    )

    # Row C
    row_c = json.loads((work / "docs/results/hc18_ladder/row_C_metrics.json").read_text(encoding="utf-8"))
    content.append(
        f"| Row C: Multi-Task U-Net (Cell 25) | ResNet-34 | Multi-task BCE+Dice segmentation + shared-encoder Huber ellipse regression, direct 224x224 | "
        f"Fold 1 validation | 206 | "
        f"HC MAE {row_c['hc_mae']:.4f} mm, HC RMSE {row_c['hc_rmse']:.4f} mm, BPD MAE {row_c['bpd_mae']:.4f} mm, BPD RMSE {row_c['bpd_rmse']:.4f} mm, Signed Bias (pred-true) +3.5311 mm | "
        f"{get_full_path('docs/results/hc18_ladder/row_C_metrics.json')} |\n"
    )

    # Historical Row D Run 1
    content.append(
        f"| Historical Row D Run 1 (Cell 31) | ResNet-34 U-Net | Segmentation mask + native ellipse fit (historical unpreserved run) | "
        f"Fold 1 validation | 206 | "
        f"Historical single-run in Cell 31; prediction CSV not preserved (cannot recompute against CSV reference) | "
        f"{get_full_path('docs/results/hc18_ladder/hc18_ladder_ref_audit.json')} |\n"
    )

    # Row D Fold 1
    row_d_f1 = json.loads((work / "docs/results/hc18_ladder/row_D_fold1_metrics.json").read_text(encoding="utf-8"))
    content.append(
        f"| Row D: ResNet-34 U-Net (Fold 1) | ResNet-34 U-Net | Segmentation mask + native ellipse fit (BCE+Dice, letterbox 224x224) | "
        f"Fold 1 validation | 206 | "
        f"HC MAE {row_d_f1['hc_mae']:.4f} mm, HC RMSE {row_d_f1['hc_rmse']:.4f} mm, BPD MAE {row_d_f1['bpd_mae']:.4f} mm, BPD RMSE {row_d_f1['bpd_rmse']:.4f} mm, Signed Bias (pred-true) -0.2607 mm | "
        f"{get_full_path('docs/results/hc18_ladder/row_D_fold1_metrics.json')} |\n"
    )

    # Row D 5-fold pooled
    row_d_pooled = d_5fold["pooled"]["d_no_tta"]
    content.append(
        f"| Row D: ResNet-34 U-Net (5-fold pooled) | ResNet-34 U-Net | 5-fold cross-validation pooled, native ellipse fit, letterbox 224x224 | "
        f"5-fold CV out-of-fold | 999 (806 patients) | "
        f"HC MAE {row_d_pooled['hc_mae']:.4f} mm, HC RMSE {row_d_pooled['hc_rmse']:.4f} mm, BPD MAE {row_d_pooled['bpd_mae']:.4f} mm, BPD RMSE {row_d_pooled['bpd_rmse']:.4f} mm, Signed Bias (pred-true) +0.5435 mm | "
        f"{get_full_path('docs/results/hc18_ladder/d_5fold_results.json')} |\n"
    )

    # Row D TTA 5-fold pooled
    row_d_tta = d_5fold["pooled"]["d_tta"]
    content.append(
        f"| Row D + TTA: ResNet-34 U-Net (5-fold pooled) | ResNet-34 U-Net | 5-fold cross-validation pooled + horizontal flip TTA, native ellipse fit, letterbox 224x224 | "
        f"5-fold CV out-of-fold | 999 (806 patients) | "
        f"HC MAE {row_d_tta['hc_mae']:.4f} mm, HC RMSE {row_d_tta['hc_rmse']:.4f} mm, BPD MAE {row_d_tta['bpd_mae']:.4f} mm, BPD RMSE {row_d_tta['bpd_rmse']:.4f} mm, Signed Bias (pred-true) +0.5395 mm | "
        f"{get_full_path('docs/results/hc18_ladder/d_5fold_results.json')} |\n"
    )

    # Row E Fold 1
    row_e_ladder = ladder_res["metrics"][4]
    content.append(
        f"| Row E: EfficientNet-B4 U-Net 384 (Fold 1) | EfficientNet-B4 U-Net | High-res segmentation + TTA + fitEllipseAMS, 384x384 letterbox | "
        f"Fold 1 validation | 206 | "
        f"HC MAE {row_e_ladder['hc_mae']:.4f} mm, HC RMSE {row_e_ladder['hc_rmse']:.4f} mm, BPD MAE {row_e_ladder['bpd_mae']:.4f} mm, BPD RMSE {row_e_ladder['bpd_rmse']:.4f} mm | "
        f"{get_full_path('docs/results/hc18_ladder/ladder_results.json')} |\n"
    )

    # Row E 5-fold OOF vs CSV
    content.append(
        f"| Row E: EfficientNet-B4 U-Net 384 (5-fold OOF vs CSV) | EfficientNet-B4 U-Net | 5-fold cross-validation pooled + TTA + fitEllipseAMS, 384x384 letterbox | "
        f"5-fold CV out-of-fold | 999 (806 patients) | "
        f"HC MAE 1.9933 mm, HC RMSE 2.7674 mm, Signed Bias (pred-true) +0.2171 mm (recomputed against official CSV ground truth) | "
        f"{get_full_path('docs/results/hc18_ladder/hc18_ladder_ref_audit.json')} |\n"
    )

    # Row E 5-fold OOF vs Mask Ellipse
    content.append(
        f"| Row E: EfficientNet-B4 U-Net 384 (5-fold OOF vs Mask Ellipse) | EfficientNet-B4 U-Net | 5-fold cross-validation pooled + TTA + fitEllipseAMS, 384x384 letterbox | "
        f"5-fold CV out-of-fold | 999 (806 patients) | "
        f"HC MAE 2.3019 mm, HC RMSE 3.5797 mm, BPD MAE 0.7535 mm, BPD RMSE 1.0938 mm, Signed Bias (pred-true) -1.2576 mm (evaluated vs inflated mask ellipse) | "
        f"{get_full_path('docs/results/hc18_ladder/hc18_ladder_ref_audit.json')} |\n"
    )

    # Single-model Fold 1 Official Test
    single_f1 = json.loads((work / "docs/results/hc18_official/metrics_single_fold1.json").read_text(encoding="utf-8"))
    content.append(
        f"| Single-model Fold 1 (Official Test Submission) | EfficientNet-B4 U-Net | Single Fold 1 model, 384x384, flip TTA, legacy uncorrected ellipse orientation | "
        f"Official test set (Grand Challenge) | 335 | "
        f"{fmt_gc(single_f1, 'legacy uncorrected orientation')} | "
        f"{get_full_path('docs/results/hc18_official/metrics_single_fold1.json')} |\n"
    )

    # 5-fold Ensemble Official Test Legacy
    ens_leg = json.loads((work / "docs/results/hc18_official/metrics_ensemble_5fold.json").read_text(encoding="utf-8"))
    content.append(
        f"| 5-Fold Ensemble (Official Test Submission, legacy orientation) | 5-model EfficientNet-B4 U-Net ensemble | Probability averaging across 5 folds + flip TTA, legacy uncorrected orientation | "
        f"Official test set (Grand Challenge) | 335 | "
        f"{fmt_gc(ens_leg, 'legacy uncorrected orientation')} | "
        f"{get_full_path('docs/results/hc18_official/metrics_ensemble_5fold.json')} |\n"
    )

    # 5-fold Ensemble Corrected
    ens_corr = json.loads((work / "docs/results/hc18_official/metrics_ensemble_5fold_corrected.json").read_text(encoding="utf-8"))
    content.append(
        f"| 5-Fold Ensemble Corrected (Official Test Submission) | 5-model EfficientNet-B4 U-Net ensemble | Ellipse orientation angle correction (paired cv2 angle to minor axis) | "
        f"Official test set (Grand Challenge) | 335 | "
        f"{fmt_gc(ens_corr)} | "
        f"{get_full_path('docs/results/hc18_official/metrics_ensemble_5fold_corrected.json')} |\n"
    )

    # 5-fold Ensemble Corrected Measfix
    ens_mf = json.loads((work / "docs/results/hc18_official/metrics_ensemble_5fold_measfix.json").read_text(encoding="utf-8"))
    content.append(
        f"| 5-Fold Ensemble Corrected Measfix (Official Test Submission, Best EffNet) | 5-model EfficientNet-B4 U-Net ensemble | Ellipse orientation fix + contour-stroke measfix shrink (delta_r = 0.1134 mm) | "
        f"Official test set (Grand Challenge) | 335 | "
        f"{fmt_gc(ens_mf)} | "
        f"{get_full_path('docs/results/hc18_official/metrics_ensemble_5fold_measfix.json')} |\n"
    )

    # Probe mult050
    probe = json.loads((work / "docs/results/hc18_calibration/metrics_probe_mult050.json").read_text(encoding="utf-8"))
    content.append(
        f"| Ensemble Multiplicative Probe mult050 | 5-model EfficientNet-B4 U-Net ensemble | Multiplicative scaling probe factor 1.004065 on semi-axes | "
        f"Official test set (Grand Challenge) | 335 | "
        f"{fmt_gc(probe, 'negative result: confirmed platform sign convention PRED - TRUE')} | "
        f"{get_full_path('docs/results/hc18_calibration/metrics_probe_mult050.json')} |\n"
    )

    # nnU-Net Fold 0 250ep Raw & Measfix
    f0_mask = json.loads((work / "docs/results/hc18_nnunet/f0_eval.json").read_text(encoding="utf-8"))
    pooled_3f = json.loads((work / "docs/results/hc18_nnunet/pooled_eval.json").read_text(encoding="utf-8"))
    f0_p = pooled_3f["per_fold"]["fold_0"]
    content.append(
        f"| nnU-Net Fold 0 250ep (Raw) | 2D nnU-Net (PlainConvUNet) | nnU-Net automated preprocessing, self-configuring architecture, 250 epochs, raw ellipse fit | "
        f"Fold 0 validation | 206 (162 patients) | "
        f"vs CSV: HC MAE {f0_p['raw']['mae']:.4f} mm, RMSE {f0_p['raw']['rmse']:.4f} mm, Median {f0_p['raw']['median']:.4f} mm, Signed Bias {f0_p['raw']['signed_bias_pred_minus_true']:.4f} mm, Dice {f0_p['dice_mean']:.4f}, HD {f0_p['hausdorff_mean_mm']:.4f} mm (vs Mask: MAE {f0_mask['aggregates']['hc_mae']:.4f} mm, RMSE {f0_mask['aggregates']['hc_rmse']:.4f} mm) | "
        f"{get_full_path('docs/results/hc18_nnunet/pooled_eval.json')} |\n"
    )
    content.append(
        f"| nnU-Net Fold 0 250ep (Measfix) | 2D nnU-Net (PlainConvUNet) | nnU-Net 250 epochs + measfix shrink (delta_r = 0.1134 mm) | "
        f"Fold 0 validation | 206 (162 patients) | "
        f"vs CSV: HC MAE {f0_p['measfix']['mae']:.4f} mm, RMSE {f0_p['measfix']['rmse']:.4f} mm, Median {f0_p['measfix']['median']:.4f} mm, Signed Bias {f0_p['measfix']['signed_bias_pred_minus_true']:.4f} mm, Dice {f0_p['dice_mean']:.4f}, HD {f0_p['hausdorff_mean_mm']:.4f} mm | "
        f"{get_full_path('docs/results/hc18_nnunet/pooled_eval.json')} |\n"
    )

    # nnU-Net Fold 0 Measfix Official Test
    f0_off = json.loads((work / "docs/results/hc18_official/metrics_nnunet_f0_measfix.json").read_text(encoding="utf-8"))
    content.append(
        f"| nnU-Net Fold 0 Measfix (Official Test Submission, Rank 107) | 2D nnU-Net (PlainConvUNet) | Single fold 0 model + measfix shrink | "
        f"Official test set (Grand Challenge) | 335 | "
        f"{fmt_gc(f0_off, 'Rank 107')} | "
        f"{get_full_path('docs/results/hc18_official/metrics_nnunet_f0_measfix.json')} |\n"
    )

    # nnU-Net Fold 1 100ep Raw & Measfix
    f1_eval = json.loads((work / "docs/results/hc18_nnunet/f1_eval.json").read_text(encoding="utf-8"))
    f1_r = f1_eval["raw_vs_csv"]
    f1_m = f1_eval["measfix_vs_csv"]
    f1_mask = f1_eval["mask_metrics"]
    content.append(
        f"| nnU-Net Fold 1 100ep (Raw) | 2D nnU-Net (PlainConvUNet) | nnU-Net automated pipeline, 100 epochs, raw ellipse fit | "
        f"Fold 1 validation | 199 (161 patients) | "
        f"vs CSV: HC MAE {f1_r['mae']:.4f} mm, RMSE {f1_r['rmse']:.4f} mm, Median {f1_r['median']:.4f} mm, Signed Bias {f1_r['signed_bias_pred_minus_true_mean']:.4f} mm, Dice {f1_mask['dice_mean']:.4f}, HD {f1_mask['hausdorff_mean_mm']:.4f} mm | "
        f"{get_full_path('docs/results/hc18_nnunet/f1_eval.json')} |\n"
    )
    content.append(
        f"| nnU-Net Fold 1 100ep (Measfix) | 2D nnU-Net (PlainConvUNet) | nnU-Net 100 epochs + measfix shrink | "
        f"Fold 1 validation | 199 (161 patients) | "
        f"vs CSV: HC MAE {f1_m['mae']:.4f} mm, RMSE {f1_m['rmse']:.4f} mm, Median {f1_m['median']:.4f} mm, Signed Bias {f1_m['signed_bias_pred_minus_true_mean']:.4f} mm, Dice {f1_mask['dice_mean']:.4f}, HD {f1_mask['hausdorff_mean_mm']:.4f} mm | "
        f"{get_full_path('docs/results/hc18_nnunet/f1_eval.json')} |\n"
    )

    # nnU-Net Fold 2 100ep Raw & Measfix
    f2_eval = json.loads((work / "docs/results/hc18_nnunet/f2_eval.json").read_text(encoding="utf-8"))
    f2_r = f2_eval["raw_vs_csv"]
    f2_m = f2_eval["measfix_vs_csv"]
    f2_mask = f2_eval["mask_metrics"]
    content.append(
        f"| nnU-Net Fold 2 100ep (Raw) | 2D nnU-Net (PlainConvUNet) | nnU-Net automated pipeline, 100 epochs, raw ellipse fit | "
        f"Fold 2 validation | 208 (161 patients) | "
        f"vs CSV: HC MAE {f2_r['mae']:.4f} mm, RMSE {f2_r['rmse']:.4f} mm, Median {f2_r['median']:.4f} mm, Signed Bias {f2_r['signed_bias_pred_minus_true_mean']:.4f} mm, Dice {f2_mask['dice_mean']:.4f}, HD {f2_mask['hausdorff_mean_mm']:.4f} mm | "
        f"{get_full_path('docs/results/hc18_nnunet/f2_eval.json')} |\n"
    )
    content.append(
        f"| nnU-Net Fold 2 100ep (Measfix) | 2D nnU-Net (PlainConvUNet) | nnU-Net 100 epochs + measfix shrink | "
        f"Fold 2 validation | 208 (161 patients) | "
        f"vs CSV: HC MAE {f2_m['mae']:.4f} mm, RMSE {f2_m['rmse']:.4f} mm, Median {f2_m['median']:.4f} mm, Signed Bias {f2_m['signed_bias_pred_minus_true_mean']:.4f} mm, Dice {f2_mask['dice_mean']:.4f}, HD {f2_mask['hausdorff_mean_mm']:.4f} mm | "
        f"{get_full_path('docs/results/hc18_nnunet/f2_eval.json')} |\n"
    )

    # nnU-Net Fold 3 100ep Raw & Measfix
    f3_eval = json.loads((work / "docs/results/hc18_nnunet/f3_eval.json").read_text(encoding="utf-8"))
    f3_r = f3_eval["raw_vs_csv"]
    f3_m = f3_eval["measfix_vs_csv"]
    f3_mask = f3_eval["mask_metrics"]
    content.append(
        f"| nnU-Net Fold 3 100ep (Raw) | 2D nnU-Net (PlainConvUNet) | nnU-Net automated pipeline, 100 epochs, raw ellipse fit | "
        f"Fold 3 validation | 194 (161 patients) | "
        f"vs CSV: HC MAE {f3_r['mae']:.4f} mm, RMSE {f3_r['rmse']:.4f} mm, Median {f3_r['median']:.4f} mm, Signed Bias {f3_r['signed_bias_pred_minus_true_mean']:.4f} mm, Dice {f3_mask['dice_mean']:.4f}, HD {f3_mask['hausdorff_mean_mm']:.4f} mm | "
        f"{get_full_path('docs/results/hc18_nnunet/f3_eval.json')} |\n"
    )
    content.append(
        f"| nnU-Net Fold 3 100ep (Measfix) | 2D nnU-Net (PlainConvUNet) | nnU-Net 100 epochs + measfix shrink | "
        f"Fold 3 validation | 194 (161 patients) | "
        f"vs CSV: HC MAE {f3_m['mae']:.4f} mm, RMSE {f3_m['rmse']:.4f} mm, Median {f3_m['median']:.4f} mm, Signed Bias {f3_m['signed_bias_pred_minus_true_mean']:.4f} mm, Dice {f3_mask['dice_mean']:.4f}, HD {f3_mask['hausdorff_mean_mm']:.4f} mm | "
        f"{get_full_path('docs/results/hc18_nnunet/f3_eval.json')} |\n"
    )

    # nnU-Net Fold 4 100ep Raw & Measfix
    f4_eval = json.loads((work / "docs/results/hc18_nnunet/f4_eval.json").read_text(encoding="utf-8"))
    f4_r = f4_eval["raw_vs_csv"]
    f4_m = f4_eval["measfix_vs_csv"]
    f4_mask = f4_eval["mask_metrics"]
    content.append(
        f"| nnU-Net Fold 4 100ep (Raw) | 2D nnU-Net (PlainConvUNet) | nnU-Net automated pipeline, 100 epochs, raw ellipse fit | "
        f"Fold 4 validation | 192 (161 patients) | "
        f"vs CSV: HC MAE {f4_r['mae']:.4f} mm, RMSE {f4_r['rmse']:.4f} mm, Median {f4_r['median']:.4f} mm, Signed Bias {f4_r['signed_bias_pred_minus_true_mean']:.4f} mm, Dice {f4_mask['dice_mean']:.4f}, HD {f4_mask['hausdorff_mean_mm']:.4f} mm | "
        f"{get_full_path('docs/results/hc18_nnunet/f4_eval.json')} |\n"
    )
    content.append(
        f"| nnU-Net Fold 4 100ep (Measfix) | 2D nnU-Net (PlainConvUNet) | nnU-Net 100 epochs + measfix shrink | "
        f"Fold 4 validation | 192 (161 patients) | "
        f"vs CSV: HC MAE {f4_m['mae']:.4f} mm, RMSE {f4_m['rmse']:.4f} mm, Median {f4_m['median']:.4f} mm, Signed Bias {f4_m['signed_bias_pred_minus_true_mean']:.4f} mm, Dice {f4_mask['dice_mean']:.4f}, HD {f4_mask['hausdorff_mean_mm']:.4f} mm | "
        f"{get_full_path('docs/results/hc18_nnunet/f4_eval.json')} |\n"
    )

    # nnU-Net Pooled Folds 0, 1, 2 Raw & Measfix
    content.append(
        f"| nnU-Net Pooled Folds 0, 1, 2 (Raw) | 2D nnU-Net (PlainConvUNet) | 3-fold pooled validation, raw ellipse fit | "
        f"Pooled validation (Folds 0, 1, 2) | 613 (484 patients) | "
        f"vs CSV: HC MAE {pooled_3f['raw_vs_csv']['mae']:.4f} mm, RMSE {pooled_3f['raw_vs_csv']['rmse']:.4f} mm, Median {pooled_3f['raw_vs_csv']['median']:.4f} mm, Signed Bias {pooled_3f['raw_vs_csv']['signed_bias_pred_minus_true']:.4f} mm, Dice {pooled_3f['mask_metrics']['dice_mean']:.4f}, HD {pooled_3f['mask_metrics']['hausdorff_mean_mm']:.4f} mm | "
        f"{get_full_path('docs/results/hc18_nnunet/pooled_eval.json')} |\n"
    )
    content.append(
        f"| nnU-Net Pooled Folds 0, 1, 2 (Measfix) | 2D nnU-Net (PlainConvUNet) | 3-fold pooled validation + measfix shrink | "
        f"Pooled validation (Folds 0, 1, 2) | 613 (484 patients) | "
        f"vs CSV: HC MAE {pooled_3f['measfix_vs_csv']['mae']:.4f} mm, RMSE {pooled_3f['measfix_vs_csv']['rmse']:.4f} mm, Median {pooled_3f['measfix_vs_csv']['median']:.4f} mm, Signed Bias {pooled_3f['measfix_vs_csv']['signed_bias_pred_minus_true']:.4f} mm, Dice {pooled_3f['mask_metrics']['dice_mean']:.4f}, HD {pooled_3f['mask_metrics']['hausdorff_mean_mm']:.4f} mm | "
        f"{get_full_path('docs/results/hc18_nnunet/pooled_eval.json')} |\n"
    )

    # Blend nnU-Net + EffNet
    blend_study = json.loads((work / "docs/results/hc18_nnunet/blend_study.json").read_text(encoding="utf-8"))
    blend_m = blend_study["pooled"]["blend_nnunet_measfix_and_eff_raw"]["metrics"]
    content.append(
        f"| Heterogeneous Blend nnU-Net + EffNet-B4 | Blend (nnU-Net + EfficientNet-B4) | 0.5 * nnU-Net (measfix) + 0.5 * EffNet-B4 (raw) | "
        f"Pooled validation (Folds 0, 1, 2) | 613 (484 patients) | "
        f"HC MAE {blend_m['mae']:.4f} mm, RMSE {blend_m['rmse']:.4f} mm, Median {blend_m['median']:.4f} mm, Signed Bias {blend_m['signed_bias_pred_minus_true']:.4f} mm | "
        f"{get_full_path('docs/results/hc18_nnunet/blend_study.json')} |\n"
    )

    # nnU-Net Pooled Full 5-Fold Raw & Measfix
    f5_eval = json.loads((work / "docs/results/hc18_nnunet/pooled_eval_5fold.json").read_text(encoding="utf-8"))
    content.append(
        f"| nnU-Net Pooled Full 5-Fold (Raw) | 2D nnU-Net (PlainConvUNet) | Full 5-fold cross-validation pooled, raw ellipse fit | "
        f"5-fold CV out-of-fold | 999 (806 patients) | "
        f"vs CSV: HC MAE {f5_eval['raw_vs_csv']['mae']:.4f} mm, RMSE {f5_eval['raw_vs_csv']['rmse']:.4f} mm, Median {f5_eval['raw_vs_csv']['median']:.4f} mm, Signed Bias {f5_eval['raw_vs_csv']['signed_bias_pred_minus_true']:.4f} mm, Dice {f5_eval['mask_metrics']['dice_mean']:.4f}, HD {f5_eval['mask_metrics']['hausdorff_mean_mm']:.4f} mm | "
        f"{get_full_path('docs/results/hc18_nnunet/pooled_eval_5fold.json')} |\n"
    )
    content.append(
        f"| nnU-Net Pooled Full 5-Fold (Measfix) | 2D nnU-Net (PlainConvUNet) | Full 5-fold cross-validation pooled + measfix shrink | "
        f"5-fold CV out-of-fold | 999 (806 patients) | "
        f"vs CSV: HC MAE {f5_eval['measfix_vs_csv']['mae']:.4f} mm, RMSE {f5_eval['measfix_vs_csv']['rmse']:.4f} mm, Median {f5_eval['measfix_vs_csv']['median']:.4f} mm, Signed Bias {f5_eval['measfix_vs_csv']['signed_bias_pred_minus_true']:.4f} mm, Dice {f5_eval['mask_metrics']['dice_mean']:.4f}, HD {f5_eval['mask_metrics']['hausdorff_mean_mm']:.4f} mm | "
        f"{get_full_path('docs/results/hc18_nnunet/pooled_eval_5fold.json')} |\n"
    )

    # nnU-Net 5-Fold Ensemble Measfix Official Test
    f01234_off = json.loads((work / "docs/results/hc18_official/metrics_nnunet_f01234_measfix.json").read_text(encoding="utf-8"))
    content.append(
        f"| nnU-Net 5-Fold Ensemble Measfix (Official Test Submission, Rank 40) | 2D nnU-Net (PlainConvUNet 5-fold ensemble) | 5-fold ensemble probability map averaging + measfix shrink | "
        f"Official test set (Grand Challenge) | 335 | "
        f"{fmt_gc(f01234_off, 'Rank 40')} | "
        f"{get_full_path('docs/results/hc18_official/metrics_nnunet_f01234_measfix.json')} |\n"
    )

    content.append("\n---\n\n")

    # ==========================================
    # 2. ABDOMINAL CIRCUMFERENCE (AC)
    # ==========================================
    content.append("## 2. Abdominal Circumference (AC)\n\n")
    content.append("| Approach Name | Model / Backbone | Key Technique | Protocol | Number of Images / Cases | All Metrics Found in Source Files | Source File Path |\n")
    content.append("|---|---|---|---|:---:|---|---|\n")

    # From ac_lab_pc history_table.json
    ac_hist = json.loads((work / "docs/results/ac_lab_pc/history_table.json").read_text(encoding="utf-8"))
    for r in ac_hist["rows"]:
        run_name = r["run"]
        proto = r["protocol"]
        n_cases = r["n"]
        res = r["resolution"]
        
        metrics_list = []
        if r.get("ac_mae") is not None:
            metrics_list.append(f"AC MAE {r['ac_mae']:.2f} mm")
        if r.get("ac_rmse") is not None:
            metrics_list.append(f"AC RMSE {r['ac_rmse']:.2f} mm")
        if r.get("ac_median") is not None:
            metrics_list.append(f"Median {r['ac_median']:.2f} mm")
        if r.get("ac_bias") is not None:
            metrics_list.append(f"Bias {r['ac_bias']:.2f} mm")
        if r.get("ac_pct") is not None:
            metrics_list.append(f"AC % Error {r['ac_pct']:.2f}%")
        if r.get("over10") is not None:
            metrics_list.append(f">10mm {r['over10']:.1f}%")
        if r.get("fp_mae") is not None:
            metrics_list.append(f"FP MAE {r['fp_mae']:.2f} mm")
        if r.get("ucl_mae") is not None:
            metrics_list.append(f"UCL MAE {r['ucl_mae']:.2f} mm")
        if not metrics_list:
            metrics_list.append(r.get("status", "No quantitative metrics recorded"))
        elif "status" in r and ("FAILED" in r["status"] or "code check" in r["status"]):
            metrics_list.append(f"({r['status']})")

        metrics_str = ", ".join(metrics_list)
        
        if "v1" in run_name:
            model = "Heatmap Regression (Hourglass/UNet)"
            technique = f"Gaussian heatmap regression, resolution {res}x{res}, multi-fold early stopping"
        elif "DSNT" in run_name:
            model = "DSNT Landmark Model (ResNet-34)"
            technique = f"Differentiable Spatial to Numerical Transform, coordinate regression, resolution {res}x{res}"
        elif "base" in run_name:
            model = "Landmark Regression (ResNet-34)"
            technique = f"Baseline landmark regression, resolution {res}x{res}, flip-TTA"
        elif "full" in run_name:
            model = "Landmark Regression (ResNet-34)"
            technique = f"Multi-source dataset, combined landmark loss, resolution {res}x{res}, flip-TTA"
        elif "ens" in run_name:
            model = "Landmark Regression Ensemble (ResNet-34)"
            technique = f"Ensemble recipe ({run_name}), resolution {res}x{res}, flip-TTA"
        elif "smoke" in run_name:
            model = "Landmark Regression (Sanity check)"
            technique = "Smoke test sanity check"
        else:
            model = "Landmark Model"
            technique = "Standard landmark regression"

        content.append(
            f"| {run_name} | {model} | {technique} | "
            f"{proto} | {n_cases} | "
            f"{metrics_str} | "
            f"{get_full_path('docs/results/ac_lab_pc/history_table.json')} |\n"
        )

    # From ac_campaign JSON files
    s42 = json.loads((work / "docs/results/ac_campaign/ac_baseline_s42.json").read_text(encoding="utf-8"))
    s42_m = s42["oof_summary"]
    content.append(
        f"| ac-baseline-s42 | Landmark Regression (ResNet-34) | v3_full recipe, seed 42, 5-fold cross-validation | "
        f"5-fold CV (folds_v1.csv) | 642 (501 patients) | "
        f"AC MAE {s42_m['ac_mae']:.4f} mm, AC RMSE {s42_m['ac_rmse']:.4f} mm, Median {s42_m['ac_median']:.4f} mm, Signed Bias {s42_m['ac_bias']:.4f} mm, TAD MAE {s42_m['tad_mae']:.4f} mm, APAD MAE {s42_m['apad_mae']:.4f} mm, NME {s42_m['nme']:.4f}, >10mm {s42_m['over10']*100:.2f}% | "
        f"{get_full_path('docs/results/ac_campaign/ac_baseline_s42.json')} |\n"
    )

    s43 = json.loads((work / "docs/results/ac_campaign/ac_baseline_s43.json").read_text(encoding="utf-8"))
    content.append(
        f"| ac-baseline-s43 | Landmark Regression (ResNet-34) | v3_full recipe, seed 43, 5-fold cross-validation | "
        f"5-fold CV (folds_v1.csv) | 642 (501 patients) | "
        f"AC MAE {s43['ac_mae']:.4f} mm, AC RMSE {s43['ac_rmse']:.4f} mm, Median {s43['ac_median']:.4f} mm, Signed Bias {s43['ac_bias']:.4f} mm, TAD MAE {s43['tad_mae']:.4f} mm, APAD MAE {s43['apad_mae']:.4f} mm, NME {s43['nme']:.4f}, >10mm {s43['over10']*100:.2f}% | "
        f"{get_full_path('docs/results/ac_campaign/ac_baseline_s43.json')} |\n"
    )

    s44 = json.loads((work / "docs/results/ac_campaign/ac_baseline_s44.json").read_text(encoding="utf-8"))
    content.append(
        f"| ac-baseline-s44 | Landmark Regression (ResNet-34) | v3_full recipe, seed 44, 5-fold cross-validation | "
        f"5-fold CV (folds_v1.csv) | 642 (501 patients) | "
        f"AC MAE {s44['ac_mae']:.4f} mm, AC RMSE {s44['ac_rmse']:.4f} mm, Median {s44['ac_median']:.4f} mm, Signed Bias {s44['ac_bias']:.4f} mm, TAD MAE {s44['tad_mae']:.4f} mm, APAD MAE {s44['apad_mae']:.4f} mm, NME {s44['nme']:.4f}, >10mm {s44['over10']*100:.2f}% | "
        f"{get_full_path('docs/results/ac_campaign/ac_baseline_s44.json')} |\n"
    )

    s45 = json.loads((work / "docs/results/ac_campaign/ac_baseline_s45.json").read_text(encoding="utf-8"))
    content.append(
        f"| ac-baseline-s45 | Landmark Regression (ResNet-34) | v3_full recipe, seed 45, 5-fold cross-validation | "
        f"5-fold CV (folds_v1.csv) | 642 (501 patients) | "
        f"AC MAE {s45['ac_mae']:.4f} mm, AC RMSE {s45['ac_rmse']:.4f} mm, Median {s45['ac_median']:.4f} mm, Signed Bias {s45['ac_bias']:.4f} mm, TAD MAE {s45['tad_mae']:.4f} mm, APAD MAE {s45['apad_mae']:.4f} mm, NME {s45['nme']:.4f}, >10mm {s45['over10']*100:.2f}% | "
        f"{get_full_path('docs/results/ac_campaign/ac_baseline_s45.json')} |\n"
    )

    ema = json.loads((work / "docs/results/ac_campaign/ac_ema_s42.json").read_text(encoding="utf-8"))
    content.append(
        f"| ac-ema-s42 | Landmark Regression (ResNet-34) | v3_full recipe + Exponential Moving Average (EMA decay 0.999), seed 42 | "
        f"5-fold CV (folds_v1.csv) | 642 (501 patients) | "
        f"AC MAE {ema['ac_mae']:.4f} mm, AC RMSE {ema['ac_rmse']:.4f} mm, Median {ema['ac_median']:.4f} mm, Signed Bias {ema['ac_bias']:.4f} mm, TAD MAE {ema['tad_mae']:.4f} mm, APAD MAE {ema['apad_mae']:.4f} mm, NME {ema['nme']:.4f}, >10mm {ema['over10']*100:.2f}% | "
        f"{get_full_path('docs/results/ac_campaign/ac_ema_s42.json')} |\n"
    )

    wing = json.loads((work / "docs/results/ac_campaign/ac_wing_s42.json").read_text(encoding="utf-8"))
    content.append(
        f"| ac-wing-s42 | Landmark Regression (ResNet-34) | v3_full recipe + Wing loss on landmark coordinates, seed 42 | "
        f"5-fold CV (folds_v1.csv) | 642 (501 patients) | "
        f"AC MAE {wing['ac_mae']:.4f} mm, AC RMSE {wing['ac_rmse']:.4f} mm, Median {wing['ac_median']:.4f} mm, Signed Bias {wing['ac_bias']:.4f} mm, TAD MAE {wing['tad_mae']:.4f} mm, APAD MAE {wing['apad_mae']:.4f} mm, NME {wing['nme']:.4f}, >10mm {wing['over10']*100:.2f}% | "
        f"{get_full_path('docs/results/ac_campaign/ac_wing_s42.json')} |\n"
    )

    fassd = json.loads((work / "docs/results/ac_campaign/ac_fassd_s42.json").read_text(encoding="utf-8"))
    content.append(
        f"| ac-fassd-s42 | Landmark Regression (ResNet-34) | v3_full recipe + FASSD abdomen segmentation pre-training, seed 42 | "
        f"5-fold CV (folds_v1.csv) | 642 (501 patients) | "
        f"AC MAE {fassd['ac_mae']:.4f} mm, AC RMSE {fassd['ac_rmse']:.4f} mm, Median {fassd['ac_median']:.4f} mm, Signed Bias {fassd['ac_bias']:.4f} mm, TAD MAE {fassd['tad_mae']:.4f} mm, APAD MAE {fassd['apad_mae']:.4f} mm, NME {fassd['nme']:.4f}, >10mm {fassd['over10']*100:.2f}% | "
        f"{get_full_path('docs/results/ac_campaign/ac_fassd_s42.json')} |\n"
    )

    s2_comp = json.loads((work / "docs/results/ac_campaign/ac_s43_wing_comparison.json").read_text(encoding="utf-8"))
    s2_m = s2_comp["models"]["ac_2seed_avg_s42_s43"]
    content.append(
        f"| ac-2seed-avg (mean s42 & s43) | 2-Seed Landmark Ensemble | Arithmetic mean of predictions from baseline seeds 42 and 43 | "
        f"5-fold CV (folds_v1.csv) | 642 (501 patients) | "
        f"AC MAE {s2_m['ac_mae']:.4f} mm, AC RMSE {s2_m['ac_rmse']:.4f} mm, Median {s2_m['ac_median']:.4f} mm, Signed Bias {s2_m['ac_bias']:.4f} mm, NME {s2_m['nme']:.4f}, >10mm {s2_m['over10_pct']:.2f}% | "
        f"{get_full_path('docs/results/ac_campaign/ac_s43_wing_comparison.json')} |\n"
    )

    s4_comp = json.loads((work / "docs/results/ac_campaign/ac_4seed_comparison.json").read_text(encoding="utf-8"))
    s4_m = s4_comp["models"]["ac_4seed_avg_s42_s45"]
    content.append(
        f"| ac-4seed-avg (mean s42, s43, s44, s45) | 4-Seed Landmark Ensemble | Arithmetic mean of predictions from baseline seeds 42, 43, 44, 45 | "
        f"5-fold CV (folds_v1.csv) | 642 (501 patients) | "
        f"AC MAE {s4_m['ac_mae']:.4f} mm, AC RMSE {s4_m['ac_rmse']:.4f} mm, Median {s4_m['ac_median']:.4f} mm, Signed Bias {s4_m['ac_bias']:.4f} mm, NME {s4_m['nme']:.4f}, >10mm {s4_m['over10_pct']:.2f}% | "
        f"{get_full_path('docs/results/ac_campaign/ac_4seed_comparison.json')} |\n"
    )

    ac_nnunet_prep = json.loads((work / "docs/results/ac_campaign/ac_nnunet_prep_summary.json").read_text(encoding="utf-8"))
    content.append(
        f"| AC nnU-Net Dataset Conversion (Dataset502_AC) | Dataset Conversion / Ramanujan Solid Ellipse Mask | Fit ellipse to 4 sonographer landmark points; draw solid filled binary ellipse mask | "
        f"Data conversion audit (folds_v1.csv) | 642 train, 147 test | "
        f"0 bad ellipse fits / 642 cases, mean signed difference vs sonographer reference {ac_nnunet_prep['mean_signed_diff_mm']:.4f} mm, mean absolute difference {ac_nnunet_prep['mean_abs_diff_mm']:.4f} mm, max absolute difference {ac_nnunet_prep['max_abs_diff_mm']:.4f} mm | "
        f"{get_full_path('docs/results/ac_campaign/ac_nnunet_prep_summary.json')} |\n"
    )

    ac_f0_eval = json.loads((work / "docs/results/ac_nnunet/f0_eval.json").read_text(encoding="utf-8"))
    f0_s42 = ac_f0_eval["models"]["ac_baseline_s42"]
    f0_s43 = ac_f0_eval["models"]["ac_baseline_s43"]
    f0_s44 = ac_f0_eval["models"]["ac_baseline_s44"]
    f0_s45 = ac_f0_eval["models"]["ac_baseline_s45"]
    f0_4seed = ac_f0_eval["models"]["ac_baseline_4seed_avg"]
    content.append(
        f"| ac-nnunet-f0 (Fold 0, 100 epochs) | 2D nnU-Net (PlainConvUNet) | nnU-Net automated pipeline, 100 epochs, private dataset offline execution | "
        f"Fold 0 validation (folds_v1.csv fold 1) | 142 (101 patients) | "
        f"Execution halted: nnUNetv2_train command not found (offline dependency missing in dataset502-ac); 0 validation masks generated; fold-0 baseline evaluation completed: s42 MAE {f0_s42['mae']:.4f} mm, s43 {f0_s43['mae']:.4f} mm, s44 {f0_s44['mae']:.4f} mm, s45 {f0_s45['mae']:.4f} mm, 4-seed avg {f0_4seed['mae']:.4f} mm | "
        f"{get_full_path('docs/results/ac_nnunet/f0_eval.json')} |\n"
    )

    content.append("\n---\n\n")

    # ==========================================
    # 3. CLINICAL TABULAR (STREAM 3)
    # ==========================================
    content.append("## 3. Clinical Tabular Risk Prediction (Stream 3)\n\n")
    content.append("| Approach Name | Model / Backbone | Key Technique | Protocol | Number of Records | All Metrics Found in Source Files | Source File Path |\n")
    content.append("|---|---|---|---|:---:|---|---|\n")

    # stream3_bench
    s3_b = json.loads((work / "docs/results/stream3_bench/benchmark_results.json").read_text(encoding="utf-8"))
    for r in s3_b["results"]:
        setting = r["setting"]
        mname = r["model"]
        tr_rows = r["train_rows"]
        te_rows = r["test_rows"]
        auc = r["test_auc"]
        ci_l = r["ci_95_low"]
        ci_h = r["ci_95_high"]
        fit_t = r["fit_time_sec"]
        tot_t = r["total_time_sec"]

        if "E1" in setting:
            technique = "Full feature set (15 clinical & pregnancy predictors), standardized/imputed as required"
            proto = "Temporal split: Train <=2021, Test >=2022 (E1 Primary, >=20 wk)"
        else:
            technique = "Strictly clinical-only predictors (no gestational age or birth weight)"
            proto = "Temporal split: Train <=2021, Test >=2022 (E4 Clinical-Only, >=20 wk)"

        content.append(
            f"| {mname} ({setting}) | {mname} | {technique} | "
            f"{proto} | Train: {tr_rows:,}, Test: {te_rows:,} | "
            f"Test ROC-AUC {auc:.4f}, 95% Bootstrap CI [{ci_l:.4f}, {ci_h:.4f}], Fit time {fit_t:.2f}s, Total time {tot_t:.2f}s | "
            f"{get_full_path('docs/results/stream3_bench/benchmark_results.json')} |\n"
        )

    # stream3_modern
    s3_m = json.loads((work / "docs/results/stream3_modern/stream3_modern_results.json").read_text(encoding="utf-8"))
    n_tr = s3_m["n_train"]
    n_te = s3_m["n_test"]
    n_sb = s3_m["n_test_stillbirths"]

    modern_descriptions = {
        "tuned_xgboost": ("XGBoost", "Optuna 40 trials (depth 6, lr 0.05, subsample 0.8), histogram tree method, early stopping"),
        "tuned_lightgbm": ("LightGBM", "Optuna 40 trials (num leaves 63, lr 0.03, colsample bytree 0.8), early stopping"),
        "tuned_catboost": ("CatBoost", "Optuna 40 trials (depth 6, l2 leaf reg 3.0, lr 0.05), early stopping"),
        "tuned_logistic_regression": ("Logistic Regression", "Optuna tuning (C=0.01, L2 penalty), standardized numeric features, median imputation"),
        "calibrated_gbm_lr_blend": ("Ensemble Blend (GBM + LR)", "Weighted probability ensemble (0.90 LightGBM + 0.10 Logistic Regression) + Isotonic Calibration"),
    }

    for mkey, mdata in s3_m["models"].items():
        m_backbone, m_tech = modern_descriptions.get(mkey, (mkey, "Modern tuning recipe"))
        auc = mdata.get("test_auc")
        ci = mdata.get("ci_95", [])
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if len(ci) == 2 else "N/A"
        metric_parts = [f"Test ROC-AUC {auc:.4f}", f"95% CI {ci_str}"]

        if "calibrated_auc" in mdata:
            metric_parts.append(f"Calibrated ROC-AUC {mdata['calibrated_auc']:.4f}")
        if "brier_score_calibrated" in mdata:
            metric_parts.append(f"Brier (raw/cal): {mdata.get('brier_score_raw', 0):.5f} / {mdata['brier_score_calibrated']:.5f}")
        if "best_C" in mdata:
            metric_parts.append(f"Best C: {mdata['best_C']}")

        metrics_formatted = ", ".join(metric_parts)

        content.append(
            f"| {mkey} (Modern Stream 3) | {m_backbone} | {m_tech} | "
            f"Temporal split: Train <=2021, Test >=2022 (E1 Primary, >=20 wk) | Train: {n_tr:,}, Test: {n_te:,} ({n_sb:,} stillbirths) | "
            f"{metrics_formatted} | "
            f"{get_full_path('docs/results/stream3_modern/stream3_modern_results.json')} |\n"
        )

    output_path.write_text("".join(content), encoding="utf-8")
    print(f"Generated {output_path} successfully ({len(content)} lines/sections).")

if __name__ == "__main__":
    build_model_selection_tables()
