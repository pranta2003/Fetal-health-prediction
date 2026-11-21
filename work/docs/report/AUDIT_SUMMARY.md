# Audit Summary

## 1. REPORT_NUMBERS
- Total rows: 117 across 16 files.
- Breakdown: AC: 44, HC18: 38, tabular: 26, head: 4, femur: 3, FASSD: 2.
- Verification script (scripts/57_verify_report_numbers.py): PASS (117/117 verified).
- Unit tests (tests/test_report_numbers.py): 1 passed (100%).

## 2. NME Audit
- AC formula (src/fetal/eval/ac_evaluator.py#L31-L38, ensemble_evaluator.py#L33-L36):
  0.5 * ( ||d_TAD|| / L_TAD_gt + ||d_APAD|| / L_APAD_gt )
- Femur formula (src/fetal/eval/femur_evaluator.py#L131-L147): ||d_FL|| / L_FL_gt
- Head formula (src/fetal/eval/head_evaluator.py#L67-L83): 0.5 * ( ||d_BPD|| / L_BPD_gt + ||d_OFD|| / L_OFD_gt )
- Test 147 (single v3_full): Def (i) landmark NME = 0.0731 +- 0.0532; Def (ii) length error: TAD = 0.0234 +- 0.0190, APAD = 0.0259 +- 0.0222 (avg = 0.0247 +- 0.0156).
- Test 147 (4-seed ens): Def (i) landmark NME: TAD = 0.0749 +- 0.0597, APAD = 0.0753 +- 0.0620 (combined = 0.0751 +- 0.0581); Def (ii) length error: TAD = 0.0250 +- 0.0207, APAD = 0.0254 +- 0.0233 (avg = 0.0252 +- 0.0170).
- OOF 642 (baseline s42): Def (i) landmark NME = 0.0808 +- 0.0647; Def (ii) length error: TAD = 0.0282 +- 0.0363, APAD = 0.0341 +- 0.0498 (avg = 0.0311 +- 0.0320).
- Discrepancy reason: 0.0731 / 0.0808 measure 2D landmark Euclidean coordinate vector error normalized by inter-landmark distance; 0.0163 / 0.0242 measure 1D scalar caliper length error (|measured - true| / true). Orthogonal endpoint displacements accumulate in 2D Euclidean distance (~0.07-0.08) but cancel in 1D scalar lengths (~0.016-0.025).

## 3. Cohort Count Audit
- Abdomen (823 published vs 789 used; 34 missing): 31 UCL images missing px_to_mm calibration in UCL metadata (UCL/annotations/Abdomen.csv); 3 images (125_3AC, 125_4AC, 125_5AC) excluded in src/fetal/data/ac_dataset.py due to corrupted acquisition geometry.
- Femur: raw disk = 895 (FP 760 + UCL 135) matches paper Table 1 (895); 28 UCL uncalibrated -> 867 calibrated.
- Head: raw disk = 1796 (FP 1637 + UCL 159) matches paper Table 1 (1796); 33 UCL uncalibrated -> 1763 calibrated.

## 4. HC18 Ladder Reference Audit
| Row | Split | N | Reference Used | Original MAE (mm) | Against CSV HC MAE (mm) | CSV Bias (mm) |
|---|---|---|---|---|---|---|
| Row A | Fold 1 | 206 | Official CSV HC | 10.6728 | 10.6728 | +5.3052 |
| Row B | Fold 1 | 206 | Official CSV HC | 11.5964 | 11.5964 | +5.6149 |
| Row C | Fold 1 | 206 | Official CSV HC | 10.5104 | 10.5104 | +3.5311 |
| Row D (f1) | Fold 1 | 206 | Official CSV HC | 1.9246 | 1.9246 | -0.2607 |
| Row D (5f) | 5-Fold | 999 | Official CSV HC | 2.2688 | 2.2688 | +0.5435 |
| Row D (+TTA) | 5-Fold | 999 | Official CSV HC | 2.2323 | 2.2323 | +0.5395 |
| Row E (OOF) | 5-Fold | 999 | Mask-Ellipse HC | 2.3019 | 1.9933 | +0.2171 |
| Row D run1 | Fold 1 | 206 | Mask-Ellipse HC | 1.9360 | Not preserved on disk | N/A |

## 5. Notebook Audit
- Notebook counts: PROOF: 17, RUNNER: 5, DRAFT: 7 (Total: 29).
- PROOF notebook paths:
  1. notebooks/benchmarks/ac_results.ipynb
  2. notebooks/benchmarks/benchmark_summary.ipynb
  3. notebooks/benchmarks/fassd_audit.ipynb
  4. notebooks/benchmarks/fassd_ensemble.ipynb
  5. notebooks/benchmarks/fassd_postproc.ipynb
  6. notebooks/benchmarks/femur_ablation.ipynb
  7. notebooks/benchmarks/femur_data.ipynb
  8. notebooks/benchmarks/femur_round2.ipynb
  9. notebooks/benchmarks/hadlock_step2.ipynb
  10. notebooks/benchmarks/hc18_ladder.ipynb
  11. notebooks/benchmarks/hc18_official.ipynb
  12. notebooks/benchmarks/hc18_oof_calibration.ipynb
  13. notebooks/benchmarks/head_external.ipynb
  14. notebooks/benchmarks/head_landmark.ipynb
  15. notebooks/benchmarks/stream3_bench.ipynb
  16. notebooks/history/ac_lab_pc/ac_lab_pc_history.ipynb
  17. notebooks/history/Capstone_fetalDataset_history.ipynb

## 6. GAPS List
- GAP-01: HC18 nnU-Net fold 0 test set inference (kernel hc18-nnunet-test-f0 running).
- GAP-02: HC18 full 5-fold nnU-Net (holds for quota refresh Sat 2026-11-21 06:00 Dhaka).
- GAP-03: AC + FASSD multi-task model (ac-fassd-s42 queued behind test inference).
- GAP-04: Modern tabular GBDT campaign (stream3-modern-s1 Optuna CPU).
- GAP-05: Landmark NME vs caliper relative length metric [CLOSED via nme_audit.json].
- GAP-06: Dataset count reconciliation across all anatomies [CLOSED via dataset_counts_audit.json].
- GAP-07: HC18 ladder ground-truth reference reconciliation [CLOSED via hc18_ladder_ref_audit.json].
