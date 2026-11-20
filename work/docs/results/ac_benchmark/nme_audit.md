# Landmark NME Formula and Normalization Audit

## 1. Codebase Citations
- `ac_evaluator`: [`src/fetal/eval/ac_evaluator.py#L31-L38`](file:///C:\fetal-github-clone\work/src/fetal/eval/ac_evaluator.py)
- `ensemble_evaluator`: [`src/fetal/eval/ensemble_evaluator.py#L33-L36`](file:///C:\fetal-github-clone\work/src/fetal/eval/ensemble_evaluator.py)
- `head_evaluator`: [`src/fetal/eval/head_evaluator.py#L67-L83`](file:///C:\fetal-github-clone\work/src/fetal/eval/head_evaluator.py)
- `femur_evaluator`: [`src/fetal/eval/femur_evaluator.py#L131-L147`](file:///C:\fetal-github-clone\work/src/fetal/eval/femur_evaluator.py)

## 2. Published Paper Citation
- **Paper**: A multi-centre, multi-device benchmark dataset for landmark-based comprehensive fetal biometry (arXiv:2512.16710)
- **Table**: Table 2
- **Caption**: "measurement error normalised by inter-landmark distance"

## 3. Comparison of Formulations

| Dataset / Model | N | Reported NME (Def i) | Def (i) TAD NME | Def (i) APAD NME | Def (ii) TAD Caliper Rel Err | Def (ii) APAD Caliper Rel Err | Def (ii) Avg Caliper Rel Err |
|---|---|---|---|---|---|---|---|
| `v3_full_single_test_147` | 147 | **0.0731 +- 0.0532** | — | — | 0.0234 +- 0.019 | 0.0259 +- 0.0222 | **0.0247 +- 0.0156** |
| `ens_4seed_test_147` | 147 | **0.0751 +- 0.0581** | 0.0749 +- 0.0597 | 0.0753 +- 0.062 | 0.025 +- 0.0207 | 0.0254 +- 0.0233 | **0.0252 +- 0.017** |
| `ac_baseline_s42_oof_642` | 642 | **0.0808 +- 0.0647** | — | — | 0.0282 +- 0.0363 | 0.0341 +- 0.0498 | **0.0311 +- 0.032** |

## 4. Key Findings

1. **Reported Numbers**: Our reported numbers (0.0731 on v3_full test and 0.0808 on OOF 642) correspond strictly to Definition (i): Landmark Euclidean Coordinate Error normalized by GT inter-landmark distance. Specifically: 0.5 * (mean(||p_tad - g_tad||) / L_tad + mean(||p_apad - g_apad||) / L_apad).
2. **Discrepancy Resolution**: Under Definition (ii) (|measured length error| / GT length), caliper relative errors are 0.0234 +- 0.0190 (TAD) and 0.0259 +- 0.0222 (APAD) on test, averaging 0.0247. When endpoints deviate in opposite perpendicular directions, their coordinate distance errors add up under Euclidean vector distance (giving ~0.073 - 0.080), but caliper length errors largely cancel, yielding relative caliper errors around 0.016 - 0.025. arXiv:2512.16710 Table 2 presents landmark NME around ~0.07-0.08 for BiometryNet, but relative length errors in other tables/benchmarks fall in the 0.0163 - 0.0242 regime.
3. **Comparability**: Our reported NME (0.0731 test, 0.0808 OOF) is directly comparable to BiometryNet's landmark NME (Table 2 M-C -> M-C: TAD 0.08 +- 0.08, APAD 0.07 +- 0.08). It is NOT directly comparable to scalar relative caliper measurement errors (0.016 - 0.024), which represent 1D length discrepancies rather than 2D coordinate localization.
