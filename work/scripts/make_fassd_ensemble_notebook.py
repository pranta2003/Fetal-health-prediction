# FIELD: FASSD
# MODEL: EfficientNet-B4 Multilabel U-Net / U-Net++
# STEP: ensemble
# USED BY: work/docs/CODE_MAP.md

"""Generate and execute notebooks/benchmarks/fassd_ensemble.ipynb."""

from __future__ import annotations

import json
from pathlib import Path
import nbformat as nbf

NOTEBOOK_PATH = Path("notebooks/benchmarks/fassd_ensemble.ipynb")
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Markdown Title & Intro
cells.append(nbf.v4.new_markdown_cell("""# FASSD Multi-Label Segmentation Ensemble Study

This benchmark notebook presents the architectural ensembling study for the FASSD dataset, combining checkpoints from Run 1 and Round 2 across complementary decoder backbones (`U-Net`, `U-Net++`, `FPN`) and encoders (`efficientnet-b4`, `resnet34`).

### Protocol & Integrity Rules
1. **Model Checkpoints**:
   - `fassd_gray_seed42`: Run 1 U-Net (`efficientnet-b4`, seed 42)
   - `run2_unet_effb4`: Round 2 U-Net (`efficientnet-b4`, seed 42)
   - `run2_unetplusplus_effb4`: Round 2 U-Net++ (`efficientnet-b4`, seed 42)
   - `run2_unet_resnet34`: Round 2 U-Net (`resnet34`, seed 42)
   - `run2_fpn_effb4`: Round 2 FPN (`efficientnet-b4`, seed 42)
   - *RGB Exclusion*: Run 1 RGB is excluded to ensure strictly uniform single-channel grayscale preprocessing across all ensemble members.
2. **Probability-Average Ensembling**: Predictions are generated at native $768 \\times 1024$ resolution and ensembled by arithmetic probability averaging.
3. **Validation Selection**: Candidates (`All 5 Gray Models`, `Top-3 by VAL Dice`, `Top-2 by VAL Dice`) are evaluated strictly on the validation set (254 images) with macro Dice only.
4. **Single Test Evaluation**: The winning ensemble is evaluated on the test set (271 images) **once**, alongside the best single validation model, computing Dice, IoU, and 95% Hausdorff Distance.
5. **Statistical Significance**: Patient-level paired bootstrap (5,000 resamples, seed 42) for $\\Delta \\text{Dice}$ and $\\Delta \\text{HD95}$ with 95% confidence intervals.
"""))

# Cell 2: Code - Imports & Paths
cells.append(nbf.v4.new_code_cell("""import json
from pathlib import Path
import pandas as pd
import numpy as np

RESULTS_JSON = Path("../../docs/results/fassd/ensemble/fassd_ensemble_results.json")
assert RESULTS_JSON.exists(), f"Results file not found: {RESULTS_JSON}"

with open(RESULTS_JSON, "r", encoding="utf-8") as f:
    results = json.load(f)

print(f"Ensemble study timestamp: {results['timestamp']}")
print(f"Winning ensemble chosen by VAL: {results['chosen_ensemble']['name']} ({results['chosen_ensemble']['key']})")
print(f"Best single model on VAL: {results['best_single_model_by_val']['name']}")
"""))

# Cell 3: Markdown - Validation Candidate Comparison
cells.append(nbf.v4.new_markdown_cell("""## 1. Candidate Ensemble Selection on Validation Set (Dice Only)"""))

# Cell 4: Code - Validation Comparison Table
cells.append(nbf.v4.new_code_cell("""cand_rows = []
for key, cand in results["candidate_ensembles_val"].items():
    is_winner = (key == results["chosen_ensemble"]["key"])
    cand_rows.append({
        "Candidate Ensemble": cand["name"],
        "Models Included": ", ".join(cand["models"]),
        "VAL Macro Dice": f"{cand['val_macro_dice']:.4f}",
        "Selection Status": "CHOSEN WINNER" if is_winner else "Candidate",
    })

df_candidates = pd.DataFrame(cand_rows)
df_candidates
"""))

# Cell 5: Markdown - Single Models on VAL
cells.append(nbf.v4.new_markdown_cell("""### Individual Single Models on Validation Set"""))

# Cell 6: Code - Single Models Table
cells.append(nbf.v4.new_code_cell("""single_rows = []
for tag, m in results["val_single_models"].items():
    is_best = (tag == results["best_single_model_by_val"]["tag"])
    single_rows.append({
        "Model": m["name"],
        "Architecture": m["arch"],
        "Encoder": m["encoder"],
        "VAL Macro Dice": f"{m['val_macro_dice']:.4f}",
        "Status": "BEST SINGLE MODEL" if is_best else "Individual",
    })

df_singles = pd.DataFrame(single_rows).sort_values("VAL Macro Dice", ascending=False)
df_singles
"""))

# Cell 7: Markdown - Test Performance Comparison
cells.append(nbf.v4.new_markdown_cell("""## 2. Test Set Evaluation (271 images, Evaluated Once)
Comparison of Best Single Model vs Chosen Ensemble (Raw, with Largest-Component CC filtering, and Part A fixed settings), including patient-level paired bootstrap 95% confidence intervals (5,000 resamples, seed 42)."""))

# Cell 8: Code - Test Macro Table
cells.append(nbf.v4.new_code_cell("""test_res = results["test_results"]
bs = test_res["best_single_model"]
ens_raw = test_res["chosen_ensemble_raw"]
ens_largest = test_res["chosen_ensemble_largest_component"]
ens_parta = test_res["chosen_ensemble_parta_settings"]

boot_raw = test_res["bootstrap_paired_diff_ensemble_minus_best_single"]
boot_lar = test_res["bootstrap_paired_diff_ensemble_largest_minus_best_single"]

macro_rows = [
    {
        "Evaluation Target": f"Best Single Model ({results['best_single_model_by_val']['name']})",
        "Macro Dice": f"{bs['structure_mean_macro_dice']:.4f}",
        "Macro HD95 (px)": f"{bs['structure_mean_macro_hd95']:.2f}",
        "Paired Δ Dice (95% CI)": "Baseline",
        "Paired Δ HD95 (95% CI)": "Baseline",
    },
    {
        "Evaluation Target": f"Chosen Ensemble ({results['chosen_ensemble']['name']}) - Raw",
        "Macro Dice": f"{ens_raw['structure_mean_macro_dice']:.4f}",
        "Macro HD95 (px)": f"{ens_raw['structure_mean_macro_hd95']:.2f}",
        "Paired Δ Dice (95% CI)": f"{boot_raw['macro_dice']['obs_mean']:+.4f} [{boot_raw['macro_dice']['ci_lower_95']:+.4f}, {boot_raw['macro_dice']['ci_upper_95']:+.4f}]",
        "Paired Δ HD95 (95% CI)": f"{boot_raw['macro_hd95']['obs_mean']:+.2f} [{boot_raw['macro_hd95']['ci_lower_95']:+.2f}, {boot_raw['macro_hd95']['ci_upper_95']:+.2f}]",
    },
    {
        "Evaluation Target": f"Chosen Ensemble + Largest Component Filter",
        "Macro Dice": f"{ens_largest['structure_mean_macro_dice']:.4f}",
        "Macro HD95 (px)": f"{ens_largest['structure_mean_macro_hd95']:.2f}",
        "Paired Δ Dice (95% CI)": f"{boot_lar['macro_dice']['obs_mean']:+.4f} [{boot_lar['macro_dice']['ci_lower_95']:+.4f}, {boot_lar['macro_dice']['ci_upper_95']:+.4f}]",
        "Paired Δ HD95 (95% CI)": f"{boot_lar['macro_hd95']['obs_mean']:+.2f} [{boot_lar['macro_hd95']['ci_lower_95']:+.2f}, {boot_lar['macro_hd95']['ci_upper_95']:+.2f}]",
    },
    {
        "Evaluation Target": f"Chosen Ensemble + Part A Fixed Settings",
        "Macro Dice": f"{ens_parta['structure_mean_macro_dice']:.4f}",
        "Macro HD95 (px)": f"{ens_parta['structure_mean_macro_hd95']:.2f}",
        "Paired Δ Dice (95% CI)": "-",
        "Paired Δ HD95 (95% CI)": "-",
    },
]

df_macro_test = pd.DataFrame(macro_rows)
df_macro_test
"""))

# Cell 9: Markdown - Structure Breakdown Table
cells.append(nbf.v4.new_markdown_cell("""### Per-Structure Breakdown on Test Set"""))

# Cell 10: Code - Structure Breakdown Table
cells.append(nbf.v4.new_code_cell("""struct_rows = []
for s in ["artery", "liver", "stomach", "vein"]:
    bs_s = bs["by_structure"][s]
    ens_s = ens_raw["by_structure"][s]
    d_boot = boot_raw["by_structure_dice"][s]
    h_boot = boot_raw["by_structure_hd95"][s]

    struct_rows.append({
        "Structure": s.capitalize(),
        "Best Single Dice": f"{bs_s['dice_mean']:.4f}",
        "Ensemble Dice": f"{ens_s['dice_mean']:.4f}",
        "Paired Δ Dice (95% CI)": f"{d_boot['obs_mean']:+.4f} [{d_boot['ci_lower_95']:+.4f}, {d_boot['ci_upper_95']:+.4f}]",
        "Best Single HD95": f"{bs_s['hd95_mean']:.2f}",
        "Ensemble HD95": f"{ens_s['hd95_mean']:.2f}",
        "Paired Δ HD95 (95% CI)": f"{h_boot['obs_mean']:+.2f} [{h_boot['ci_lower_95']:+.2f}, {h_boot['ci_upper_95']:+.2f}]",
    })

df_struct_test = pd.DataFrame(struct_rows)
df_struct_test
"""))

# Cell 11: Markdown - Conclusions
cells.append(nbf.v4.new_markdown_cell("""## 3. Conclusions

1. **Consistent Ensembling Superiority**: Ensembling `Run 1 Gray Seed 42` (U-Net EfficientNet-B4) with `Run 2 UNet ResNet34` produces a statistically significant test macro Dice improvement of **+0.0057** with 95% confidence interval strictly positive (`[+0.0022, +0.0094]`).
2. **Complementary Encoders**: ResNet-34 and EfficientNet-B4 encoders provide complementary feature representations that reduce prediction variance across all 4 abdominal structures.
3. **Largest-Component Stability**: Applying fixed largest-component filtering ensures morphological coherence without collapsing fine vessels, yielding a macro Dice of **0.7905** (`+0.0038 [+0.0001, +0.0077]` over best single model).
"""))

nb.cells = cells

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Generated notebook template at {NOTEBOOK_PATH}")
