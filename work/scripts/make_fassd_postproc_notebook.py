"""Generate notebooks/benchmarks/fassd_postproc.ipynb."""
from __future__ import annotations

import json
from pathlib import Path
import nbformat as nbf

NOTEBOOK_PATH = Path("notebooks/benchmarks/fassd_postproc.ipynb")
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Markdown Title & Intro
cells.append(nbf.v4.new_markdown_cell("""# FASSD Post-Processing Study (Round 2)

This notebook presents the evaluation of candidate post-processing strategies on the saved checkpoints from FASSD Run 1 (`fassd_gray_seed42` and `fassd_rgb_seed42`).

### Experimental Protocol
1. **Inference & Native Resolution**: Checkpoints are run on validation (254 images, 26 patients) and test (271 images, 25 patients) using the native $768 \\times 1024$ resolution (resizing input to $384 \\times 512$, forward pass, sigmoid probability output, bilinear interpolation back to $768 \\times 1024$).
2. **Probability Caching**: Probability maps are cached as float16 `.npz` files outside the repository.
3. **Hyperparameter Tuning (VAL Only)**:
   - Threshold grid: $[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]$
   - Morphology candidates: `none`, `largest` (keep largest connected component), and `min_area` with thresholds $\\{50, 100, 200, 500, 1000, 2000\\}$ pixels.
   - Selected strictly on validation set macro/structure Dice.
4. **Single-Pass Test Evaluation**: Selected settings are applied to the test set **once**.
5. **Statistical Rigor**:
   - Patient-level paired bootstrap (5,000 resamples, seed 42) for $\\Delta \\text{Dice}$ and $\\Delta \\text{HD95}$ with 95% confidence intervals.
   - Explicit accounting of empty ground-truth images and false-positive rates on empty slices.
   - Disambiguation of macro Dice definitions.
"""))

# Cell 2: Code - Imports & Paths
cells.append(nbf.v4.new_code_cell("""import json
from pathlib import Path
import pandas as pd
import numpy as np

RESULTS_JSON = Path("../../docs/results/fassd/postproc/fassd_postproc_results.json")
assert RESULTS_JSON.exists(), f"Results file not found: {RESULTS_JSON}"

with open(RESULTS_JSON, "r", encoding="utf-8") as f:
    results = json.load(f)

print(f"Models evaluated: {list(results.keys())}")
"""))

# Cell 3: Markdown - Selected Hyperparameters
cells.append(nbf.v4.new_markdown_cell("""## 1. Selected Post-Processing Configurations (Tuned on VAL)"""))

# Cell 4: Code - Selected Configs Table
cells.append(nbf.v4.new_code_cell("""config_rows = []
for model_name, m_data in results.items():
    sel = m_data["selected_config"]
    for struct, best in sel.items():
        config_rows.append({
            "Model": model_name,
            "Structure": struct,
            "Selected Threshold": best["threshold"],
            "Morphology Type": best["morph_type"],
            "Min Area (px)": best["min_area"],
            "VAL Dice": f"{best['val_dice']:.4f}",
            "VAL IoU": f"{best['val_iou']:.4f}",
        })

df_configs = pd.DataFrame(config_rows)
df_configs
"""))

# Cell 5: Markdown - Test Performance & Paired Bootstrap CI
cells.append(nbf.v4.new_markdown_cell("""## 2. Test Performance: Baseline vs Post-Processed
The table below compares baseline ($T=0.5$, no morphology) against post-processed metrics on the test set (271 images, 25 patients), along with patient-level paired bootstrap 95% confidence intervals (5,000 resamples, seed 42)."""))

# Cell 6: Code - Test Metrics Table
cells.append(nbf.v4.new_code_cell("""test_rows = []
for model_name, m_data in results.items():
    test_data = m_data["test"]
    base_structs = test_data["baseline"]["by_structure"]
    post_structs = test_data["postproc"]["by_structure"]
    boot_structs = test_data["paired_diffs_5000boot_ci"]["by_structure"]
    
    for struct, base in base_structs.items():
        post = post_structs[struct]
        b_dice = boot_structs[struct]["dice"]
        b_hd = boot_structs[struct]["hd95"]
        
        test_rows.append({
            "Model": model_name,
            "Structure": struct,
            "Base Dice": f"{base['dice_mean']:.4f}",
            "Post Dice": f"{post['dice_mean']:.4f}",
            "Δ Dice (95% CI)": f"{b_dice['mean_diff']:+.4f} [{b_dice['ci_lower_95']:+.4f}, {b_dice['ci_upper_95']:+.4f}]",
            "Base HD95": f"{base['hd95_mean']:.2f}",
            "Post HD95": f"{post['hd95_mean']:.2f}",
            "Δ HD95 (95% CI)": f"{b_hd['mean_diff']:+.2f} [{b_hd['ci_lower_95']:+.2f}, {b_hd['ci_upper_95']:+.2f}]",
        })

df_test = pd.DataFrame(test_rows)
df_test
"""))

# Cell 7: Markdown - Empty Ground Truth Analysis
cells.append(nbf.v4.new_markdown_cell("""## 3. Empty Ground Truth Handling & False Positive Rates
For structures that are anatomically absent in certain scan slices ($|GT| = 0$), we report:
- Total number of empty GT slices in the test set.
- Detection count (slices where the model correctly predicts 0 pixels).
- Non-empty prediction rate on empty GT (slices with false positive predictions)."""))

# Cell 8: Code - Empty GT Table
cells.append(nbf.v4.new_code_cell("""empty_rows = []
for model_name, m_data in results.items():
    test_data = m_data["test"]
    base_structs = test_data["baseline"]["by_structure"]
    post_structs = test_data["postproc"]["by_structure"]
    for struct, base in base_structs.items():
        post = post_structs[struct]
        empty_rows.append({
            "Model": model_name,
            "Structure": struct,
            "Empty GT Images": base["n_empty_gt"],
            "Base Correct Empty Rate": f"{base['empty_gt_detection_rate']*100:.1f}%",
            "Base Non-Empty Pred Rate": f"{base['non_empty_pred_rate_on_empty_gt']*100:.1f}%",
            "Post Correct Empty Rate": f"{post['empty_gt_detection_rate']*100:.1f}%",
            "Post Non-Empty Pred Rate": f"{post['non_empty_pred_rate_on_empty_gt']*100:.1f}%",
        })

df_empty = pd.DataFrame(empty_rows)
df_empty
"""))

# Cell 9: Markdown - Macro Dice Disambiguation
cells.append(nbf.v4.new_markdown_cell("""## 4. Macro Dice Metric Disambiguation
We clarify the distinct formulations of Macro Dice:
1. **Per-Image Macro Dice**: Macro-average across non-empty structures present on each image.
2. **Image-Level Mean Macro Dice**: Mean of per-image macro Dice values across all test images.
3. **Structure-Level Macro Dice**: Unweighted arithmetic mean of individual structure-level Dice means.
4. **Patient-Level Macro Dice**: Macro-averaged across images belonging to each patient, providing the basis for patient-level paired bootstrap resampling.
"""))

# Cell 10: Code - Summary Print
cells.append(nbf.v4.new_code_cell(r"""print("=== SUMMARY METRICS ===")
for model_name, m_data in results.items():
    test_data = m_data["test"]
    b_eval = test_data["baseline"]
    p_eval = test_data["postproc"]
    m_dice_diff = test_data["paired_diffs_5000boot_ci"]["macro_dice"]
    m_hd_diff = test_data["paired_diffs_5000boot_ci"]["macro_hd95"]
    print(f"\n{model_name}:")
    print(f"  Structure-Mean Macro Dice:  Base = {b_eval['structure_macro_dice_mean']:.4f} -> Post = {p_eval['structure_macro_dice_mean']:.4f}")
    print(f"  Image-Level Macro Dice:     Base = {b_eval['image_level_macro_dice_mean']:.4f} -> Post = {p_eval['image_level_macro_dice_mean']:.4f}")
    print(f"  Patient-Level Macro Dice:   Base = {b_eval['patient_level_macro_dice_mean']:.4f} -> Post = {p_eval['patient_level_macro_dice_mean']:.4f}")
    print(f"  Paired Δ Dice (5k boot CI): {m_dice_diff['mean_diff']:+.4f} [{m_dice_diff['ci_lower_95']:+.4f}, {m_dice_diff['ci_upper_95']:+.4f}]")
    print(f"  Patient-Level Macro HD95:   Base = {b_eval['patient_level_macro_hd95_mean']:.2f} px -> Post = {p_eval['patient_level_macro_hd95_mean']:.2f} px")
    print(f"  Paired Δ HD95 (5k boot CI): {m_hd_diff['mean_diff']:+.2f} px [{m_hd_diff['ci_lower_95']:+.2f}, {m_hd_diff['ci_upper_95']:+.2f}]")
"""))

nb.cells = cells
with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook generated successfully at {NOTEBOOK_PATH}")
