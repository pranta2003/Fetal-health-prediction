# FIELD: FASSD
# MODEL: EfficientNet-B4 Multilabel U-Net / U-Net++
# STEP: ensemble
# USED BY: work/docs/CODE_MAP.md, work/docs/report/EVIDENCE_MAP.md

"""FASSD Multilabel Segmentation Ensemble Study Runner.

Performs probability-averaging ensembling of available FASSD checkpoints,
evaluates candidate ensembles on VAL (macro Dice only), selects the winning
ensemble, and evaluates on TEST once with patient-level paired bootstrap CIs.
Memory-optimized for lightweight execution.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pandas as pd

from src.fetal.seg.fassd_ensemble import (
    STRUCTURE_NAMES,
    average_probabilities,
    evaluate_ensemble_slice,
    select_candidate_models,
)
from src.fetal.seg.fassd_postproc import run_patient_paired_bootstrap


def load_ground_truth_masks(
    df_subset: pd.DataFrame,
    masks_dir: Path,
) -> dict[str, np.ndarray]:
    """Load bit-flag PNG masks unpacked into (4, H, W) uint8 binary arrays."""
    gt_dict = {}
    for _, row in df_subset.iterrows():
        fname = str(row["filename"])
        stem = Path(fname).stem
        m_file = masks_dir / f"{stem}.png"
        if not m_file.exists():
            m_file = masks_dir / fname
        mask_png = cv2.imread(str(m_file), cv2.IMREAD_UNCHANGED)
        if mask_png is None:
            raise FileNotFoundError(f"Failed to read mask at {m_file}")

        # Bit flags: artery=1, liver=2, stomach=4, vein=8
        h, w = mask_png.shape[:2]
        unpacked = np.zeros((4, h, w), dtype=np.uint8)
        unpacked[0] = ((mask_png & 1) > 0).astype(np.uint8)
        unpacked[1] = ((mask_png & 2) > 0).astype(np.uint8)
        unpacked[2] = ((mask_png & 4) > 0).astype(np.uint8)
        unpacked[3] = ((mask_png & 8) > 0).astype(np.uint8)
        gt_dict[stem] = unpacked
    return gt_dict


def load_single_prob(cache_root: Path, model_tag: str, split: str, stem: str) -> np.ndarray:
    """Load a single float16 probability map from disk."""
    p_file = cache_root / model_tag / split / f"{stem}.npz"
    if not p_file.exists():
        raise FileNotFoundError(f"Missing cached probabilities at {p_file}")
    with np.load(p_file) as data:
        return data["prob"]


def load_model_probs_split(
    cache_root: Path,
    model_tag: str,
    split: str,
    stems: list[str],
) -> dict[str, np.ndarray]:
    """Load all probability maps for one model in a split."""
    probs = {}
    for s in stems:
        probs[s] = load_single_prob(cache_root, model_tag, split, s)
    return probs


def build_ensemble_probs_split(
    cache_root: Path,
    model_tags: list[str],
    split: str,
    stems: list[str],
) -> dict[str, np.ndarray]:
    """Construct averaged probabilities for an ensemble on-the-fly per image."""
    ens_probs = {}
    for s in stems:
        maps = [load_single_prob(cache_root, tag, split, s) for tag in model_tags]
        ens_probs[s] = average_probabilities(maps)
    return ens_probs


def main() -> None:
    parser = argparse.ArgumentParser(description="FASSD Ensemble Study")
    parser.add_argument("--data-root", type=str, default=r"E:\BUBT-CSE-4th yr\Capstone\data\kaggle_fassd_v2")
    parser.add_argument("--split-csv", type=str, default="docs/results/fassd/split_v2.csv")
    parser.add_argument("--prob-cache-dir", type=str, default=r"E:\BUBT-CSE-4th yr\Capstone\data\fassd_pred_prob")
    parser.add_argument("--output-dir", type=str, default="docs/results/fassd/ensemble")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    masks_dir = data_root / "masks"
    split_csv_path = Path(args.split_csv)
    cache_root = Path(args.prob_cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("FASSD MULTI-LABEL SEGMENTATION ENSEMBLE STUDY")
    print("=================================================================")

    # Load split metadata
    df = pd.read_csv(split_csv_path)
    df_val = df[df["split"] == "val"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)
    val_stems = [Path(f).stem for f in df_val["filename"]]
    test_stems = [Path(f).stem for f in df_test["filename"]]
    print(f"Dataset splits loaded: VAL = {len(df_val)} images, TEST = {len(df_test)} images.")

    # Model specifications
    models_spec = [
        {"tag": "fassd_gray_seed42", "name": "Run 1 Gray Seed 42 (UNet EffB4)", "arch": "unet", "encoder": "efficientnet-b4"},
        {"tag": "run2_unet_effb4", "name": "Run 2 UNet EffB4", "arch": "unet", "encoder": "efficientnet-b4"},
        {"tag": "run2_unetplusplus_effb4", "name": "Run 2 UNet++ EffB4", "arch": "unetplusplus", "encoder": "efficientnet-b4"},
        {"tag": "run2_unet_resnet34", "name": "Run 2 UNet ResNet34", "arch": "unet", "encoder": "resnet34"},
        {"tag": "run2_fpn_effb4", "name": "Run 2 FPN EffB4", "arch": "fpn", "encoder": "efficientnet-b4"},
    ]

    # Verify probability maps are present
    print("\n[Step 1] Verifying cached probability maps...")
    for m in models_spec:
        tag = m["tag"]
        v_dir = cache_root / tag / "val"
        t_dir = cache_root / tag / "test"
        v_count = len(list(v_dir.glob("*.npz")))
        t_count = len(list(t_dir.glob("*.npz")))
        assert v_count == 254, f"Expected 254 val npz in {v_dir}, found {v_count}"
        assert t_count == 271, f"Expected 271 test npz in {t_dir}, found {t_count}"
        print(f"  {m['name']:<35} -> OK (val={v_count}, test={t_count})")

    # Load ground truth masks
    print("\n[Step 2] Loading ground-truth masks...")
    val_gts = load_ground_truth_masks(df_val, masks_dir)
    test_gts = load_ground_truth_masks(df_test, masks_dir)

    # Evaluate each single model on VAL (macro Dice only)
    print("\n[Step 3] Evaluating individual models on VAL (Dice only)...")
    val_macro_dices: dict[str, float] = {}
    val_metrics_by_model: dict[str, dict[str, Any]] = {}

    for m in models_spec:
        tag = m["tag"]
        probs = load_model_probs_split(cache_root, tag, "val", val_stems)
        res = evaluate_ensemble_slice(
            probs_dict=probs,
            gts_dict=val_gts,
            df_subset=df_val,
            threshold=0.5,
            morph_type="none",
            min_area=0,
            compute_hd=False,
        )
        del probs
        gc.collect()

        val_metrics_by_model[tag] = res
        val_macro_dices[tag] = res["structure_mean_macro_dice"]
        print(f"  {m['name']:<35} VAL Macro Dice: {res['structure_mean_macro_dice']:.4f} (Per-Image Mean: {res['per_image_macro_dice_mean']:.4f})")

    # Determine best single model by VAL macro Dice
    best_single_tag = max(val_macro_dices.keys(), key=lambda k: val_macro_dices[k])
    best_single_name = next(m["name"] for m in models_spec if m["tag"] == best_single_tag)
    print(f"\n--> Best Single Model on VAL: {best_single_name} ({best_single_tag}) with VAL Macro Dice = {val_macro_dices[best_single_tag]:.4f}")

    # Form candidate ensembles
    candidate_definitions = {
        "all_gray": {
            "name": "All 5 Gray Models",
            "models": list(val_macro_dices.keys()),
        },
        "top3_val": {
            "name": "Top-3 by VAL Dice",
            "models": select_candidate_models(val_macro_dices, "top3"),
        },
        "top2_val": {
            "name": "Top-2 by VAL Dice",
            "models": select_candidate_models(val_macro_dices, "top2"),
        },
    }

    print("\n[Step 4] Evaluating Candidate Ensembles on VAL (Dice only)...")
    candidate_val_results: dict[str, dict[str, Any]] = {}
    candidate_val_dices: dict[str, float] = {}

    for cand_key, cand_info in candidate_definitions.items():
        cand_models = cand_info["models"]
        ens_probs = build_ensemble_probs_split(cache_root, cand_models, "val", val_stems)
        res = evaluate_ensemble_slice(
            probs_dict=ens_probs,
            gts_dict=val_gts,
            df_subset=df_val,
            threshold=0.5,
            morph_type="none",
            min_area=0,
            compute_hd=False,
        )
        del ens_probs
        gc.collect()

        candidate_val_results[cand_key] = res
        candidate_val_dices[cand_key] = res["structure_mean_macro_dice"]
        models_str = ", ".join(cand_models)
        print(f"  Candidate: {cand_info['name']:<25} | VAL Macro Dice: {res['structure_mean_macro_dice']:.4f} | Models: [{models_str}]")

    # Select winning ensemble strictly by VAL macro Dice
    chosen_cand_key = max(candidate_val_dices.keys(), key=lambda k: candidate_val_dices[k])
    chosen_cand = candidate_definitions[chosen_cand_key]
    print(f"\n--> CHOSEN ENSEMBLE BY VAL: {chosen_cand['name']} ({chosen_cand_key}) with VAL Macro Dice = {candidate_val_dices[chosen_cand_key]:.4f}")

    # Evaluate on TEST ONCE
    print("\n[Step 5] Evaluating on TEST ONCE for chosen ensemble and best single model...")
    chosen_models = chosen_cand["models"]

    # 1. Best Single Model on TEST
    print(f"  Evaluating Best Single Model ({best_single_name}) on TEST with HD95...")
    best_single_test_probs = load_model_probs_split(cache_root, best_single_tag, "test", test_stems)
    best_single_test_res = evaluate_ensemble_slice(
        probs_dict=best_single_test_probs,
        gts_dict=test_gts,
        df_subset=df_test,
        threshold=0.5,
        morph_type="none",
        min_area=0,
        compute_hd=True,
    )
    # We keep best_single_test_res, but can discard probs if needed
    del best_single_test_probs
    gc.collect()

    # 2. Chosen Ensemble on TEST
    print(f"  Building and Evaluating Chosen Ensemble ({chosen_cand['name']}) on TEST with HD95...")
    chosen_ens_test_probs = build_ensemble_probs_split(cache_root, chosen_models, "test", test_stems)

    chosen_ens_test_res = evaluate_ensemble_slice(
        probs_dict=chosen_ens_test_probs,
        gts_dict=test_gts,
        df_subset=df_test,
        threshold=0.5,
        morph_type="none",
        min_area=0,
        compute_hd=True,
    )

    # 3. Chosen Ensemble on TEST with Fixed Largest Component Post-Processing
    print("  Evaluating Chosen Ensemble with Largest Component (fixed threshold 0.5, largest cc)...")
    chosen_ens_largest_test_res = evaluate_ensemble_slice(
        probs_dict=chosen_ens_test_probs,
        gts_dict=test_gts,
        df_subset=df_test,
        threshold=0.5,
        morph_type="largest",
        min_area=0,
        compute_hd=True,
    )

    # 4. Fixed postproc settings from Part A
    fixed_postproc_settings = {
        "threshold": {"artery": 0.60, "liver": 0.30, "stomach": 0.35, "vein": 0.50},
        "morph_type": {"artery": "largest", "liver": "none", "stomach": "min_area", "vein": "largest"},
        "min_area": {"artery": 0, "liver": 0, "stomach": 200, "vein": 0},
    }
    print("  Evaluating Chosen Ensemble with Part A fixed postproc settings...")
    chosen_ens_parta_test_res = evaluate_ensemble_slice(
        probs_dict=chosen_ens_test_probs,
        gts_dict=test_gts,
        df_subset=df_test,
        threshold=fixed_postproc_settings["threshold"],
        morph_type=fixed_postproc_settings["morph_type"],
        min_area=fixed_postproc_settings["min_area"],
        compute_hd=True,
    )
    del chosen_ens_test_probs
    gc.collect()

    # Compute Patient-Level Paired Bootstrap (5000 resamples, seed 42)
    print("\n[Step 6] Running Patient-Level Paired Bootstrap (5000 resamples, seed 42)...")
    patients = sorted(list(best_single_test_res["patient_mean_macro_dices"].keys()))

    bootstrap_macro_dice_diff = {
        p: chosen_ens_test_res["patient_mean_macro_dices"][p] - best_single_test_res["patient_mean_macro_dices"][p]
        for p in patients
    }
    boot_macro_dice = run_patient_paired_bootstrap(bootstrap_macro_dice_diff, n_bootstraps=args.n_bootstrap, seed=args.seed)

    bootstrap_macro_hd95_diff = {
        p: chosen_ens_test_res["patient_mean_macro_hd95s"][p] - best_single_test_res["patient_mean_macro_hd95s"][p]
        for p in patients
    }
    boot_macro_hd95 = run_patient_paired_bootstrap(bootstrap_macro_hd95_diff, n_bootstraps=args.n_bootstrap, seed=args.seed)

    # Bootstrap per structure
    boot_struct_dice = {}
    boot_struct_hd95 = {}
    for s in STRUCTURE_NAMES:
        s_p_dice_diff = {
            p: chosen_ens_test_res["patient_mean_dices_by_struct"][s][p] - best_single_test_res["patient_mean_dices_by_struct"][s][p]
            for p in patients
            if p in chosen_ens_test_res["patient_mean_dices_by_struct"][s] and p in best_single_test_res["patient_mean_dices_by_struct"][s]
        }
        boot_struct_dice[s] = run_patient_paired_bootstrap(s_p_dice_diff, n_bootstraps=args.n_bootstrap, seed=args.seed)

        s_p_hd_diff = {
            p: chosen_ens_test_res["patient_mean_hd95s_by_struct"][s][p] - best_single_test_res["patient_mean_hd95s_by_struct"][s][p]
            for p in patients
            if p in chosen_ens_test_res["patient_mean_hd95s_by_struct"][s] and p in best_single_test_res["patient_mean_hd95s_by_struct"][s]
        }
        boot_struct_hd95[s] = run_patient_paired_bootstrap(s_p_hd_diff, n_bootstraps=args.n_bootstrap, seed=args.seed)

    # Bootstrap for Ensemble + Largest vs Best Single Model
    boot_largest_macro_dice_diff = {
        p: chosen_ens_largest_test_res["patient_mean_macro_dices"][p] - best_single_test_res["patient_mean_macro_dices"][p]
        for p in patients
    }
    boot_largest_macro_dice = run_patient_paired_bootstrap(boot_largest_macro_dice_diff, n_bootstraps=args.n_bootstrap, seed=args.seed)

    boot_largest_macro_hd95_diff = {
        p: chosen_ens_largest_test_res["patient_mean_macro_hd95s"][p] - best_single_test_res["patient_mean_macro_hd95s"][p]
        for p in patients
    }
    boot_largest_macro_hd95 = run_patient_paired_bootstrap(boot_largest_macro_hd95_diff, n_bootstraps=args.n_bootstrap, seed=args.seed)

    # Save summary results JSON
    print("\n[Step 7] Writing JSON and Markdown Reports...")
    results_payload = {
        "study": "fassd_ensemble_study",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_evaluated": [m["name"] for m in models_spec],
        "rgb_exclusion_reason": "Run 1 RGB skipped from ensemble to maintain single-channel grayscale preprocessing uniformity.",
        "val_single_models": {
            m["tag"]: {
                "name": m["name"],
                "arch": m["arch"],
                "encoder": m["encoder"],
                "val_macro_dice": val_macro_dices[m["tag"]],
            }
            for m in models_spec
        },
        "best_single_model_by_val": {
            "tag": best_single_tag,
            "name": best_single_name,
            "val_macro_dice": val_macro_dices[best_single_tag],
        },
        "candidate_ensembles_val": {
            k: {
                "name": v["name"],
                "models": v["models"],
                "val_macro_dice": candidate_val_dices[k],
            }
            for k, v in candidate_definitions.items()
        },
        "chosen_ensemble": {
            "key": chosen_cand_key,
            "name": chosen_cand["name"],
            "models": chosen_cand["models"],
            "val_macro_dice": candidate_val_dices[chosen_cand_key],
        },
        "test_results": {
            "best_single_model": {
                "structure_mean_macro_dice": best_single_test_res["structure_mean_macro_dice"],
                "structure_mean_macro_hd95": best_single_test_res["structure_mean_macro_hd95"],
                "by_structure": best_single_test_res["by_structure"],
            },
            "chosen_ensemble_raw": {
                "structure_mean_macro_dice": chosen_ens_test_res["structure_mean_macro_dice"],
                "structure_mean_macro_hd95": chosen_ens_test_res["structure_mean_macro_hd95"],
                "by_structure": chosen_ens_test_res["by_structure"],
            },
            "chosen_ensemble_largest_component": {
                "structure_mean_macro_dice": chosen_ens_largest_test_res["structure_mean_macro_dice"],
                "structure_mean_macro_hd95": chosen_ens_largest_test_res["structure_mean_macro_hd95"],
                "by_structure": chosen_ens_largest_test_res["by_structure"],
            },
            "chosen_ensemble_parta_settings": {
                "structure_mean_macro_dice": chosen_ens_parta_test_res["structure_mean_macro_dice"],
                "structure_mean_macro_hd95": chosen_ens_parta_test_res["structure_mean_macro_hd95"],
                "by_structure": chosen_ens_parta_test_res["by_structure"],
            },
            "bootstrap_paired_diff_ensemble_minus_best_single": {
                "macro_dice": {
                    "obs_mean": boot_macro_dice[0],
                    "ci_lower_95": boot_macro_dice[1],
                    "ci_upper_95": boot_macro_dice[2],
                },
                "macro_hd95": {
                    "obs_mean": boot_macro_hd95[0],
                    "ci_lower_95": boot_macro_hd95[1],
                    "ci_upper_95": boot_macro_hd95[2],
                },
                "by_structure_dice": {
                    s: {
                        "obs_mean": boot_struct_dice[s][0],
                        "ci_lower_95": boot_struct_dice[s][1],
                        "ci_upper_95": boot_struct_dice[s][2],
                    }
                    for s in STRUCTURE_NAMES
                },
                "by_structure_hd95": {
                    s: {
                        "obs_mean": boot_struct_hd95[s][0],
                        "ci_lower_95": boot_struct_hd95[s][1],
                        "ci_upper_95": boot_struct_hd95[s][2],
                    }
                    for s in STRUCTURE_NAMES
                },
            },
            "bootstrap_paired_diff_ensemble_largest_minus_best_single": {
                "macro_dice": {
                    "obs_mean": boot_largest_macro_dice[0],
                    "ci_lower_95": boot_largest_macro_dice[1],
                    "ci_upper_95": boot_largest_macro_dice[2],
                },
                "macro_hd95": {
                    "obs_mean": boot_largest_macro_hd95[0],
                    "ci_lower_95": boot_largest_macro_hd95[1],
                    "ci_upper_95": boot_largest_macro_hd95[2],
                },
            },
        },
    }

    json_path = output_dir / "fassd_ensemble_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Saved JSON results to {json_path}")

    # Generate Markdown Report
    md_lines = [
        "# FASSD Multi-Label Segmentation Ensemble Benchmark Report",
        "",
        "## 1. Study Configuration",
        "",
        "- **Models Evaluated**: 5 Single-Channel Grayscale Models",
        "  1. `fassd_gray_seed42`: Run 1 U-Net (`efficientnet-b4`, seed 42)",
        "  2. `run2_unet_effb4`: Run 2 U-Net (`efficientnet-b4`, seed 42)",
        "  3. `run2_unetplusplus_effb4`: Run 2 U-Net++ (`efficientnet-b4`, seed 42)",
        "  4. `run2_unet_resnet34`: Run 2 U-Net (`resnet34`, seed 42)",
        "  5. `run2_fpn_effb4`: Run 2 FPN (`efficientnet-b4`, seed 42)",
        "- **Modality Uniformity**: Run 1 RGB is excluded to maintain strictly uniform single-channel grayscale preprocessing.",
        f"- **Dataset Split**: `split_v2.csv` (VAL = {len(df_val)} images, TEST = {len(df_test)} images).",
        "- **Ensemble Selection Protocol**: Evaluated on VAL with macro Dice only; TEST evaluated once on the chosen ensemble.",
        "- **Bootstrap Evaluation**: Patient-level paired bootstrap (5,000 resamples, seed 42).",
        "",
        "## 2. Validation Candidate Comparison (Selection by VAL Macro Dice)",
        "",
        "| Candidate Configuration | Models Included | VAL Macro Dice | Per-Image Mean Dice | Status |",
        "|:---|:---|:---:|:---:|:---:|",
    ]

    for k, v in candidate_definitions.items():
        is_chosen = (k == chosen_cand_key)
        status = "**Chosen Winner**" if is_chosen else "Candidate"
        cand_d = candidate_val_results[k]
        md_lines.append(
            f"| **{v['name']}** | {', '.join(v['models'])} | {cand_d['structure_mean_macro_dice']:.4f} | {cand_d['per_image_macro_dice_mean']:.4f} | {status} |"
        )

    md_lines.extend([
        "",
        f"**Chosen Ensemble**: `{chosen_cand['name']}` ({chosen_cand_key}) selected strictly by highest validation macro Dice (`{candidate_val_dices[chosen_cand_key]:.4f}`).",
        f"**Best Single Model on VAL**: `{best_single_name}` ({best_single_tag}) with VAL macro Dice = `{val_macro_dices[best_single_tag]:.4f}`.",
        "",
        "## 3. Test Set Performance (271 images, Evaluated Once)",
        "",
        "### Macro Performance Summary",
        "",
        "| Evaluation Target | Macro Dice | Macro HD95 (px) | Paired $\\Delta$ Dice vs Best Single (95% CI) | Paired $\\Delta$ HD95 vs Best Single (95% CI) |",
        "|:---|:---:|:---:|:---:|:---:|",
        f"| **Best Single Model** (`{best_single_name}`) | {best_single_test_res['structure_mean_macro_dice']:.4f} | {best_single_test_res['structure_mean_macro_hd95']:.2f} | Baseline | Baseline |",
        f"| **Chosen Ensemble** (Raw, threshold 0.5) | {chosen_ens_test_res['structure_mean_macro_dice']:.4f} | {chosen_ens_test_res['structure_mean_macro_hd95']:.2f} | {boot_macro_dice[0]:+.4f} [{boot_macro_dice[1]:+.4f}, {boot_macro_dice[2]:+.4f}] | {boot_macro_hd95[0]:+.2f} [{boot_macro_hd95[1]:+.2f}, {boot_macro_hd95[2]:+.2f}] |",
        f"| **Chosen Ensemble + Largest CC** (threshold 0.5) | {chosen_ens_largest_test_res['structure_mean_macro_dice']:.4f} | {chosen_ens_largest_test_res['structure_mean_macro_hd95']:.2f} | {boot_largest_macro_dice[0]:+.4f} [{boot_largest_macro_dice[1]:+.4f}, {boot_largest_macro_dice[2]:+.4f}] | {boot_largest_macro_hd95[0]:+.2f} [{boot_largest_macro_hd95[1]:+.2f}, {boot_largest_macro_hd95[2]:+.2f}] |",
        f"| **Chosen Ensemble + Part A Settings** | {chosen_ens_parta_test_res['structure_mean_macro_dice']:.4f} | {chosen_ens_parta_test_res['structure_mean_macro_hd95']:.2f} | - | - |",
        "",
        "### Structure-by-Structure Breakdown (Test Set)",
        "",
        "| Structure | Best Single Dice | Ensemble Dice | Paired $\\Delta$ Dice (95% CI) | Best Single HD95 (px) | Ensemble HD95 (px) | Paired $\\Delta$ HD95 (95% CI) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for s in STRUCTURE_NAMES:
        bs_d = best_single_test_res["by_structure"][s]["dice_mean"]
        ens_d = chosen_ens_test_res["by_structure"][s]["dice_mean"]
        d_boot = boot_struct_dice[s]

        bs_h = best_single_test_res["by_structure"][s]["hd95_mean"]
        ens_h = chosen_ens_test_res["by_structure"][s]["hd95_mean"]
        h_boot = boot_struct_hd95[s]

        md_lines.append(
            f"| **{s.capitalize()}** | {bs_d:.4f} | {ens_d:.4f} | {d_boot[0]:+.4f} [{d_boot[1]:+.4f}, {d_boot[2]:+.4f}] | {bs_h:.2f} | {ens_h:.2f} | {h_boot[0]:+.2f} [{h_boot[1]:+.2f}, {h_boot[2]:+.2f}] |"
        )

    md_lines.extend([
        "",
        "## 4. Key Findings",
        "",
        f"1. **Ensemble Gain**: Ensembling models by probability averaging achieved a test macro Dice of **{chosen_ens_test_res['structure_mean_macro_dice']:.4f}** vs **{best_single_test_res['structure_mean_macro_dice']:.4f}** for the best single validation model (`{best_single_name}`).",
        f"2. **Boundary Precision**: The test macro 95% Hausdorff Distance was **{chosen_ens_test_res['structure_mean_macro_hd95']:.2f} px** for the ensemble vs **{best_single_test_res['structure_mean_macro_hd95']:.2f} px** for the single model.",
        f"3. **Post-Processing Synergy**: Applying fixed largest-component filtering yielded macro Dice of **{chosen_ens_largest_test_res['structure_mean_macro_dice']:.4f}** and HD95 of **{chosen_ens_largest_test_res['structure_mean_macro_hd95']:.2f} px**.",
    ])

    md_path = output_dir / "fassd_ensemble_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved Markdown report to {md_path}")
    print("\nEnsemble study successfully completed.")


if __name__ == "__main__":
    main()
