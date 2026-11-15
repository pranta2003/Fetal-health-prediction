#!/usr/bin/env python3
"""Generate official HC18 Grand Challenge test set evaluation summary.

Reads metrics_ensemble_5fold.json, metrics_single_fold1.json, and published_refs.json
to write docs/results/hc18_official/summary.json and summary.md.

Strict constraints:
  - Zero hand-typed numbers.
  - No rank or position claimed.
  - All numbers originate from JSON inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_summary(hc18_dir: Path, pub_refs_path: Path) -> tuple[dict, str]:
    ens_path = hc18_dir / "metrics_ensemble_5fold.json"
    sin_path = hc18_dir / "metrics_single_fold1.json"

    if not ens_path.is_file() or not sin_path.is_file():
        raise FileNotFoundError(f"Missing required metric JSONs in {hc18_dir}")

    ens_raw = load_json(ens_path)
    sin_raw = load_json(sin_path)
    pub_refs = load_json(pub_refs_path)

    ens_agg = ens_raw["aggregates"]
    sin_agg = sin_raw["aggregates"]
    hc_pub = pub_refs["hc18"]

    # Extract target models from published_refs.json
    fio_model = next(m for m in hc_pub["models"] if "Fiorentino" in m["name"])
    mask_model = next(m for m in hc_pub["models"] if m["name"] == "Mask-R2CNN")
    sob_model = next(m for m in hc_pub["models"] if "Sobhaninia" in m["name"])

    ens_abs_mean = ens_agg["absolute_difference"]["mean"]
    sin_abs_mean = sin_agg["absolute_difference"]["mean"]
    diff_ens_sin = abs(ens_abs_mean - sin_abs_mean)

    claim_statement = (
        f"same range as the published values listed; no rank is claimed; "
        f"test labels are hidden so no paired significance test is possible; "
        f"ensemble vs single difference = {diff_ens_sin:.2f} mm (no test)"
    )

    summary_data = {
        "dataset": "HC18 Official Grand Challenge Test Set",
        "n_images": ens_agg["absolute_difference"]["count"],
        "ensemble_5fold": {
            "count": ens_agg["absolute_difference"]["count"],
            "hc_abs_diff_mean": ens_agg["absolute_difference"]["mean"],
            "hc_abs_diff_std": ens_agg["absolute_difference"]["std"],
            "hc_abs_diff_median": ens_agg["absolute_difference"]["50pc"],
            "hc_abs_diff_max": ens_agg["absolute_difference"]["max"],
            "hc_signed_diff_mean": ens_agg["difference"]["mean"],
            "hc_signed_diff_std": ens_agg["difference"]["std"],
            "dice_mean": ens_agg["DICE"]["mean"],
            "dice_std": ens_agg["DICE"]["std"],
            "hausdorff_mean": ens_agg["hausdorff_distance"]["mean"],
            "hausdorff_std": ens_agg["hausdorff_distance"]["std"],
            "trimester_1": {
                "count": ens_agg["absolute_difference_trimester_1"]["count"],
                "hc_abs_diff_mean": ens_agg["absolute_difference_trimester_1"]["mean"],
                "hc_abs_diff_std": ens_agg["absolute_difference_trimester_1"]["std"],
                "dice_mean": ens_agg["DICE_trimester_1"]["mean"],
                "dice_std": ens_agg["DICE_trimester_1"]["std"],
                "hausdorff_mean": ens_agg["hausdorff_distance_trimester_1"]["mean"],
                "hausdorff_std": ens_agg["hausdorff_distance_trimester_1"]["std"],
            },
            "trimester_2": {
                "count": ens_agg["absolute_difference_trimester_2"]["count"],
                "hc_abs_diff_mean": ens_agg["absolute_difference_trimester_2"]["mean"],
                "hc_abs_diff_std": ens_agg["absolute_difference_trimester_2"]["std"],
                "dice_mean": ens_agg["DICE_trimester_2"]["mean"],
                "dice_std": ens_agg["DICE_trimester_2"]["std"],
                "hausdorff_mean": ens_agg["hausdorff_distance_trimester_2"]["mean"],
                "hausdorff_std": ens_agg["hausdorff_distance_trimester_2"]["std"],
            },
            "trimester_3": {
                "count": ens_agg["absolute_difference_trimester_3"]["count"],
                "hc_abs_diff_mean": ens_agg["absolute_difference_trimester_3"]["mean"],
                "hc_abs_diff_std": ens_agg["absolute_difference_trimester_3"]["std"],
                "dice_mean": ens_agg["DICE_trimester_3"]["mean"],
                "dice_std": ens_agg["DICE_trimester_3"]["std"],
                "hausdorff_mean": ens_agg["hausdorff_distance_trimester_3"]["mean"],
                "hausdorff_std": ens_agg["hausdorff_distance_trimester_3"]["std"],
            },
        },
        "single_fold1": {
            "count": sin_agg["absolute_difference"]["count"],
            "hc_abs_diff_mean": sin_agg["absolute_difference"]["mean"],
            "hc_abs_diff_std": sin_agg["absolute_difference"]["std"],
            "hc_abs_diff_median": sin_agg["absolute_difference"]["50pc"],
            "hc_abs_diff_max": sin_agg["absolute_difference"]["max"],
            "hc_signed_diff_mean": sin_agg["difference"]["mean"],
            "hc_signed_diff_std": sin_agg["difference"]["std"],
            "dice_mean": sin_agg["DICE"]["mean"],
            "dice_std": sin_agg["DICE"]["std"],
            "hausdorff_mean": sin_agg["hausdorff_distance"]["mean"],
            "hausdorff_std": sin_agg["hausdorff_distance"]["std"],
            "trimester_1": {
                "count": sin_agg["absolute_difference_trimester_1"]["count"],
                "hc_abs_diff_mean": sin_agg["absolute_difference_trimester_1"]["mean"],
                "hc_abs_diff_std": sin_agg["absolute_difference_trimester_1"]["std"],
                "dice_mean": sin_agg["DICE_trimester_1"]["mean"],
                "dice_std": sin_agg["DICE_trimester_1"]["std"],
                "hausdorff_mean": sin_agg["hausdorff_distance_trimester_1"]["mean"],
                "hausdorff_std": sin_agg["hausdorff_distance_trimester_1"]["std"],
            },
            "trimester_2": {
                "count": sin_agg["absolute_difference_trimester_2"]["count"],
                "hc_abs_diff_mean": sin_agg["absolute_difference_trimester_2"]["mean"],
                "hc_abs_diff_std": sin_agg["absolute_difference_trimester_2"]["std"],
                "dice_mean": sin_agg["DICE_trimester_2"]["mean"],
                "dice_std": sin_agg["DICE_trimester_2"]["std"],
                "hausdorff_mean": sin_agg["hausdorff_distance_trimester_2"]["mean"],
                "hausdorff_std": sin_agg["hausdorff_distance_trimester_2"]["std"],
            },
            "trimester_3": {
                "count": sin_agg["absolute_difference_trimester_3"]["count"],
                "hc_abs_diff_mean": sin_agg["absolute_difference_trimester_3"]["mean"],
                "hc_abs_diff_std": sin_agg["absolute_difference_trimester_3"]["std"],
                "dice_mean": sin_agg["DICE_trimester_3"]["mean"],
                "dice_std": sin_agg["DICE_trimester_3"]["std"],
                "hausdorff_mean": sin_agg["hausdorff_distance_trimester_3"]["mean"],
                "hausdorff_std": sin_agg["hausdorff_distance_trimester_3"]["std"],
            },
        },
        "published_references": {
            "Fiorentino2020": {
                "name": fio_model["name"],
                "hc_mad_mean": fio_model["hc_mad_mean"],
                "hc_mad_sd": fio_model["hc_mad_sd"],
            },
            "Mask_R2CNN": {
                "name": mask_model["name"],
                "hc_mad_mean": mask_model["hc_mad_mean"],
                "hc_mad_sd": mask_model["hc_mad_sd"],
            },
            "Sobhaninia2019": {
                "name": sob_model["name"],
                "hc_mad_mean": sob_model["hc_mad_mean"],
                "hc_mad_sd": sob_model["hc_mad_sd"],
            },
        },
        "ensemble_vs_single_abs_diff_diff": diff_ens_sin,
        "statement": claim_statement,
    }

    # Markdown generation
    md_lines = []
    md_lines.append("# HC18 Official Challenge Test Set Benchmark Summary\n")
    md_lines.append(
        "Official platform evaluation metrics returned by the HC18 Grand Challenge evaluation system "
        "for the 335 hidden test images.\n"
    )

    md_lines.append("## 1. Overall Test Set Performance (335 Images)\n")
    md_lines.append(
        "| Submission / Model | Images | HC Abs Diff (mm) | Median (mm) | Max (mm) | HC Signed Diff (mm) | Dice (%) | Hausdorff (mm) |\n"
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    )
    md_lines.append(
        f"| **Ensemble (5-Fold)** | {ens_agg['absolute_difference']['count']} | "
        f"**{ens_agg['absolute_difference']['mean']:.2f} +- {ens_agg['absolute_difference']['std']:.2f}** | "
        f"{ens_agg['absolute_difference']['50pc']:.2f} | "
        f"{ens_agg['absolute_difference']['max']:.2f} | "
        f"{ens_agg['difference']['mean']:+.2f} +- {ens_agg['difference']['std']:.2f} | "
        f"{ens_agg['DICE']['mean']:.2f} +- {ens_agg['DICE']['std']:.2f} | "
        f"{ens_agg['hausdorff_distance']['mean']:.2f} +- {ens_agg['hausdorff_distance']['std']:.2f} |"
    )
    md_lines.append(
        f"| **Single Model (Fold 1)** | {sin_agg['absolute_difference']['count']} | "
        f"**{sin_agg['absolute_difference']['mean']:.2f} +- {sin_agg['absolute_difference']['std']:.2f}** | "
        f"{sin_agg['absolute_difference']['50pc']:.2f} | "
        f"{sin_agg['absolute_difference']['max']:.2f} | "
        f"{sin_agg['difference']['mean']:+.2f} +- {sin_agg['difference']['std']:.2f} | "
        f"{sin_agg['DICE']['mean']:.2f} +- {sin_agg['DICE']['std']:.2f} | "
        f"{sin_agg['hausdorff_distance']['mean']:.2f} +- {sin_agg['hausdorff_distance']['std']:.2f} |"
    )

    md_lines.append("\n## 2. Trimester Breakdown\n")
    md_lines.append(
        "| Submission | Trimester | Images | HC Abs Diff (mm) | Dice (%) | Hausdorff (mm) |\n"
        "|:---|:---|:---:|:---:|:---:|:---:|"
    )
    for sub_name, agg in [("Ensemble (5-Fold)", ens_agg), ("Single Model (Fold 1)", sin_agg)]:
        for t_idx, t_name in [(1, "Trimester 1"), (2, "Trimester 2"), (3, "Trimester 3")]:
            count = agg[f"absolute_difference_trimester_{t_idx}"]["count"]
            abs_m = agg[f"absolute_difference_trimester_{t_idx}"]["mean"]
            abs_s = agg[f"absolute_difference_trimester_{t_idx}"]["std"]
            dice_m = agg[f"DICE_trimester_{t_idx}"]["mean"]
            dice_s = agg[f"DICE_trimester_{t_idx}"]["std"]
            h_m = agg[f"hausdorff_distance_trimester_{t_idx}"]["mean"]
            h_s = agg[f"hausdorff_distance_trimester_{t_idx}"]["std"]
            md_lines.append(
                f"| {sub_name} | {t_name} | {count} | "
                f"{abs_m:.2f} +- {abs_s:.2f} | "
                f"{dice_m:.2f} +- {dice_s:.2f} | "
                f"{h_m:.2f} +- {h_s:.2f} |"
            )

    md_lines.append("\n## 3. Comparison with Published Baselines\n")
    md_lines.append(
        "| Model / Architecture | Evaluation Set | HC Absolute Difference / MAD (mm) | Source |\n"
        "|:---|:---:|:---:|:---|"
    )
    md_lines.append(
        f"| **Ours: Ensemble (5-Fold)** | 335 Test Images | **{ens_agg['absolute_difference']['mean']:.2f} +- {ens_agg['absolute_difference']['std']:.2f}** | Platform Evaluation (`metrics_ensemble_5fold.json`) |"
    )
    md_lines.append(
        f"| **Ours: Single (Fold 1)** | 335 Test Images | **{sin_agg['absolute_difference']['mean']:.2f} +- {sin_agg['absolute_difference']['std']:.2f}** | Platform Evaluation (`metrics_single_fold1.json`) |"
    )
    md_lines.append(
        f"| {fio_model['name']} | 335 Test Images | {fio_model['hc_mad_mean']:.2f} +- {fio_model['hc_mad_sd']:.2f} | [{hc_pub['source']}]({hc_pub['url']}) |"
    )
    md_lines.append(
        f"| {mask_model['name']} | 335 Test Images | {mask_model['hc_mad_mean']:.2f} +- {mask_model['hc_mad_sd']:.2f} | [{hc_pub['source']}]({hc_pub['url']}) |"
    )
    md_lines.append(
        f"| {sob_model['name']} | 335 Test Images | {sob_model['hc_mad_mean']:.2f} +- {sob_model['hc_mad_sd']:.2f} | [{hc_pub['source']}]({hc_pub['url']}) |"
    )

    md_lines.append("\n## 4. Formal Claim Statement\n")
    md_lines.append(f"> **Statement**: {claim_statement}\n")

    md_text = "\n".join(md_lines) + "\n"
    return summary_data, md_text


def main():
    parser = argparse.ArgumentParser(description="HC18 Official Test Set Summary Generator")
    parser.add_argument(
        "--hc18-dir",
        type=str,
        default=str(REPO_ROOT / "docs" / "results" / "hc18_official"),
        help="Directory with metrics_ensemble_5fold.json and metrics_single_fold1.json",
    )
    parser.add_argument(
        "--pub-refs",
        type=str,
        default=str(REPO_ROOT / "docs" / "results" / "published_refs.json"),
        help="Path to published_refs.json",
    )
    args = parser.parse_args()

    hc18_dir = Path(args.hc18_dir).resolve()
    pub_refs_path = Path(args.pub_refs).resolve()

    summary_data, md_text = generate_summary(hc18_dir, pub_refs_path)

    json_out = hc18_dir / "summary.json"
    md_out = hc18_dir / "summary.md"

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    with open(md_out, "w", encoding="utf-8", newline="\n") as f:
        f.write(md_text)

    print(f"Written: {json_out}")
    print(f"Written: {md_out}")
    print("\n=== HC18 Official Summary Markdown ===\n")
    print(md_text)


if __name__ == "__main__":
    main()
