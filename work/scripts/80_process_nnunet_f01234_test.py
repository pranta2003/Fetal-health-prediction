#!/usr/bin/env python3

# FIELD: HC18
# MODEL: nnU-Net
# STEP: smoke_test
# USED BY: work/docs/CODE_MAP.md, work/docs/report/REPORT_NUMBERS.json

"""80_process_nnunet_f01234_test.py: Process 5-fold nnU-Net test outputs, candidates, and blend.

Tasks:
1. Load GPU test outputs from C:/fetal-data/kaggle_outputs/hc18_nnunet_test_f01234
2. Validate and write candidates/submission_nnunet_f01234_raw.csv and candidates/submission_nnunet_f01234_measfix.csv
3. Validate: 335 rows, columns, filenames, LF, SHA-256, HC differences vs f012_measfix and ens5_measfix
4. Assert mean HC difference vs f012_measfix <= 0.3 mm
5. Build candidates/submission_blend_nnunetf01234_ens5_measfix.csv (circular mean mod pi)
6. Save summary to docs/results/hc18_nnunet/test_f01234_candidates_summary.json
"""

import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path("C:/fetal-github-clone/work")
CAND_DIR = REPO_ROOT / "candidates"
CAND_DIR.mkdir(parents=True, exist_ok=True)

GPU_DIR = Path(r"C:\fetal-data\kaggle_outputs\hc18_nnunet_test_f01234")
CPU_DIR = Path(r"C:\fetal-data\kaggle_outputs\hc18_nnunet_test_f01234_cpu")

EXPECTED_COLS = ["filename", "center_x_mm", "center_y_mm", "semi_axes_a_mm", "semi_axes_b_mm", "angle_rad"]


def ramanujan_hc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.pi * (3.0 * (a + b) - np.sqrt((3.0 * a + b) * (a + 3.0 * b)))


def circular_mean_pi(theta1: np.ndarray, theta2: np.ndarray) -> np.ndarray:
    s = np.sin(2.0 * theta1) + np.sin(2.0 * theta2)
    c = np.cos(2.0 * theta1) + np.cos(2.0 * theta2)
    mean_2theta = np.arctan2(s, c)
    return mean_2theta / 2.0


def main():
    print("=" * 60)
    print("STEP 1: Copy and normalize GPU outputs to candidates/")
    print("=" * 60)

    for name in ["submission_nnunet_f01234_raw.csv", "submission_nnunet_f01234_measfix.csv"]:
        src_file = GPU_DIR / name
        assert src_file.is_file(), f"Missing expected GPU output file: {src_file}"
        text = src_file.read_text(encoding="utf-8")
        text_lf = text.replace("\r\n", "\n").replace("\r", "\n")
        if not text_lf.endswith("\n"):
            text_lf += "\n"
        dst_file = CAND_DIR / name
        dst_file.write_bytes(text_lf.encode("utf-8"))
        print(f"Wrote {dst_file} with LF line endings ({dst_file.stat().st_size} bytes)")

    # Reference CSVs
    f012_mf_path = CAND_DIR / "submission_nnunet_f012_measfix.csv"
    assert f012_mf_path.is_file(), f"Missing {f012_mf_path}"
    df_f012_mf = pd.read_csv(f012_mf_path)
    df_f012_mf["hc"] = ramanujan_hc(df_f012_mf["semi_axes_a_mm"].to_numpy(), df_f012_mf["semi_axes_b_mm"].to_numpy())

    ens_path = CAND_DIR / "submission_ensemble_5fold_corrected_measfix.csv"
    assert ens_path.is_file(), f"Missing {ens_path}"
    df_ens = pd.read_csv(ens_path)
    df_ens["hc_ens"] = ramanujan_hc(df_ens["semi_axes_a_mm"].to_numpy(), df_ens["semi_axes_b_mm"].to_numpy())

    results = {}
    for name in ["submission_nnunet_f01234_raw.csv", "submission_nnunet_f01234_measfix.csv"]:
        p = CAND_DIR / name
        raw_bytes = p.read_bytes()
        assert b"\r\n" not in raw_bytes, f"Found CRLF in {name}"
        assert b"\r" not in raw_bytes, f"Found CR in {name}"
        sha256 = hashlib.sha256(raw_bytes).hexdigest()

        df = pd.read_csv(p)
        assert len(df) == 335, f"Expected 335 rows, got {len(df)}"
        assert list(df.columns) == EXPECTED_COLS, f"Column mismatch in {name}: {df.columns}"
        assert (df["filename"] == df_ens["filename"]).all(), f"Filename mismatch in {name}"
        assert (df["semi_axes_a_mm"] >= df["semi_axes_b_mm"]).all(), f"Found a < b in {name}"

        df["hc"] = ramanujan_hc(df["semi_axes_a_mm"].to_numpy(), df["semi_axes_b_mm"].to_numpy())

        # diff vs f012 measfix
        diff_vs_f012 = df["hc"] - df_f012_mf["hc"]
        mean_diff_f012 = float(diff_vs_f012.mean())
        abs_diff_f012 = float(diff_vs_f012.abs().mean())

        # diff vs 5-fold ensemble measfix
        diff_vs_ens = df["hc"] - df_ens["hc_ens"]
        mean_diff_ens = float(diff_vs_ens.mean())
        abs_diff_ens = float(diff_vs_ens.abs().mean())

        results[name] = {
            "sha256": sha256,
            "rows": len(df),
            "mean_hc": float(df["hc"].mean()),
            "mean_diff_vs_f012_mf": mean_diff_f012,
            "abs_diff_vs_f012_mf": abs_diff_f012,
            "mean_diff_vs_ens_mf": mean_diff_ens,
            "abs_diff_vs_ens_mf": abs_diff_ens,
        }

        print("=" * 60)
        print(f"Candidate: {name}")
        print(f"  Rows: {len(df)}")
        print(f"  Line Endings: LF")
        print(f"  SHA256: {sha256}")
        print(f"  Mean HC: {df['hc'].mean():.4f} mm")
        print(f"  Mean diff vs f012_measfix (pred - f012): {mean_diff_f012:+.4f} mm (abs: {abs_diff_f012:.4f} mm)")
        print(f"  Mean diff vs 5-fold ens measfix (pred - ens): {mean_diff_ens:+.4f} mm (abs: {abs_diff_ens:.4f} mm)")

        # Threshold check for measfix
        if name == "submission_nnunet_f01234_measfix.csv":
            assert abs_diff_f012 <= 0.3, f"Mean absolute difference vs f012 ({abs_diff_f012:.4f} mm) exceeds 0.3 mm!"
            print(f"  [CHECK PASSED] Mean abs diff vs f012 is {abs_diff_f012:.4f} mm <= 0.3 mm.")

    print("\n" + "=" * 60)
    print("STEP 2: Build blend candidate submission_blend_nnunetf01234_ens5_measfix.csv")
    print("=" * 60)

    df_nn5 = pd.read_csv(CAND_DIR / "submission_nnunet_f01234_measfix.csv")
    th_nn5 = df_nn5["angle_rad"].to_numpy()
    th_ens = df_ens["angle_rad"].to_numpy()

    # Angular difference modulo pi
    diff_angle = (th_ens - th_nn5 + math.pi / 2.0) % math.pi - math.pi / 2.0
    max_angle_diff = float(np.max(np.abs(diff_angle)))
    print(f"Angle difference modulo pi: max={max_angle_diff:.4f} rad, mean={np.mean(np.abs(diff_angle)):.4f} rad")
    assert max_angle_diff <= 0.5, f"Angle difference exceeded 0.5 rad: {max_angle_diff}"

    blend_cx = (df_nn5["center_x_mm"].to_numpy() + df_ens["center_x_mm"].to_numpy()) / 2.0
    blend_cy = (df_nn5["center_y_mm"].to_numpy() + df_ens["center_y_mm"].to_numpy()) / 2.0
    blend_a = (df_nn5["semi_axes_a_mm"].to_numpy() + df_ens["semi_axes_a_mm"].to_numpy()) / 2.0
    blend_b = (df_nn5["semi_axes_b_mm"].to_numpy() + df_ens["semi_axes_b_mm"].to_numpy()) / 2.0
    blend_angle = circular_mean_pi(th_nn5, th_ens)

    assert (blend_a >= blend_b).all(), "Blend semi-axes violate a >= b"

    df_blend = pd.DataFrame({
        "filename": df_nn5["filename"],
        "center_x_mm": np.round(blend_cx, 6),
        "center_y_mm": np.round(blend_cy, 6),
        "semi_axes_a_mm": np.round(blend_a, 6),
        "semi_axes_b_mm": np.round(blend_b, 6),
        "angle_rad": np.round(blend_angle, 6),
    })

    blend_path = CAND_DIR / "submission_blend_nnunetf01234_ens5_measfix.csv"
    blend_csv_str = df_blend.to_csv(index=False, lineterminator="\n")
    blend_path.write_bytes(blend_csv_str.encode("utf-8"))

    raw_blend_bytes = blend_path.read_bytes()
    assert b"\r\n" not in raw_blend_bytes, "Found CRLF in blend candidate!"
    blend_sha256 = hashlib.sha256(raw_blend_bytes).hexdigest()

    hc_nn5 = ramanujan_hc(df_nn5["semi_axes_a_mm"].to_numpy(), df_nn5["semi_axes_b_mm"].to_numpy())
    hc_ens = df_ens["hc_ens"].to_numpy()
    hc_blend = ramanujan_hc(df_blend["semi_axes_a_mm"].to_numpy(), df_blend["semi_axes_b_mm"].to_numpy())

    diff_blend_vs_nn5 = hc_blend - hc_nn5
    diff_blend_vs_ens = hc_blend - hc_ens

    print(f"Candidate: {blend_path.name}")
    print(f"  Rows: {len(df_blend)}")
    print(f"  Line Endings: LF")
    print(f"  SHA256: {blend_sha256}")
    print(f"  Mean HC: {hc_blend.mean():.4f} mm")
    print(f"  Mean diff vs nnunet_f01234_measfix: {diff_blend_vs_nn5.mean():+.4f} mm (abs: {np.abs(diff_blend_vs_nn5).mean():.4f} mm)")
    print(f"  Mean diff vs 5-fold ens measfix:    {diff_blend_vs_ens.mean():+.4f} mm (abs: {np.abs(diff_blend_vs_ens).mean():.4f} mm)")

    results[blend_path.name] = {
        "sha256": blend_sha256,
        "rows": len(df_blend),
        "mean_hc": float(hc_blend.mean()),
        "max_angle_diff_mod_pi": max_angle_diff,
        "mean_diff_vs_nnunet_f01234": float(diff_blend_vs_nn5.mean()),
        "abs_diff_vs_nnunet_f01234": float(np.abs(diff_blend_vs_nn5).mean()),
        "mean_diff_vs_ens_mf": float(diff_blend_vs_ens.mean()),
        "abs_diff_vs_ens_mf": float(np.abs(diff_blend_vs_ens).mean()),
    }

    # Save summary json
    out_json = REPO_ROOT / "docs" / "results" / "hc18_nnunet" / "test_f01234_candidates_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "gpu_kernel": "hc18-nnunet-test-f01234",
            "candidates": results,
        }, f, indent=2)
    print(f"\nSaved summary to {out_json}")


if __name__ == "__main__":
    main()
