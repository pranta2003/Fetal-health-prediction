# FIELD: HC18
# MODEL: nnU-Net
# STEP: smoke_test
# USED BY: work/docs/CODE_MAP.md, work/docs/report/REPORT_NUMBERS.json

import hashlib
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path("C:/fetal-github-clone/work")
cand_dir = REPO_ROOT / "candidates"
cand_dir.mkdir(parents=True, exist_ok=True)

gpu_dir = Path(r"C:\fetal-data\kaggle_outputs\hc18_nnunet_test_f012")
cpu_dir = Path(r"C:\fetal-data\kaggle_outputs\hc18_nnunet_test_f012_cpu")

def ramanujan_hc(a, b):
    return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))

# 1. Compare GPU vs CPU image by image
print("Comparing GPU vs CPU predictions...")
df_gpu_raw = pd.read_csv(gpu_dir / "submission_nnunet_f012_raw.csv")
df_cpu_raw = pd.read_csv(cpu_dir / "submission_nnunet_f012_raw.csv")
df_gpu_mf = pd.read_csv(gpu_dir / "submission_nnunet_f012_measfix.csv")
df_cpu_mf = pd.read_csv(cpu_dir / "submission_nnunet_f012_measfix.csv")

assert (df_gpu_raw["filename"] == df_cpu_raw["filename"]).all()
hc_gpu_raw = ramanujan_hc(df_gpu_raw["semi_axes_a_mm"], df_gpu_raw["semi_axes_b_mm"])
hc_cpu_raw = ramanujan_hc(df_cpu_raw["semi_axes_a_mm"], df_cpu_raw["semi_axes_b_mm"])
hc_gpu_mf = ramanujan_hc(df_gpu_mf["semi_axes_a_mm"], df_gpu_mf["semi_axes_b_mm"])
hc_cpu_mf = ramanujan_hc(df_cpu_mf["semi_axes_a_mm"], df_cpu_mf["semi_axes_b_mm"])

max_diff_raw = (hc_gpu_raw - hc_cpu_raw).abs().max()
max_diff_mf = (hc_gpu_mf - hc_cpu_mf).abs().max()
mean_diff_raw = (hc_gpu_raw - hc_cpu_raw).abs().mean()
print(f"Max absolute HC difference between GPU and CPU (raw): {max_diff_raw:.6f} mm")
print(f"Mean absolute HC difference between GPU and CPU (raw): {mean_diff_raw:.6f} mm")
print(f"Max absolute HC difference between GPU and CPU (measfix): {max_diff_mf:.6f} mm")

# 2. Prepare candidates (from GPU kernel output, normalized to LF)
for name in ["submission_nnunet_f012_raw.csv", "submission_nnunet_f012_measfix.csv"]:
    src_file = gpu_dir / name
    text = src_file.read_text(encoding="utf-8")
    text_lf = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text_lf.endswith("\n"):
        text_lf += "\n"
    dst_file = cand_dir / name
    dst_file.write_bytes(text_lf.encode("utf-8"))
    print(f"Wrote {name} to {dst_file} with LF line endings")

# 3. Load reference CSVs
f0_mf_path = cand_dir / "submission_nnunet_f0_measfix.csv"
df_f0_mf = pd.read_csv(f0_mf_path)
df_f0_mf["hc"] = ramanujan_hc(df_f0_mf["semi_axes_a_mm"], df_f0_mf["semi_axes_b_mm"])

ens_path = cand_dir / "submission_ensemble_5fold_corrected_measfix.csv"
df_ens = pd.read_csv(ens_path)
df_ens["hc_ens"] = ramanujan_hc(df_ens["semi_axes_a_mm"], df_ens["semi_axes_b_mm"])

expected_cols = ["filename", "center_x_mm", "center_y_mm", "semi_axes_a_mm", "semi_axes_b_mm", "angle_rad"]

results = {}
for name in ["submission_nnunet_f012_raw.csv", "submission_nnunet_f012_measfix.csv"]:
    p = cand_dir / name
    b = p.read_bytes()
    assert b"\r\n" not in b, f"CRLF in {name}"
    assert b"\r" not in b, f"CR in {name}"
    sha256 = hashlib.sha256(b).hexdigest()
    
    df = pd.read_csv(p)
    assert len(df) == 335, f"Expected 335 rows, got {len(df)}"
    assert list(df.columns) == expected_cols, f"Wrong cols: {df.columns}"
    assert (df["filename"] == df_ens["filename"]).all(), "Filename mismatch"
    
    df["hc"] = ramanujan_hc(df["semi_axes_a_mm"], df["semi_axes_b_mm"])
    
    # diff vs f0 measfix
    diff_vs_f0 = df["hc"] - df_f0_mf["hc"]
    mean_diff_f0 = diff_vs_f0.mean()
    abs_diff_f0 = diff_vs_f0.abs().mean()
    
    # diff vs 5-fold ensemble measfix
    diff_vs_ens = df["hc"] - df_ens["hc_ens"]
    mean_diff_ens = diff_vs_ens.mean()
    abs_diff_ens = diff_vs_ens.abs().mean()
    
    results[name] = {
        "sha256": sha256,
        "rows": len(df),
        "mean_hc": df["hc"].mean(),
        "mean_diff_vs_f0_mf": mean_diff_f0,
        "abs_diff_vs_f0_mf": abs_diff_f0,
        "mean_diff_vs_ens": mean_diff_ens,
        "abs_diff_vs_ens": abs_diff_ens,
    }
    print("=" * 60)
    print(f"Candidate: {name}")
    print(f"  Rows: {len(df)}")
    print(f"  Line Endings: LF")
    print(f"  SHA256: {sha256}")
    print(f"  Mean HC: {df['hc'].mean():.4f} mm")
    print(f"  Mean diff vs f0_measfix (pred - f0): {mean_diff_f0:+.4f} mm (abs: {abs_diff_f0:.4f} mm)")
    print(f"  Mean diff vs 5-fold ens measfix (pred - ens): {mean_diff_ens:+.4f} mm (abs: {abs_diff_ens:.4f} mm)")

import json
out_json = REPO_ROOT / "docs" / "results" / "hc18_nnunet" / "test_f012_candidates_summary.json"
out_json.parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({
        "comparison_gpu_vs_cpu": {
            "max_abs_diff_raw": max_diff_raw,
            "mean_abs_diff_raw": mean_diff_raw,
            "max_abs_diff_measfix": max_diff_mf,
        },
        "candidates": results,
    }, f, indent=2)
print(f"\nSaved summary to {out_json}")
