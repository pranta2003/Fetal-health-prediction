import hashlib
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
cand_dir = REPO_ROOT / "candidates"
cand_dir.mkdir(parents=True, exist_ok=True)
src_dir = Path(r"E:\BUBT-CSE-4th yr\Capstone\kaggle_outputs\hc18_nnunet_test_f0")

for name in ["submission_nnunet_f0_raw.csv", "submission_nnunet_f0_measfix.csv"]:
    src = src_dir / name
    dst = cand_dir / name
    shutil.copy2(src, dst)
    print(f"Copied {name} to {dst}")

def ramanujan_hc(a, b):
    return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))

ens_path = cand_dir / "submission_ensemble_5fold_corrected_measfix.csv"
df_ens = pd.read_csv(ens_path)
df_ens["hc_ens"] = ramanujan_hc(df_ens["semi_axes_a_mm"], df_ens["semi_axes_b_mm"])

for name in ["submission_nnunet_f0_raw.csv", "submission_nnunet_f0_measfix.csv"]:
    p = cand_dir / name
    b = p.read_bytes()
    assert b"\r\n" not in b, f"CRLF in {name}"
    assert b"\r" not in b, f"CR in {name}"
    sha256 = hashlib.sha256(b).hexdigest()
    df = pd.read_csv(p)
    assert len(df) == 335, f"Expected 335 rows, got {len(df)}"
    expected_cols = ["filename", "center_x_mm", "center_y_mm", "semi_axes_a_mm", "semi_axes_b_mm", "angle_rad"]
    assert list(df.columns) == expected_cols, f"Wrong cols: {df.columns}"
    assert (df["filename"] == df_ens["filename"]).all(), "Filename mismatch with ensemble"
    df["hc"] = ramanujan_hc(df["semi_axes_a_mm"], df["semi_axes_b_mm"])
    diff = df["hc"] - df_ens["hc_ens"]
    mean_diff = diff.mean()
    abs_diff = diff.abs().mean()
    print("=" * 60)
    print(f"File: {name}")
    print(f"  Rows: {len(df)}")
    print(f"  Line Endings: strictly LF")
    print(f"  SHA256: {sha256}")
    print(f"  Mean HC: {df['hc'].mean():.4f} mm")
    print(f"  Mean HC diff vs ensemble (pred - ens): {mean_diff:+.4f} mm")
    print(f"  Mean absolute HC diff vs ensemble: {abs_diff:.4f} mm")
