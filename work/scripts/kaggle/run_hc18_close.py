"""Kaggle runner script for HC18 OOF evaluation and official test inference.

Clones repo from master, installs dependencies, links datasets, executes:
- scripts/11_hc18_oof_report.py
- scripts/12_hc18_test_inference.py
and writes all artifacts to /kaggle/working/out.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


def run_cmd(cmd: list[str] | str, check: bool = True, cwd: str | Path | None = None) -> subprocess.CompletedProcess:
    print(f"\n>>> Running: {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    result = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with returncode {result.returncode}: {cmd}")
    return result


def find_file(root: Path, target_name: str) -> Path:
    for p in root.rglob(target_name):
        return p
    raise FileNotFoundError(f"Could not find {target_name} under {root}")


def main():
    print("==================================================")
    print("=== HC18 Kaggle Close Runner (OOF + Test Set) ===")
    print("==================================================")

    working_dir = Path("/kaggle/working")
    input_dir = Path("/kaggle/input")
    out_dir = working_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pip install segmentation-models-pytorch
    print("\n[Step 1/5] Installing segmentation-models-pytorch==0.5.0...")
    run_cmd([sys.executable, "-m", "pip", "install", "--no-warn-script-location", "segmentation-models-pytorch==0.5.0"])

    # 2. Clone repo from master
    repo_dir = working_dir / "repo"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    print(f"\n[Step 2/5] Cloning repository from master to {repo_dir}...")
    run_cmd(["git", "clone", "--depth", "1", "https://github.com/pranta2003/Fetal-health-prediction.git", str(repo_dir)])

    # Repo workspace directory is repo/work
    work_dir = repo_dir / "work"
    if not work_dir.exists():
        raise FileNotFoundError(f"Expected work directory at {work_dir}")

    # 3. Locate datasets and link hc18-ultrasound as <fd>/HC18
    print("\n[Step 3/5] Locating input datasets and setting up links...")
    print(f"Scanning {input_dir}...")
    for item in input_dir.iterdir():
        print(f"  Input mount: {item.name}")

    # Find HC18 dataset root by searching for training_set_pixel_size_and_HC.csv
    train_csv = find_file(input_dir, "training_set_pixel_size_and_HC.csv")
    hc18_source_dir = train_csv.parent
    print(f"Found HC18 dataset at: {hc18_source_dir}")

    fd_dir = working_dir / "fd"
    fd_dir.mkdir(parents=True, exist_ok=True)
    hc18_link = fd_dir / "HC18"
    if hc18_link.exists() or hc18_link.is_symlink():
        hc18_link.unlink()
    try:
        os.symlink(str(hc18_source_dir), str(hc18_link))
        print(f"Created symlink: {hc18_link} -> {hc18_source_dir}")
    except OSError:
        shutil.copytree(str(hc18_source_dir), str(hc18_link))
        print(f"Copied dataset: {hc18_source_dir} -> {hc18_link}")

    # Find model weights directory by searching for hc18_seg_fold1_eff_b4_384.pt
    fold1_file = find_file(input_dir, "hc18_seg_fold1_eff_b4_384.pt")
    model_dir = fold1_file.parent
    print(f"Found model weights at: {model_dir}")
    for fold in range(1, 6):
        expected_pt = model_dir / f"hc18_seg_fold{fold}_eff_b4_384.pt"
        if not expected_pt.exists():
            raise FileNotFoundError(f"Missing fold checkpoint: {expected_pt}")
        print(f"  Verified fold {fold}: {expected_pt.name} ({expected_pt.stat().st_size} bytes)")

    # Test images path
    test_set_dir = hc18_link / "test_set" / "test_set"
    if not test_set_dir.exists():
        test_set_dir = hc18_link / "test_set"
    test_csv = hc18_link / "test_set_pixel_size.csv"
    if not test_csv.exists():
        test_csv = find_file(input_dir, "test_set_pixel_size.csv")

    print(f"HC18 training CSV: {train_csv}")
    print(f"HC18 test dir:     {test_set_dir}")
    print(f"HC18 test CSV:     {test_csv}")

    # 4. Run scripts/11_hc18_oof_report.py
    print("\n[Step 4/5] Running scripts/11_hc18_oof_report.py...")
    oof_script = work_dir / "scripts" / "11_hc18_oof_report.py"
    oof_cmd = [
        sys.executable,
        str(oof_script),
        "--data-dir", str(hc18_link),
        "--model-dir", str(model_dir),
        "--out-dir", str(out_dir),
        "--device", "cuda",
        "--tolerance", "0.02",
    ]
    run_cmd(oof_cmd, cwd=work_dir)

    # 5. Run scripts/12_hc18_test_inference.py
    print("\n[Step 5/5] Running scripts/12_hc18_test_inference.py...")
    test_script = work_dir / "scripts" / "12_hc18_test_inference.py"
    test_cmd = [
        sys.executable,
        str(test_script),
        "--test-dir", str(test_set_dir),
        "--test-csv", str(test_csv),
        "--model-dir", str(model_dir),
        "--out-dir", str(out_dir),
        "--device", "cuda",
    ]
    run_cmd(test_cmd, cwd=work_dir)

    print("\n==================================================")
    print("=== RUN COMPLETED SUCCESSFULLY ===")
    print("==================================================")
    print(f"Artifacts in {out_dir}:")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name} ({f.stat().st_size} bytes)")
        shutil.copy2(f, working_dir / f.name)


if __name__ == "__main__":
    main()
