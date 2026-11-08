"""Kaggle runner script for HC18 historical approach ladder (Benchmark 2B).

Clones repo branch bench-hc18-ladder, installs dependencies, runs:
- scripts/19_hc18_ladder.py
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
    print("=== HC18 Ladder Benchmark Runner (Grouped Fold 1) ===")
    print("==================================================")

    working_dir = Path("/kaggle/working")
    input_dir = Path("/kaggle/input")
    out_dir = working_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pip install dependencies
    print("\n[Step 1/4] Installing dependencies: segmentation-models-pytorch, timm...")
    run_cmd([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-warn-script-location",
        "segmentation-models-pytorch==0.5.0",
        "timm>=1.0.0",
    ])

    # 2. Clone repository from branch bench-hc18-ladder
    repo_dir = working_dir / "repo"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    print(f"\n[Step 2/4] Cloning repository branch bench-hc18-ladder to {repo_dir}...")
    run_cmd([
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        "bench-hc18-ladder",
        "https://github.com/pranta2003/Fetal-health-prediction.git",
        str(repo_dir),
    ])

    work_dir = repo_dir / "work"
    if not work_dir.exists():
        raise FileNotFoundError(f"Expected work directory at {work_dir}")

    # 3. Locate HC18 dataset
    print("\n[Step 3/4] Locating HC18 dataset in /kaggle/input...")
    for item in input_dir.iterdir():
        print(f"  Input mount: {item.name}")

    train_csv = find_file(input_dir, "training_set_pixel_size_and_HC.csv")
    hc18_source_dir = train_csv.parent
    print(f"Found HC18 dataset at: {hc18_source_dir}")

    oof_csv = work_dir / "docs" / "results" / "hc18" / "oof_predictions.csv"
    if not oof_csv.exists():
        raise FileNotFoundError(f"Missing oof_predictions.csv at {oof_csv}")

    # 4. Run scripts/19_hc18_ladder.py
    print("\n[Step 4/4] Executing scripts/19_hc18_ladder.py on CUDA...")
    ladder_script = work_dir / "scripts" / "19_hc18_ladder.py"
    ladder_cmd = [
        sys.executable,
        str(ladder_script),
        "--data-dir", str(hc18_source_dir),
        "--oof-csv", str(oof_csv),
        "--out-dir", str(out_dir),
        "--device", "cuda",
        "--epochs", "100",
    ]
    run_cmd(ladder_cmd, cwd=work_dir)

    print("\n==================================================")
    print("=== LADDER RUN COMPLETED SUCCESSFULLY ===")
    print("==================================================")
    print(f"Artifacts in {out_dir}:")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name} ({f.stat().st_size} bytes)")
        shutil.copy2(f, working_dir / f.name)


if __name__ == "__main__":
    main()
