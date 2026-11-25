#!/usr/bin/env python3

# FIELD: General
# MODEL: None
# STEP: utility
# USED BY: work/docs/CODE_MAP.md

"""Test the dataset discovery code on a mock Kaggle input layout."""

import shutil
from pathlib import Path

MOCK_ROOT = Path("C:/fetal-github-clone/work/scratch/mock_kaggle/input")
if MOCK_ROOT.exists():
    shutil.rmtree(MOCK_ROOT)

mock_ds = MOCK_ROOT / "hc18-ultrasound"
mock_train = mock_ds / "training_set"
mock_test = mock_ds / "test_set" / "test_set"
mock_kernel = MOCK_ROOT / "hc18-nnunet-f0-250ep" / "nnUNet_results"

mock_train.mkdir(parents=True, exist_ok=True)
mock_test.mkdir(parents=True, exist_ok=True)
mock_kernel.mkdir(parents=True, exist_ok=True)

# Populate mock training_set with 1998 mock files
for i in range(999):
    (mock_train / f"{i:03d}_HC.png").touch()
    (mock_train / f"{i:03d}_HC_Annotation.png").touch()

# Populate mock test_set with exactly 335 mock files
for i in range(335):
    (mock_test / f"{i:03d}_HC.png").touch()

print("Mock folder created.")

# Run discovery code
entries = []
for p in MOCK_ROOT.iterdir():
    if p.is_dir():
        for sub in list(p.iterdir())[:10]:
            entries.append(str(sub))
    else:
        entries.append(str(p))

print("First 20 entries under mock input/*:")
for e in entries[:20]:
    print(" ", e)

exact_subpath = Path("test_set/test_set")
test_src_dir = None
for base in MOCK_ROOT.iterdir():
    candidate = base / exact_subpath
    if candidate.is_dir():
        test_src_dir = candidate
        break

assert test_src_dir is not None, f"Could not find exact subpath {exact_subpath} under any folder in {MOCK_ROOT}"
test_pngs = sorted([p for p in test_src_dir.iterdir() if p.name.endswith(".png") and not p.name.endswith("_Annotation.png")])
print(f"Discovered test directory: {test_src_dir}")
print(f"Test PNG count: {len(test_pngs)}")
assert len(test_pngs) == 335, f"Expected exactly 335 test PNGs, got {len(test_pngs)}"
print("Mock discovery test PASSED successfully!")

# Clean up mock directory
shutil.rmtree(MOCK_ROOT.parent)
