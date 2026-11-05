"""Tests for HC18 patient-grouped cross-validation splitting."""

import numpy as np
import pytest
from fetal.data.hc18_split import patient_id_from_name, make_grouped_folds


def test_patient_id_from_name():
    """Verify patient_id extraction handles single and multiple scan naming."""
    assert patient_id_from_name("245_HC.png") == "245"
    assert patient_id_from_name("245_2HC.png") == "245"
    assert patient_id_from_name("000_HC.png") == "000"
    assert patient_id_from_name("014_3HC.png") == "014"
    assert patient_id_from_name("/some/path/to/999_HC.png") == "999"
    assert patient_id_from_name(r"C:\data\806_2HC.png") == "806"


def test_grouped_folds_no_patient_leakage():
    """Verify no patient appears in both train and validation sets of any fold."""
    names = [
        "001_HC.png", "001_2HC.png",
        "002_HC.png",
        "003_HC.png", "003_2HC.png", "003_3HC.png",
        "004_HC.png",
        "005_HC.png",
        "006_HC.png", "006_2HC.png",
        "007_HC.png",
        "008_HC.png",
        "009_HC.png",
        "010_HC.png", "010_2HC.png",
        "011_HC.png",
        "012_HC.png",
    ]
    folds = make_grouped_folds(names, n_splits=5, seed=42)
    assert len(folds) == 5

    groups = [patient_id_from_name(n) for n in names]

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        tr_patients = set(groups[i] for i in tr_idx)
        val_patients = set(groups[i] for i in val_idx)
        overlap = tr_patients & val_patients
        assert len(overlap) == 0, f"Fold {fold_idx} has patient overlap: {overlap}"


def test_grouped_folds_exact_val_partition():
    """Verify every image is in exactly one validation fold."""
    names = [
        "001_HC.png", "001_2HC.png",
        "002_HC.png",
        "003_HC.png", "003_2HC.png",
        "004_HC.png",
        "005_HC.png",
        "006_HC.png",
        "007_HC.png",
        "008_HC.png",
    ]
    folds = make_grouped_folds(names, n_splits=5, seed=42)

    val_indices = []
    for _, val_idx in folds:
        val_indices.extend(val_idx.tolist())

    assert len(val_indices) == len(names)
    assert sorted(val_indices) == list(range(len(names)))


def test_grouped_folds_patient_balance():
    """Verify fold sizes in unique patients are within 1 patient of balanced."""
    # 13 unique patients with varying number of images
    names = [
        f"pt{i:02d}_{suffix}.png"
        for i in range(13)
        for suffix in (["HC", "2HC"] if i % 3 == 0 else ["HC"])
    ]
    folds = make_grouped_folds(names, n_splits=5, seed=123)

    patient_counts_per_val_fold = []
    for _, val_idx in folds:
        val_patients = set(patient_id_from_name(names[i]) for i in val_idx)
        patient_counts_per_val_fold.append(len(val_patients))

    # 13 patients / 5 folds -> folds must have either 2 or 3 patients (diff <= 1)
    max_pts = max(patient_counts_per_val_fold)
    min_pts = min(patient_counts_per_val_fold)
    assert max_pts - min_pts <= 1, f"Patient counts {patient_counts_per_val_fold} differ by >1"


def test_grouped_folds_deterministic():
    """Verify split is deterministic for a given seed."""
    names = [
        f"pt{i:03d}_{s}.png"
        for i in range(25)
        for s in (["HC", "2HC"] if i % 2 == 0 else ["HC"])
    ]
    folds1 = make_grouped_folds(names, n_splits=5, seed=42)
    folds2 = make_grouped_folds(names, n_splits=5, seed=42)

    assert len(folds1) == len(folds2)
    for (tr1, val1), (tr2, val2) in zip(folds1, folds2):
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(val1, val2)
