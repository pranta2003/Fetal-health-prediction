"""HC18 patient-grouped cross-validation splitting.

Prevents data leakage across folds by ensuring all ultrasound images
belonging to the same patient are assigned to the same split.
"""

from __future__ import annotations

import os
from typing import Sequence
import numpy as np
from sklearn.model_selection import GroupKFold


def patient_id_from_name(name: str) -> str:
    """Extract patient ID prefix from an HC18 image filename.

    HC18 filenames follow patterns like '245_HC.png', '245_2HC.png', etc.
    The patient identifier is the token before the first underscore.

    Args:
        name: Image filename or path (e.g. '245_HC.png' or '/path/to/245_2HC.png').

    Returns:
        String patient ID prefix (e.g. '245').
    """
    base = os.path.basename(name)
    return base.split("_")[0]


def make_grouped_folds(
    names: Sequence[str],
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate patient-grouped cross-validation folds.

    Groups unique patient IDs, shuffles them deterministically using `seed`,
    and distributes them across `n_splits` folds using sklearn GroupKFold.
    No patient appears in both train and validation sets of any fold.

    Args:
        names: Sequence of image filenames or paths.
        n_splits: Number of cross-validation folds (default: 5).
        seed: Random seed for deterministic patient shuffling (default: 42).

    Returns:
        List of (train_idx, val_idx) tuples as numpy integer arrays.
    """
    if len(names) == 0:
        return []

    groups = np.array([patient_id_from_name(n) for n in names])
    indices = np.arange(len(names))

    try:
        gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(gkf.split(indices, groups=groups))
    except TypeError:
        # Fallback for scikit-learn versions where GroupKFold lacks shuffle/random_state
        rng = np.random.default_rng(seed)
        unique_patients = np.unique(groups)
        rng.shuffle(unique_patients)
        split_patients = np.array_split(unique_patients, n_splits)

        folds = []
        for test_pts in split_patients:
            val_mask = np.isin(groups, test_pts)
            val_idx = indices[val_mask]
            tr_idx = indices[~val_mask]
            folds.append((tr_idx, val_idx))
        return folds


def split_patient_inner_holdout(
    train_indices: Sequence[int] | np.ndarray,
    names: Sequence[str],
    holdout_fraction: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split train indices into inner-train and inner-holdout by patient ID.

    Ensures zero patient leakage between inner-train and inner-holdout sets.

    Args:
        train_indices: Sequence or array of integer indices belonging to the outer train set.
        names: Full sequence of filenames/paths corresponding to dataset indices.
        holdout_fraction: Fraction of unique training patients to allocate to inner holdout (0.0 to 1.0).
        seed: Random seed for deterministic patient shuffling.

    Returns:
        Tuple of (inner_train_idx, inner_holdout_idx) as numpy integer arrays.
    """
    train_idx_arr = np.asarray(train_indices, dtype=int)
    if holdout_fraction <= 0.0 or len(train_idx_arr) == 0:
        return train_idx_arr, np.array([], dtype=int)

    train_names = [names[i] for i in train_idx_arr]
    patient_ids = np.array([patient_id_from_name(n) for n in train_names])
    unique_patients = np.unique(patient_ids)

    n_unique = len(unique_patients)
    n_holdout = max(1, int(round(n_unique * holdout_fraction)))
    if n_holdout >= n_unique:
        n_holdout = max(1, n_unique - 1)

    rng = np.random.default_rng(seed)
    shuffled_patients = unique_patients.copy()
    rng.shuffle(shuffled_patients)

    holdout_patients = set(shuffled_patients[:n_holdout])
    inner_holdout_mask = np.isin(patient_ids, list(holdout_patients))

    inner_holdout_idx = train_idx_arr[inner_holdout_mask]
    inner_train_idx = train_idx_arr[~inner_holdout_mask]

    return inner_train_idx, inner_holdout_idx
