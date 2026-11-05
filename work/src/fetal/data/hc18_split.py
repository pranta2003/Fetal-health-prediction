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
