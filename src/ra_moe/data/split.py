"""
Train/validation/test splitting for RA-MoE-4E.

This module refactors the sequential 80/10/10 splitting procedure
used in the supplied final experimental implementation.

The original implementation uses ``train_test_split`` with
``shuffle=False`` so that the existing row order is preserved.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def split_indices(
    n_observations: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split aligned observation indices into train, validation, and test sets.

    The procedure reproduces the supplied final implementation:

    1. 80% training / 20% temporary split.
    2. Split the temporary 20% equally into validation and test sets.
    3. Preserve the existing observation order with ``shuffle=False``.

    Parameters
    ----------
    n_observations
        Total number of aligned observations.

    seed
        Random-state value retained from the original implementation.
        With ``shuffle=False``, it does not affect the ordering.

    Returns
    -------
    train_indices
        Indices for the training subset.

    validation_indices
        Indices for the validation subset.

    test_indices
        Indices for the test subset.
    """
    indices = np.arange(n_observations)

    train_indices, temporary_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        shuffle=False,
    )

    validation_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=0.5,
        random_state=seed,
        shuffle=False,
    )

    return (
        train_indices,
        validation_indices,
        test_indices,
    )
