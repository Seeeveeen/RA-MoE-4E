"""
Chronological dataset splitting for RA-MoE-4E.

The public reproducibility pipeline uses calendar-date boundaries
rather than row-position boundaries. This prevents observations from
the same trading date from being distributed across different splits
and makes the temporal evaluation protocol explicit.

This differs from the supplied final experimental implementation,
which performed an 80/10/10 split by row position after temporal
feature construction with ``shuffle=False``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitIndices:
    """Indices defining chronological train, validation, and test sets."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def chronological_date_split(
    df: pd.DataFrame,
    train_fraction: float = 0.80,
    validation_fraction: float = 0.10,
    date_column: str = "date",
) -> SplitIndices:
    """
    Split observations chronologically using unique calendar dates.

    Parameters
    ----------
    df
        Dataset aligned with all feature matrices and temporal windows.

    train_fraction
        Fraction of unique dates assigned to training.

    validation_fraction
        Fraction of unique dates assigned to validation.

        The remaining dates are assigned to the test set.

    date_column
        Name of the calendar-date column.

    Returns
    -------
    SplitIndices
        Integer row indices for train, validation, and test subsets.

    Notes
    -----
    Splitting is performed on unique dates rather than individual rows.
    Therefore, all observations from the same trading date remain in
    the same subset.
    """
    if date_column not in df.columns:
        raise ValueError(
            f"Date column {date_column!r} is not present in the dataset."
        )

    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            "train_fraction must satisfy 0 < train_fraction < 1."
        )

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must satisfy "
            "0 < validation_fraction < 1."
        )

    if train_fraction + validation_fraction >= 1.0:
        raise ValueError(
            "train_fraction + validation_fraction must be less than 1."
        )

    dates = pd.to_datetime(
        df[date_column],
        errors="coerce",
    )

    if dates.isna().any():
        raise ValueError(
            "Missing or invalid dates detected before dataset splitting."
        )

    unique_dates = np.sort(
        dates.unique()
    )

    if len(unique_dates) < 3:
        raise ValueError(
            "At least three unique dates are required for "
            "train/validation/test splitting."
        )

    train_end = int(
        len(unique_dates) * train_fraction
    )

    validation_end = int(
        len(unique_dates)
        * (train_fraction + validation_fraction)
    )

    # Guarantee that each split contains at least one date.
    train_end = max(
        1,
        min(train_end, len(unique_dates) - 2),
    )

    validation_end = max(
        train_end + 1,
        min(validation_end, len(unique_dates) - 1),
    )

    train_dates = unique_dates[:train_end]

    validation_dates = unique_dates[
        train_end:validation_end
    ]

    test_dates = unique_dates[
        validation_end:
    ]

    train_mask = dates.isin(train_dates)
    validation_mask = dates.isin(validation_dates)
    test_mask = dates.isin(test_dates)

    return SplitIndices(
        train=np.flatnonzero(
            train_mask.to_numpy()
        ),
        validation=np.flatnonzero(
            validation_mask.to_numpy()
        ),
        test=np.flatnonzero(
            test_mask.to_numpy()
        ),
    )


def validate_chronological_split(
    df: pd.DataFrame,
    split: SplitIndices,
    date_column: str = "date",
) -> None:
    """
    Validate temporal separation between dataset subsets.

    The expected ordering is:

        max(train date)
            <
        min(validation date)
            <=
        max(validation date)
            <
        min(test date)

    Raises
    ------
    ValueError
        If the split is empty, overlapping, or not chronological.
    """
    if (
        len(split.train) == 0
        or len(split.validation) == 0
        or len(split.test) == 0
    ):
        raise ValueError(
            "Train, validation, and test splits must all be non-empty."
        )

    dates = pd.to_datetime(
        df[date_column]
    ).reset_index(drop=True)

    train_dates = dates.iloc[
        split.train
    ]

    validation_dates = dates.iloc[
        split.validation
    ]

    test_dates = dates.iloc[
        split.test
    ]

    if train_dates.max() >= validation_dates.min():
        raise ValueError(
            "Training and validation periods are not "
            "strictly separated."
        )

    if validation_dates.max() >= test_dates.min():
        raise ValueError(
            "Validation and test periods are not "
            "strictly separated."
        )


def subset_array(
    array: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """
    Select rows from an aligned NumPy array.

    This works for both two-dimensional feature matrices and
    three-dimensional temporal-window arrays.
    """
    return array[indices]
