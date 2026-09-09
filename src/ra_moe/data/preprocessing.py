"""
Basic preprocessing utilities for RA-MoE-4E.

This module handles column normalization, schema validation, numeric
conversion, and basic option-level derived features.

Temporal and rolling features are constructed separately in
``temporal.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import COLUMN_ALIASES, REQUIRED_COLUMNS


NUMERIC_COLUMNS = (
    "S",
    "K",
    "sigma",
    "tau",
    "r",
    "q",
    "market_price",
    "volume",
    "open_interest",
    "delta",
    "gamma",
    "vega",
    "theta",
)


def normalize_s(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw dataset column names to the internal RA-MoE schema.

    Parameters
    ----------
    df
        Raw option dataset.

    Returns
    -------
    pandas.DataFrame
        Copy of the dataset with recognized aliases renamed.
    """
    df = df.copy()

    rename_map = {
        source: target
        for source, target in COLUMN_ALIASES.items()
        if source in df.columns and target not in df.columns
    }

    return df.rename(columns=rename_map)


def validate_required_columns(df: pd.DataFrame) -> None:
    """
    Validate that all core dataset columns are available.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )


def convert_numeric_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert numerical columns to numeric dtype.

    This preserves the supplied final implementation:
    invalid values are replaced with zero, and missing optional
    numeric columns are initialized to zero.
    """
    df = df.copy()

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = (
                pd.to_numeric(
                    df[column],
                    errors="coerce",
                )
                .fillna(0.0)
            )
        else:
            df[column] = 0.0

    return df


def normalize_time_to_maturity(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize time-to-maturity to years.

    The original experimental code interpreted tau values greater than
    10 as being expressed in days and divided the entire column by 365.
    This behavior is preserved here for compatibility.
    """
    df = df.copy()

    if df["tau"].dropna().empty:
        return df

    if df["tau"].max() > 10:
        df["tau"] = df["tau"] / 365.0

    return df


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct option-level features that do not require rolling history.

    Currently constructs:

    - moneyness = S / K
    - relative bid-ask spread when bid/offer data are available
    """
    df = df.copy()

    df["moneyness"] = (
        df["S"] / (df["K"] + 1e-12)
    )

    if {
        "best_bid",
        "best_offer",
    }.issubset(df.columns):
        df["spread"] = (
            df["best_offer"] - df["best_bid"]
        ) / (
            df["market_price"].abs() + 1e-12
        )

    elif "spread" not in df.columns:
        df["spread"] = 0.0

    return df


def remove_invalid_contracts(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove observations with invalid spot or strike prices.

    This preserves the validity filter used in the supplied final
    implementation.
    """
    return df.loc[
        (df["S"] > 0)
        & (df["K"] > 0)
    ].copy()


def preprocess_basic(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the basic preprocessing pipeline.

    Processing order:

    1. Normalize column names.
    2. Validate required columns.
    3. Parse dates.
    4. Convert numerical columns.
    5. Normalize time-to-maturity.
    6. Remove invalid contracts.
    7. Construct basic option-level features.

    Returns
    -------
    pandas.DataFrame
        Preprocessed option-level dataset.
    """
    df = normalize_columns(df)

    validate_required_columns(df)

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = convert_numeric_columns(df)

    df = normalize_time_to_maturity(df)

    df = remove_invalid_contracts(df)

    df = add_basic_features(df)

    return df.reset_index(drop=True)
