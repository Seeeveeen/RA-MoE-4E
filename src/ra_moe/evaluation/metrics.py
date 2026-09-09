"""
Core evaluation metrics for RA-MoE-4E.

This module refactors the basic pricing and gating diagnostics used in
the original evaluation scripts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


def compute_metrics(
    df: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """
    Compute RA-MoE-4E pricing-error metrics.

    Returns
    -------
    mse
        Mean squared error.

    mae
        Mean absolute error.

    rmse
        Root mean squared error.

    rpe
        Mean relative pricing error.
    """
    y_true = df["market_price"]
    y_pred = df["ra_moe_pred"]

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(mse)

    rpe = np.mean(
        np.abs(y_pred - y_true)
        / (np.abs(y_true) + 1e-8)
    )

    return mse, mae, rmse, rpe


def compute_bsm_metrics(
    df: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """
    Compute the same pricing-error metrics for the BSM baseline.
    """
    y_true = df["market_price"]
    y_pred = df["bs_pred"]

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(mse)

    rpe = np.mean(
        np.abs(y_pred - y_true)
        / (np.abs(y_true) + 1e-8)
    )

    return mse, mae, rmse, rpe


def baseline_comparison(
    df: pd.DataFrame,
) -> tuple[float, float]:
    """
    Compare RA-MoE-4E MSE with the BSM baseline.

    Returns
    -------
    bsm_mse
        BSM mean squared error.

    improvement
        Relative MSE improvement of RA-MoE-4E over BSM, in percent.
    """
    y_true = df["market_price"]

    hybrid_mse = mean_squared_error(
        y_true,
        df["ra_moe_pred"],
    )

    bsm_mse = mean_squared_error(
        y_true,
        df["bs_pred"],
    )

    improvement = (
        (bsm_mse - hybrid_mse)
        / bsm_mse
        * 100
    )

    return bsm_mse, improvement


def gating_analysis(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute mean and standard deviation of expert routing weights.
    """
    weights = df[
        [
            "w_bs",
            "w_resid",
            "w_mlp",
            "w_trans",
        ]
    ]

    return (
        weights.mean(),
        weights.std(),
    )


def error_stats(
    df: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """
    Summarize the RA-MoE-4E pricing-error distribution.
    """
    error = (
        df["ra_moe_pred"]
        - df["market_price"]
    )

    return (
        np.mean(error),
        np.std(error),
        pd.Series(error).skew(),
        pd.Series(error).kurt(),
    )
