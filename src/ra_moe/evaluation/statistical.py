"""
Statistical comparison utilities for RA-MoE-4E.

This module refactors the Diebold-Mariano test used in the supplied
final experimental implementation to compare RA-MoE-4E and BSM
pricing errors.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def dm_test(
    model_errors: np.ndarray,
    baseline_errors: np.ndarray,
    h: int = 1,
    power: int = 2,
) -> tuple[float, float]:
    """
    Perform the Diebold-Mariano test used in the original experiment.

    Parameters
    ----------
    model_errors
        Pricing errors from RA-MoE-4E.

    baseline_errors
        Pricing errors from the BSM baseline.

    h
        Forecast horizon used in the variance calculation.
        The original implementation uses h=1.

    power
        Power applied to absolute errors when constructing the
        loss differential. The original implementation uses power=2.

    Returns
    -------
    dm_statistic
        Diebold-Mariano test statistic.

    p_value
        Two-sided p-value based on the standard normal distribution.
    """
    loss_differential = (
        np.abs(model_errors) ** power
        - np.abs(baseline_errors) ** power
    )

    n_observations = len(
        loss_differential
    )

    mean_differential = np.mean(
        loss_differential
    )

    def autocovariance(
        values: np.ndarray,
        lag: int,
    ) -> float:
        if lag == 0:
            return np.mean(
                (values - values.mean()) ** 2
            )

        return np.mean(
            (values[lag:] - values.mean())
            * (
                values[:-lag]
                - values.mean()
            )
        )

    autocovariances = [
        autocovariance(
            loss_differential,
            lag,
        )
        for lag in range(h)
    ]

    variance = (
        autocovariances[0]
        + 2 * sum(
            autocovariances[1:]
        )
    )

    dm_statistic = (
        mean_differential
        / np.sqrt(
            variance / n_observations
        )
    )

    dm_statistic *= np.sqrt(
        (
            n_observations
            + 1
            - 2 * h
            + h * (h - 1)
            / n_observations
        )
        / n_observations
    )

    p_value = (
        2
        * (
            1
            - stats.norm.cdf(
                np.abs(dm_statistic)
            )
        )
    )

    return (
        float(dm_statistic),
        float(p_value),
    )
