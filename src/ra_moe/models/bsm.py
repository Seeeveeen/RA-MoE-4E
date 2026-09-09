"""
Black-Scholes-Merton pricing utilities for RA-MoE-4E.

This module implements the theory-driven expert used in the
Regime-Aware Mixture-of-Experts (RA-MoE-4E) framework.

Both NumPy and PyTorch implementations are provided:

- ``bsm_price_numpy`` is used for preprocessing and baseline evaluation.
- ``bsm_price_torch`` provides a tensor-compatible implementation.

The functions support European call and put options with a continuous
dividend yield.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
from scipy.stats import norm


OptionType = Literal["call", "put"]

_EPS = 1e-12


def bsm_price_numpy(
    spot,
    strike,
    rate,
    dividend_yield,
    volatility,
    time_to_maturity,
    option_type: OptionType = "call",
):
    """
    Compute Black-Scholes-Merton option prices using NumPy.

    Parameters
    ----------
    spot
        Current underlying asset price.
    strike
        Option strike price.
    rate
        Continuously compounded risk-free interest rate.
    dividend_yield
        Continuous dividend yield.
    volatility
        Implied volatility.
    time_to_maturity
        Time to maturity in years.
    option_type
        Either ``"call"`` or ``"put"``.

    Returns
    -------
    numpy.ndarray
        Black-Scholes-Merton option prices.
    """
    _validate_option_type(option_type)

    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    rate = np.asarray(rate, dtype=float)
    dividend_yield = np.asarray(dividend_yield, dtype=float)
    volatility = np.asarray(volatility, dtype=float)
    time_to_maturity = np.asarray(time_to_maturity, dtype=float)

    sqrt_tau = np.sqrt(np.maximum(time_to_maturity, _EPS))

    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (
            np.log(spot / strike)
            + (
                rate
                - dividend_yield
                + 0.5 * volatility**2
            )
            * time_to_maturity
        ) / (volatility * sqrt_tau + _EPS)

        d2 = d1 - volatility * sqrt_tau

    call_price = (
        spot
        * np.exp(-dividend_yield * time_to_maturity)
        * norm.cdf(d1)
        - strike
        * np.exp(-rate * time_to_maturity)
        * norm.cdf(d2)
    )

    if option_type == "call":
        return call_price

    put_price = (
        strike
        * np.exp(-rate * time_to_maturity)
        * norm.cdf(-d2)
        - spot
        * np.exp(-dividend_yield * time_to_maturity)
        * norm.cdf(-d1)
    )

    return put_price


def bsm_price_torch(
    spot: torch.Tensor,
    strike: torch.Tensor,
    rate: torch.Tensor,
    dividend_yield: torch.Tensor,
    volatility: torch.Tensor,
    time_to_maturity: torch.Tensor,
    option_type: OptionType = "call",
) -> torch.Tensor:
    """
    Compute Black-Scholes-Merton option prices using PyTorch.

    This implementation preserves tensor operations and can therefore
    be used inside PyTorch-based workflows.

    Parameters
    ----------
    spot
        Current underlying asset price.
    strike
        Option strike price.
    rate
        Continuously compounded risk-free interest rate.
    dividend_yield
        Continuous dividend yield.
    volatility
        Implied volatility.
    time_to_maturity
        Time to maturity in years.
    option_type
        Either ``"call"`` or ``"put"``.

    Returns
    -------
    torch.Tensor
        Black-Scholes-Merton option prices.
    """
    _validate_option_type(option_type)

    sqrt_tau = torch.sqrt(
        torch.clamp(time_to_maturity, min=_EPS)
    )

    denominator = volatility * sqrt_tau + _EPS

    d1 = (
        torch.log(spot / strike)
        + (
            rate
            - dividend_yield
            + 0.5 * volatility**2
        )
        * time_to_maturity
    ) / denominator

    d2 = d1 - volatility * sqrt_tau

    normal_cdf_d1 = _normal_cdf_torch(d1)
    normal_cdf_d2 = _normal_cdf_torch(d2)

    call_price = (
        spot
        * torch.exp(-dividend_yield * time_to_maturity)
        * normal_cdf_d1
        - strike
        * torch.exp(-rate * time_to_maturity)
        * normal_cdf_d2
    )

    if option_type == "call":
        return call_price

    put_price = (
        strike
        * torch.exp(-rate * time_to_maturity)
        * _normal_cdf_torch(-d2)
        - spot
        * torch.exp(-dividend_yield * time_to_maturity)
        * _normal_cdf_torch(-d1)
    )

    return put_price


def _normal_cdf_torch(x: torch.Tensor) -> torch.Tensor:
    """Standard normal cumulative distribution function in PyTorch."""
    return 0.5 * (
        1.0 + torch.erf(x / math.sqrt(2.0))
    )


def _validate_option_type(option_type: str) -> None:
    """Validate the supported option type."""
    if option_type not in {"call", "put"}:
        raise ValueError(
            f"option_type must be 'call' or 'put', got {option_type!r}"
        )
