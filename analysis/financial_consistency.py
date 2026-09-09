"""
Post-hoc financial-consistency diagnostics for RA-MoE-4E.

This script refactors the financial-consistency analysis used in the
original SPX experiment.

The analysis evaluates:

1. Strike-monotonicity violations.
2. Discrete strike-convexity violations.
3. Calendar-spread violations.
4. Delta-hedging residuals.

Important
---------
These tests are empirical diagnostics on a filtered subset of SPX
call options. They should not be interpreted as theoretical guarantees
that RA-MoE-4E satisfies no-arbitrage conditions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(
    path: Path,
) -> pd.DataFrame:
    """
    Load and filter the option-pricing results used in the original
    financial-consistency analysis.

    The supplied implementation restricts the analysis to:

    - call options;
    - positive spot, strike, and time to maturity;
    - 0.95 < moneyness < 1.05;
    - maturity greater than seven days;
    - positive volume when available;
    - market price greater than 0.05.
    """
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    # Call options only.
    df = df[
        df["cp_flag"] == "C"
    ]

    # Valid observations.
    df = df[
        (df["tau"] > 0)
        & (df["S"] > 0)
        & (df["K"] > 0)
    ]

    # Approximately at-the-money options.
    df = df[
        (df["moneyness"] > 0.95)
        & (df["moneyness"] < 1.05)
    ]

    # Exclude ultra-short maturities.
    df = df[
        df["tau"] > 7 / 365
    ]

    # Exclude zero-volume observations when volume is available.
    if "volume" in df.columns:
        df = df[
            df["volume"] > 0
        ]

    # Exclude negligible market prices.
    df = df[
        df["market_price"] > 0.05
    ]

    return df


def monotonicity_violation(
    df: pd.DataFrame,
    price_column: str,
) -> float:
    """
    Compute the strike-monotonicity violation rate for call options.

    Within each date and maturity group, call prices are expected not
    to increase as strike increases.
    """
    violations = 0
    total = 0

    grouped = df.groupby(
        ["date", "tau"]
    )

    for _, group in grouped:
        group = group.sort_values(
            "K"
        )

        prices = group[
            price_column
        ].to_numpy()

        for index in range(
            len(prices) - 1
        ):
            total += 1

            if (
                prices[index + 1]
                > prices[index]
            ):
                violations += 1

    if total == 0:
        return np.nan

    return violations / total


def convexity_violation(
    df: pd.DataFrame,
    price_column: str,
) -> float:
    """
    Compute the discrete strike-convexity violation rate.

    Within each date and maturity group, neighboring option prices are
    evaluated using the second difference:

        P[i+1] - 2 * P[i] + P[i-1]

    A negative second difference is counted as a violation.

    Notes
    -----
    This preserves the original experimental diagnostic and does not
    adjust the second difference for unequal spacing between strikes.
    """
    violations = 0
    total = 0

    grouped = df.groupby(
        ["date", "tau"]
    )

    for _, group in grouped:
        group = group.sort_values(
            "K"
        )

        prices = group[
            price_column
        ].to_numpy()

        if len(prices) < 3:
            continue

        for index in range(
            1,
            len(prices) - 1,
        ):
            total += 1

            second_difference = (
                prices[index + 1]
                - 2 * prices[index]
                + prices[index - 1]
            )

            if second_difference < 0:
                violations += 1

    if total == 0:
        return np.nan

    return violations / total


def calendar_violation(
    df: pd.DataFrame,
    price_column: str,
) -> float:
    """
    Compute the calendar-spread violation rate.

    Within each date and strike group, the original analysis counts a
    violation when a longer-maturity call is cheaper than the preceding
    shorter-maturity call.
    """
    violations = 0
    total = 0

    grouped = df.groupby(
        ["date", "K"]
    )

    for _, group in grouped:
        group = group.sort_values(
            "tau"
        )

        prices = group[
            price_column
        ].to_numpy()

        for index in range(
            len(prices) - 1
        ):
            total += 1

            if (
                prices[index + 1]
                < prices[index]
            ):
                violations += 1

    if total == 0:
        return np.nan

    return violations / total


def delta_hedging(
    df: pd.DataFrame,
    price_column: str,
) -> tuple[float, float]:
    """
    Compute the delta-hedging diagnostics used in the original analysis.

    Observations are grouped by option identifier and ordered by date.
    For consecutive observations, the hedging residual is accumulated as:

        delta_price - delta * delta_spot

    where ``delta`` is the supplied option delta from the dataset.

    Returns
    -------
    rmse
        Root mean squared accumulated hedging residual across contracts.

    variance
        Variance of accumulated hedging residuals across contracts.

    Notes
    -----
    The hedge ratio is the supplied option delta. It is not a
    model-derived RA-MoE-4E delta.
    """
    pnl_list = []

    df = df.sort_values(
        ["optionid", "date"]
    )

    grouped = df.groupby(
        "optionid"
    )

    for _, group in grouped:
        group = group.sort_values(
            "date"
        )

        if len(group) < 2:
            continue

        pnl = 0.0

        for index in range(
            len(group) - 1
        ):
            row_t = group.iloc[index]
            row_t1 = group.iloc[index + 1]

            delta_spot = (
                row_t1["S"]
                - row_t["S"]
            )

            delta = row_t["delta"]

            delta_price = (
                row_t1[price_column]
                - row_t[price_column]
            )

            pnl += (
                delta_price
                - delta * delta_spot
            )

        pnl_list.append(
            pnl
        )

    pnl_array = np.asarray(
        pnl_list
    )

    rmse = np.sqrt(
        np.mean(
            pnl_array ** 2
        )
    )

    variance = np.var(
        pnl_array
    )

    return (
        float(rmse),
        float(variance),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the original RA-MoE-4E financial-consistency "
            "diagnostics on SPX result data."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help=(
            "Path to the SPX result CSV containing market prices, "
            "BSM predictions, RA-MoE predictions, and option features."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "financial_results"
        ),
        help=(
            "Directory used to save the financial-consistency tables."
        ),
    )

    args = parser.parse_args()

    df = load_data(
        args.data
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nFiltered sample size: {len(df):,}"
    )

    # -----------------------------------------------------------------
    # Strike monotonicity and discrete convexity
    # -----------------------------------------------------------------

    monotonicity_bsm = (
        monotonicity_violation(
            df,
            "bs_pred",
        )
    )

    monotonicity_hybrid = (
        monotonicity_violation(
            df,
            "ra_moe_pred",
        )
    )

    convexity_bsm = (
        convexity_violation(
            df,
            "bs_pred",
        )
    )

    convexity_hybrid = (
        convexity_violation(
            df,
            "ra_moe_pred",
        )
    )

    no_arbitrage_table = pd.DataFrame(
        {
            "Model": [
                "Black-Scholes",
                "RA-MoE-4E",
            ],
            "Monotonicity Violation": [
                monotonicity_bsm,
                monotonicity_hybrid,
            ],
            "Convexity Violation": [
                convexity_bsm,
                convexity_hybrid,
            ],
        }
    )

    no_arbitrage_path = (
        args.output_dir
        / "no_arbitrage.csv"
    )

    no_arbitrage_table.to_csv(
        no_arbitrage_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Calendar spread
    # -----------------------------------------------------------------

    calendar_bsm = (
        calendar_violation(
            df,
            "bs_pred",
        )
    )

    calendar_hybrid = (
        calendar_violation(
            df,
            "ra_moe_pred",
        )
    )

    calendar_table = pd.DataFrame(
        {
            "Model": [
                "Black-Scholes",
                "RA-MoE-4E",
            ],
            "Calendar Violation": [
                calendar_bsm,
                calendar_hybrid,
            ],
        }
    )

    calendar_path = (
        args.output_dir
        / "calendar.csv"
    )

    calendar_table.to_csv(
        calendar_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Delta hedging
    # -----------------------------------------------------------------

    (
        hedge_bsm_rmse,
        hedge_bsm_variance,
    ) = delta_hedging(
        df,
        "bs_pred",
    )

    (
        hedge_hybrid_rmse,
        hedge_hybrid_variance,
    ) = delta_hedging(
        df,
        "ra_moe_pred",
    )

    hedging_table = pd.DataFrame(
        {
            "Model": [
                "Black-Scholes",
                "RA-MoE-4E",
            ],
            "Hedging RMSE": [
                hedge_bsm_rmse,
                hedge_hybrid_rmse,
            ],
            "PnL Variance": [
                hedge_bsm_variance,
                hedge_hybrid_variance,
            ],
        }
    )

    hedging_path = (
        args.output_dir
        / "hedging.csv"
    )

    hedging_table.to_csv(
        hedging_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Display results
    # -----------------------------------------------------------------

    print(
        "\nFinancial-consistency diagnostics\n"
    )

    print(
        "Strike monotonicity and discrete convexity:"
    )
    print(
        no_arbitrage_table
    )

    print(
        "\nCalendar-spread consistency:"
    )
    print(
        calendar_table
    )

    print(
        "\nDelta-hedging diagnostics:"
    )
    print(
        hedging_table
    )

    print(
        f"\nSaved results to: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
