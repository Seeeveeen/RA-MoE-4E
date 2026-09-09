"""
Cross-market gating analysis for RA-MoE-4E.

This script compares the average routing weights learned for SPX
and AAPL in the original experimental analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WEIGHT_COLUMNS = [
    "w_bs",
    "w_resid",
    "w_mlp",
    "w_trans",
]

EXPERT_NAMES = [
    "BSM",
    "Residual",
    "Static MLP",
    "Transformer",
]


def load_data(
    path: Path,
) -> pd.DataFrame:
    """Load an RA-MoE-4E result file."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    return df


def compute_average_weights(
    df: pd.DataFrame,
) -> pd.Series:
    """Compute average routing weights across observations."""
    return df[WEIGHT_COLUMNS].mean()


def plot_cross_market(
    spx_weights: pd.Series,
    aapl_weights: pd.Series,
    output_path: Path,
) -> None:
    """Plot SPX and AAPL average routing weights."""
    x = np.arange(
        len(EXPERT_NAMES)
    )

    width = 0.35

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        x - width / 2,
        spx_weights.values,
        width,
        label="SPX",
    )

    plt.bar(
        x + width / 2,
        aapl_weights.values,
        width,
        label="AAPL",
    )

    plt.xticks(
        x,
        EXPERT_NAMES,
    )

    plt.ylabel(
        "Average Gating Weight"
    )

    plt.title(
        "Cross-Market Comparison of Gating Allocations"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare average RA-MoE-4E routing weights "
            "between SPX and AAPL."
        )
    )

    parser.add_argument(
        "--spx",
        type=Path,
        required=True,
        help="Path to the SPX result CSV.",
    )

    parser.add_argument(
        "--aapl",
        type=Path,
        required=True,
        help="Path to the AAPL result CSV.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "cross_market_gating.png"
        ),
        help="Output figure path.",
    )

    args = parser.parse_args()

    spx_df = load_data(
        args.spx
    )

    aapl_df = load_data(
        args.aapl
    )

    spx_weights = compute_average_weights(
        spx_df
    )

    aapl_weights = compute_average_weights(
        aapl_df
    )

    print("\nSPX average routing weights:")
    print(spx_weights)

    print("\nAAPL average routing weights:")
    print(aapl_weights)

    plot_cross_market(
        spx_weights=spx_weights,
        aapl_weights=aapl_weights,
        output_path=args.output,
    )

    print(
        f"\nFigure saved to: {args.output}"
    )


if __name__ == "__main__":
    main()
