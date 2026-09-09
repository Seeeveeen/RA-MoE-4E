"""
Generate summary figures for the RA-MoE-4E README.

The figures are generated from the public aggregate pricing summary
rather than from proprietary row-level option data.

Outputs
-------
1. pricing_performance.png
2. cross_market_gating.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_pricing_performance(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot relative MSE improvement over BSM together with RMSE changes.
    """
    markets = df["Market"]

    improvement = df[
        "MSE_Improvement_Pct"
    ]

    ra_moe_rmse = df[
        "RA_MoE_RMSE"
    ]

    bsm_rmse = df[
        "BSM_RMSE"
    ]

    fig, ax = plt.subplots(
        figsize=(8, 4.8)
    )

    bars = ax.barh(
        markets,
        improvement,
    )

    ax.set_xlabel(
        "Relative MSE improvement vs BSM (%)"
    )

    ax.set_title(
        "RA-MoE-4E Pricing Performance"
    )

    ax.set_xlim(
        0,
        improvement.max() * 1.25,
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    for (
        bar,
        improvement_value,
        ra_rmse,
        bs_rmse,
    ) in zip(
        bars,
        improvement,
        ra_moe_rmse,
        bsm_rmse,
    ):
        y_position = (
            bar.get_y()
            + bar.get_height() / 2
        )

        ax.text(
            improvement_value + 0.2,
            y_position,
            f"{improvement_value:.2f}%",
            va="center",
            fontweight="bold",
        )

        ax.text(
            0.25,
            y_position,
            (
                f"RMSE "
                f"{bs_rmse:.4f} → "
                f"{ra_rmse:.4f}"
            ),
            va="center",
        )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_cross_market_gating(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Compare average expert-routing weights between SPX and AAPL.
    """
    experts = [
        "BSM",
        "Residual MLP",
        "Static MLP",
        "Transformer",
    ]

    weight_columns = [
        "BSM_Weight",
        "Residual_Weight",
        "Static_MLP_Weight",
        "Transformer_Weight",
    ]

    x = np.arange(
        len(experts)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(9, 5.2)
    )

    for offset, (_, row) in zip(
        [-width / 2, width / 2],
        df.iterrows(),
    ):
        weights = (
            row[weight_columns]
            .to_numpy(dtype=float)
            * 100
        )

        bars = ax.bar(
            x + offset,
            weights,
            width,
            label=row["Market"],
        )

        for bar in bars:
            height = bar.get_height()

            ax.text(
                (
                    bar.get_x()
                    + bar.get_width() / 2
                ),
                height + 1.0,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_ylabel(
        "Average routing weight (%)"
    )

    ax.set_title(
        "Cross-Market Expert Allocation"
    )

    ax.set_xticks(
        x,
        experts,
    )

    ax.set_ylim(
        0,
        105,
    )

    ax.legend(
        frameon=False,
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate public RA-MoE-4E "
            "summary figures."
        )
    )

    parser.add_argument(
        "--results",
        type=Path,
        default=Path(
            "results/pricing_summary.csv"
        ),
        help=(
            "Path to the canonical aggregate "
            "pricing summary."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for generated figures.",
    )

    args = parser.parse_args()

    df = pd.read_csv(
        args.results
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pricing_path = (
        args.output_dir
        / "pricing_performance.png"
    )

    gating_path = (
        args.output_dir
        / "cross_market_gating.png"
    )

    plot_pricing_performance(
        df=df,
        output_path=pricing_path,
    )

    plot_cross_market_gating(
        df=df,
        output_path=gating_path,
    )

    print(
        f"Saved: {pricing_path}"
    )

    print(
        f"Saved: {gating_path}"
    )


if __name__ == "__main__":
    main()
