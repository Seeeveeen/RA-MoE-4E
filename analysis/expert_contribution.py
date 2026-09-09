"""
Post-hoc expert contribution analysis for RA-MoE-4E.

This script refactors the expert-sensitivity analysis used in the
original SPX and AAPL experiments.

Important
---------
This is a post-hoc analysis of predictions from an already trained
RA-MoE-4E model. Reduced expert configurations are not retrained.

The Static MLP and Transformer analyses remove the selected expert
output and renormalize the remaining routing weights.

The Residual analysis follows the original experiment by attenuating
the learned residual correction to 30% of its original magnitude
rather than fully removing the residual expert.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bsm_pred: np.ndarray,
    full_mse: float | None = None,
) -> tuple[float, float, float]:
    """
    Compute the metrics used in the original contribution analysis.

    Returns
    -------
    mse
        Mean squared error.

    improvement_vs_bsm
        Relative MSE improvement over the BSM baseline, in percent.

    deterioration_vs_full
        Relative MSE deterioration compared with the full RA-MoE-4E
        model, in percent.
    """
    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    bsm_mse = mean_squared_error(
        y_true,
        bsm_pred,
    )

    improvement_vs_bsm = (
        (bsm_mse - mse)
        / bsm_mse
        * 100
    )

    if full_mse is not None:
        deterioration_vs_full = (
            (mse - full_mse)
            / full_mse
            * 100
        )
    else:
        deterioration_vs_full = 0.0

    return (
        mse,
        improvement_vs_bsm,
        deterioration_vs_full,
    )


def compute_average_weights(
    weight_list: list[pd.Series],
    names: list[str],
) -> dict[str, float]:
    """
    Renormalize remaining routing weights observation by observation
    and report their average percentage allocation.
    """
    total = np.sum(
        weight_list,
        axis=0,
    )

    normalized = [
        np.mean(weight / total) * 100
        for weight in weight_list
    ]

    return {
        name: round(value, 4)
        for name, value in zip(
            names,
            normalized,
        )
    }


def run_contribution_analysis(
    df: pd.DataFrame,
    residual_attenuation: float = 0.3,
) -> tuple[
    dict[str, tuple[float, float, float]],
    dict[str, dict[str, float]],
]:
    """
    Run the original post-hoc expert contribution analysis.

    Parameters
    ----------
    df
        Result DataFrame containing market prices, model predictions,
        expert outputs, and routing weights.

    residual_attenuation
        Fraction of the learned residual correction retained in the
        residual sensitivity analysis.

        The original analysis uses 0.3.
    """
    y_true = df[
        "market_price"
    ].to_numpy()

    bsm_pred = df[
        "bs_pred"
    ].to_numpy()

    full_pred = df[
        "ra_moe_pred"
    ].to_numpy()

    full_mse = mean_squared_error(
        y_true,
        full_pred,
    )

    results = {}
    weight_results = {}

    # -----------------------------------------------------------------
    # Full model
    # -----------------------------------------------------------------

    results["Full Model"] = compute_metrics(
        y_true=y_true,
        y_pred=full_pred,
        bsm_pred=bsm_pred,
        full_mse=full_mse,
    )

    weight_results["Full Model"] = {
        "BSM": round(
            df["w_bs"].mean() * 100,
            4,
        ),
        "Residual": round(
            df["w_resid"].mean() * 100,
            4,
        ),
        "Static MLP": round(
            df["w_mlp"].mean() * 100,
            4,
        ),
        "Transformer": round(
            df["w_trans"].mean() * 100,
            4,
        ),
    }

    # -----------------------------------------------------------------
    # BSM-only baseline
    # -----------------------------------------------------------------

    results["BSM-only"] = compute_metrics(
        y_true=y_true,
        y_pred=bsm_pred,
        bsm_pred=bsm_pred,
        full_mse=full_mse,
    )

    weight_results["BSM-only"] = {
        "BSM": 100.0,
        "Residual": 0.0,
        "Static MLP": 0.0,
        "Transformer": 0.0,
    }

    # -----------------------------------------------------------------
    # Residual-correction attenuation
    #
    # This preserves the original prediction-level sensitivity
    # experiment. It is not a complete removal of Expert 2.
    # -----------------------------------------------------------------

    residual_correction = (
        df["expert_resid"]
        - df["bs_pred"]
    )

    attenuated_residual = (
        df["bs_pred"]
        + residual_attenuation
        * residual_correction
    )

    pred_residual_attenuated = (
        df["w_bs"]
        * df["expert_bs"]
        + df["w_resid"]
        * attenuated_residual
        + df["w_mlp"]
        * df["expert_mlp"]
        + df["w_trans"]
        * df["expert_trans"]
    )

    results[
        "Residual correction x0.3"
    ] = compute_metrics(
        y_true=y_true,
        y_pred=pred_residual_attenuated.to_numpy(),
        bsm_pred=bsm_pred,
        full_mse=full_mse,
    )

    # No redistributed-weight table is reported for this condition.
    # The residual expert remains active with its original routing
    # weight; only its correction magnitude is attenuated.

    # -----------------------------------------------------------------
    # Remove Static MLP
    # -----------------------------------------------------------------

    remaining_without_mlp = (
        df["w_bs"]
        + df["w_resid"]
        + df["w_trans"]
    )

    pred_without_mlp = (
        (
            df["w_bs"]
            / remaining_without_mlp
        )
        * df["expert_bs"]
        + (
            df["w_resid"]
            / remaining_without_mlp
        )
        * df["expert_resid"]
        + (
            df["w_trans"]
            / remaining_without_mlp
        )
        * df["expert_trans"]
    )

    results["Without Static MLP"] = (
        compute_metrics(
            y_true=y_true,
            y_pred=pred_without_mlp.to_numpy(),
            bsm_pred=bsm_pred,
            full_mse=full_mse,
        )
    )

    weight_results[
        "Without Static MLP"
    ] = compute_average_weights(
        weight_list=[
            df["w_bs"],
            df["w_resid"],
            df["w_trans"],
        ],
        names=[
            "BSM",
            "Residual",
            "Transformer",
        ],
    )

    # -----------------------------------------------------------------
    # Remove Transformer
    # -----------------------------------------------------------------

    remaining_without_transformer = (
        df["w_bs"]
        + df["w_resid"]
        + df["w_mlp"]
    )

    pred_without_transformer = (
        (
            df["w_bs"]
            / remaining_without_transformer
        )
        * df["expert_bs"]
        + (
            df["w_resid"]
            / remaining_without_transformer
        )
        * df["expert_resid"]
        + (
            df["w_mlp"]
            / remaining_without_transformer
        )
        * df["expert_mlp"]
    )

    results["Without Transformer"] = (
        compute_metrics(
            y_true=y_true,
            y_pred=pred_without_transformer.to_numpy(),
            bsm_pred=bsm_pred,
            full_mse=full_mse,
        )
    )

    weight_results[
        "Without Transformer"
    ] = compute_average_weights(
        weight_list=[
            df["w_bs"],
            df["w_resid"],
            df["w_mlp"],
        ],
        names=[
            "BSM",
            "Residual",
            "Static MLP",
        ],
    )

    return results, weight_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run post-hoc RA-MoE-4E expert "
            "contribution analysis."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help=(
            "Path to an RA-MoE-4E result CSV containing "
            "expert outputs and routing weights."
        ),
    )

    parser.add_argument(
        "--market",
        type=str,
        required=True,
        help="Market label, e.g. SPX or AAPL.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for output tables.",
    )

    args = parser.parse_args()

    df = pd.read_csv(
        args.data
    )

    results, weight_results = (
        run_contribution_analysis(df)
    )

    results_df = pd.DataFrame(
        results,
        index=[
            "MSE",
            "Improvement_vs_BSM(%)",
            "Deterioration_vs_Full(%)",
        ],
    ).T.round(4)

    weights_df = pd.DataFrame(
        weight_results
    ).T.round(4)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        args.output_dir
        / f"expert_contribution_{args.market}.csv"
    )

    weights_path = (
        args.output_dir
        / f"routing_weights_{args.market}.csv"
    )

    results_df.to_csv(
        results_path
    )

    weights_df.to_csv(
        weights_path
    )

    print(
        "\nExpert contribution analysis\n"
    )
    print(results_df)

    print(
        "\nRouting-weight redistribution\n"
    )
    print(weights_df)

    print(
        f"\nSaved: {results_path}"
    )
    print(
        f"Saved: {weights_path}"
    )


if __name__ == "__main__":
    main()
