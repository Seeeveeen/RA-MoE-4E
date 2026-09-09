"""
Training entry point for RA-MoE-4E.

This script connects the modular model, data, and training components
while preserving the core workflow of the supplied final experimental
implementation.

The script performs:

1. Data loading and feature construction.
2. Sequential 80/10/10 train-validation-test splitting.
3. PyTorch Dataset and DataLoader construction.
4. RA-MoE-4E model initialization.
5. Multi-stage model training.
6. Final test-set evaluation.

Result saving, statistical tests, and diagnostic plotting are handled
separately from this core training entry point.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------
# Make the local src/ package importable when this script is executed
# directly from the repository.
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from ra_moe.data import (  # noqa: E402
    OptionDataset,
    load_data,
    split_indices,
)
from ra_moe.models import RAMoE4E  # noqa: E402
from ra_moe.training import (  # noqa: E402
    evaluate_model,
    train_model,
)


# ---------------------------------------------------------------------
# Original experiment settings
# ---------------------------------------------------------------------

SEED = 42

BATCH_SIZE = 512

SEQUENCE_LENGTH = 10

LEARNING_RATE = 5e-5

RESIDUAL_PRETRAIN_EPOCHS = 5

EXPERT_PRETRAIN_EPOCHS = 10

JOINT_EPOCHS = 20

JOINT_LEARNING_RATE = 1e-4

EARLY_STOPPING_PATIENCE = 5


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Only the data path and optional sampling fraction are exposed here.
    Model and training defaults preserve the supplied final experiment.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train the RA-MoE-4E option-pricing model."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help=(
            "Path to the cleaned option CSV file. "
            "Raw OptionMetrics data are not included in this repository."
        ),
    )

    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of aligned observations used for training. "
            "Default: 1.0, matching the full supplied experiment."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the RA-MoE-4E training experiment."""
    args = parse_args()

    # -----------------------------------------------------------------
    # Reproducibility settings
    # -----------------------------------------------------------------

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # -----------------------------------------------------------------
    # Load and prepare aligned model inputs
    # -----------------------------------------------------------------

    (
        static_features,
        temporal_features,
        targets,
        bsm_prices,
        residual_extra_features,
        gate_extra_features,
        aligned_df,
    ) = load_data(
        data_path=str(args.data),
        sample_fraction=args.sample_fraction,
        sequence_length=SEQUENCE_LENGTH,
        temporal_group_key="optionid",
        seed=SEED,
    )

    print(
        f"Loaded {len(targets):,} aligned observations."
    )

    print(
        "Feature shapes:"
        f"\n  static:       {static_features.shape}"
        f"\n  temporal:     {temporal_features.shape}"
        f"\n  residual:     {residual_extra_features.shape}"
        f"\n  gate extra:   {gate_extra_features.shape}"
    )

    # -----------------------------------------------------------------
    # Sequential 80/10/10 split
    # -----------------------------------------------------------------

    (
        train_indices,
        validation_indices,
        test_indices,
    ) = split_indices(
        n_observations=len(targets),
        seed=SEED,
    )

    # Training subset
    static_train = static_features[train_indices]
    temporal_train = temporal_features[train_indices]
    targets_train = targets[train_indices]
    bsm_train = bsm_prices[train_indices]
    residual_train = residual_extra_features[
        train_indices
    ]
    gate_train = gate_extra_features[
        train_indices
    ]

    # Validation subset
    static_validation = static_features[
        validation_indices
    ]
    temporal_validation = temporal_features[
        validation_indices
    ]
    targets_validation = targets[
        validation_indices
    ]
    bsm_validation = bsm_prices[
        validation_indices
    ]
    residual_validation = residual_extra_features[
        validation_indices
    ]
    gate_validation = gate_extra_features[
        validation_indices
    ]

    # Test subset
    static_test = static_features[test_indices]
    temporal_test = temporal_features[test_indices]
    targets_test = targets[test_indices]
    bsm_test = bsm_prices[test_indices]
    residual_test = residual_extra_features[
        test_indices
    ]
    gate_test = gate_extra_features[
        test_indices
    ]

    print(
        "Dataset split:"
        f"\n  train:      {len(train_indices):,}"
        f"\n  validation: {len(validation_indices):,}"
        f"\n  test:       {len(test_indices):,}"
    )

    # -----------------------------------------------------------------
    # PyTorch datasets
    # -----------------------------------------------------------------

    train_dataset = OptionDataset(
        static_features=static_train,
        temporal_features=temporal_train,
        targets=targets_train,
        bsm_prices=bsm_train,
        static_scaler=None,
        gate_extra_features=gate_train,
        residual_extra_features=residual_train,
    )

    validation_dataset = OptionDataset(
        static_features=static_validation,
        temporal_features=temporal_validation,
        targets=targets_validation,
        bsm_prices=bsm_validation,
        static_scaler=train_dataset.scaler,
        gate_extra_features=gate_validation,
        residual_extra_features=residual_validation,
    )

    test_dataset = OptionDataset(
        static_features=static_test,
        temporal_features=temporal_test,
        targets=targets_test,
        bsm_prices=bsm_test,
        static_scaler=train_dataset.scaler,
        gate_extra_features=gate_test,
        residual_extra_features=residual_test,
    )

    # -----------------------------------------------------------------
    # DataLoaders
    #
    # Preserve the supplied final implementation:
    # - training batch size = 512
    # - validation/test batch size = 1024
    # - training shuffle = True
    # - training drop_last = True
    # -----------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
    )

    # -----------------------------------------------------------------
    # Infer model dimensions from the actual prepared datasets
    # -----------------------------------------------------------------

    static_dim = train_dataset.static_features.shape[1]

    residual_dim = (
        train_dataset.residual_features.shape[1]
    )

    gate_dim = train_dataset.gate_features.shape[1]

    temporal_dim = (
        train_dataset.temporal_features.shape[2]
    )

    print(
        "Model input dimensions:"
        f"\n  static_dim:   {static_dim}"
        f"\n  residual_dim: {residual_dim}"
        f"\n  gate_dim:     {gate_dim}"
        f"\n  temporal_dim: {temporal_dim}"
    )

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------

    model = RAMoE4E(
        static_dim=static_dim,
        residual_dim=residual_dim,
        gate_dim=gate_dim,
        temporal_dim=temporal_dim,
        residual_scale=5.0,
        gate_temperature=4.0,
        transformer_d_model=64,
        transformer_nhead=4,
        transformer_num_layers=2,
    ).to(device)

    # -----------------------------------------------------------------
    # Multi-stage training
    # -----------------------------------------------------------------

    print("Starting RA-MoE-4E training...")

    training_history = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        residual_pretrain_epochs=(
            RESIDUAL_PRETRAIN_EPOCHS
        ),
        expert_pretrain_epochs=(
            EXPERT_PRETRAIN_EPOCHS
        ),
        joint_epochs=JOINT_EPOCHS,
        joint_learning_rate=(
            JOINT_LEARNING_RATE
        ),
        patience=EARLY_STOPPING_PATIENCE,
    )

    # -----------------------------------------------------------------
    # Final test evaluation
    # -----------------------------------------------------------------

    test_result = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
    )

    bsm_test_array = (
        test_dataset.bsm_prices
        .cpu()
        .numpy()
    )

    target_test_array = (
        test_dataset.targets
        .cpu()
        .numpy()
    )

    bsm_mse = np.mean(
        (
            target_test_array
            - bsm_test_array
        )
        ** 2
    )

    hybrid_mse = test_result["mse"]

    if bsm_mse != 0:
        improvement = (
            (bsm_mse - hybrid_mse)
            / bsm_mse
            * 100.0
        )
    else:
        improvement = 0.0

    print("\nFinal test evaluation")
    print("---------------------")
    print(
        f"RA-MoE-4E MSE: {hybrid_mse:.6f}"
    )
    print(
        f"BSM baseline MSE: {bsm_mse:.6f}"
    )
    print(
        "Relative MSE improvement vs BSM: "
        f"{improvement:.2f}%"
    )

    average_weights = (
        test_result["weights"]
        .mean(axis=0)
    )

    print("\nAverage routing weights")
    print("-----------------------")
    print(
        f"BSM:         {average_weights[0]:.4f}"
    )
    print(
        f"Residual:    {average_weights[1]:.4f}"
    )
    print(
        f"Static MLP:  {average_weights[2]:.4f}"
    )
    print(
        f"Transformer: {average_weights[3]:.4f}"
    )

    # Retain the history object for future result/output integration.
    _ = training_history

    # Retain alignment information for future evaluation scripts.
    _ = aligned_df


if __name__ == "__main__":
    main()
