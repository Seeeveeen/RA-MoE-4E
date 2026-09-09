"""
Epoch-level training and evaluation utilities for RA-MoE-4E.

This module refactors the ``train_epoch`` and ``eval_model`` functions
from the supplied final experimental implementation.

The original training behavior is preserved. Potential methodological
or implementation issues are documented separately rather than
silently corrected.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .losses import entropy_loss, load_balance_loss


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    alpha_ent: float = 4.0,
    alpha_bal: float = 12.0,
    alpha_thy: float = 1e-3,
    device: torch.device | str = "cpu",
) -> float:
    """
    Train RA-MoE-4E for one epoch.

    This preserves the loss composition used in the supplied final
    implementation:

    - pricing MSE;
    - routing entropy term;
    - load-balancing penalty;
    - BSM routing-weight penalty;
    - optional monotonicity penalty.

    Parameters
    ----------
    model
        RA-MoE-4E model.

    loader
        Training DataLoader.

    optimizer
        PyTorch optimizer.

    epoch
        Current epoch index. Retained from the original function
        signature.

    alpha_ent
        Weight applied to the entropy term.

    alpha_bal
        Weight applied to the load-balancing term.

    alpha_thy
        Weight applied to the optional monotonicity term.

    device
        Device used for model computation.

    Returns
    -------
    float
        Average total training loss over the dataset.
    """
    model.train()

    total_loss = 0.0

    criterion = nn.MSELoss(
        reduction="mean"
    )

    for batch in loader:
        targets = batch["y"].to(device)
        bsm_prices = batch["bs"].to(device)
        residual_features = batch["x_resid"].to(device)
        static_features = batch["x_static"].to(device)
        temporal_features = batch["temporal"].to(device)
        gate_features = batch["x_gate"].to(device)

        optimizer.zero_grad()

        predictions, weights, _ = model(
            bsm_prices,
            residual_features,
            static_features,
            temporal_features,
            gate_features,
        )

        # -------------------------------------------------------------
        # Main pricing loss
        # -------------------------------------------------------------

        pricing_loss = criterion(
            predictions,
            targets,
        )

        # -------------------------------------------------------------
        # Routing regularization
        # -------------------------------------------------------------

        entropy = entropy_loss(weights)

        balance = load_balance_loss(weights)

        loss = (
            pricing_loss
            + alpha_ent * entropy
            + alpha_bal * balance
        )

        # -------------------------------------------------------------
        # BSM routing-weight penalty
        #
        # The supplied implementation penalizes BSM routing weights
        # above 0.6.
        # -------------------------------------------------------------

        bsm_weight = weights[:, 0]

        bsm_penalty = torch.mean(
            torch.relu(
                bsm_weight - 0.6
            )
            ** 2
        )

        loss = loss + bsm_penalty

        # -------------------------------------------------------------
        # Optional monotonicity penalty
        #
        # This block preserves the supplied final implementation.
        # Its gradient behavior is retained here rather than silently
        # modified.
        # -------------------------------------------------------------

        if alpha_thy > 0:
            try:
                dataset = loader.dataset

                scale_s = (
                    dataset.scaler.scale_[0]
                    if hasattr(dataset, "scaler")
                    else 1.0
                )

                relative_step = 1e-3

                delta_scaled = (
                    relative_step * scale_s
                )

                static_features_plus = (
                    static_features.clone()
                )

                static_features_plus[:, 0] = (
                    static_features_plus[:, 0]
                    + delta_scaled
                )

                residual_features_plus = (
                    residual_features.clone()
                )

                residual_features_plus[:, 0] = (
                    residual_features_plus[:, 0]
                    + delta_scaled
                )

                gate_features_plus = (
                    gate_features.clone()
                )

                gate_features_plus[:, 0] = (
                    gate_features_plus[:, 0]
                    + delta_scaled
                )

                predictions_plus, _, _ = model(
                    bsm_prices,
                    residual_features_plus,
                    static_features_plus,
                    temporal_features,
                    gate_features_plus,
                )

                with torch.no_grad():
                    predictions_plus, _, _ = model(
                        bsm_prices,
                        residual_features_plus,
                        static_features_plus,
                        temporal_features,
                        gate_features_plus,
                    )

                difference = (
                    predictions_plus
                    - predictions
                )

                negative_part = torch.relu(
                    -difference
                )

                theory_loss = (
                    negative_part.mean()
                )

                loss = (
                    loss
                    + alpha_thy * theory_loss
                )

            except Exception:
                pass

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * targets.shape[0]
        )

    return (
        total_loss
        / len(loader.dataset)
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
) -> dict[str, np.ndarray | float]:
    """
    Evaluate RA-MoE-4E on an aligned dataset.

    Returns the pricing predictions together with expert routing
    weights and individual expert outputs, matching the information
    collected by the supplied final implementation.
    """
    model.eval()

    targets_list = []
    predictions_list = []
    weights_list = []
    experts_list = []

    with torch.no_grad():
        for batch in loader:
            targets = batch["y"].to(device)
            bsm_prices = batch["bs"].to(device)
            residual_features = batch["x_resid"].to(device)
            static_features = batch["x_static"].to(device)
            temporal_features = batch["temporal"].to(device)
            gate_features = batch["x_gate"].to(device)

            predictions, weights, experts = model(
                bsm_prices,
                residual_features,
                static_features,
                temporal_features,
                gate_features,
            )

            targets_list.append(
                targets.cpu().numpy()
            )

            predictions_list.append(
                predictions.cpu().numpy()
            )

            weights_list.append(
                weights.cpu().numpy()
            )

            experts_list.append(
                experts.cpu().numpy()
            )

    targets = np.concatenate(
        targets_list
    )

    predictions = np.concatenate(
        predictions_list
    )

    weights = np.concatenate(
        weights_list
    )

    experts = np.concatenate(
        experts_list
    )

    mse = np.mean(
        (targets - predictions) ** 2
    )

    return {
        "mse": float(mse),
        "y": targets,
        "pred": predictions,
        "weights": weights,
        "experts": experts,
    }
