"""
Training workflow for RA-MoE-4E.

This module refactors the multi-stage training procedure used in the
supplied final experimental implementation.

The implemented workflow is:

1. Residual-expert pretraining against market price minus BSM price.
2. Expert pretraining with the gating network frozen.
3. Joint fine-tuning with the gating network unfrozen.
4. Validation-based early stopping and restoration of the best state.

This preserves the supplied final implementation rather than adding
the separate gate-only training stage described in the thesis.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader

from .epoch import evaluate_model, train_epoch


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device | str,
    learning_rate: float = 5e-5,
    residual_pretrain_epochs: int = 5,
    expert_pretrain_epochs: int = 10,
    joint_epochs: int = 20,
    joint_learning_rate: float = 1e-4,
    patience: int = 5,
) -> dict[str, list[float]]:
    """
    Run the supplied RA-MoE-4E multi-stage training workflow.

    Parameters
    ----------
    model
        Complete RA-MoE-4E model.

    train_loader
        Training DataLoader.

    validation_loader
        Validation DataLoader.

    device
        PyTorch computation device.

    learning_rate
        Learning rate used for residual and expert pretraining.

    residual_pretrain_epochs
        Number of residual-expert pretraining epochs.

    expert_pretrain_epochs
        Number of epochs with the gate frozen.

    joint_epochs
        Maximum number of joint fine-tuning epochs.

    joint_learning_rate
        Learning rate used during joint fine-tuning.

    patience
        Early-stopping patience during joint fine-tuning.

    Returns
    -------
    dict
        Training and validation loss histories.
    """

    # -----------------------------------------------------------------
    # Stage 1: residual-expert pretraining
    # -----------------------------------------------------------------

    residual_optimizer = torch.optim.Adam(
        model.residual_expert.parameters(),
        lr=learning_rate,
    )

    residual_criterion = nn.MSELoss()

    for _ in range(residual_pretrain_epochs):
        model.residual_expert.train()

        for batch in train_loader:
            residual_optimizer.zero_grad()

            targets = batch["y"].to(device)
            bsm_prices = batch["bs"].to(device)
            residual_features = batch["x_resid"].to(device)

            residual_target = (
                targets - bsm_prices
            )

            residual_prediction = model.residual_expert(
                residual_features
            )

            residual_loss = residual_criterion(
                residual_prediction,
                residual_target,
            )

            residual_loss.backward()
            residual_optimizer.step()

    # -----------------------------------------------------------------
    # Stage 2: expert pretraining with gate frozen
    # -----------------------------------------------------------------

    for parameter in model.gate.parameters():
        parameter.requires_grad = False

    expert_optimizer = torch.optim.Adam(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=learning_rate,
    )

    pretrain_train_losses = []
    pretrain_validation_losses = []

    for epoch in range(expert_pretrain_epochs):
        training_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=expert_optimizer,
            epoch=epoch,
            device=device,
        )

        validation_result = evaluate_model(
            model=model,
            loader=validation_loader,
            device=device,
        )

        pretrain_train_losses.append(
            training_loss
        )

        pretrain_validation_losses.append(
            validation_result["mse"]
        )

    # -----------------------------------------------------------------
    # Stage 3: joint fine-tuning
    # -----------------------------------------------------------------

    for parameter in model.gate.parameters():
        parameter.requires_grad = True

    joint_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=joint_learning_rate,
    )

    joint_train_losses = []
    joint_validation_losses = []

    best_validation = float("inf")
    best_state = None
    counter = 0

    for epoch in range(joint_epochs):
        training_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=joint_optimizer,
            epoch=epoch,
            device=device,
        )

        validation_result = evaluate_model(
            model=model,
            loader=validation_loader,
            device=device,
        )

        validation_mse = validation_result[
            "mse"
        ]

        joint_train_losses.append(
            training_loss
        )

        joint_validation_losses.append(
            validation_mse
        )

        if validation_mse < best_validation:
            best_validation = validation_mse

            best_state = copy.deepcopy(
                model.state_dict()
            )

            counter = 0

        else:
            counter += 1

        if counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    return {
        "pretrain_train_losses": pretrain_train_losses,
        "pretrain_validation_losses": pretrain_validation_losses,
        "joint_train_losses": joint_train_losses,
        "joint_validation_losses": joint_validation_losses,
    }
