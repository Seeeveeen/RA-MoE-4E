"""
MLP-based experts for the RA-MoE-4E framework.

This module implements two specialized neural experts:

- ResidualExpert:
  learns systematic corrections relative to the BSM theoretical anchor.

- StaticMLPExpert:
  learns nonlinear cross-sectional relationships directly from
  static market and option features.

Both experts use a shared feed-forward MLP building block while
retaining distinct architectures and research roles.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class FeedForwardMLP(nn.Module):
    """
    Generic feed-forward multilayer perceptron.

    Parameters
    ----------
    input_dim
        Number of input features.
    hidden_dims
        Width of each hidden layer.
    dropout
        Dropout probability applied after each hidden-layer activation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        layers: list[nn.Module] = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("All hidden dimensions must be positive.")

            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Produce one scalar prediction per observation.

        Parameters
        ----------
        x
            Tensor of shape ``(batch_size, input_dim)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch_size,)``.
        """
        return self.network(x).squeeze(-1)


class ResidualExpert(nn.Module):
    """
    Expert 2: residual-correction network.

    This lightweight MLP is designed to learn systematic pricing
    corrections relative to the Black-Scholes-Merton theoretical
    baseline.

    The original experimental architecture uses two hidden layers
    with 128 and 64 units.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.mlp = FeedForwardMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict a residual correction signal.

        The conversion of this residual signal into the full Expert 2
        price is handled by the complete RA-MoE model rather than by
        this module.
        """
        return self.mlp(x)


class StaticMLPExpert(nn.Module):
    """
    Expert 3: nonlinear cross-sectional pricing network.

    This deeper MLP directly models nonlinear relationships among
    static option and market features.

    The original experimental architecture uses three hidden layers
    with 256, 128, and 64 units.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 128, 64),
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        self.mlp = FeedForwardMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict an option price from static cross-sectional features."""
        return self.mlp(x)
