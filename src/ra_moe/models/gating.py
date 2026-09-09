"""
Regime-aware gating network for RA-MoE-4E.

The gating network acts as the routing mechanism of the
Mixture-of-Experts framework. It maps regime-aware and
option-state features to a probability distribution over
the four specialized experts.

The resulting routing weights are non-negative and sum to one.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class GatingNetwork(nn.Module):
    """
    Regime-aware routing network for four experts.

    Parameters
    ----------
    input_dim
        Number of gating input features.
    hidden_dims
        Widths of the hidden layers.
    num_experts
        Number of experts in the mixture.
    temperature
        Temperature used in the softmax routing distribution.

        Lower values produce sharper expert allocations, while
        higher values produce smoother routing distributions.

    Notes
    -----
    The supplied final experimental implementation used:

    - hidden dimensions: (128, 64)
    - number of experts: 4
    - temperature: 4.0
    - LayerNorm followed by ReLU in each hidden layer
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        num_experts: int = 4,
        temperature: float = 4.0,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if not hidden_dims:
            raise ValueError(
                "hidden_dims must contain at least one layer."
            )

        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")

        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")

        layers: list[nn.Module] = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError(
                    "All hidden dimensions must be positive."
                )

            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                ]
            )

            current_dim = hidden_dim

        layers.append(
            nn.Linear(current_dim, num_experts)
        )

        self.network = nn.Sequential(*layers)

        self.num_experts = num_experts
        self.temperature = temperature

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute state-dependent routing weights.

        Parameters
        ----------
        x
            Gating input tensor with shape
            ``(batch_size, input_dim)``.

        Returns
        -------
        torch.Tensor
            Routing weights with shape
            ``(batch_size, num_experts)``.

            Each row is a probability distribution over experts.
        """
        logits = self.network(x)

        weights = torch.softmax(
            logits / self.temperature,
            dim=-1,
        )

        return weights
