"""
Transformer-based temporal expert for RA-MoE-4E.

This module implements Expert 4 of the original experimental
RA-MoE-4E codebase. It processes fixed-length sequences of market
features using a Transformer encoder and produces one scalar
prediction per observation.

Important
---------
This implementation intentionally preserves the supplied final
experimental code. In particular, positional encoding is not added
here, even though it was described in the thesis methodology.
That discrepancy is documented separately for reproducibility.
"""

from __future__ import annotations

import torch
from torch import nn


class TransformerExpert(nn.Module):
    """
    Expert 4: Transformer-based temporal expert.

    Parameters
    ----------
    input_dim
        Number of temporal features at each time step.
    d_model
        Transformer embedding dimension.
    nhead
        Number of attention heads.
    num_layers
        Number of Transformer encoder layers.
    feedforward_multiplier
        Multiplier controlling the hidden dimension of the
        feed-forward subnetwork inside each Transformer layer.

    Notes
    -----
    The original experimental configuration used:

    - d_model = 64
    - nhead = 4
    - num_layers = 2
    - feed-forward dimension = 4 * d_model

    The supplied final implementation did not include an explicit
    positional encoding.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        feedforward_multiplier: int = 4,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if d_model <= 0:
            raise ValueError("d_model must be positive.")

        if nhead <= 0:
            raise ValueError("nhead must be positive.")

        if d_model % nhead != 0:
            raise ValueError(
                "d_model must be divisible by nhead."
            )

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.embedding = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=feedforward_multiplier * d_model,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Produce one temporal-expert prediction per observation.

        Parameters
        ----------
        x
            Temporal input tensor with shape
            ``(batch_size, sequence_length, input_dim)``.

        Returns
        -------
        torch.Tensor
            Predictions with shape ``(batch_size,)``.
        """
        hidden = self.embedding(x)

        encoded = self.encoder(hidden)

        # Equivalent to global average pooling across the
        # sequence dimension in the original implementation.
        pooled = encoded.mean(dim=1)

        return self.head(pooled).squeeze(-1)
