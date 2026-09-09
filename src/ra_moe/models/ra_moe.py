"""
Complete RA-MoE-4E model.

This module combines four heterogeneous experts with a
regime-aware gating network:

1. Black-Scholes-Merton theoretical expert
2. Residual-correction MLP expert
3. Static cross-sectional MLP expert
4. Transformer-based temporal expert

The final prediction is a state-dependent weighted combination
of the four expert outputs.
"""

from __future__ import annotations

import torch
from torch import nn

from .gating import GatingNetwork
from .mlp import ResidualExpert, StaticMLPExpert
from .transformer import TransformerExpert


class RAMoE4E(nn.Module):
    """
    Regime-Aware Mixture-of-Experts with four specialized experts.

    Parameters
    ----------
    static_dim
        Number of input features for the static MLP expert.

    residual_dim
        Number of input features for the residual-correction expert.

    gate_dim
        Number of input features for the gating network.

    temporal_dim
        Number of features at each temporal time step.

    residual_scale
        Scaling factor applied to the learned residual correction.

        The supplied final experimental implementation used 5.0.

    gate_temperature
        Temperature of the softmax routing distribution.

        The supplied final experimental implementation used 4.0.

    transformer_d_model
        Transformer embedding dimension.

    transformer_nhead
        Number of Transformer attention heads.

    transformer_num_layers
        Number of Transformer encoder layers.
    """

    NUM_EXPERTS = 4

    def __init__(
        self,
        static_dim: int,
        residual_dim: int,
        gate_dim: int,
        temporal_dim: int,
        residual_scale: float = 5.0,
        gate_temperature: float = 4.0,
        transformer_d_model: int = 64,
        transformer_nhead: int = 4,
        transformer_num_layers: int = 2,
    ) -> None:
        super().__init__()

        if residual_scale <= 0:
            raise ValueError("residual_scale must be greater than zero.")

        self.residual_scale = residual_scale

        self.residual_expert = ResidualExpert(
            input_dim=residual_dim,
        )

        self.static_expert = StaticMLPExpert(
            input_dim=static_dim,
        )

        self.temporal_expert = TransformerExpert(
            input_dim=temporal_dim,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
        )

        self.gate = GatingNetwork(
            input_dim=gate_dim,
            num_experts=self.NUM_EXPERTS,
            temperature=gate_temperature,
        )

    def forward(
        self,
        bsm_price: torch.Tensor,
        residual_features: torch.Tensor,
        static_features: torch.Tensor,
        temporal_features: torch.Tensor,
        gate_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the RA-MoE-4E prediction.

        Parameters
        ----------
        bsm_price
            Precomputed BSM price for each observation,
            shape ``(batch_size,)`` or ``(batch_size, 1)``.

        residual_features
            Input features for the residual-correction expert.

        static_features
            Input features for the static MLP expert.

        temporal_features
            Sequential features with shape
            ``(batch_size, sequence_length, temporal_dim)``.

        gate_features
            State-dependent features used by the gating network.

        Returns
        -------
        hybrid_prediction
            Final dynamically weighted prediction,
            shape ``(batch_size,)``.

        routing_weights
            State-dependent expert weights,
            shape ``(batch_size, 4)``.

        expert_outputs
            Individual expert predictions,
            shape ``(batch_size, 4)``.
        """
        if bsm_price.ndim > 1:
            bsm_price = bsm_price.squeeze(-1)

        # Expert 1: theory-driven BSM anchor.
        expert_bsm = bsm_price

        # Expert 2: BSM anchor plus learned residual correction.
        residual_correction = self.residual_expert(
            residual_features
        )

        expert_residual = (
            bsm_price
            + self.residual_scale * residual_correction
        )

        # Expert 3: nonlinear cross-sectional predictor.
        expert_static = self.static_expert(
            static_features
        )

        # Expert 4: temporal Transformer predictor.
        expert_temporal = self.temporal_expert(
            temporal_features
        )

        expert_outputs = torch.stack(
            [
                expert_bsm,
                expert_residual,
                expert_static,
                expert_temporal,
            ],
            dim=1,
        )

        routing_weights = self.gate(
            gate_features
        )

        hybrid_prediction = torch.sum(
            routing_weights * expert_outputs,
            dim=1,
        )

        return (
            hybrid_prediction,
            routing_weights,
            expert_outputs,
        )
