"""
PyTorch dataset for RA-MoE-4E.

This module refactors the OptionDataset used in the supplied final
experimental implementation.

The original scaling behavior is preserved:
- static features use StandardScaler;
- validation/test static features can reuse the training scaler;
- residual and gating extra features are standardized using the
  statistics of the dataset instance in which they are supplied.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class OptionDataset(Dataset):
    """
    Dataset containing aligned inputs for the four-expert model.

    Parameters
    ----------
    static_features
        Static feature matrix used by Expert 3.

    temporal_features
        Temporal windows used by Expert 4.

    targets
        Observed market option prices.

    bsm_prices
        Precomputed Black-Scholes-Merton prices.

    static_scaler
        Optional StandardScaler fitted on the training static features.
        If None, a new scaler is fitted.

    gate_extra_features
        Additional regime/state features appended to the standardized
        static representation for the gating network.

    residual_extra_features
        Additional features appended to the standardized static
        representation for the residual-correction expert.
    """

    def __init__(
        self,
        static_features: np.ndarray,
        temporal_features: np.ndarray,
        targets: np.ndarray,
        bsm_prices: np.ndarray,
        static_scaler: StandardScaler | None = None,
        gate_extra_features: np.ndarray | None = None,
        residual_extra_features: np.ndarray | None = None,
    ) -> None:
        self.static_features_np = static_features.astype(np.float32)
        self.temporal_features_np = temporal_features.astype(np.float32)
        self.targets_np = targets.astype(np.float32)
        self.bsm_prices_np = bsm_prices.astype(np.float32)

        # Static-feature scaling.
        # Training data fit a new scaler; validation and test data
        # can reuse the scaler fitted on the training set.
        if static_scaler is None:
            self.scaler = StandardScaler()
            self.static_features_np = self.scaler.fit_transform(
                self.static_features_np
            )
        else:
            self.scaler = static_scaler
            self.static_features_np = self.scaler.transform(
                self.static_features_np
            )

        self.static_features = torch.tensor(
            self.static_features_np,
            dtype=torch.float32,
        )

        self.temporal_features = torch.tensor(
            self.temporal_features_np,
            dtype=torch.float32,
        )

        self.targets = torch.tensor(
            self.targets_np,
            dtype=torch.float32,
        )

        self.bsm_prices = torch.tensor(
            self.bsm_prices_np,
            dtype=torch.float32,
        )

        # The supplied final implementation initializes both the
        # residual and gating inputs from the standardized static
        # feature representation.
        self.residual_features = self.static_features.clone()
        self.gate_features = self.static_features.clone()

        if gate_extra_features is not None:
            gate_extra = gate_extra_features.astype(np.float32)

            gate_mean = gate_extra.mean(
                axis=0,
                keepdims=True,
            )

            gate_std = (
                gate_extra.std(
                    axis=0,
                    keepdims=True,
                )
                + 1e-8
            )

            gate_extra_scaled = (
                gate_extra - gate_mean
            ) / gate_std

            self.gate_features = torch.cat(
                [
                    self.gate_features,
                    torch.tensor(
                        gate_extra_scaled,
                        dtype=torch.float32,
                    ),
                ],
                dim=1,
            )

        if residual_extra_features is not None:
            residual_extra = residual_extra_features.astype(
                np.float32
            )

            residual_mean = residual_extra.mean(
                axis=0,
                keepdims=True,
            )

            residual_std = (
                residual_extra.std(
                    axis=0,
                    keepdims=True,
                )
                + 1e-8
            )

            residual_extra_scaled = (
                residual_extra - residual_mean
            ) / residual_std

            self.residual_features = torch.cat(
                [
                    self.residual_features,
                    torch.tensor(
                        residual_extra_scaled,
                        dtype=torch.float32,
                    ),
                ],
                dim=1,
            )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        return {
            "bs": self.bsm_prices[index],
            "x_static": self.static_features[index],
            "x_resid": self.residual_features[index],
            "x_gate": self.gate_features[index],
            "temporal": self.temporal_features[index],
            "y": self.targets[index],
        }
