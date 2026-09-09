"""
Loss components used by the RA-MoE-4E training procedure.

This module refactors the entropy and load-balancing terms from the
supplied final experimental implementation.

The original mathematical behavior is preserved here. Any discrepancy
between the implementation and the thesis description is documented
separately rather than silently corrected.
"""

from __future__ import annotations

import torch


def entropy_loss(
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the entropy term used in the supplied implementation.

    Parameters
    ----------
    weights
        Expert routing probabilities with shape
        ``(batch_size, num_experts)``.

    Returns
    -------
    torch.Tensor
        Mean routing entropy across the batch.
    """
    return -torch.mean(
        torch.sum(
            weights
            * torch.log(weights + 1e-8),
            dim=1,
        )
    )


def load_balance_loss(
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the expert load-balancing penalty.

    The penalty measures squared deviation of the average routing
    weights from uniform expert utilization.
    """
    average_weights = weights.mean(dim=0)

    target = (
        torch.ones_like(average_weights)
        / len(average_weights)
    )

    return torch.sum(
        (average_weights - target) ** 2
    )
