# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Closed-form linear mappers between model states.

Ridge regression is gradient-free, needs a few thousand aligned positions,
and is the baseline every learned translator has to beat. Its held-out R2
per target layer is also the cheapest viability signal for a model pair.
"""

from dataclasses import dataclass

import torch


def ridge_fit(
    x: torch.Tensor, y: torch.Tensor, lam: float = 1e-2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve ``y ~= x @ W + b`` in float64 for ``x: [N, d_in]``, ``y: [N, d_out]``."""
    x64 = x.to(torch.float64)
    y64 = y.to(torch.float64)
    x_mean, y_mean = x64.mean(0, keepdim=True), y64.mean(0, keepdim=True)
    xc, yc = x64 - x_mean, y64 - y_mean
    gram = xc.T @ xc
    gram.diagonal().add_(lam)
    weight = torch.linalg.solve(gram, xc.T @ yc)
    bias = y_mean - x_mean @ weight
    return weight.to(torch.float32), bias.squeeze(0).to(torch.float32)


def r2_score(y: torch.Tensor, y_hat: torch.Tensor) -> float:
    """Coefficient of determination over all elements."""
    y = y.to(torch.float32)
    ss_res = ((y - y_hat.to(torch.float32)) ** 2).sum()
    ss_tot = ((y - y.mean(0, keepdim=True)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def select_source_layers(scores: dict[int, float], k: int) -> tuple[int, ...]:
    """The ``k`` source layers with the highest scores, in layer order."""
    top = sorted(scores, key=lambda layer: scores[layer], reverse=True)[:k]
    return tuple(sorted(top))


@dataclass
class LinearMapper:
    """An affine map from concatenated source features to one target tensor."""

    weight: torch.Tensor
    bias: torch.Tensor
    src_layers: tuple[int, ...]
    target: str

    @classmethod
    def fit(
        cls,
        x: torch.Tensor,
        y: torch.Tensor,
        src_layers: tuple[int, ...],
        target: str,
        lam: float = 1e-2,
    ) -> "LinearMapper":
        weight, bias = ridge_fit(x, y, lam)
        return cls(weight, bias, src_layers, target)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x.to(torch.float32) @ self.weight + self.bias).to(x.dtype)
