# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Position-free keys: strip and re-apply rotary embeddings.

A key stored in the cache carries the source model's rotation at the source
model's position. Rotation is orthogonal, so it is exactly invertible given
the position and frequency table. Mappers are fit on stripped keys so they
are independent of position, context length and the two models' RoPE
parameters; the target's rotation is applied at the target's positions
after mapping.
"""

import torch


def rope_inv_freq(rotary_dim: int, base: float) -> torch.Tensor:
    """Default inverse frequencies, matching ``RotaryEmbedding``."""
    exponent = torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim
    return 1.0 / (base**exponent)


def rope_cos_sin(
    positions: torch.Tensor, inv_freq: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :].to(
        positions.device
    )
    return freqs.cos(), freqs.sin()


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    is_neox_style: bool,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate ``x`` of shape ``[T, H, D]`` by ``positions`` (or by their negation).

    Only the first ``2 * len(inv_freq)`` dims are rotated; the rest pass
    through, as in partial-rotary models. Matches
    ``RotaryEmbedding.forward_native`` for both layouts.
    """
    rotary_dim = 2 * inv_freq.shape[0]
    cos, sin = rope_cos_sin(positions, inv_freq)
    if inverse:
        sin = -sin
    cos = cos[:, None, :]
    sin = sin[:, None, :]

    x_rot = x[..., :rotary_dim].to(torch.float32)
    x_pass = x[..., rotary_dim:]
    if is_neox_style:
        x1, x2 = x_rot.chunk(2, dim=-1)
    else:
        x1, x2 = x_rot[..., ::2], x_rot[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        out = torch.cat((o1, o2), dim=-1)
    else:
        out = torch.stack((o1, o2), dim=-1).flatten(-2)
    return torch.cat((out.to(x.dtype), x_pass), dim=-1)


def strip_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """Undo the rotation applied at ``positions``."""
    return apply_rope(x, positions, inv_freq, is_neox_style, inverse=True)
