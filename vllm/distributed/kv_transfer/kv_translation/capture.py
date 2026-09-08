# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture per-layer residual streams and pre-RoPE keys/values from an HF model.

Used by the offline viability study: it needs the target's cache contents
in the position-free form a mapper predicts (keys after any QK-norm but
before rotation) alongside the residual stream feeding each layer.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class CapturedStates:
    """States for one sequence of ``T`` tokens.

    Attributes:
        residual: Input to each decoder layer plus the final hidden state,
            each ``[T, hidden_size]``.
        keys: Per layer, keys after projection and QK-norm, before RoPE,
            ``[T, num_kv_heads, head_dim]``.
        values: Per layer, projected values ``[T, num_kv_heads, head_dim]``.
    """

    residual: list[torch.Tensor]
    keys: dict[int, torch.Tensor] = field(default_factory=dict)
    values: dict[int, torch.Tensor] = field(default_factory=dict)


def _attention_modules(model):
    return [layer.self_attn for layer in model.model.layers]


@torch.no_grad()
def capture_states(model, input_ids: torch.Tensor) -> CapturedStates:
    """Run ``model`` on ``input_ids`` (``[T]``) and capture its states."""
    num_tokens = input_ids.shape[0]
    keys: dict[int, torch.Tensor] = {}
    values: dict[int, torch.Tensor] = {}
    handles = []

    def hook(store, index):
        def _hook(module, args, output):
            store[index] = output.detach().reshape(num_tokens, -1)

        return _hook

    for index, attn in enumerate(_attention_modules(model)):
        key_module = getattr(attn, "k_norm", None) or attn.k_proj
        handles.append(key_module.register_forward_hook(hook(keys, index)))
        handles.append(attn.v_proj.register_forward_hook(hook(values, index)))
    try:
        out = model(input_ids=input_ids[None], output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()

    head_dim = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads
    )
    to_heads = lambda t: t.reshape(num_tokens, -1, head_dim)  # noqa: E731
    return CapturedStates(
        residual=[h[0] for h in out.hidden_states],
        keys={i: to_heads(k) for i, k in keys.items()},
        values={i: to_heads(v) for i, v in values.items()},
    )
