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
        residual: Input to each decoder layer, each ``[T, hidden_size]``,
            followed by the output of the final norm (not a residual; do
            not use it as a source tap).
        keys: Per layer, keys after projection and QK-norm, before RoPE,
            ``[T, num_kv_heads, head_dim]``.
        values: Per layer, projected values ``[T, num_kv_heads, head_dim]``.
    """

    residual: list[torch.Tensor]
    keys: dict[int, torch.Tensor] = field(default_factory=dict)
    values: dict[int, torch.Tensor] = field(default_factory=dict)


def decoder(model):
    """The decoder stack, unwrapping multimodal wrappers such as Gemma 3."""
    inner = model.model
    return getattr(inner, "language_model", inner)


def text_config(model):
    config = model.config
    return getattr(config, "text_config", config)


def _attention_modules(model):
    modules = []
    for index, layer in enumerate(decoder(model).layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None or not hasattr(attn, "k_proj") or not hasattr(attn, "v_proj"):
            raise NotImplementedError(
                f"layer {index} has no k_proj/v_proj attention; MLA and hybrid "
                "SSM targets need the vLLM-side fill path, not the HF study"
            )
        modules.append(attn)
    return modules


@torch.no_grad()
def capture_states(model, input_ids: torch.Tensor) -> CapturedStates:
    """Run ``model`` on ``input_ids`` (``[T]``) and capture its states."""
    num_tokens = input_ids.shape[0]
    keys: dict[int, torch.Tensor] = {}
    values: dict[int, torch.Tensor] = {}
    handles = []

    def hook(store, index):
        def _hook(module, args, output):
            out = output.detach()
            if out.dim() == 4 and out.shape[2] == num_tokens:
                out = out.transpose(1, 2)  # k_norm applied after the head transpose
            store[index] = out.reshape(num_tokens, -1)

        return _hook

    for index, attn in enumerate(_attention_modules(model)):
        key_module = getattr(attn, "k_norm", None) or attn.k_proj
        handles.append(key_module.register_forward_hook(hook(keys, index)))
        handles.append(attn.v_proj.register_forward_hook(hook(values, index)))
    try:
        device = next(model.parameters()).device
        out = model(input_ids=input_ids[None].to(device), output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()

    config = text_config(model)
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    to_heads = lambda t: t.reshape(num_tokens, -1, head_dim)  # noqa: E731
    return CapturedStates(
        residual=[h[0] for h in out.hidden_states],
        keys={i: to_heads(k) for i, k in keys.items()},
        values={i: to_heads(v) for i, v in values.items()},
    )
