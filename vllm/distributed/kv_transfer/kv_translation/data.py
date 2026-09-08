# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aligned (source, target) examples for fitting and evaluating translators."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_translation.alignment import (
    TokenSpans,
    align_spans,
)
from vllm.distributed.kv_transfer.kv_translation.capture import (
    CapturedStates,
    capture_states,
)
from vllm.distributed.kv_transfer.kv_translation.chat import (
    align_chats,
    render_chat,
)

StateKind = Literal["src", "hidden", "keys", "values"]


@dataclass
class AlignedExample:
    """One text split into a target-token prefix and continuation.

    ``src_pos[j]`` is the source position whose state stands in for target
    prefix position ``j``, or -1 for tokens with no counterpart (special or
    template tokens), which the target computes natively. ``exact[j]`` marks
    positions whose source and target states have seen identical text.
    """

    src: TokenSpans
    tgt: TokenSpans
    src_states: CapturedStates
    tgt_states: CapturedStates
    src_pos: np.ndarray
    exact: np.ndarray
    cont_ids: torch.Tensor

    @property
    def content(self) -> np.ndarray:
        return self.src_pos >= 0

    @property
    def num_target_layers(self) -> int:
        return len(self.tgt_states.residual) - 1

    def exact_states(self, kind: StateKind, layer: int) -> torch.Tensor:
        """Rows of one state at the exactly aligned positions."""
        tgt_pos = np.flatnonzero(self.exact)
        if kind == "src":
            return self.src_states.residual[layer][self.src_pos[tgt_pos]]
        if kind == "hidden":
            return self.tgt_states.residual[layer][tgt_pos]
        if kind == "keys":
            return self.tgt_states.keys[layer][tgt_pos].flatten(1)
        return self.tgt_states.values[layer][tgt_pos].flatten(1)

    def source_features(self, layers: Sequence[int]) -> torch.Tensor:
        """Source residuals ``[num_content, len(layers), d_src]`` standing in
        for the content positions, in target order."""
        pos = torch.as_tensor(self.src_pos[self.content])
        return torch.stack(
            [self.src_states.residual[layer][pos] for layer in layers], dim=1
        )

    def target_residuals(self) -> torch.Tensor:
        """Target residual stream ``[num_content, num_layers, d_tgt]`` at the
        content positions (layer inputs, not the final hidden state)."""
        pos = torch.as_tensor(self.content)
        layers = self.tgt_states.residual[: self.num_target_layers]
        return torch.stack([h[pos] for h in layers], dim=1)


def source_position_policy(alignment, allow_lookahead: bool = False) -> np.ndarray:
    """Which source state stands in for each target position.

    Causal by default: the nearest source token that has seen no more text
    than the target token, so the translated state never depends on bytes
    the target position has not read. ``allow_lookahead`` instead takes
    the nearest source token that has seen at least the target's text,
    which leaks up to one token of future text and is only an upper bound.
    """
    if not allow_lookahead:
        return alignment.src_before.copy()
    pos = np.where(alignment.exact, alignment.src_before, alignment.src_after)
    return np.where(pos >= 0, pos, alignment.src_before)


def prepare_example(
    src_model,
    src_tokenizer,
    tgt_model,
    tgt_tokenizer,
    text: str,
    prefix_frac: float = 0.75,
    allow_lookahead: bool = False,
) -> AlignedExample | None:
    """Split ``text`` at a target-token boundary and align the prefix."""
    tgt_full = TokenSpans.from_tokenizer(tgt_tokenizer, text)
    n_prefix = int(len(tgt_full) * prefix_frac)
    if n_prefix < 2 or n_prefix >= len(tgt_full):
        return None
    prefix_text = text[: tgt_full.offsets[n_prefix - 1, 1]]
    tgt = TokenSpans(tgt_full.ids[:n_prefix], tgt_full.offsets[:n_prefix])
    src = TokenSpans.from_tokenizer(src_tokenizer, prefix_text)
    return _build_example(
        src_model,
        tgt_model,
        src,
        tgt,
        align_spans(src, tgt),
        torch.as_tensor(tgt_full.ids[n_prefix:]),
        allow_lookahead,
    )


def prepare_chat_example(
    src_model,
    src_tokenizer,
    tgt_model,
    tgt_tokenizer,
    messages: list[dict],
    allow_lookahead: bool = False,
) -> AlignedExample | None:
    """Render ``messages[:-1]`` through each model's own chat template as
    the prefix and use the final assistant message as the continuation.
    Template tokens have no source counterpart and keep native states."""
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return None
    src_chat = render_chat(src_tokenizer, messages[:-1])
    tgt_chat = render_chat(tgt_tokenizer, messages[:-1])
    alignment = align_chats(src_chat, tgt_chat)
    if not alignment.exact.any():
        return None
    cont_ids = tgt_tokenizer(messages[-1]["content"], add_special_tokens=False)[
        "input_ids"
    ]
    if not cont_ids:
        return None
    return _build_example(
        src_model,
        tgt_model,
        src_chat.spans,
        tgt_chat.spans,
        alignment,
        torch.as_tensor(cont_ids),
        allow_lookahead,
    )


def _build_example(
    src_model, tgt_model, src, tgt, alignment, cont_ids, allow_lookahead
):
    return AlignedExample(
        src=src,
        tgt=tgt,
        src_states=capture_states(src_model, torch.as_tensor(src.ids)),
        tgt_states=capture_states(tgt_model, torch.as_tensor(tgt.ids)),
        src_pos=source_position_policy(alignment, allow_lookahead),
        exact=alignment.exact,
        cont_ids=cont_ids,
    )


def features(
    states: CapturedStates, layers: Sequence[int], positions: np.ndarray
) -> torch.Tensor:
    idx = torch.as_tensor(positions)
    return torch.cat([states.residual[layer][idx] for layer in layers], dim=-1)


def stack_exact(
    examples: Sequence[AlignedExample], kind: StateKind, layer: int
) -> torch.Tensor:
    return torch.cat(
        [ex.exact_states(kind, layer) for ex in examples if ex.exact.any()]
    )


def stack_features(
    examples: Sequence[AlignedExample], layers: Sequence[int]
) -> torch.Tensor:
    return torch.cat([stack_exact(examples, "src", s) for s in layers], dim=-1)
