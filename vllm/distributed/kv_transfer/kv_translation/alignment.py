# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Position alignment between two tokenizations of the same text.

A causal model's state at token ``i`` summarizes the text up to the end
offset of token ``i``. Two tokenizers split the same string at different
boundaries, so the only positions whose states are comparable are those
whose end offsets coincide. This module finds them, and for every other
target token records the nearest source tokens on either side so a
translator can pick a policy (nearest-before, nearest-after, or pool).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class TokenSpans:
    """Token ids and their ``[start, end)`` character offsets in one string."""

    ids: np.ndarray
    offsets: np.ndarray

    @classmethod
    def from_tokenizer(
        cls, tokenizer, text: str, add_special_tokens: bool = True
    ) -> "TokenSpans":
        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=add_special_tokens,
        )
        return cls(
            np.asarray(enc["input_ids"], dtype=np.int64),
            np.asarray(enc["offset_mapping"], dtype=np.int64).reshape(-1, 2),
        )

    @property
    def is_content(self) -> np.ndarray:
        """Tokens covering at least one character (special tokens cover none)."""
        return self.offsets[:, 1] > self.offsets[:, 0]

    def __len__(self) -> int:
        return len(self.ids)


@dataclass
class SpanAlignment:
    """Per target token, the comparable source positions.

    Attributes:
        src_before: Index of the source content token with the largest end
            offset ``<=`` the target token's end offset, or -1.
        src_after: Index of the source content token with the smallest end
            offset ``>=`` the target token's end offset, or -1.
        exact: True where a source content token ends exactly where the
            target token ends, so ``src_before == src_after``.
        tgt_content: True for target tokens covering at least one character.
    """

    src_before: np.ndarray
    src_after: np.ndarray
    exact: np.ndarray
    tgt_content: np.ndarray

    @property
    def boundary_agreement(self) -> float:
        """Fraction of target content tokens with an exactly aligned source."""
        if not self.tgt_content.any():
            return 0.0
        return float(self.exact[self.tgt_content].mean())

    def exact_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """``(src_positions, tgt_positions)`` of exactly aligned tokens."""
        tgt = np.flatnonzero(self.exact)
        return self.src_before[tgt], tgt


def _last_index_per_end(spans: TokenSpans) -> tuple[np.ndarray, np.ndarray]:
    """Distinct end offsets of content tokens and the last token index of each.

    Byte-level tokenizers can split one character into several tokens that
    share its character span; only the last of them has seen the whole
    character, so it is the one whose state is comparable.
    """
    idx = np.flatnonzero(spans.is_content)
    ends = spans.offsets[idx, 1]
    if len(ends) > 1 and np.any(np.diff(ends) < 0):
        raise ValueError("token end offsets must be non-decreasing")
    uniq_ends = np.unique(ends)
    last = idx[np.searchsorted(ends, uniq_ends, side="right") - 1]
    return uniq_ends, last


def align_spans(src: TokenSpans, tgt: TokenSpans) -> SpanAlignment:
    """Align target tokens to source tokens by end offset.

    Both spans must come from the same string. Special tokens (empty spans)
    are never aligned to or from, and a target token is ``exact`` only if
    it is the last token ending at its offset.
    """
    src_ends, src_last = _last_index_per_end(src)
    _, tgt_last = _last_index_per_end(tgt)

    tgt_content = tgt.is_content
    tgt_ends = tgt.offsets[:, 1]
    if len(src_ends) == 0:
        none: np.ndarray = np.full(len(tgt), -1, dtype=np.int64)
        return SpanAlignment(none, none.copy(), np.zeros(len(tgt), dtype=bool), tgt_content)
    after = np.searchsorted(src_ends, tgt_ends, side="left")
    before = np.searchsorted(src_ends, tgt_ends, side="right") - 1
    has_after = tgt_content & (after < len(src_ends))
    has_before = tgt_content & (before >= 0)
    src_after = np.where(has_after, src_last[np.minimum(after, len(src_ends) - 1)], -1)
    src_before = np.where(has_before, src_last[np.maximum(before, 0)], -1)

    is_last_of_end: np.ndarray = np.zeros(len(tgt), dtype=bool)
    is_last_of_end[tgt_last] = True
    exact = has_before & is_last_of_end
    exact[exact] = src.offsets[src_before[exact], 1] == tgt_ends[exact]
    return SpanAlignment(src_before, src_after, exact, tgt_content)
