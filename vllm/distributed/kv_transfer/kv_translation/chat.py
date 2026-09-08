# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Template-aware alignment of rendered chats across tokenizers.

Two models render the same conversation into different strings: the
message contents are shared bytes, the role markers and separators are
not. Content tokens are aligned per message on message-relative offsets;
template tokens have no counterpart and are left for the target to
compute natively.
"""

from dataclasses import dataclass

import numpy as np

from vllm.distributed.kv_transfer.kv_translation.alignment import (
    SpanAlignment,
    TokenSpans,
    align_spans,
)


@dataclass
class RenderedChat:
    """A conversation rendered through one tokenizer's chat template.

    Attributes:
        text: The rendered string.
        spans: Its tokenization with character offsets.
        message_index: Per token, the index of the message whose content
            fully contains it, or -1 for template tokens and tokens that
            straddle a content boundary.
        content_offsets: Per message, ``[start, end)`` of its content in
            ``text``; ``(-1, -1)`` if the template altered the content so
            it could not be located.
    """

    text: str
    spans: TokenSpans
    message_index: np.ndarray
    content_offsets: np.ndarray

    def message_tokens(self, message: int) -> np.ndarray:
        return np.flatnonzero(self.message_index == message)


def locate_contents(text: str, contents: list[str]) -> np.ndarray:
    """Find each message's content in order; a miss yields ``(-1, -1)``."""
    offsets: np.ndarray = np.full((len(contents), 2), -1, dtype=np.int64)
    cursor = 0
    for i, content in enumerate(contents):
        if not content:
            continue
        start = text.find(content, cursor)
        if start < 0:
            continue
        offsets[i] = (start, start + len(content))
        cursor = start + len(content)
    return offsets


def render_chat(
    tokenizer, messages: list[dict], add_generation_prompt: bool = True
) -> RenderedChat:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    spans = TokenSpans.from_tokenizer(tokenizer, text, add_special_tokens=False)
    content_offsets = locate_contents(text, [m["content"] for m in messages])
    message_index: np.ndarray = np.full(len(spans), -1, dtype=np.int64)
    starts, ends = spans.offsets[:, 0], spans.offsets[:, 1]
    for i, (cs, ce) in enumerate(content_offsets):
        if cs < 0:
            continue
        inside = (starts >= cs) & (ends <= ce) & (ends > starts)
        message_index[inside] = i
    return RenderedChat(text, spans, message_index, content_offsets)


def align_chats(src: RenderedChat, tgt: RenderedChat) -> SpanAlignment:
    """Align target content tokens to source content tokens message by
    message. Template tokens are non-content on both sides."""
    num_tgt = len(tgt.spans)
    src_before: np.ndarray = np.full(num_tgt, -1, dtype=np.int64)
    src_after: np.ndarray = np.full(num_tgt, -1, dtype=np.int64)
    exact: np.ndarray = np.zeros(num_tgt, dtype=bool)
    tgt_content = tgt.message_index >= 0

    def relative(chat: RenderedChat, message: int) -> tuple[np.ndarray, TokenSpans]:
        tokens = chat.message_tokens(message)
        offsets = chat.spans.offsets[tokens] - chat.content_offsets[message, 0]
        return tokens, TokenSpans(chat.spans.ids[tokens], offsets)

    for message in range(len(tgt.content_offsets)):
        src_tokens, src_rel = relative(src, message)
        tgt_tokens, tgt_rel = relative(tgt, message)
        if len(src_tokens) == 0 or len(tgt_tokens) == 0:
            continue
        part = align_spans(src_rel, tgt_rel)
        src_before[tgt_tokens] = np.where(
            part.src_before >= 0, src_tokens[np.maximum(part.src_before, 0)], -1
        )
        src_after[tgt_tokens] = np.where(
            part.src_after >= 0, src_tokens[np.maximum(part.src_after, 0)], -1
        )
        exact[tgt_tokens] = part.exact
    return SpanAlignment(src_before, src_after, exact, tgt_content)
