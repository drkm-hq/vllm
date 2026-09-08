# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import numpy as np
import pytest

from vllm.distributed.kv_transfer.kv_translation.chat import (
    align_chats,
    locate_contents,
    render_chat,
)

HEADER_TEMPLATE = (
    "{{ '<|begin_of_text|>' }}"
    "{% for m in messages %}"
    "{{ '<|start_header_id|>' + m['role'] + '<|end_header_id|>\\n\\n' }}"
    "{{ m['content'] + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)
CHATML_TEMPLATE = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)
MESSAGES = [
    {"role": "system", "content": "You translate kv caches between models."},
    {"role": "user", "content": "The kv cache is paged into blocks of 16 tokens."},
    {"role": "assistant", "content": "Prefix caching hashes each block."},
    {"role": "user", "content": "So a model switch breaks the cache?"},
]


@pytest.fixture(scope="module")
def header_tokenizer(bpe_tokenizer):
    tok = copy.deepcopy(bpe_tokenizer)
    tok.add_special_tokens(
        {
            "additional_special_tokens": [
                "<|begin_of_text|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>",
            ]
        }
    )
    tok.chat_template = HEADER_TEMPLATE
    return tok


@pytest.fixture(scope="module")
def chatml_tokenizer(unigram_tokenizer):
    tok = copy.deepcopy(unigram_tokenizer)
    tok.add_special_tokens(
        {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}
    )
    tok.chat_template = CHATML_TEMPLATE
    return tok


def test_locate_contents_is_sequential_and_tolerates_misses():
    text = "a: hello\nb: hello world\n"
    offsets = locate_contents(text, ["hello", "hello world", "missing", ""])
    np.testing.assert_array_equal(offsets, [[3, 8], [12, 23], [-1, -1], [-1, -1]])


@pytest.mark.parametrize("fixture", ["header_tokenizer", "chatml_tokenizer"])
def test_render_separates_template_from_content(request, fixture):
    tok = request.getfixturevalue(fixture)
    chat = render_chat(tok, MESSAGES)
    assert (chat.content_offsets >= 0).all()
    for i, message in enumerate(MESSAGES):
        tokens = chat.message_tokens(i)
        assert len(tokens) > 0
        covered = chat.text[
            chat.spans.offsets[tokens[0], 0] : chat.spans.offsets[tokens[-1], 1]
        ]
        assert covered.strip() == message["content"].strip()
    template_tokens = np.flatnonzero(chat.message_index < 0)
    template_text = "".join(
        chat.text[s:e] for s, e in chat.spans.offsets[template_tokens]
    )
    assert "<|" in template_text
    assert not any(m["content"] in template_text for m in MESSAGES)


def test_align_chats_matches_content_per_message(header_tokenizer, chatml_tokenizer):
    src = render_chat(header_tokenizer, MESSAGES)
    tgt = render_chat(chatml_tokenizer, MESSAGES)
    alignment = align_chats(src, tgt)
    src_pos, tgt_pos = alignment.exact_pairs()
    assert len(tgt_pos) > 0
    # Aligned pairs belong to the same message and end at the same
    # message-relative offset, so the text seen so far is identical.
    np.testing.assert_array_equal(
        src.message_index[src_pos], tgt.message_index[tgt_pos]
    )
    src_rel = (
        src.spans.offsets[src_pos, 1]
        - src.content_offsets[src.message_index[src_pos], 0]
    )
    tgt_rel = (
        tgt.spans.offsets[tgt_pos, 1]
        - tgt.content_offsets[tgt.message_index[tgt_pos], 0]
    )
    np.testing.assert_array_equal(src_rel, tgt_rel)
    # Template tokens on the target side never get a source.
    template = tgt.message_index < 0
    assert not alignment.tgt_content[template].any()
    assert (alignment.src_before[template] == -1).all()
    assert alignment.boundary_agreement > 0.5


def test_align_chats_is_identity_for_same_template(header_tokenizer):
    chat = render_chat(header_tokenizer, MESSAGES)
    alignment = align_chats(chat, chat)
    src_pos, tgt_pos = alignment.exact_pairs()
    np.testing.assert_array_equal(src_pos, tgt_pos)
    assert alignment.boundary_agreement == 1.0
