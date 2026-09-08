# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest

from vllm.distributed.kv_transfer.kv_translation import TokenSpans, align_spans

TEXT = "The KV cache is paged into blocks of 16 tokens; naïve façade (ünïcode)."


def test_same_tokenizer_aligns_to_identity(bpe_tokenizer):
    spans = TokenSpans.from_tokenizer(bpe_tokenizer, TEXT)
    alignment = align_spans(spans, spans)
    src_pos, tgt_pos = alignment.exact_pairs()
    np.testing.assert_array_equal(src_pos, tgt_pos)
    # Every distinct end offset is represented by exactly one aligned token.
    distinct_ends = np.unique(spans.offsets[spans.is_content, 1])
    np.testing.assert_array_equal(spans.offsets[tgt_pos, 1], distinct_ends)


def test_cross_tokenizer_exact_positions_share_end_offsets(
    bpe_tokenizer, unigram_tokenizer
):
    src = TokenSpans.from_tokenizer(bpe_tokenizer, TEXT)
    tgt = TokenSpans.from_tokenizer(unigram_tokenizer, TEXT)
    alignment = align_spans(src, tgt)
    src_pos, tgt_pos = alignment.exact_pairs()
    assert len(tgt_pos) > 0
    np.testing.assert_array_equal(src.offsets[src_pos, 1], tgt.offsets[tgt_pos, 1])
    assert np.all(np.diff(src_pos) > 0), "exact matches must stay monotone"
    # Every target content token brackets its end offset from both sides.
    content = alignment.tgt_content
    before = alignment.src_before[content]
    after = alignment.src_after[content]
    ends = tgt.offsets[content, 1]
    has_before = before >= 0
    has_after = after >= 0
    assert np.all(src.offsets[before[has_before], 1] <= ends[has_before])
    assert np.all(src.offsets[after[has_after], 1] >= ends[has_after])


def test_special_tokens_are_never_aligned(bpe_tokenizer, unigram_tokenizer):
    src = TokenSpans.from_tokenizer(bpe_tokenizer, TEXT)
    ids = np.concatenate([[unigram_tokenizer.bos_token_id], [7, 8]])
    offsets = np.array([[0, 0], [0, 3], [3, 6]])
    tgt = TokenSpans(ids, offsets)
    alignment = align_spans(src, tgt)
    assert not alignment.tgt_content[0]
    assert not alignment.exact[0]
    assert alignment.src_before[0] == -1 and alignment.src_after[0] == -1


def test_rejects_non_monotone_source():
    src = TokenSpans(np.array([1, 2]), np.array([[0, 5], [2, 4]]))
    tgt = TokenSpans(np.array([1]), np.array([[0, 4]]))
    with pytest.raises(ValueError, match="non-decreasing"):
        align_spans(src, tgt)
