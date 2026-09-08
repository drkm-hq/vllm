# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Architectures the HF study must handle beyond Llama and Qwen3: a
Gemma-3-shaped target (key norm after the head transpose, sliding-window
layers with a local rotary, a global layer), chat-rendered examples whose
template tokens keep native states, and a loud failure for attention
modules without k_proj/v_proj.
"""

import copy

import numpy as np
import pytest
import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from vllm.distributed.kv_transfer.kv_translation.capture import capture_states
from vllm.distributed.kv_transfer.kv_translation.data import (
    prepare_chat_example,
    prepare_example,
)
from vllm.distributed.kv_transfer.kv_translation.study import (
    continuation_divergence,
    evaluate_predictor,
    fit_pair_mappers,
    prepare_examples,
    translated_cache,
)

from .test_chat import CHATML_TEMPLATE, HEADER_TEMPLATE, MESSAGES


@pytest.fixture(scope="module")
def tiny_gemma3(unigram_tokenizer):
    torch.manual_seed(2)
    config = Gemma3TextConfig(
        vocab_size=len(unigram_tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=4,
        max_position_embeddings=256,
    )
    assert "full_attention" in config.layer_types
    assert "sliding_attention" in config.layer_types
    return Gemma3ForCausalLM(config).eval()


def test_gemma3_capture_reshapes_post_transpose_key_norm(
    tiny_gemma3, unigram_tokenizer, corpus
):
    ids = torch.as_tensor(unigram_tokenizer(corpus[0])["input_ids"])
    states = capture_states(tiny_gemma3, ids)
    cfg = tiny_gemma3.config
    assert states.keys[0].shape == (len(ids), cfg.num_key_value_heads, cfg.head_dim)
    # The hook output is [1, H, T, D]; a plain reshape would interleave
    # heads and positions. Recompute the keys directly to check.
    attn = tiny_gemma3.model.layers[0].self_attn
    h = tiny_gemma3.model.layers[0].input_layernorm(states.residual[0])
    k = attn.k_proj(h).view(len(ids), cfg.num_key_value_heads, cfg.head_dim)
    torch.testing.assert_close(states.keys[0], attn.k_norm(k))


def test_gemma3_layer_zero_oracle_with_sliding_and_global_layers(
    tiny_gemma3, unigram_tokenizer, corpus
):
    args = (tiny_gemma3, unigram_tokenizer, tiny_gemma3, unigram_tokenizer)
    examples = prepare_examples(*args, corpus[:30])
    long_enough = [
        ex for ex in examples if len(ex.tgt) > 2 * tiny_gemma3.config.sliding_window
    ]
    assert long_enough, "need prefixes longer than the window to exercise cropping"
    mappers = fit_pair_mappers(examples, top_k=1, lam=1e-6)
    for ex in long_enough[:3]:
        cache = translated_cache(
            tiny_gemma3, mappers, ex, 0, predicted_hidden=ex.tgt_states.residual[0]
        )
        kl, agree = continuation_divergence(tiny_gemma3, ex, cache)
        assert kl < 1e-5 and agree == 1.0
    # Pure translation of the model into itself runs through the sliding
    # crop and the per-layer rotary and stays close to native.
    num_layers = tiny_gemma3.config.num_hidden_layers
    report = evaluate_predictor(tiny_gemma3, mappers, long_enough[:4], [num_layers])
    assert report.handoff[0].kl_mean < 0.5 * report.handoff[0].kl_control_mean


def test_chat_examples_keep_template_positions_native(
    tiny_llama, bpe_tokenizer, tiny_qwen3, unigram_tokenizer
):
    src_tok, tgt_tok = copy.deepcopy(bpe_tokenizer), copy.deepcopy(unigram_tokenizer)
    src_tok.add_special_tokens(
        {
            "additional_special_tokens": [
                "<|begin_of_text|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>",
            ]
        }
    )
    src_tok.chat_template = HEADER_TEMPLATE
    tgt_tok.add_special_tokens(
        {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}
    )
    tgt_tok.chat_template = CHATML_TEMPLATE
    # The added special tokens need embedding rows in both models.
    src_model, tgt_model = copy.deepcopy(tiny_llama), copy.deepcopy(tiny_qwen3)
    torch.manual_seed(0)
    src_model.resize_token_embeddings(len(src_tok))
    tgt_model.resize_token_embeddings(len(tgt_tok))
    reply = {"role": "assistant", "content": "Yes, unless the cache is translated."}
    messages = MESSAGES + [reply]
    assert (
        prepare_chat_example(src_model, src_tok, tgt_model, tgt_tok, MESSAGES) is None
    )
    ex = prepare_chat_example(src_model, src_tok, tgt_model, tgt_tok, messages)
    assert ex is not None
    template = ~ex.content
    assert 0 < template.mean() < 1
    assert (
        ex.cont_ids.tolist()
        == tgt_tok(reply["content"], add_special_tokens=False)["input_ids"]
    )
    # Template positions must carry the target's own states in every layer
    # of a fully translated cache.
    mappers = fit_pair_mappers([ex], top_k=1)
    num_layers = tgt_model.config.num_hidden_layers
    cache = translated_cache(tgt_model, mappers, ex, num_layers)
    native = tgt_model(
        input_ids=torch.as_tensor(ex.tgt.ids)[None], use_cache=True
    ).past_key_values
    keep = torch.as_tensor(template)
    for layer in range(num_layers):
        torch.testing.assert_close(
            cache.layers[layer].keys[:, :, keep], native.layers[layer].keys[:, :, keep]
        )
    report = evaluate_predictor(tgt_model, mappers, [ex], [num_layers])
    assert abs(report.native_position_fraction - template.mean()) < 1e-6


def test_causal_policy_never_reads_ahead(
    tiny_llama, bpe_tokenizer, tiny_qwen3, unigram_tokenizer, corpus
):
    ex = prepare_example(
        tiny_llama, bpe_tokenizer, tiny_qwen3, unigram_tokenizer, corpus[0]
    )
    content = np.flatnonzero(ex.content)
    src_end = ex.src.offsets[ex.src_pos[content], 1]
    tgt_end = ex.tgt.offsets[content, 1]
    assert (src_end <= tgt_end).all()
    ahead = prepare_example(
        tiny_llama,
        bpe_tokenizer,
        tiny_qwen3,
        unigram_tokenizer,
        corpus[0],
        allow_lookahead=True,
    )
    content = np.flatnonzero(ahead.content)
    assert (
        ahead.src.offsets[ahead.src_pos[content], 1] >= ahead.tgt.offsets[content, 1]
    ).any()


def test_attention_without_kv_projections_fails_loudly(tiny_qwen3):
    broken = copy.deepcopy(tiny_qwen3)
    del broken.model.layers[1].self_attn.k_proj
    with pytest.raises(NotImplementedError, match="k_proj"):
        capture_states(broken, torch.tensor([1, 2, 3]))
