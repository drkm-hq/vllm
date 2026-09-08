# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metrics that decide the kill criteria: acceptance rate and greedy run
length as a draft, the mismatched-document control, the rank-capacity
bound, and hub isolation under onboarding.
"""

import itertools

import torch

from vllm.distributed.kv_transfer.kv_translation.study import (
    acceptance_metrics,
    capacity_bound,
    continuation_logits,
    evaluate_predictor,
    fit_pair_mappers,
    mismatched_control,
    prepare_examples,
    translated_cache,
)
from vllm.distributed.kv_transfer.kv_translation.train import (
    TrainConfig,
    calibrate,
    train_translator,
)
from vllm.distributed.kv_transfer.kv_translation.translator import (
    HubConfig,
    HubTranslator,
)


def test_acceptance_metrics_on_identical_and_disjoint_distributions():
    logits = torch.randn(6, 20)
    alpha, run = acceptance_metrics(logits, logits)
    assert abs(alpha - 1.0) < 1e-6 and run == 6
    shifted = logits.roll(1, dims=-1)
    alpha, run = acceptance_metrics(logits, shifted)
    assert 0.0 <= alpha < 1.0 and run == 0
    mixed = logits.clone()
    mixed[3:] = shifted[3:]
    assert acceptance_metrics(logits, mixed)[1] == 3


def test_native_cache_is_a_perfect_draft(tiny_qwen3, unigram_tokenizer, corpus):
    args = (tiny_qwen3, unigram_tokenizer, tiny_qwen3, unigram_tokenizer)
    ex = prepare_examples(*args, corpus[:3])[0]
    mappers = fit_pair_mappers(prepare_examples(*args, corpus[3:30]), top_k=1)
    cache = translated_cache(
        tiny_qwen3, mappers, ex, 0, predicted_hidden=ex.tgt_states.residual[0]
    )
    alpha, run = acceptance_metrics(*continuation_logits(tiny_qwen3, ex, cache))
    assert abs(alpha - 1.0) < 1e-5 and run == len(ex.cont_ids)


def test_mismatched_control_is_worse_than_translation(
    tiny_qwen3, unigram_tokenizer, corpus
):
    args = (tiny_qwen3, unigram_tokenizer, tiny_qwen3, unigram_tokenizer)
    train, evals = (
        prepare_examples(*args, corpus[:50]),
        prepare_examples(*args, corpus[50:58]),
    )
    mappers = fit_pair_mappers(train, top_k=1, lam=1e-6)
    control = mismatched_control(evals[0], evals[1])
    assert control.tgt is evals[0].tgt and control.src_states is evals[1].src_states
    assert (control.src_pos[control.content] < len(evals[1].src)).all()
    report = evaluate_predictor(tiny_qwen3, mappers, evals)
    result = report.handoff[0]
    assert result.kl_control_mean > 10 * result.kl_mean
    assert result.acceptance_rate > 0.95
    assert result.greedy_run_length_mean > 0


def test_capacity_bound_is_monotone_and_saturates(
    tiny_qwen3, unigram_tokenizer, corpus
):
    args = (tiny_qwen3, unigram_tokenizer, tiny_qwen3, unigram_tokenizer)
    examples = prepare_examples(*args, corpus[:30])
    dim = tiny_qwen3.config.hidden_size
    bound = capacity_bound(examples, ranks=(1, 8, dim // 2, dim))
    for layer, by_rank in bound.items():
        values = [by_rank[r] for r in (1, 8, dim // 2, dim)]
        assert all(0.0 <= v <= 1.0 + 1e-6 for v in values)
        assert values == sorted(values)
        assert abs(by_rank[dim] - 1.0) < 1e-6


def test_onboarding_with_frozen_hub_leaves_existing_target_untouched(
    tiny_llama, bpe_tokenizer, tiny_qwen3, unigram_tokenizer, corpus
):
    src = (tiny_llama, bpe_tokenizer)
    to_qwen = prepare_examples(*src, tiny_qwen3, unigram_tokenizer, corpus[:20])
    to_llama = prepare_examples(*src, tiny_llama, bpe_tokenizer, corpus[:20])
    torch.manual_seed(0)
    translator = HubTranslator(
        HubConfig(
            src_layers=(1, 3),
            src_dim=tiny_llama.config.hidden_size,
            latent_dim=32,
            targets={"qwen": (tiny_qwen3.config.num_hidden_layers, 96)},
            context_layers=1,
        )
    )
    calibrate(translator, "qwen", to_qwen)
    train_translator(translator, "qwen", itertools.cycle(to_qwen), TrainConfig(steps=5))
    hub_state = {
        k: v.clone()
        for k, v in translator.state_dict().items()
        if not k.startswith("heads.")
    }
    probe = to_qwen[0].source_features((1, 3))
    before = translator(probe, "qwen").clone()

    translator.add_target("llama", tiny_llama.config.num_hidden_layers, 64)
    calibrate(translator, "llama", to_llama, hub_frozen=True)
    train_translator(
        translator,
        "llama",
        itertools.cycle(to_llama),
        TrainConfig(steps=5, train_hub=False),
    )
    torch.testing.assert_close(translator(probe, "qwen"), before, atol=0, rtol=0)
    for k, v in translator.state_dict().items():
        if not k.startswith("heads."):
            assert torch.equal(v, hub_state[k]), k
    assert translator(probe, "llama").shape == (probe.shape[0], 3, 64)
