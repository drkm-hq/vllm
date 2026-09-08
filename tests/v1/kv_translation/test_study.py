# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The study's cache mechanics have two exact oracles on tiny random models:
handing off at layer 0 with the true embeddings must reproduce native
logits, and translating a model into itself must be near-lossless.
Cross-family runs only check plumbing, since random weights share nothing.
"""

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_translation.study import (
    continuation_divergence,
    fit_pair_mappers,
    heldout_r2,
    prepare_example,
    run_pair_study,
    translated_cache,
)


@pytest.fixture(scope="module")
def same_model_examples(tiny_qwen3, unigram_tokenizer, corpus):
    args = (tiny_qwen3, unigram_tokenizer, tiny_qwen3, unigram_tokenizer)
    examples = [prepare_example(*args, text) for text in corpus[:80]]
    return [ex for ex in examples if ex is not None]


def test_handoff_at_layer_zero_with_true_embeddings_is_native(
    tiny_qwen3, same_model_examples
):
    mappers = fit_pair_mappers(same_model_examples[:60], top_k=1, lam=1e-3)
    ex = same_model_examples[-1]
    cache = translated_cache(
        tiny_qwen3,
        mappers,
        ex,
        handoff_layer=0,
        predicted_hidden=ex.tgt_states.residual[0],
    )
    kl, agree = continuation_divergence(tiny_qwen3, ex, cache)
    assert kl < 1e-6
    assert agree == 1.0


@pytest.mark.parametrize("mode", ["resid", "kv"])
def test_self_translation_is_near_lossless(tiny_qwen3, same_model_examples, mode):
    train, evals = same_model_examples[:60], same_model_examples[60:]
    mappers = fit_pair_mappers(train, top_k=1, lam=1e-6)
    r2 = heldout_r2(mappers, evals)
    num_layers = tiny_qwen3.config.num_hidden_layers
    # Layer 0 rows are embeddings; the tiny corpus covers fewer distinct
    # tokens than dimensions, so unseen tokens are not in the fit span.
    assert r2["hidden"][0] > 0.99
    assert all(r2["hidden"][t] > 0.999 for t in range(1, num_layers))
    assert all(mappers.src_layers[t] == (t,) for t in range(num_layers))
    kls, baseline = [], []
    for i, ex in enumerate(evals):
        cache = translated_cache(tiny_qwen3, mappers, ex, num_layers, mode=mode)
        kl, _ = continuation_divergence(tiny_qwen3, ex, cache)
        kls.append(kl)
        # A cache prefilled from a different text of the same length.
        other = evals[(i + 1) % len(evals)]
        if len(other.tgt) >= len(ex.tgt):
            ids = torch.as_tensor(other.tgt.ids[: len(ex.tgt)])[None]
            wrong = tiny_qwen3(input_ids=ids, use_cache=True).past_key_values
            baseline.append(continuation_divergence(tiny_qwen3, ex, wrong)[0])
    # Native projections of an identity-mapped residual are exact except
    # for layer-0 rows of unseen tokens; direct key/value prediction also
    # pays for the norms it cannot represent linearly. Both must sit far
    # below what a wrong prefix cache costs.
    assert baseline, "no example had a longer partner for the wrong cache"
    if mode == "resid":
        assert float(np.median(kls)) < 1e-6
    assert max(kls) < 0.5 * float(np.median(baseline))


def test_cross_family_study_runs(
    tiny_llama, bpe_tokenizer, tiny_qwen3, unigram_tokenizer, corpus
):
    report = run_pair_study(
        tiny_llama,
        bpe_tokenizer,
        tiny_qwen3,
        unigram_tokenizer,
        train_texts=corpus[:50],
        eval_texts=corpus[50:60],
        top_k=2,
        handoff_layers=[0, 2, tiny_qwen3.config.num_hidden_layers],
    )
    num_layers = tiny_qwen3.config.num_hidden_layers
    assert 0.0 < report.boundary_agreement <= 1.0
    assert set(report.r2_keys) == set(range(num_layers))
    assert all(len(layers) == 2 for layers in report.src_layers.values())
    assert [h.handoff_layer for h in report.handoff] == [0, 2, num_layers]
    assert report.handoff[0].native_layer_fraction == 1.0
    assert all(torch.isfinite(torch.tensor(h.kl_mean)) for h in report.handoff)
    assert all(0.0 <= h.top1_agreement <= 1.0 for h in report.handoff)
