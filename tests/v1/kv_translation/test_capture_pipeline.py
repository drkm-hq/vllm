# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end CPU check of capture -> align -> fit on tiny random models.

Random weights carry no cross-model structure, so the cross-family case
only checks plumbing. The same-model case is the correctness oracle: a
model's own residual stream must be exactly linearly predictable from
itself, and its keys/values must be exactly predictable from the layer
input that produces them.
"""

import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_translation import (
    LinearMapper,
    TokenSpans,
    align_spans,
    r2_score,
)
from vllm.distributed.kv_transfer.kv_translation.capture import capture_states


def _gather(model, tokenizer, texts):
    spans = [TokenSpans.from_tokenizer(tokenizer, t) for t in texts]
    states = [capture_states(model, torch.tensor(s.ids)) for s in spans]
    return spans, states


def test_capture_shapes(tiny_qwen3, unigram_tokenizer, corpus):
    spans = TokenSpans.from_tokenizer(unigram_tokenizer, corpus[0])
    states = capture_states(tiny_qwen3, torch.tensor(spans.ids))
    cfg = tiny_qwen3.config
    assert len(states.residual) == cfg.num_hidden_layers + 1
    assert states.residual[0].shape == (len(spans), cfg.hidden_size)
    assert set(states.keys) == set(range(cfg.num_hidden_layers))
    assert states.keys[0].shape == (
        len(spans),
        cfg.num_key_value_heads,
        cfg.head_dim,
    )
    assert states.values[1].shape == states.keys[1].shape


def test_same_model_states_are_exactly_predictable(
    tiny_qwen3, unigram_tokenizer, corpus
):
    spans, states = _gather(tiny_qwen3, unigram_tokenizer, corpus[:60])
    layer = 2
    x = torch.cat([s.residual[layer] for s in states])
    n = x.shape[0]
    split = int(0.8 * n)
    # Residual -> residual at the same layer is the identity.
    mapper = LinearMapper.fit(x[:split], x[:split], (layer,), f"h:{layer}")
    assert r2_score(x[split:], mapper(x[split:])) > 0.999
    # Values are a linear function of the RMS-normalized layer input, so
    # the fit is near-exact on the normalized features.
    normed = tiny_qwen3.model.layers[layer].input_layernorm(x).detach()
    v = torch.cat([s.values[layer].flatten(1) for s in states])
    mapper = LinearMapper.fit(normed[:split], v[:split], (layer,), f"v:{layer}")
    assert r2_score(v[split:], mapper(normed[split:])) > 0.999


def test_cross_family_pipeline_runs(
    tiny_llama, tiny_qwen3, bpe_tokenizer, unigram_tokenizer, corpus
):
    texts = corpus[:40]
    src_spans, src_states = _gather(tiny_llama, bpe_tokenizer, texts)
    tgt_spans, tgt_states = _gather(tiny_qwen3, unigram_tokenizer, texts)
    xs, ys = [], []
    agreement = []
    for ss, st, ts, tt in zip(src_spans, src_states, tgt_spans, tgt_states):
        alignment = align_spans(ss, ts)
        agreement.append(alignment.boundary_agreement)
        src_pos, tgt_pos = alignment.exact_pairs()
        xs.append(torch.cat([r[src_pos] for r in st.residual], dim=-1))
        ys.append(tt.keys[1][tgt_pos].flatten(1))
    assert np.mean(agreement) > 0.5
    x, y = torch.cat(xs), torch.cat(ys)
    assert x.shape[0] == y.shape[0] > 100
    mapper = LinearMapper.fit(x, y, tuple(range(len(src_states[0].residual))), "k:1")
    out = mapper(x)
    assert out.shape == y.shape
    assert torch.isfinite(out).all()
