# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Training is checked on the one pair with a known answer: a model into
itself, where the residual stream at layer l is exactly recoverable from
the source's layer l. Cross-family runs only check that every attempt
(ridge, token-local hub, contextual hub, handoff hybrid) goes through the
same evaluation and that distillation gradients reach the translator.
"""

import itertools

import pytest
import torch

from vllm.distributed.kv_transfer.kv_translation.study import (
    evaluate_predictor,
    fit_pair_mappers,
    prepare_examples,
)
from vllm.distributed.kv_transfer.kv_translation.train import (
    TrainConfig,
    calibrate,
    residual_loss,
    train_translator,
)
from vllm.distributed.kv_transfer.kv_translation.translator import (
    HubConfig,
    HubTranslator,
    TranslatorPredictor,
)


def hub_for(
    src_model, tgt_model, name, src_layers, latent_dim, context_layers=0, seed=0
):
    torch.manual_seed(seed)
    config = HubConfig(
        src_layers=src_layers,
        src_dim=src_model.config.hidden_size,
        latent_dim=latent_dim,
        targets={
            name: (tgt_model.config.num_hidden_layers, tgt_model.config.hidden_size)
        },
        context_layers=context_layers,
        context_heads=4,
    )
    return HubTranslator(config)


@pytest.fixture(scope="module")
def self_examples(tiny_qwen3, unigram_tokenizer, corpus):
    args = (tiny_qwen3, unigram_tokenizer, tiny_qwen3, unigram_tokenizer)
    return prepare_examples(*args, corpus[:80])


def test_residual_loss_is_zero_at_target_and_scale_free():
    target = torch.randn(10, 3, 8) * torch.tensor([1.0, 10.0, 100.0])[None, :, None]
    assert residual_loss(target, target) < 1e-6
    noisy = target + 0.1 * target.std((0, 2), keepdim=True) * torch.randn_like(target)
    per_layer_equal = residual_loss(noisy, target)
    assert 0.0 < per_layer_equal < 0.1


def test_self_translation_training_recovers_residuals(tiny_qwen3, self_examples):
    train, evals = self_examples[:60], self_examples[60:]
    layers = tuple(range(tiny_qwen3.config.num_hidden_layers))
    translator = hub_for(tiny_qwen3, tiny_qwen3, "q", layers, latent_dim=128)
    calibrate(translator, "q", train)
    before = min(
        evaluate_predictor(
            tiny_qwen3, TranslatorPredictor(translator, "q"), evals
        ).r2_hidden.values()
    )
    cfg = TrainConfig(steps=600, lr=3e-3, weight_decay=0.0, log_every=200, seed=0)
    log = train_translator(
        translator, "q", itertools.cycle(train), cfg, eval_examples=evals
    )
    assert log[-1]["residual_loss"] < 0.25 * log[0]["residual_loss"]
    report = evaluate_predictor(tiny_qwen3, TranslatorPredictor(translator, "q"), evals)
    after = min(report.r2_hidden.values())
    assert after > before
    # Measured 0.88 on this fixture. The map is exactly linear (ridge finds
    # it to R2 ~ 1), but 600 Adam steps on ~30-token examples leave the
    # network short of the exact solution; the bound guards convergence.
    assert after > 0.75, report.r2_hidden
    assert report.handoff[0].kl_mean < 1e-2


def test_training_is_deterministic(tiny_qwen3, self_examples):
    train = self_examples[:20]
    layers = (1, 2)
    outs = []
    for _ in range(2):
        translator = hub_for(tiny_qwen3, tiny_qwen3, "q", layers, latent_dim=32, seed=1)
        train_translator(
            translator, "q", itertools.cycle(train), TrainConfig(steps=5, seed=1)
        )
        outs.append(translator(train[0].source_features(layers), "q"))
    torch.testing.assert_close(outs[0], outs[1], atol=0, rtol=0)


def test_distillation_changes_the_update(tiny_qwen3, self_examples):
    """The KL term must contribute gradient: the same seeded step with and
    without it must end in different parameters, and the logged loss must
    carry the KL."""
    train = self_examples[:8]
    outcomes = {}
    for weight in (0.0, 1.0):
        translator = hub_for(
            tiny_qwen3, tiny_qwen3, "q", (1, 2), latent_dim=32, context_layers=1, seed=3
        )
        cfg = TrainConfig(steps=1, kl_weight=weight, kl_every=1, handoff_layer=2)
        log = train_translator(
            translator, "q", itertools.cycle(train), cfg, tgt_model=tiny_qwen3
        )
        outcomes[weight] = (
            log[0],
            [p.detach().clone() for p in translator.parameters()],
        )
    plain, distilled = outcomes[0.0], outcomes[1.0]
    assert "kl" not in plain[0] and distilled[0]["kl"] > 0
    assert distilled[0]["loss"] > distilled[0]["residual_loss"]
    assert plain[0]["loss"] == plain[0]["residual_loss"]
    assert any(not torch.equal(a, b) for a, b in zip(plain[1], distilled[1]))
    assert all(torch.isfinite(p).all() for p in distilled[1])
    assert not any(p.requires_grad for p in tiny_qwen3.parameters())


def test_all_attempts_share_one_evaluation(
    tiny_llama, bpe_tokenizer, tiny_qwen3, unigram_tokenizer, corpus
):
    args = (tiny_llama, bpe_tokenizer, tiny_qwen3, unigram_tokenizer)
    train, evals = (
        prepare_examples(*args, corpus[:40]),
        prepare_examples(*args, corpus[40:48]),
    )
    num_layers = tiny_qwen3.config.num_hidden_layers
    handoffs = [2, num_layers]
    attempts = {"ridge": fit_pair_mappers(train, top_k=2)}
    for name, context_layers in (("hub", 0), ("hub_ctx", 1)):
        translator = hub_for(tiny_llama, tiny_qwen3, "q", (1, 3), 32, context_layers)
        calibrate(translator, "q", train)
        train_translator(translator, "q", itertools.cycle(train), TrainConfig(steps=10))
        attempts[name] = TranslatorPredictor(translator, "q")
    reports = {
        name: evaluate_predictor(tiny_qwen3, predictor, evals, handoffs)
        for name, predictor in attempts.items()
    }
    for report in reports.values():
        assert set(report.r2_hidden) == set(range(num_layers))
        assert [h.handoff_layer for h in report.handoff] == handoffs
        assert all(0.0 <= h.top1_agreement <= 1.0 for h in report.handoff)
        assert all(torch.isfinite(torch.tensor(h.kl_mean)) for h in report.handoff)
    assert reports["ridge"].r2_keys is not None
    assert reports["hub"].r2_keys is None
