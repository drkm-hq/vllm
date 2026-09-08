# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.distributed.kv_transfer.kv_translation.translator import (
    HubConfig,
    HubTranslator,
    TranslatorPredictor,
)

CONFIG = HubConfig(
    src_layers=(1, 3),
    src_dim=64,
    latent_dim=48,
    targets={"a": (4, 96), "b": (3, 32)},
    head_rank=16,
    context_layers=1,
    context_heads=4,
)


def make(config: HubConfig = CONFIG, seed: int = 0) -> HubTranslator:
    torch.manual_seed(seed)
    return HubTranslator(config).eval()


def test_forward_shapes_per_target():
    translator = make()
    x = torch.randn(7, 2, 64)
    assert translator(x, "a").shape == (7, 4, 96)
    assert translator(x, "b").shape == (7, 3, 32)
    assert translator.num_parameters("a") < translator.num_parameters()


def test_deterministic_given_seed_and_input():
    x = torch.randn(9, 2, 64)
    out1 = make(seed=3)(x, "a")
    out2 = make(seed=3)(x, "a")
    torch.testing.assert_close(out1, out2, atol=0, rtol=0)
    torch.testing.assert_close(make(seed=3)(x, "a"), out1, atol=0, rtol=0)
    assert not torch.allclose(make(seed=4)(x, "a"), out1)


def test_context_block_is_causal():
    translator = make()
    x = torch.randn(10, 2, 64)
    out = translator(x, "a")
    x2 = x.clone()
    x2[6:] += 1.0
    out2 = translator(x2, "a")
    torch.testing.assert_close(out[:6], out2[:6])
    assert not torch.allclose(out[6:], out2[6:])


def test_token_local_without_context_layers():
    config = HubConfig(src_layers=(0,), src_dim=16, latent_dim=8, targets={"a": (2, 8)})
    translator = make(config)
    x = torch.randn(5, 1, 16)
    single = torch.cat([translator(x[i : i + 1], "a") for i in range(5)])
    torch.testing.assert_close(translator(x, "a"), single)


def test_calibrate_sets_mean_and_scale():
    translator = make()
    residuals = torch.randn(200, 4, 96) * 3 + 2
    translator.heads["a"].calibrate(residuals)
    torch.testing.assert_close(translator.heads["a"].bias, residuals.mean(0))
    assert torch.allclose(translator.heads["a"].scale, torch.full((4,), 3.0), atol=0.3)


def test_save_load_roundtrip(tmp_path):
    translator = make()
    translator.add_target("c", 2, 16)
    path = tmp_path / "hub.pt"
    translator.save(str(path))
    loaded = HubTranslator.load(str(path))
    x = torch.randn(4, 2, 64)
    for target in ("a", "b", "c"):
        torch.testing.assert_close(loaded(x, target), translator(x, target))


@pytest.mark.parametrize("context_layers", [0, 1])
def test_flops_scale_with_context(context_layers):
    config = HubConfig(
        src_layers=(0,),
        src_dim=16,
        latent_dim=8,
        targets={"a": (2, 8)},
        context_layers=context_layers,
    )
    translator = make(config)
    short, long = (
        translator.flops_per_token("a", 1),
        translator.flops_per_token("a", 1000),
    )
    assert (long > short) == (context_layers > 0)


def test_predictor_caches_per_example_without_aliasing(
    tiny_qwen3, unigram_tokenizer, corpus
):
    from vllm.distributed.kv_transfer.kv_translation.data import prepare_example

    args = (tiny_qwen3, unigram_tokenizer, tiny_qwen3, unigram_tokenizer)
    dim = tiny_qwen3.config.hidden_size
    config = HubConfig(
        src_layers=(1, 2),
        src_dim=dim,
        latent_dim=32,
        targets={"q": (tiny_qwen3.config.num_hidden_layers, dim)},
    )
    predictor = TranslatorPredictor(make(config), "q")
    ex = prepare_example(*args, corpus[0])
    h1 = predictor.predict_hidden(ex, 1)
    assert h1.shape == (int(ex.content.sum()), dim)
    assert predictor.predict_all(ex) is predictor.predict_all(ex)
    # A fresh example of a different length must never see a stale result.
    for text in corpus[1:6]:
        other = prepare_example(*args, text)
        assert predictor.predict_all(other).shape[0] == int(other.content.sum())
        del other
    predictor.reset()
    assert predictor._pred is None
