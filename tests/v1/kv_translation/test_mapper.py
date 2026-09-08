# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.distributed.kv_transfer.kv_translation import (
    LinearMapper,
    r2_score,
    ridge_fit,
    select_source_layers,
)


def test_ridge_recovers_planted_affine_map():
    torch.manual_seed(0)
    n, d_in, d_out = 4096, 64, 32
    x = torch.randn(n, d_in)
    weight = torch.randn(d_in, d_out) / d_in**0.5
    bias = torch.randn(d_out)
    y = x @ weight + bias + 0.01 * torch.randn(n, d_out)
    w_hat, b_hat = ridge_fit(x[:3000], y[:3000], lam=1e-3)
    torch.testing.assert_close(w_hat, weight, atol=2e-2, rtol=0)
    torch.testing.assert_close(b_hat, bias, atol=2e-2, rtol=0)
    assert r2_score(y[3000:], x[3000:] @ w_hat + b_hat) > 0.99


def test_linear_mapper_roundtrip_preserves_dtype():
    torch.manual_seed(0)
    x = torch.randn(256, 16)
    y = x[:, :8] * 2 + 1
    mapper = LinearMapper.fit(x, y, src_layers=(0, 2), target="k:3")
    out = mapper(x.to(torch.bfloat16))
    assert out.dtype == torch.bfloat16
    assert r2_score(y, out.float()) > 0.99


def test_select_source_layers_orders_by_layer():
    scores = {0: 0.1, 1: 0.9, 2: 0.5, 3: 0.8}
    assert select_source_layers(scores, 2) == (1, 3)
    assert select_source_layers(scores, 10) == (0, 1, 2, 3)
