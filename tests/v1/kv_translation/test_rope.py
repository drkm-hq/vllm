# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.kv_transfer.kv_translation import (
    apply_rope,
    rope_inv_freq,
    strip_rope,
)
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

T, H, D = 40, 3, 64


@pytest.fixture
def cpu_vllm_config():
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(config):
        yield config


@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("base", [10_000.0, 500_000.0, 1_000_000.0])
def test_matches_vllm_rotary_embedding(cpu_vllm_config, is_neox_style, base):
    torch.manual_seed(0)
    rope = RotaryEmbedding(
        head_size=D,
        rotary_dim=D,
        max_position_embeddings=1024,
        base=base,
        is_neox_style=is_neox_style,
        dtype=torch.float32,
    )
    positions = torch.arange(T)
    query = torch.randn(T, H * D)
    key = torch.randn(T, H * D)
    _, key_ref = rope.forward_native(positions, query.clone(), key.clone())

    inv_freq = rope_inv_freq(D, base)
    key_ours = apply_rope(key.view(T, H, D), positions, inv_freq, is_neox_style)
    torch.testing.assert_close(key_ours.reshape(T, H * D), key_ref)

    stripped = strip_rope(key_ref.view(T, H, D), positions, inv_freq, is_neox_style)
    torch.testing.assert_close(stripped.reshape(T, H * D), key, atol=1e-5, rtol=0)


def test_stripped_keys_are_position_free():
    torch.manual_seed(0)
    inv_freq = rope_inv_freq(D, 500_000.0)
    key = torch.randn(1, H, D).expand(T, H, D)
    positions = torch.arange(100, 100 + T)
    rotated = apply_rope(key, positions, inv_freq, True)
    assert not torch.allclose(rotated[0], rotated[-1])
    stripped = strip_rope(rotated, positions, inv_freq, True)
    torch.testing.assert_close(stripped, key, atol=1e-5, rtol=0)


def test_partial_rotary_passes_tail_through():
    torch.manual_seed(0)
    inv_freq = rope_inv_freq(D // 2, 10_000.0)
    key = torch.randn(T, H, D)
    rotated = apply_rope(key, torch.arange(T), inv_freq, True)
    torch.testing.assert_close(rotated[..., D // 2 :], key[..., D // 2 :])
    assert not torch.allclose(rotated[1:, :, : D // 2], key[1:, :, : D // 2])
