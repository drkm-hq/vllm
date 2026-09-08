# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Learned hub translator between models' residual streams.

The source's residual stream at a few layers is encoded into a shared
latent per position; per-target-layer heads decode the latent into the
target's residual stream, which the target's own projections turn into
its native cache. Onboarding another target adds heads, not an encoder.

Two capacities are expressed by one class: ``context_layers == 0`` is a
token-local map; ``context_layers >= 1`` adds causal attention over the
latent sequence, the size class of a speculative-decoding draft head, so
the translator can repair tokenizer-boundary mismatch from neighbours.
Every path is a deterministic function of its inputs.
"""

import weakref
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from vllm.distributed.kv_transfer.kv_translation.data import AlignedExample


@dataclass
class HubConfig:
    src_layers: tuple[int, ...]
    src_dim: int
    latent_dim: int
    targets: dict[str, tuple[int, int]] = field(default_factory=dict)
    head_rank: int | None = None
    context_layers: int = 0
    context_heads: int = 4
    mlp_mult: int = 2


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight).to(x.dtype)


class GatedMLP(nn.Module):
    def __init__(self, dim: int, mult: int):
        super().__init__()
        self.up = nn.Linear(dim, 2 * mult * dim, bias=False)
        self.down = nn.Linear(mult * dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class CausalBlock(nn.Module):
    """Pre-norm causal self-attention plus gated MLP over one sequence.

    Uses ``scaled_dot_product_attention`` with ``is_causal`` so no ``[T, T]``
    mask is materialized and flash kernels apply at long prefixes.
    """

    def __init__(self, dim: int, heads: int, mult: int):
        super().__init__()
        if dim % heads:
            raise ValueError(f"latent_dim {dim} not divisible by {heads} heads")
        self.heads = heads
        self.attn_norm = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = GatedMLP(dim, mult)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        num_tokens = z.shape[0]
        q, k, v = (
            t.view(num_tokens, self.heads, -1).transpose(0, 1)
            for t in self.qkv(self.attn_norm(z)).chunk(3, dim=-1)
        )
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=num_tokens > 1)
        z = z + self.out(attn.transpose(0, 1).reshape(num_tokens, -1))
        return z + self.mlp(self.mlp_norm(z))


class LayerHeads(nn.Module):
    """Per-target-layer affine heads from the latent, optionally low-rank.

    ``scale`` and ``bias`` are calibrated to the target's per-layer residual
    statistics so an untrained head predicts the layer mean at the right
    magnitude.
    """

    def __init__(
        self, latent_dim: int, num_layers: int, out_dim: int, rank: int | None
    ):
        super().__init__()
        self.rank = rank
        if rank is None:
            self.weight = nn.Parameter(torch.empty(num_layers, latent_dim, out_dim))
            nn.init.normal_(self.weight, std=1.0 / latent_dim**0.5)
        else:
            self.down = nn.Parameter(torch.empty(num_layers, latent_dim, rank))
            self.up = nn.Parameter(torch.empty(num_layers, rank, out_dim))
            nn.init.normal_(self.down, std=1.0 / latent_dim**0.5)
            nn.init.normal_(self.up, std=1.0 / rank**0.5)
        self.scale = nn.Parameter(torch.ones(num_layers))
        self.bias = nn.Parameter(torch.zeros(num_layers, out_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Broadcast matmuls read the weights in place; einsum would copy them.
        if self.rank is None:
            out = torch.matmul(z[None], self.weight)
        else:
            out = torch.matmul(torch.matmul(z[None], self.down), self.up)
        out = out.transpose(0, 1)
        return out * self.scale[None, :, None] + self.bias[None]

    @torch.no_grad()
    def calibrate(self, residuals: torch.Tensor) -> None:
        """Set ``bias`` to the per-layer mean and ``scale`` to the per-layer
        RMS of ``residuals`` ``[N, num_layers, out_dim]``."""
        self.bias.copy_(residuals.float().mean(0))
        centered = residuals.float() - self.bias[None]
        self.scale.copy_(centered.pow(2).mean((0, 2)).sqrt())


class HubTranslator(nn.Module):
    def __init__(self, config: HubConfig):
        super().__init__()
        self.config = config
        k = len(config.src_layers)
        # Static per-layer scales keep each position's residual magnitude,
        # which per-token normalization would discard; calibrated from data.
        self.register_buffer("src_rms", torch.ones(k))
        self.src_weight = nn.Parameter(torch.ones(k, config.src_dim))
        self.encode_in = nn.Linear(k * config.src_dim, config.latent_dim, bias=False)
        self.encode_norm = RMSNorm(config.latent_dim)
        self.encode_mlp = GatedMLP(config.latent_dim, config.mlp_mult)
        self.context = nn.ModuleList(
            [
                CausalBlock(config.latent_dim, config.context_heads, config.mlp_mult)
                for _ in range(config.context_layers)
            ]
        )
        self.heads = nn.ModuleDict()
        for name, (num_layers, dim) in config.targets.items():
            self.add_target(name, num_layers, dim)

    @torch.no_grad()
    def calibrate_source(self, src_feats: torch.Tensor) -> None:
        """Set per-layer input scales from ``[N, k, d_src]`` source residuals."""
        self.src_rms.copy_(src_feats.float().pow(2).mean((0, 2)).sqrt())

    def add_target(self, name: str, num_layers: int, dim: int) -> None:
        self.config.targets[name] = (num_layers, dim)
        self.heads[name] = LayerHeads(
            self.config.latent_dim, num_layers, dim, self.config.head_rank
        )

    def encode(self, src_feats: torch.Tensor) -> torch.Tensor:
        """``[T, k, d_src]`` source residuals to ``[T, latent_dim]``."""
        scaled = src_feats / self.src_rms[None, :, None] * self.src_weight[None]
        z = self.encode_in(scaled.flatten(1))
        z = z + self.encode_mlp(self.encode_norm(z))
        for block in self.context:
            z = block(z)
        return z

    def decode(self, z: torch.Tensor, target: str) -> torch.Tensor:
        """``[T, latent_dim]`` to the target's ``[T, num_layers, d_tgt]``."""
        return self.heads[target](z)

    def forward(self, src_feats: torch.Tensor, target: str) -> torch.Tensor:
        return self.decode(self.encode(src_feats), target)

    def num_parameters(self, target: str | None = None) -> int:
        shared = sum(
            p.numel() for n, p in self.named_parameters() if not n.startswith("heads.")
        )
        if target is None:
            return shared + sum(p.numel() for p in self.heads.parameters())
        return shared + sum(p.numel() for p in self.heads[target].parameters())

    def flops_per_token(self, target: str, context_len: int = 1) -> int:
        """Approximate FLOPs per translated position for the translator
        alone, counting the attention block against ``context_len``
        positions. The target's own projections in the fill, and any native
        top layers, are costed separately by the study."""
        dense = 2 * self.num_parameters(target)
        attn = 4 * self.config.latent_dim * context_len * len(self.context)
        return dense + attn

    def save(self, path: str) -> None:
        torch.save({"config": asdict(self.config), "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str) -> "HubTranslator":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        cfg = blob["config"]
        cfg["src_layers"] = tuple(cfg["src_layers"])
        cfg["targets"] = {k: tuple(v) for k, v in cfg["targets"].items()}
        model = cls(HubConfig(**cfg))
        model.load_state_dict(blob["state"])
        return model


class TranslatorPredictor:
    """Adapts a ``HubTranslator`` to the study's ``ResidualPredictor``.

    The translator runs once per example over all content positions; the
    result is kept for the most recent example, held through a weak
    reference so a freed example can never alias a new one.
    """

    def __init__(self, translator: HubTranslator, target: str):
        self.translator = translator
        self.target = target
        self._last: weakref.ref[AlignedExample] | None = None
        self._pred: torch.Tensor | None = None

    def preset(self, ex: AlignedExample, pred: torch.Tensor) -> None:
        """Use an already computed prediction (e.g. one carrying gradients)."""
        self._last, self._pred = weakref.ref(ex), pred

    def predict_all(self, ex: AlignedExample) -> torch.Tensor:
        """``[num_content, num_layers, d_tgt]``."""
        if self._pred is None or self._last is None or self._last() is not ex:
            feats = ex.source_features(self.translator.config.src_layers)
            self.preset(ex, self.translator(feats, self.target))
        assert self._pred is not None
        return self._pred

    def predict_hidden(self, ex: AlignedExample, layer: int) -> torch.Tensor:
        return self.predict_all(ex)[:, layer]

    def reset(self) -> None:
        self._last, self._pred = None, None
