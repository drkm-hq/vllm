# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline viability study for one (source, target) model pair.

Measures how well the target's per-layer state is linearly predictable
from the source's residual stream, then what the prediction error costs
downstream: KL divergence and top-1 agreement of the target's next-token
distribution on a continuation whose prefix cache was translated rather
than natively prefilled.

A handoff layer ``L`` translates the cache of layers below ``L`` and
natively recomputes layers ``L`` and above from the predicted residual
stream at ``L``. ``L == num_layers`` is pure translation; ``L == 0`` with
an exact residual is the native oracle. Runs with HF models only; it is a
measurement tool, not the serving path.
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DynamicCache

from vllm.distributed.kv_transfer.kv_translation.alignment import (
    TokenSpans,
    align_spans,
)
from vllm.distributed.kv_transfer.kv_translation.capture import (
    CapturedStates,
    capture_states,
)
from vllm.distributed.kv_transfer.kv_translation.mapper import (
    LinearMapper,
    r2_score,
    ridge_fit,
    select_source_layers,
)

StateKind = Literal["src", "hidden", "keys", "values"]


@dataclass
class AlignedExample:
    """One text split into a target-token prefix and continuation."""

    src: TokenSpans
    tgt: TokenSpans
    src_states: CapturedStates
    tgt_states: CapturedStates
    src_pos: np.ndarray
    exact: np.ndarray
    cont_ids: torch.Tensor

    @property
    def content(self) -> np.ndarray:
        return self.src_pos >= 0

    def exact_states(self, kind: StateKind, layer: int) -> torch.Tensor:
        """Rows of one state at the exactly aligned positions."""
        tgt_pos = np.flatnonzero(self.exact)
        if kind == "src":
            return self.src_states.residual[layer][self.src_pos[tgt_pos]]
        if kind == "hidden":
            return self.tgt_states.residual[layer][tgt_pos]
        if kind == "keys":
            return self.tgt_states.keys[layer][tgt_pos].flatten(1)
        return self.tgt_states.values[layer][tgt_pos].flatten(1)


def source_position_policy(alignment) -> np.ndarray:
    """Exact match if any, else the nearest source state that has seen at
    least the target token's text, else the nearest one before it."""
    pos = np.where(alignment.exact, alignment.src_before, alignment.src_after)
    return np.where(pos >= 0, pos, alignment.src_before)


def prepare_example(
    src_model,
    src_tokenizer,
    tgt_model,
    tgt_tokenizer,
    text: str,
    prefix_frac: float = 0.75,
) -> AlignedExample | None:
    tgt_full = TokenSpans.from_tokenizer(tgt_tokenizer, text)
    n_prefix = int(len(tgt_full) * prefix_frac)
    if n_prefix < 2 or n_prefix >= len(tgt_full):
        return None
    prefix_text = text[: tgt_full.offsets[n_prefix - 1, 1]]
    tgt = TokenSpans(tgt_full.ids[:n_prefix], tgt_full.offsets[:n_prefix])
    src = TokenSpans.from_tokenizer(src_tokenizer, prefix_text)
    alignment = align_spans(src, tgt)
    return AlignedExample(
        src=src,
        tgt=tgt,
        src_states=capture_states(src_model, torch.as_tensor(src.ids)),
        tgt_states=capture_states(tgt_model, torch.as_tensor(tgt.ids)),
        src_pos=source_position_policy(alignment),
        exact=alignment.exact,
        cont_ids=torch.as_tensor(tgt_full.ids[n_prefix:]),
    )


def features(
    states: CapturedStates, layers: Sequence[int], positions: np.ndarray
) -> torch.Tensor:
    idx = torch.as_tensor(positions)
    return torch.cat([states.residual[layer][idx] for layer in layers], dim=-1)


def stack_exact(
    examples: Sequence[AlignedExample], kind: StateKind, layer: int
) -> torch.Tensor:
    return torch.cat(
        [ex.exact_states(kind, layer) for ex in examples if ex.exact.any()]
    )


def stack_features(
    examples: Sequence[AlignedExample], layers: Sequence[int]
) -> torch.Tensor:
    return torch.cat([stack_exact(examples, "src", s) for s in layers], dim=-1)


@dataclass
class PairMappers:
    """Per target layer: chosen source layers and mappers to its states."""

    src_layers: dict[int, tuple[int, ...]]
    hidden: dict[int, LinearMapper]
    keys: dict[int, LinearMapper]
    values: dict[int, LinearMapper]


def fit_pair_mappers(
    examples: Sequence[AlignedExample],
    top_k: int = 3,
    lam: float = 1.0,
    max_score_rows: int = 4096,
) -> PairMappers:
    """Two stages: score each source layer for each target layer by
    single-layer residual R2, then fit mappers on the top-k concatenation."""
    num_src = len(examples[0].src_states.residual)
    num_tgt = len(examples[0].tgt_states.residual) - 1
    src_feats = {s: stack_exact(examples, "src", s) for s in range(num_src)}
    n = min(src_feats[0].shape[0], max_score_rows)
    split = int(0.8 * n)

    mappers = PairMappers({}, {}, {}, {})
    for t in range(num_tgt):
        hidden = stack_exact(examples, "hidden", t)
        scores = {}
        for s in range(num_src):
            x, y = src_feats[s][:n], hidden[:n]
            w, b = ridge_fit(x[:split], y[:split], lam)
            scores[s] = r2_score(y[split:], x[split:] @ w + b)
        layers = select_source_layers(scores, top_k)
        x = torch.cat([src_feats[s] for s in layers], dim=-1)
        mappers.src_layers[t] = layers
        mappers.hidden[t] = LinearMapper.fit(x, hidden, layers, f"h:{t}", lam)
        keys, values = (
            stack_exact(examples, "keys", t),
            stack_exact(examples, "values", t),
        )
        mappers.keys[t] = LinearMapper.fit(x, keys, layers, f"k:{t}", lam)
        mappers.values[t] = LinearMapper.fit(x, values, layers, f"v:{t}", lam)
    return mappers


def heldout_r2(
    mappers: PairMappers, examples: Sequence[AlignedExample]
) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {"hidden": {}, "keys": {}, "values": {}}
    for t, layers in mappers.src_layers.items():
        x = stack_features(examples, layers)
        for name in out:
            y = stack_exact(examples, name, t)  # type: ignore[arg-type]
            out[name][t] = r2_score(y, getattr(mappers, name)[t](x))
    return out


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_target_rope(model, keys: torch.Tensor) -> torch.Tensor:
    """Rotate ``keys`` ``[1, H, T, D]`` with the model's own rotary module."""
    num_tokens = keys.shape[2]
    positions = torch.arange(num_tokens, device=keys.device)[None]
    cos, sin = model.model.rotary_emb(keys, positions)
    cos, sin = cos[:, None], sin[:, None]
    return keys * cos + _rotate_half(keys) * sin


def translated_layer_kv(
    model, mappers: PairMappers, ex: AlignedExample, layer: int, mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predicted post-RoPE keys and values ``[1, H, T, D]`` for content
    positions of layer ``layer``; non-content rows are zero."""
    num_tokens = len(ex.tgt)
    content = np.flatnonzero(ex.content)
    x = features(ex.src_states, mappers.src_layers[layer], ex.src_pos[content])
    attn = model.model.layers[layer].self_attn
    num_kv_heads = model.config.num_key_value_heads
    if mode == "kv":
        k = mappers.keys[layer](x)
        v = mappers.values[layer](x)
    elif mode == "resid":
        h = model.model.layers[layer].input_layernorm(mappers.hidden[layer](x))
        k = attn.k_proj(h).view(len(content), num_kv_heads, -1)
        k_norm = getattr(attn, "k_norm", None)
        k = (k_norm(k) if k_norm is not None else k).flatten(1)
        v = attn.v_proj(h)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    full_k = torch.zeros(num_tokens, k.shape[1], dtype=k.dtype)
    full_v = torch.zeros(num_tokens, v.shape[1], dtype=v.dtype)
    full_k[content] = k
    full_v[content] = v

    def to_cache(t: torch.Tensor) -> torch.Tensor:
        return t.view(1, num_tokens, num_kv_heads, -1).transpose(1, 2)

    return _apply_target_rope(model, to_cache(full_k)), to_cache(full_v)


@torch.no_grad()
def translated_cache(
    model,
    mappers: PairMappers,
    ex: AlignedExample,
    handoff_layer: int,
    mode: str = "resid",
    predicted_hidden: torch.Tensor | None = None,
) -> DynamicCache:
    """Cache for the prefix: layers below ``handoff_layer`` translated,
    layers from it on recomputed natively from the residual at the handoff.

    Special tokens (no source counterpart) keep native states in every
    layer. ``predicted_hidden`` overrides the mapper at the handoff layer,
    which makes the exact residual an oracle for the mechanics.
    """
    num_layers = model.config.num_hidden_layers
    content = ex.content
    prefix_ids = torch.as_tensor(ex.tgt.ids)[None]
    handles = []
    if handoff_layer < num_layers:
        if predicted_hidden is None:
            layers = mappers.src_layers[handoff_layer]
            x = features(ex.src_states, layers, ex.src_pos[content])
            predicted_hidden = ex.tgt_states.residual[handoff_layer].clone()
            predicted_hidden[torch.as_tensor(content)] = mappers.hidden[handoff_layer](
                x
            )

        def swap_input(module, args, kwargs, h=predicted_hidden):
            return (h[None].to(args[0].dtype),) + tuple(args[1:]), kwargs

        handles.append(
            model.model.layers[handoff_layer].register_forward_pre_hook(
                swap_input, with_kwargs=True
            )
        )
    try:
        native = model(input_ids=prefix_ids, use_cache=True).past_key_values
    finally:
        for handle in handles:
            handle.remove()

    keep_native = torch.as_tensor(~content)
    cache = DynamicCache(config=model.config)
    for layer in range(num_layers):
        k, v = native.layers[layer].keys, native.layers[layer].values
        if layer < handoff_layer:
            k_hat, v_hat = translated_layer_kv(model, mappers, ex, layer, mode)
            k_hat[:, :, keep_native] = k[:, :, keep_native]
            v_hat[:, :, keep_native] = v[:, :, keep_native]
            k, v = k_hat.to(k.dtype), v_hat.to(v.dtype)
        cache.update(k, v, layer)
    return cache


@dataclass
class HandoffResult:
    handoff_layer: int
    native_layer_fraction: float
    kl_mean: float
    top1_agreement: float


@torch.no_grad()
def continuation_divergence(
    model, ex: AlignedExample, cache: DynamicCache
) -> tuple[float, float]:
    """Mean ``KL(native || translated)`` and top-1 agreement over the
    continuation's next-token distributions."""
    prefix = torch.as_tensor(ex.tgt.ids)[None]
    cont = ex.cont_ids[None]
    ref = model(input_ids=torch.cat([prefix, cont], dim=1)).logits[0, prefix.shape[1] :]
    got = model(input_ids=cont, past_key_values=cache, use_cache=True).logits[0]
    log_p, log_q = F.log_softmax(ref.float(), -1), F.log_softmax(got.float(), -1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1).mean()
    agree = (ref.argmax(-1) == got.argmax(-1)).float().mean()
    return float(kl), float(agree)


@dataclass
class StudyReport:
    boundary_agreement: float
    src_layers: dict[int, tuple[int, ...]]
    r2: dict[str, dict[int, float]]
    handoff: list[HandoffResult] = field(default_factory=list)


def run_pair_study(
    src_model,
    src_tokenizer,
    tgt_model,
    tgt_tokenizer,
    train_texts: Sequence[str],
    eval_texts: Sequence[str],
    top_k: int = 3,
    lam: float = 1.0,
    handoff_layers: Sequence[int] | None = None,
    mode: str = "resid",
    prefix_frac: float = 0.75,
) -> StudyReport:
    def prepare(texts: Sequence[str]) -> list[AlignedExample]:
        examples = [
            prepare_example(
                src_model, src_tokenizer, tgt_model, tgt_tokenizer, t, prefix_frac
            )
            for t in texts
        ]
        return [ex for ex in examples if ex is not None]

    train, evals = prepare(train_texts), prepare(eval_texts)
    mappers = fit_pair_mappers(train, top_k, lam)
    agreement = float(np.mean([ex.exact[ex.tgt.is_content].mean() for ex in evals]))
    report = StudyReport(agreement, mappers.src_layers, heldout_r2(mappers, evals))
    num_layers = tgt_model.config.num_hidden_layers
    for layer in handoff_layers if handoff_layers is not None else [num_layers]:
        kls, agrees = [], []
        for ex in evals:
            cache = translated_cache(tgt_model, mappers, ex, layer, mode)
            kl, agree = continuation_divergence(tgt_model, ex, cache)
            kls.append(kl)
            agrees.append(agree)
        native_fraction = (num_layers - layer) / num_layers
        report.handoff.append(
            HandoffResult(
                layer, native_fraction, float(np.mean(kls)), float(np.mean(agrees))
            )
        )
    return report


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="source model name or path")
    parser.add_argument("--tgt", required=True, help="target model name or path")
    parser.add_argument("--texts", required=True, help="file with one text per line")
    parser.add_argument("--eval-frac", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--mode", choices=["resid", "kv"], default="resid")
    parser.add_argument("--handoff-layers", type=int, nargs="*", default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    def load(name: str):
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        return model.to(args.device).eval()

    src_model, tgt_model = load(args.src), load(args.tgt)
    src_tok = AutoTokenizer.from_pretrained(args.src)
    tgt_tok = AutoTokenizer.from_pretrained(args.tgt)
    with open(args.texts) as f:
        texts = [line.rstrip("\n") for line in f if line.strip()]
    n_eval = max(1, int(len(texts) * args.eval_frac))
    report = run_pair_study(
        src_model,
        src_tok,
        tgt_model,
        tgt_tok,
        texts[n_eval:],
        texts[:n_eval],
        top_k=args.top_k,
        lam=args.lam,
        handoff_layers=args.handoff_layers,
        mode=args.mode,
    )
    print(json.dumps(asdict(report), indent=1))


if __name__ == "__main__":
    main()
