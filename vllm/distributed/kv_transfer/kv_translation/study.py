# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline viability study for one (source, target) model pair.

Measures how well the target's per-layer state is predictable from the
source's residual stream, then what the prediction error costs downstream:
KL divergence and top-1 agreement of the target's next-token distribution
on a continuation whose prefix cache was translated rather than natively
prefilled.

Any ``ResidualPredictor`` can be measured: the closed-form ridge mappers
fit here, or a trained ``HubTranslator``. A handoff layer ``L`` translates
the cache of layers below ``L`` and natively recomputes layers ``L`` and
above from the predicted residual stream at ``L``. ``L == num_layers`` is
pure translation; ``L == 0`` with an exact residual is the native oracle.
Runs with HF models only; it is a measurement tool, not the serving path.
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DynamicCache

from vllm.distributed.kv_transfer.kv_translation.data import (
    AlignedExample,
    features,
    prepare_example,
    stack_exact,
    stack_features,
)
from vllm.distributed.kv_transfer.kv_translation.mapper import (
    LinearMapper,
    r2_score,
    ridge_fit,
    select_source_layers,
)


class ResidualPredictor(Protocol):
    """Predicts the target residual stream at an example's content positions."""

    def predict_hidden(self, ex: AlignedExample, layer: int) -> torch.Tensor:
        """``[num_content, d_tgt]`` in target order."""
        ...


@dataclass
class PairMappers:
    """Per target layer: chosen source layers and closed-form mappers."""

    src_layers: dict[int, tuple[int, ...]]
    hidden: dict[int, LinearMapper]
    keys: dict[int, LinearMapper]
    values: dict[int, LinearMapper]

    def _features(self, ex: AlignedExample, layer: int) -> torch.Tensor:
        return features(ex.src_states, self.src_layers[layer], ex.src_pos[ex.content])

    def predict_hidden(self, ex: AlignedExample, layer: int) -> torch.Tensor:
        return self.hidden[layer](self._features(ex, layer))

    def predict_kv(
        self, ex: AlignedExample, layer: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._features(ex, layer)
        return self.keys[layer](x), self.values[layer](x)


def fit_pair_mappers(
    examples: Sequence[AlignedExample],
    top_k: int = 3,
    lam: float = 1.0,
    max_score_rows: int = 4096,
) -> PairMappers:
    """Two stages: score each source layer for each target layer by
    single-layer residual R2, then fit mappers on the top-k concatenation."""
    num_src = len(examples[0].src_states.residual)
    num_tgt = examples[0].num_target_layers
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


@torch.no_grad()
def predictor_hidden_r2(
    predictor: ResidualPredictor, examples: Sequence[AlignedExample]
) -> dict[int, float]:
    """Held-out R2 of the predicted residual at exactly aligned positions."""
    num_layers = examples[0].num_target_layers
    out = {}
    for layer in range(num_layers):
        preds, targets = [], []
        for ex in examples:
            exact_in_content = torch.as_tensor(ex.exact[ex.content])
            preds.append(predictor.predict_hidden(ex, layer)[exact_in_content])
            targets.append(ex.exact_states("hidden", layer))
        out[layer] = r2_score(torch.cat(targets), torch.cat(preds))
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
    model, predictor: ResidualPredictor, ex: AlignedExample, layer: int, mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predicted post-RoPE keys and values ``[1, H, T, D]`` for content
    positions of layer ``layer``; non-content rows are zero.

    ``resid`` pushes the predicted residual through the target's own
    projections and QK-norm; ``kv`` needs a predictor with ``predict_kv``.
    """
    num_tokens = len(ex.tgt)
    content = np.flatnonzero(ex.content)
    attn = model.model.layers[layer].self_attn
    num_kv_heads = model.config.num_key_value_heads
    if mode == "resid":
        h = model.model.layers[layer].input_layernorm(
            predictor.predict_hidden(ex, layer)
        )
        k = attn.k_proj(h).view(len(content), num_kv_heads, -1)
        k_norm = getattr(attn, "k_norm", None)
        k = (k_norm(k) if k_norm is not None else k).flatten(1)
        v = attn.v_proj(h)
    elif mode == "kv":
        predict_kv = getattr(predictor, "predict_kv", None)
        if predict_kv is None:
            raise ValueError("mode 'kv' needs a predictor with predict_kv")
        k, v = predict_kv(ex, layer)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    full_k = torch.zeros(num_tokens, k.shape[1], dtype=k.dtype)
    full_v = torch.zeros(num_tokens, v.shape[1], dtype=v.dtype)
    full_k[content] = k
    full_v[content] = v

    def to_cache(t: torch.Tensor) -> torch.Tensor:
        return t.view(1, num_tokens, num_kv_heads, -1).transpose(1, 2)

    return _apply_target_rope(model, to_cache(full_k)), to_cache(full_v)


def translated_cache(
    model,
    predictor: ResidualPredictor,
    ex: AlignedExample,
    handoff_layer: int,
    mode: str = "resid",
    predicted_hidden: torch.Tensor | None = None,
) -> DynamicCache:
    """Cache for the prefix: layers below ``handoff_layer`` translated,
    layers from it on recomputed natively from the residual at the handoff.

    Special tokens (no source counterpart) keep native states in every
    layer. ``predicted_hidden`` overrides the predictor at the handoff
    layer, which makes the exact residual an oracle for the mechanics.
    Gradients flow through the translated entries when the caller
    enables them; the frozen target's own prefill runs without.
    """
    num_layers = model.config.num_hidden_layers
    content = ex.content
    prefix_ids = torch.as_tensor(ex.tgt.ids)[None]
    handles = []
    if handoff_layer < num_layers:
        if predicted_hidden is None:
            predicted_hidden = ex.tgt_states.residual[handoff_layer].clone()
            predicted_hidden[torch.as_tensor(content)] = predictor.predict_hidden(
                ex, handoff_layer
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
            k_hat, v_hat = translated_layer_kv(model, predictor, ex, layer, mode)
            k_hat[:, :, keep_native] = k[:, :, keep_native]
            v_hat[:, :, keep_native] = v[:, :, keep_native]
            k, v = k_hat.to(k.dtype), v_hat.to(v.dtype)
        cache.update(k, v, layer)
    return cache


def continuation_logits(
    model, ex: AlignedExample, cache: DynamicCache
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native and translated next-token logits over the continuation."""
    prefix = torch.as_tensor(ex.tgt.ids)[None]
    cont = ex.cont_ids[None]
    with torch.no_grad():
        full = model(input_ids=torch.cat([prefix, cont], dim=1)).logits
    ref = full[0, prefix.shape[1] :]
    got = model(input_ids=cont, past_key_values=cache, use_cache=True).logits[0]
    return ref, got


def divergence(
    ref: torch.Tensor, got: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean ``KL(native || translated)`` and top-1 agreement."""
    log_p, log_q = F.log_softmax(ref.float(), -1), F.log_softmax(got.float(), -1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1).mean()
    agree = (ref.argmax(-1) == got.argmax(-1)).float().mean()
    return kl, agree


@torch.no_grad()
def acceptance_metrics(ref: torch.Tensor, got: torch.Tensor) -> tuple[float, int]:
    """Speculative acceptance rate ``E[sum_v min(p, q)]`` of the translated
    distribution as a draft for the native one, and the greedy run length:
    leading continuation positions whose argmax agrees with native."""
    p, q = F.softmax(ref.float(), -1), F.softmax(got.float(), -1)
    alpha = torch.minimum(p, q).sum(-1).mean()
    agree = ref.argmax(-1) == got.argmax(-1)
    run = int(agree.shape[0]) if bool(agree.all()) else int((~agree).int().argmax())
    return float(alpha), run


@torch.no_grad()
def continuation_divergence(
    model, ex: AlignedExample, cache: DynamicCache
) -> tuple[float, float]:
    kl, agree = divergence(*continuation_logits(model, ex, cache))
    return float(kl), float(agree)


@dataclass
class HandoffResult:
    """Scores for one handoff layer.

    ``kl_control_mean`` is the same measurement with the cache translated
    from a *different* document's source states: if it is not far worse
    than ``kl_mean``, nothing about the context is being transferred.
    """

    handoff_layer: int
    native_layer_fraction: float
    kl_mean: float
    top1_agreement: float
    acceptance_rate: float
    greedy_run_length_mean: float
    kl_control_mean: float


@dataclass
class StudyReport:
    boundary_agreement: float
    r2_hidden: dict[int, float]
    handoff: list[HandoffResult] = field(default_factory=list)
    src_layers: dict[int, tuple[int, ...]] | None = None
    r2_keys: dict[int, float] | None = None
    r2_values: dict[int, float] | None = None
    r2_bound_by_rank: dict[int, dict[int, float]] | None = None


def mismatched_control(ex: AlignedExample, other: AlignedExample) -> AlignedExample:
    """``ex`` with its source states replaced by another document's, so the
    translator sees a real but unrelated context at every position."""
    src_pos = np.minimum(ex.src_pos, len(other.src) - 1)
    src_pos = np.where(ex.src_pos >= 0, src_pos, -1)
    return AlignedExample(
        src=other.src,
        tgt=ex.tgt,
        src_states=other.src_states,
        tgt_states=ex.tgt_states,
        src_pos=src_pos,
        exact=ex.exact,
        cont_ids=ex.cont_ids,
    )


@torch.no_grad()
def capacity_bound(
    examples: Sequence[AlignedExample], ranks: Sequence[int]
) -> dict[int, dict[int, float]]:
    """R2 ceiling of a rank-``r`` affine head per target layer: the fraction
    of the layer's residual variance in its top ``r`` principal directions.
    Decides whether a head rank can work before any training spend."""
    out: dict[int, dict[int, float]] = {}
    for layer in range(examples[0].num_target_layers):
        h = stack_exact(examples, "hidden", layer).double()
        h = h - h.mean(0, keepdim=True)
        eig = torch.linalg.eigvalsh(h.T @ h).flip(0).clamp_min(0)
        energy = eig.cumsum(0) / eig.sum().clamp_min(1e-12)
        out[layer] = {
            r: float(energy[min(r, len(energy)) - 1]) if r > 0 else 0.0 for r in ranks
        }
    return out


@torch.no_grad()
def evaluate_predictor(
    tgt_model,
    predictor: ResidualPredictor,
    evals: Sequence[AlignedExample],
    handoff_layers: Sequence[int] | None = None,
    mode: str = "resid",
    capacity_ranks: Sequence[int] = (),
) -> StudyReport:
    """Score a predictor on held-out examples."""
    agreement = float(np.mean([ex.exact[ex.tgt.is_content].mean() for ex in evals]))
    report = StudyReport(agreement, predictor_hidden_r2(predictor, evals))
    if capacity_ranks:
        report.r2_bound_by_rank = capacity_bound(evals, capacity_ranks)
    if isinstance(predictor, PairMappers):
        r2 = heldout_r2(predictor, evals)
        report.src_layers = predictor.src_layers
        report.r2_keys, report.r2_values = r2["keys"], r2["values"]
    num_layers = tgt_model.config.num_hidden_layers
    for layer in handoff_layers if handoff_layers is not None else [num_layers]:
        rows = []
        for i, ex in enumerate(evals):
            cache = translated_cache(tgt_model, predictor, ex, layer, mode)
            ref, got = continuation_logits(tgt_model, ex, cache)
            kl, agree = divergence(ref, got)
            alpha, run = acceptance_metrics(ref, got)
            control = mismatched_control(ex, evals[(i + 1) % len(evals)])
            control_cache = translated_cache(tgt_model, predictor, control, layer, mode)
            kl_control, _ = continuation_divergence(tgt_model, ex, control_cache)
            rows.append((float(kl), float(agree), alpha, float(run), kl_control))
        kl, agree, alpha, run, kl_control = (float(v) for v in np.mean(rows, axis=0))
        report.handoff.append(
            HandoffResult(
                handoff_layer=layer,
                native_layer_fraction=(num_layers - layer) / num_layers,
                kl_mean=kl,
                top1_agreement=agree,
                acceptance_rate=alpha,
                greedy_run_length_mean=run,
                kl_control_mean=kl_control,
            )
        )
    return report


def prepare_examples(
    src_model, src_tokenizer, tgt_model, tgt_tokenizer, texts, prefix_frac=0.75
) -> list[AlignedExample]:
    examples = [
        prepare_example(
            src_model, src_tokenizer, tgt_model, tgt_tokenizer, t, prefix_frac
        )
        for t in texts
    ]
    return [ex for ex in examples if ex is not None]


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
    predictor: ResidualPredictor | None = None,
    capacity_ranks: Sequence[int] = (),
) -> StudyReport:
    """Fit closed-form mappers on ``train_texts`` (unless a trained
    ``predictor`` is given) and score on ``eval_texts``."""
    args = (src_model, src_tokenizer, tgt_model, tgt_tokenizer)
    evals = prepare_examples(*args, eval_texts, prefix_frac)
    if predictor is None:
        train = prepare_examples(*args, train_texts, prefix_frac)
        predictor = fit_pair_mappers(train, top_k, lam)
    return evaluate_predictor(
        tgt_model, predictor, evals, handoff_layers, mode, capacity_ranks
    )


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
    parser.add_argument("--translator", default=None, help="trained HubTranslator .pt")
    parser.add_argument(
        "--capacity-ranks", type=int, nargs="*", default=[512, 1024, 2048]
    )
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
    predictor = None
    if args.translator:
        from vllm.distributed.kv_transfer.kv_translation.translator import (
            HubTranslator,
            TranslatorPredictor,
        )

        translator = HubTranslator.load(args.translator).to(args.device)
        predictor = TranslatorPredictor(translator, args.tgt)
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
        predictor=predictor,
        capacity_ranks=args.capacity_ranks,
    )
    print(json.dumps(asdict(report), indent=1))


if __name__ == "__main__":
    main()
