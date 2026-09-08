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
import inspect
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DynamicCache

from vllm.distributed.kv_transfer.kv_translation.capture import (
    decoder,
    text_config,
)
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
    predictor: ResidualPredictor,
    examples: Sequence[AlignedExample],
    exact_only: bool = True,
) -> dict[int, float]:
    """Held-out R2 of the predicted residual per target layer, at exactly
    aligned content positions or, with ``exact_only`` off, at the inexact
    ones the position policy has to stand in for."""
    num_layers = examples[0].num_target_layers
    out = {}
    for layer in range(num_layers):
        preds, targets = [], []
        for ex in examples:
            content = torch.as_tensor(ex.content)
            select = torch.as_tensor(ex.exact[ex.content])
            if not exact_only:
                select = ~select
            if not select.any():
                continue
            preds.append(predictor.predict_hidden(ex, layer)[select])
            targets.append(ex.tgt_states.residual[layer][content][select])
        out[layer] = (
            r2_score(torch.cat(targets), torch.cat(preds)) if preds else float("nan")
        )
    return out


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_target_rope(model, layer: int, keys: torch.Tensor) -> torch.Tensor:
    """Rotate ``keys`` ``[1, H, T, D]`` with the rotary module the model uses
    for ``layer`` (local or global on sliding-window models), over the
    rotary dims only; the rest pass through as in partial-rotary models."""
    stack = decoder(model)
    rotary = stack.rotary_emb
    kwargs = {}
    layer_types = getattr(text_config(model), "layer_types", None)
    if layer_types is not None:
        if "layer_type" in inspect.signature(rotary.forward).parameters:
            kwargs["layer_type"] = layer_types[layer]
        elif layer_types[layer] == "sliding_attention":
            rotary = getattr(stack, "rotary_emb_local", rotary)
    num_tokens = keys.shape[2]
    positions = torch.arange(num_tokens, device=keys.device)[None]
    cos, sin = rotary(keys, positions, **kwargs)
    cos, sin = cos[:, None].to(keys.dtype), sin[:, None].to(keys.dtype)
    rotary_dim = cos.shape[-1]
    rot, rest = keys[..., :rotary_dim], keys[..., rotary_dim:]
    return torch.cat((rot * cos + _rotate_half(rot) * sin, rest), dim=-1)


def layer_kv_from_residual(
    model, layer: int, hidden: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Post-RoPE keys and values ``[1, H, T, D]`` that the target itself
    computes from a layer input ``hidden`` ``[T, d]``: its own input norm,
    projections, QK-norm and rotary. Differentiable in ``hidden``."""
    block = decoder(model).layers[layer]
    attn = block.self_attn
    num_kv_heads = text_config(model).num_key_value_heads
    num_tokens = hidden.shape[0]
    h = block.input_layernorm(hidden.to(attn.k_proj.weight.dtype))
    k = attn.k_proj(h).view(num_tokens, num_kv_heads, -1)
    k_norm = getattr(attn, "k_norm", None)
    if k_norm is not None:
        k = k_norm(k)
    v = attn.v_proj(h).view(num_tokens, num_kv_heads, -1)
    k, v = k[None].transpose(1, 2), v[None].transpose(1, 2)
    return _apply_target_rope(model, layer, k), v


def capture_layer_inputs(
    model,
    prefix_ids: torch.Tensor,
    handoff_layer: int,
    predicted_hidden: torch.Tensor | None,
) -> list[torch.Tensor]:
    """Run the prefix natively, substituting ``predicted_hidden`` as the input
    of ``handoff_layer`` if given, and return every layer's input ``[T, d]``
    as computed in that run. Layers below the handoff see native inputs;
    layers from it on see what follows from the prediction."""
    num_layers = text_config(model).num_hidden_layers
    inputs: list[torch.Tensor | None] = [None] * num_layers
    handles = []
    if predicted_hidden is not None:

        def swap_input(module, args, kwargs, h=predicted_hidden):
            return (h[None].to(args[0].dtype),) + tuple(args[1:]), kwargs

        handles.append(
            decoder(model)
            .layers[handoff_layer]
            .register_forward_pre_hook(swap_input, with_kwargs=True)
        )

    def capture(index):
        def _hook(module, args):
            inputs[index] = args[0][0]

        return _hook

    for index, block in enumerate(decoder(model).layers):
        handles.append(block.register_forward_pre_hook(capture(index)))
    try:
        model(input_ids=prefix_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    assert all(h is not None for h in inputs)
    return inputs  # type: ignore[return-value]


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

    Every entry is produced by the target's own projections from a layer
    input, so sliding-window and per-layer rotary semantics are the
    model's own. Positions without a source counterpart (special and
    template tokens) keep native inputs in every layer. ``predicted_hidden``
    overrides the predictor at the handoff layer, which makes the exact
    residual an oracle for the mechanics. ``mode`` ``kv`` writes a
    predictor's direct key/value predictions at content positions instead.
    Gradients flow through the predicted entries when the caller enables
    them; the target's own prefill runs without.
    """
    num_layers = text_config(model).num_hidden_layers
    content = torch.as_tensor(ex.content)
    device = next(model.parameters()).device
    prefix_ids = torch.as_tensor(ex.tgt.ids)[None].to(device)
    if handoff_layer < num_layers and predicted_hidden is None:
        predicted_hidden = ex.tgt_states.residual[handoff_layer].clone()
        predicted_hidden[content] = predictor.predict_hidden(ex, handoff_layer).to(
            predicted_hidden.dtype
        )
    with torch.no_grad():
        inputs = capture_layer_inputs(
            model,
            prefix_ids,
            handoff_layer,
            predicted_hidden if handoff_layer < num_layers else None,
        )
    if mode not in ("resid", "kv"):
        raise ValueError(f"unknown mode {mode!r}")

    cache = DynamicCache(config=model.config)
    for layer in range(num_layers):
        hidden = inputs[layer]
        if layer < handoff_layer and mode == "resid":
            hidden = hidden.clone()
            hidden[content] = predictor.predict_hidden(ex, layer).to(hidden.dtype)
        k, v = layer_kv_from_residual(model, layer, hidden)
        if layer < handoff_layer and mode == "kv":
            predict_kv = getattr(predictor, "predict_kv", None)
            if predict_kv is None:
                raise ValueError("mode 'kv' needs a predictor with predict_kv")
            k_hat, v_hat = predict_kv(ex, layer)
            num_kv_heads = text_config(model).num_key_value_heads
            k_hat = k_hat.view(1, -1, num_kv_heads, k.shape[-1]).transpose(1, 2)
            v_hat = v_hat.view(1, -1, num_kv_heads, v.shape[-1]).transpose(1, 2)
            k, v = k.clone(), v.clone()
            k[:, :, content] = _apply_target_rope(
                model, layer, _scatter_positions(k_hat, content, k.shape)
            )[:, :, content].to(k.dtype)
            v[:, :, content] = v_hat.to(v.dtype)
        cache.update(k, v, layer)
    return cache


def _scatter_positions(
    values: torch.Tensor, content: torch.Tensor, shape: torch.Size
) -> torch.Tensor:
    """Place ``[1, H, C, D]`` rows at the content positions of a zero
    ``[1, H, T, D]`` tensor so position-dependent rotary applies correctly."""
    full = torch.zeros(shape, dtype=values.dtype, device=values.device)
    full[:, :, content] = values
    return full


def continuation_logits(
    model, ex: AlignedExample, cache: DynamicCache
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native and translated next-token logits over the continuation."""
    device = next(model.parameters()).device
    prefix = torch.as_tensor(ex.tgt.ids)[None].to(device)
    cont = ex.cont_ids[None].to(device)
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
    """``native_position_fraction`` is the share of prefix positions with
    no source counterpart (special and template tokens), which every
    layer keeps native; it is an oracle share, reported next to the
    layer-wise one."""

    boundary_agreement: float
    native_position_fraction: float
    r2_hidden: dict[int, float]
    r2_hidden_inexact: dict[int, float] = field(default_factory=dict)
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
    native_fraction = float(np.mean([(~ex.content).mean() for ex in evals]))
    report = StudyReport(
        agreement,
        native_fraction,
        predictor_hidden_r2(predictor, evals),
        predictor_hidden_r2(predictor, evals, exact_only=False),
    )
    if capacity_ranks:
        report.r2_bound_by_rank = capacity_bound(evals, capacity_ranks)
    if isinstance(predictor, PairMappers):
        r2 = heldout_r2(predictor, evals)
        report.src_layers = predictor.src_layers
        report.r2_keys, report.r2_values = r2["keys"], r2["values"]
    num_layers = text_config(tgt_model).num_hidden_layers
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
