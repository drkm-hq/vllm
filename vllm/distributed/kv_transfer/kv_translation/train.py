# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Train a ``HubTranslator`` between two frozen models.

The bill is that of a speculative-decoding draft head: both models stay
frozen, every step captures their residual streams on a batch of text and
regresses the target's per-layer residual from the source's. An optional
distillation term pushes the predicted residual through the target's own
projections into a cache and matches the target's next-token distribution
on a continuation, which is the quantity the serving path cares about.
Runs are seeded and deterministic for a fixed example order.
"""

import argparse
import itertools
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vllm.distributed.kv_transfer.kv_translation.capture import text_config
from vllm.distributed.kv_transfer.kv_translation.data import (
    AlignedExample,
    prepare_example,
)
from vllm.distributed.kv_transfer.kv_translation.study import (
    continuation_logits,
    divergence,
    predictor_hidden_r2,
    translated_cache,
)
from vllm.distributed.kv_transfer.kv_translation.translator import (
    HubConfig,
    HubTranslator,
    TranslatorPredictor,
)


@dataclass
class TrainConfig:
    steps: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 0
    kl_weight: float = 0.0
    kl_every: int = 1
    handoff_layer: int | None = None
    log_every: int = 50
    train_hub: bool = True


def residual_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-layer MSE normalized by the layer's power, plus cosine distance,
    averaged over layers. Layers with large residual norms do not dominate."""
    pred, target = pred.float(), target.float()
    power = target.pow(2).mean((0, 2)) + 1e-6
    mse = (pred - target).pow(2).mean((0, 2)) / power
    cos = 1 - F.cosine_similarity(pred, target, dim=-1).mean(0)
    return (mse + cos).mean()


def freeze(model) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def calibrate(
    translator: HubTranslator,
    target: str,
    examples: Sequence[AlignedExample],
    hub_frozen: bool = False,
) -> None:
    """Set output statistics for ``target`` and, unless the hub is frozen,
    the shared input scales. A frozen hub is never touched by onboarding."""
    if not hub_frozen:
        layers = translator.config.src_layers
        translator.calibrate_source(
            torch.cat([ex.source_features(layers) for ex in examples])
        )
    translator.heads[target].calibrate(
        torch.cat([ex.target_residuals() for ex in examples])
    )


def train_translator(
    translator: HubTranslator,
    target: str,
    examples: Iterable[AlignedExample],
    cfg: TrainConfig,
    tgt_model=None,
    eval_examples: Sequence[AlignedExample] | None = None,
) -> list[dict[str, float]]:
    """Optimize ``translator`` on a stream of aligned examples.

    ``tgt_model`` is needed only for the distillation term; it is frozen.
    With ``cfg.train_hub`` off only the target's own heads are optimized,
    so onboarding a target leaves every other target's translation
    bit-identical. Returns a log of losses and, when ``eval_examples`` is
    given, held-out residual R2 at every ``log_every`` steps.
    """
    torch.manual_seed(cfg.seed)
    if tgt_model is not None:
        freeze(tgt_model)
        tgt_model.eval()
    stream: Iterator[AlignedExample] = iter(examples)
    params = (
        list(translator.parameters())
        if cfg.train_hub
        else list(translator.heads[target].parameters())
    )
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    log: list[dict[str, float]] = []
    for step in range(cfg.steps):
        ex = next(stream)
        translator.train()
        pred = translator(ex.source_features(translator.config.src_layers), target)
        loss = residual_loss(pred, ex.target_residuals())
        entry = {"step": float(step), "residual_loss": loss.item()}
        if cfg.kl_weight > 0 and step % cfg.kl_every == 0:
            if tgt_model is None:
                raise ValueError("kl_weight > 0 needs tgt_model")
            predictor = TranslatorPredictor(translator, target)
            predictor.preset(ex, pred)
            handoff = cfg.handoff_layer
            if handoff is None:
                handoff = tgt_model.config.num_hidden_layers
            cache = translated_cache(tgt_model, predictor, ex, handoff)
            kl, agree = divergence(*continuation_logits(tgt_model, ex, cache))
            loss = loss + cfg.kl_weight * kl
            entry["kl"] = kl.item()
            entry["top1_agreement"] = agree.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        entry["loss"] = loss.item()
        if eval_examples is not None and (step + 1) % cfg.log_every == 0:
            translator.eval()
            r2 = predictor_hidden_r2(
                TranslatorPredictor(translator, target), eval_examples
            )
            entry["eval_r2_mean"] = sum(r2.values()) / len(r2)
            entry["eval_r2_min"] = min(r2.values())
        log.append(entry)
    translator.eval()
    return log


def default_source_layers(src_model, count: int = 6) -> tuple[int, ...]:
    """Evenly spaced layer inputs, excluding the post-norm final state."""
    num_layers = text_config(src_model).num_hidden_layers
    picks = torch.linspace(1, num_layers - 1, count).round().int().tolist()
    return tuple(sorted(set(picks)))


def example_stream(
    src_model, src_tokenizer, tgt_model, tgt_tokenizer, texts: Sequence[str]
) -> Iterator[AlignedExample]:
    """Re-capture frozen-model states per step so nothing is stored."""
    for text in itertools.cycle(texts):
        ex = prepare_example(src_model, src_tokenizer, tgt_model, tgt_tokenizer, text)
        if ex is not None:
            yield ex


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--tgt", required=True)
    parser.add_argument("--texts", required=True, help="file with one text per line")
    parser.add_argument("--out", required=True, help="where to save the translator")
    parser.add_argument("--src-layers", type=int, nargs="+", default=None)
    parser.add_argument("--latent-dim", type=int, default=2048)
    parser.add_argument("--head-rank", type=int, default=512)
    parser.add_argument("--context-layers", type=int, default=1)
    parser.add_argument("--context-heads", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--kl-weight", type=float, default=0.0)
    parser.add_argument("--kl-every", type=int, default=4)
    parser.add_argument("--eval-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
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
    freeze(src_model)
    src_tok = AutoTokenizer.from_pretrained(args.src)
    tgt_tok = AutoTokenizer.from_pretrained(args.tgt)
    with open(args.texts) as f:
        texts = [line.rstrip("\n") for line in f if line.strip()]
    n_eval = max(1, int(len(texts) * args.eval_frac))
    eval_texts, train_texts = texts[:n_eval], texts[n_eval:]

    src_layers = tuple(args.src_layers or default_source_layers(src_model))
    tgt_layers, tgt_dim = (
        text_config(tgt_model).num_hidden_layers,
        text_config(tgt_model).hidden_size,
    )
    torch.manual_seed(args.seed)
    config = HubConfig(
        src_layers=src_layers,
        src_dim=text_config(src_model).hidden_size,
        latent_dim=args.latent_dim,
        targets={args.tgt: (tgt_layers, tgt_dim)},
        head_rank=args.head_rank,
        context_layers=args.context_layers,
        context_heads=args.context_heads,
    )
    translator = HubTranslator(config).to(args.device)
    print(f"translator params: {translator.num_parameters(args.tgt) / 1e6:.1f}M")

    models = (src_model, src_tok, tgt_model, tgt_tok)
    evals = [
        ex for ex in (prepare_example(*models, t) for t in eval_texts) if ex is not None
    ]
    calibration = [
        ex
        for ex in (prepare_example(*models, t) for t in train_texts[: len(evals)])
        if ex is not None
    ]
    calibrate(translator, args.tgt, calibration)
    cfg = TrainConfig(
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        kl_weight=args.kl_weight,
        kl_every=args.kl_every,
        log_every=100,
    )
    log = train_translator(
        translator,
        args.tgt,
        example_stream(*models, train_texts),
        cfg,
        tgt_model,
        evals,
    )
    for entry in log:
        if "eval_r2_mean" in entry:
            print(entry)
    translator.save(args.out)


if __name__ == "__main__":
    main()
