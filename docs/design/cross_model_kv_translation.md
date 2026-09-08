# Cross-Model KV Translation

Keep the prompt cache warm across a model switch by translating the
model that prefilled a conversation into the state of the model that
takes over, across completely different model families.

This document records the design, what "lossless" can and cannot mean
here, the attempts implemented on this branch, how they are measured,
the serving path in vLLM, and the experiments that decide whether the
approach ships. It was produced with a multi-agent design and adversarial
review pass; section 11 lists the claims that were attacked and how they
were resolved. Code lives in
`vllm/distributed/kv_transfer/kv_translation/` with tests under
`tests/v1/kv_translation/`.

[TOC]

## 1. Problem and what "lossless" means

Model A prefills a conversation; the router switches to model B. B must
continue as if it had read the conversation itself. Today it re-prefills
the whole prefix through its own stack, the dominant cost of a switch,
growing with every turn.

The requirement is a generalizable solution: N models in a pool with O(N)
learned artifacts, spanning tokenizers, RoPE parameterizations, layer
counts, KV head layouts, QK-norm placement, sliding windows, MLA latents
and Mamba state.

### 1.1 What is known

Bit-exact reconstruction of B's cache, cheaper than B's own prefill, is
not known to be possible for an unrelated cross-family pair, and this
design does not attempt it. It is not an impossibility theorem: the
tokens determine the cache, but determinism fixes the output, never the
evaluation cost. The systems argument is stronger than the complexity
one: any exact shortcut has to beat dense matmul, not attention, so even
a free-attention oracle caps an exact speedup near 1.15x.

Three classes are carved out because they are exact today:

- **Memoization.** vLLM already returns bit-exact cache at zero cost when
  the same model saw the same prefix. This is the baseline to beat in a
  multi-turn workload.
- **Co-designed pools.** Frozen-bottom fine-tunes and shared-prefill
  training give exact reuse by construction. A third-party pool cannot be
  retrained, which is the only reason this design exists.
- **Exact reparameterizations.** Weight folding, fused versus split QKV,
  head permutation, TP resharding, LoRA deltas. These are basis changes
  with a closed-form map. The study probes for this class before fitting
  anything.

### 1.2 The rungs actually delivered

| Rung | Statement | Cost |
| --- | --- | --- |
| L1 mechanical exactness at the handoff | Given the target's true layer-L input at every prefix position, and no attention layer at or above L sharing KV with a layer below L, the prefix cache for layers L and above and the logits at prefix positions equal what the target computes itself, under pinned numerics. Generated tokens from the second one on attend below L into translated state and are not exact. | free |
| L2 structural exactness on a subset | Layer-0 K/V is a context-free function of the token, so it is exact wherever token boundaries coincide. Template scaffolding and the trailing native window are computed natively. Reported as a fraction. | free |
| L3 format compatibility | The cache is written through the target's own projections at the target's own slots. Paging, chunked prefill and MLA latent layout work unmodified; prefix caching, sliding-window trimming and FP8 KV each need a change (section 5). | free |
| L4 output-distribution losslessness (Mode V) | The served token sequence is distributed exactly as under native prefill by treating the translated cache as a draft state and verifying against a genuine native prefill. Buys latency, not compute: full prefill FLOPs are still paid, off the critical path. | ~1.4x native FLOPs, ~2x transient prefix KV |

**Mode F**, translate plus native top layers with no verification, is not
lossless. It is governed by a measured, gated population contract
(section 6.4) and saves roughly 2.7x prefill FLOPs. Mode F and Mode V
trade against each other; no design collapses them.

No rung repairs **blind-spot inheritance**: a translator re-expresses
only what the source encoded. A small-to-large handoff gives the large
model the small model's reading of the prompt. Escalation traffic is
therefore served by Mode V only (section 4.5).

## 2. The representation

### 2.1 Residual stream, not K/V

Every published cross-model transfer method maps K/V tensors and inherits
head count, head dim, RoPE convention, QK-norm placement and cache layout
as hard constraints, which is why none has evaluated an MLA or Mamba
target.

K, V, MLA latents and SSM inputs are all deterministic projections of the
residual stream. So the translator predicts the target's **residual
stream** and the target runs its own `k_proj`, `v_proj`, QK-norm, RoPE at
its own positions, `kv_a_proj` plus `kv_a_layernorm`, or conv1d plus
gated-delta scan. Every architectural discontinuity disappears inside the
model file. The translator only ever sees `[T, H]` float tensors, and RoPE
is never predicted.

Architectures where the premise fails are refused rather than degraded:
non-rotational position encodings (mrope, xdrope), non-residual
inter-layer state (Gemma 3n AltUp), non-residual cross-layer inputs
(Zamba2), encoder-decoder models. MoE routers are the most consequential
soft case: routing is a top-k argmax over the residual, so small residual
error flips expert selection discontinuously and no continuous metric
sees it. Expert-agreement rate is a first-class metric with its own kill
criterion.

### 2.2 The hub

The pool-level interlingua is the **anchor model's raw tapped
residuals** at a fixed, versioned set of layers. Each spoke (each target,
and each non-anchor source) owns a **private trunk** from those taps to
a latent, plus per-target-layer heads. There is no shared learned trunk
that every spoke depends on: a shared trunk fixes capacity before the
pool is known and invalidates every spoke when retrained.

In code, `HubTranslator` expresses both configurations. One translator
per (anchor, target) with a single target is the private-trunk form. A
translator with several targets and `train_hub=False` on every target
after the first is the shared-codec form, which the design permits only
as a versioned transport codec certified per spoke. A test proves that
onboarding a target with the hub frozen leaves every other target's
translation bit-identical.

Hub identity is a deployment invariant: the tapped residual depends on
weights, tap set, capture dtype, TP degree and any LoRA, so
`hub_id = sha256(weights ‖ taps ‖ trunk_or_codec ‖ dtype ‖ tp ‖ lora)`
is published by the anchor engine and checked fail-closed by the
consumer. Anchor replacement is an O(N) retraining event; O(N) onboarding
holds within an anchor epoch.

### 2.3 Handoff at one layer

Predicting the target's state at every layer produces a cache on no
trajectory the target could have produced. The translator predicts well
at **one** layer L and the target computes layers L and above itself.
Layers below L are translated too, but their state is read only by newly
generated tokens attending below L, the most local part of the
computation. Low-layer error is localized, not eliminated, which is why
L1's horizon is one token.

L is constrained: with cross-layer KV sharing (Gemma 3n, Gemma 4) every
sharing layer at or above L must share only with layers at or above L.
The maximum legal L is computed at model load; pairs with no legal L are
ineligible.

### 2.4 MLA and hybrid SSM

MLA (DeepSeek V3/R1) is the easiest leg: from a predicted residual the
fill is one small projection per layer. DeepSeek V3.2 adds a second
per-layer cache for its sparse-attention indexer that must be filled too,
and is gated out until it is. Hybrids are the honest weak point: the
target runs its own scan over the translated residual sequence, which is
cheap in FLOPs but sequential, and error injected into a recurrence
accumulates. Predicting block-boundary SSM states is a research branch,
not load-bearing. Sliding-window layers must be translated for every
position, because whether a position is out of window depends on the
reader's length, not the writer's.

## 3. Alignment across tokenizers and templates

All indexing is by **character end offset** into the rendered text,
never by token index. A causal model's state at token i summarizes the
text up to that token's end offset, so two tokenizations are comparable
exactly where end offsets coincide.

`alignment.py` builds token spans with offsets, classifies each target
token as exact, before, or after, reports boundary agreement, treats
zero-width spans as non-content and, for byte-level tokenizers that split
one character into several tokens, keeps only the last token ending at
each offset. `chat.py` locates each message's content in the rendered
template, labels tokens as content or template (tokens straddling a
boundary count as template), and aligns content per message on
message-relative offsets; template tokens are never translated.

The study's position policy for non-exact target tokens is the nearest
source token that has seen at least the target token's text. That token
has seen up to one token's worth of text beyond the target position, a
within-token leak that flatters offline numbers slightly. Serving fills
those positions from the causal context block instead (section 5.3).

**Eligibility, fail-closed.** A request falls back to native prefill when
the tokenizer cannot emit offsets (the Mistral tokenizer reports
`is_fast` but takes no offsets argument, so `is_fast` is not a valid
probe), when the request is multimodal, or when boundary agreement is
below 0.80. Agreement between 0.60 and 0.80 routes to the contextual
translator. Multimodal and mrope exclusions remove the vision-language
models from the pool; certification must publish what fraction of real
switching traffic survives eligibility.

**Keys and provenance.** Today's block hash carries no model identity at
all, a live collision hazard for any multi-model pool. Translated blocks
need per-block provenance (`hub_id`, translator hash, ordered source
chain, L, native-tail width, alignment digest, gate verdict) emitted for
every block by `generate_block_hash_extra_keys`, not a request-level
`cache_salt` that is applied only at block 0. After Mode V verification
the request's verified blocks are re-keyed to the unsalted native hash,
which is what makes multi-turn amortization work. The cross-model lookup
key is a per-turn chain of content hashes over the canonicalized
conversation, template-free and tokenizer-free.

## 4. Translator, training, and the attempts

### 4.1 Attempts

All attempts share `data.py` (alignment and capture) and `study.py`
(translated cache, handoff sweep, metrics), so their numbers compare
directly.

| Attempt | What is learned | Training | Deterministic |
| --- | --- | --- | --- |
| A0 ridge | closed-form affine map per target layer from the top-k source layers (`mapper.py`) | none, a solve | yes |
| A1 hub | calibrated static input scales, shared latent, per-target-layer low-rank heads (`translator.py`, `context_layers=0`) | residual regression | yes |
| A2 hub with context | A1 plus one causal attention block over the latent sequence, the size class of a speculative-decoding draft head | residual regression, optional KL distillation through the target | yes |
| A3 handoff | any of the above below layer L, native recompute from L (`study.py`) | none extra | yes |

Per-token normalization of the source residual was tried and rejected:
it discards the position's magnitude, which the target residual depends
on, and capped self-translation R² at 0.85. Calibrated static per-layer
scales keep it.

### 4.2 Shape and the FLOPs trap

Per spoke: a private trunk from k = 6 to 8 raw taps to a 2048-dim
latent, then per bottom layer a rank-512 head, about 4 MFLOP per token
per layer. This factorization is load-bearing: a dense map from
concatenated anchor residuals to a target layer costs 100 to 220 MFLOP
per token per layer, 20 to 45% of the layer it replaces, and silently
destroys the economics. The CLI defaults (`--latent-dim 2048`,
`--head-rank 512`) follow this.

Capacity is bounded and checked before training: every predicted
residual lies in a rank-512 affine subspace, so the R² ceiling per layer
is the fraction of residual variance in its top 512 principal directions.
`capacity_bound` in the study computes it in minutes from the covariance
spectrum. Head rank should be allocated from the measured R² profile,
not set uniformly.

### 4.3 Objective and stages

R² is a diagnostic, never an acceptance criterion. The implemented loss
is a per-layer power-normalized MSE plus cosine distance on the residual
(`train.py`), with an optional continuation-KL term that pushes the
prediction through the target's own projections into a cache and matches
the target's next-token distribution. The full design adds an
attention-sensitivity-weighted state loss, a per-channel range penalty
so translated K/V stay inside the target's static FP8 KV scales, and a
router-agreement term for MoE targets.

Stages: A0 probes for an exact reparameterization (R² near 1 with a
low-rank remainder means a closed-form map, not a learned adapter). A
fits ridge from raw taps, giving a warm start, the per-layer R² profile,
and a usable degraded spoke in minutes. B trains the residual losses on
roughly 2M aligned tokens. C trains the continuation-KL loss on 50 to
200M tokens; this backpropagates through the frozen target's stack, a
cost comparable to fine-tuning the target, which is the honest price of
the "draft head" comparison. Paired data is free: no labels, and anchor
features are spoke-independent.

Everything is seeded and deterministic for a fixed example order; the
test suite checks bitwise reproducibility of training on CPU. On GPU,
deterministic algorithms must be enabled explicitly, and the einsum and
attention kernels used by the translator are deterministic under them.

### 4.4 Onboarding model N+1

Two artifacts: a decoder (as target) and, if the model must also be a
source, an encoder into the hub's coordinates, plus a permanently
archived anchor teacher. Estimated 200 to 800 accelerator-hours for an
8B spoke; Stage A alone yields a Mode-V-only spoke in minutes.
Certification is not O(N): adding a model creates 2N ordered routes.

### 4.5 Direction policy

A translator re-expresses only what the source encoded. Mode F is
disallowed when the source is less capable than the target. Escalation
is served by Mode V, where the translated cache is only a draft.

## 5. Serving path in vLLM

### 5.1 An ordinary prefill that is cheaper

vLLM carries one `num_computed_tokens` per request, shared by every
layer and KV cache group; per-layer accounting is inexpressible. So a
translated request runs an **ordinary prefill of all T tokens that
happens to be a cheaper forward**: chunked prefill, paging, prefix
caching and preemption keep working. Three corrections to the naive
version of this claim:

- The exact-fallback channel (`get_block_ids_with_load_errors`) does
  mutate the scalar, and it is single-KV-cache-group only, so Gemma 3,
  Qwen3-Next and Nemotron-H have no exact fallback today.
- Mode V needs a second cursor, `num_verified_tokens`, and a low-priority
  verify branch.
- Depth is not per-row: one model call serves the whole batch, so fill
  chunks must be co-scheduled only with other fills at the same L or run
  as their own step.

### 5.2 Extension points

- **Source capture.** The EAGLE-3 aux-hidden-state hook already emits
  the true residual at an arbitrary layer list, and
  `extract_hidden_states` parks it in a paged KV-cache group
  (`HiddenStateCacheSpec`). Needed: a proposer-free owner (today it
  asserts a speculative config), the `SupportsEagle3` mixin for Gemma 3,
  and a loud failure instead of the silent skip when a request exceeds
  the drafter's max length.
- **The fill.** `unified_kv_cache_update` and `unified_mla_kv_cache_update`
  write the paged cache with no attention kernel. A per-family `fill_kv`
  beside `forward` (dense GQA, MLA, GDN/Mamba) must replicate the
  KV-sharing guard. The fill must run **inside** the normal forward
  context with real per-group metadata: nulling attention metadata makes
  Mamba, GDN and the sparse-MLA indexer take their profile-run branch
  and write nothing, silently.
- **Transport.** A thin connector shaped like `OffloadingConnector`
  moves the hub payload (about 4 KB per token).
- **Hashing.** Relax the single-group early return in
  `resolve_kv_cache_block_sizes` so a pool-wide `prefix_match_unit` is
  honored.

### 5.3 Cache-safety requirements

- **Never publish a block with unwritten slots.** Freshly allocated KV
  memory is not zeroed and a masked slot holds the previous tenant's K/V
  verbatim; a stale large-norm key can suppress the whole context. Fill
  every slot below L, writing unaligned positions from the causal
  context block, or thread a write-coverage block mask into caching and
  zero-fill masked slots.
- **Never publish unverified translated blocks.** Blocks enter the
  shared prefix cache on every prefill chunk, before any gate evidence.
  Route translated spans through `delay_cache_blocks` and cache them only
  when the gate clears them or a chunk has been natively repaired.
- **Never mutate a published block.** Mode V verification runs as a
  shadow request with its own blocks, never in place.

Security follows: a translated cache is unauthenticated state accepted
from another model that enters a shared, content-addressed cache.
Provenance keying, deferred caching and Mode V's verifier are the
containment, and they are built early.

## 6. Guarantee mechanism

### 6.1 Mode V as a shadow request

Issue the native prefill as a second request id with the same prompt and
no translation provenance. When it completes, append the drafted tokens
to the shadow as ordinary scheduled tokens, score them in that forward,
run rejection sampling, truncate, retire the draft and continue on the
shadow. The verifier draws a **fresh** uniform and uses the stored full
draft distribution; reusing the draw-time uniform is provably biased
toward the draft. Under greedy decoding the check degenerates to argmax
equality and needs no stored distribution, so a greedy-only tier is
cheap. Drafts are bounded to 128 tokens; constrained decoding is
excluded until an FSM rewind exists; retraction rate is a published SLO.

### 6.2 The gate

Computed before any token is emitted, at roughly 1 to 2% of native
prefill: an exact recompute of the first tenth of the prefix (the only
non-proxy signal, free in Mode V), quantizer bin margin, alignment
coverage, a rank-64 Mahalanobis out-of-distribution sketch, local
self-consistency of the target's own layer map on translated inputs, the
echo test (teacher-forced NLL of the prompt's own tokens under the
translated cache), and router agreement for MoE targets. The gate never
uses the source model's next-token distribution as reference: the two
models disagree by construction, and agreement with the source cannot
detect a faithful rendering of a wrong understanding. Whether the gate
signals predict realized divergence is the load-bearing empirical unknown.

### 6.3 Mode F's contract

For served requests, including prefix-cache hits on translated blocks,
the probability that committed output diverges from native greedy output
within 128 tokens is at most epsilon at confidence 0.95, estimated by
natively prefilling a random 2% and bounded by Clopper-Pearson. Admission
rate and eligibility rate are published alongside epsilon.

### 6.4 Metrics

In order of importance: **acceptance rate** (the expected
`sum_v min(p, q)` of the translated distribution as a draft for native,
which prices both Mode V's retraction rate and Mode F's viability),
**greedy run length** to first divergence, **router agreement**,
per-position KL, and quantizer-exact fractions. The **mismatched-document
control** is mandatory: report quality with the cache translated from a
different document next to every number, and require the pairing effect
(correct minus mismatched) to be at least 90% of the presence effect
(correct minus zeroed). Task-level evaluation (needle-in-haystack,
rare-entity recall, multi-hop) is mandatory for output-affecting changes.

The study implements acceptance rate, greedy run length, KL, top-1 and
the mismatched control today.

## 7. Cost model

Llama-3.1-8B target, 8K prompt, per layer per token: QKV 50 + O 34 +
SwiGLU 352 + attention 67 = 503 MFLOP; 16.1 GFLOP per token native.

| Component | MFLOP/token | vs native |
| --- | --- | --- |
| 22 bottom-layer rank-512 heads | ~370 | 2.3% |
| Fill (target's own K/V projections, RoPE, scatter) | ~370 | 2.3% |
| Gate | ~275 | 1.7% |
| Native top 10 layers | 5030 | 31.2% |
| Mode F total | ~6050 | 37.6%, a 2.7x saving |
| Mode V with native top layers | | ~1.39x native |

Costs not in the table that must be measured: the attention-metadata
build the fill cannot skip, loss of compiled graphs under an eager fill,
the same-L co-scheduling constraint, and the hub payload (34 MB at 8K,
134 MB at 32K per switch). Mode V's real price is a transient KV
reservation plus full prefill FLOPs; it buys latency.

Baselines that must be beaten: **eager background native prefill on the
target** when a switch is predicted (trivially exact, native blocks
reusable by every later turn), SpecPrefill (up to 7.66x TTFT reduction,
training-free), and simply serving the smaller model.

## 8. Model pair and runbook

| | Primary pair | Dev pair |
| --- | --- | --- |
| Models (both directions) | `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen3-8B` | `meta-llama/Llama-3.2-1B-Instruct`, `Qwen/Qwen3-1.7B` |
| Tokenizer | 128K BPE vs 151K BPE | same |
| RoPE | theta 500k with Llama-3 scaling vs theta 1M | same |
| Layers / hidden | 32 / 4096 vs 36 / 4096 | 16 / 2048 vs 28 / 2048 |
| KV heads x head dim | 8 x 128 vs 8 x 128 | 8 x 64 vs 8 x 128 |
| QK-norm | no vs yes | no vs yes |

The primary pair is the easiest genuinely cross-family case: matched KV
geometry isolates the tokenizer, RoPE, depth and QK-norm differences.
The dev pair has mismatched head dims, which any K/V-space mapper cannot
handle and the residual approach does not notice; both fit on one 24 GB
GPU. Gemma-3-1B joins the dev pool once it has the residual tap.

```bash
# Closed-form baseline, capacity bound, and the handoff curve.
python -m vllm.distributed.kv_transfer.kv_translation.study \
    --src meta-llama/Llama-3.1-8B-Instruct --tgt Qwen/Qwen3-8B \
    --texts texts.txt --handoff-layers 0 9 18 27 36

# A1: token-local hub, residual loss only.
python -m vllm.distributed.kv_transfer.kv_translation.train \
    --src meta-llama/Llama-3.1-8B-Instruct --tgt Qwen/Qwen3-8B \
    --texts texts.txt --out hub_a1.pt --context-layers 0 --steps 4000

# A2: contextual hub with distillation through the target every 4th step.
python -m vllm.distributed.kv_transfer.kv_translation.train \
    --src meta-llama/Llama-3.1-8B-Instruct --tgt Qwen/Qwen3-8B \
    --texts texts.txt --out hub_a2.pt --context-layers 1 \
    --steps 4000 --kl-weight 0.5 --kl-every 4

# Score a trained attempt on the same metrics and handoff curve.
python -m vllm.distributed.kv_transfer.kv_translation.study \
    --src meta-llama/Llama-3.1-8B-Instruct --tgt Qwen/Qwen3-8B \
    --texts texts.txt --translator hub_a2.pt --handoff-layers 0 9 18 27 36
```

`texts.txt` holds one document per line; a few thousand lines of the
conversational and code data used to train speculators is enough for the
residual loss. Run each pair in both directions.

## 9. Experiment plan

- **E0, CPU, done.** Oracles on tiny random fixtures: handing off at
  layer 0 with the true embeddings reproduces native logits; a model
  translated into itself is near-lossless; template tokens keep native
  states; training is bitwise reproducible; onboarding with a frozen hub
  leaves existing targets untouched; the mismatched control is far worse
  than translation.
- **E0-vllm, prerequisite to E1.** Under vLLM with `VLLM_BATCH_INVARIANT`,
  supplying the true layer-L state reproduces native logits exactly for
  every legal L, and does not for an L above the KV-sharing bound.
  Decides whether L1 can be stated bitwise at all.
- **E0c, tokenizers only.** Boundary agreement for every ordered pair
  over prose, code, math, multilingual and rendered chat.
- **E1, one GPU, one day.** The study on the dev pool, all ordered
  pairs: per-layer R², capacity bound, handoff sweep, KL, top-1,
  acceptance rate, greedy run length, mismatched control. If the best L
  gives R² below 0.5 and KL never falls below 0.15, the program dies for
  one GPU-day.
- **E2, three days.** Rank-512 adapter with the full loss; ablate each
  term; fit the gate calibration curve and its AUROC.
- **E3, one week.** Scale to 8B, build the hub, measure one-hop versus
  two-hop and few-shot onboarding against a pairwise mapper.
- **E4, one week.** MLA and hybrid legs, SSM error growth by context
  length.
- **E5 and E6, four to five weeks.** vLLM integration, then Mode V as a
  shadow request verified by exact output equality across a thousand
  requests, plus the repeated-switch drift experiment and the
  eager-prefill baseline.

## 10. Kill criteria

- **K1** Boundary agreement below 0.80 on real traffic: pair ineligible.
- **K2** Held-out residual R² at the best handoff layer below 0.50, or a
  rank-512 capacity ceiling already below 0.50: kill the pair.
- **K3** Pairing effect below 90% of presence effect: nothing is
  transmitted, regardless of accuracy.
- **K4** Top-1 below 0.90 or mean KL above 0.15 nats under serving
  numerics including FP8 KV: Mode F unshippable.
- **K5** Native-recompute fraction above 0.60: under 1.7x saving, loses
  to SpecPrefill or to serving the smaller model.
- **K6** Two-hop KL above 1.5x one-hop: the hub does not compose.
- **K7** Hybrid SSM cost above 25% of native or error growth above 2x
  from 1K to 8K: drop the hybrid leg.
- **K8** Mode V retraction above 20%, or TTFT saving below 1.5x against
  eager background prefill.
- **K10** MoE router agreement below 0.95: MoE targets are Mode V only.
- **K-gate** Gate AUROC for predicting divergence below 0.70: no
  per-request admission; the product is Mode V plus the exact
  first-chunk audit.

## 11. Adversarial review log

Six load-bearing claims were each attacked twice; nine of twelve
arguments were accepted as refutations and the design revised.

| Claim | Verdict | Resolution |
| --- | --- | --- |
| Bit-exact cross-model KV cheaper than prefill is impossible | 1 of 2 refuted | Restated as "not known and not attempted"; determinism does not bound evaluation cost. |
| Exact layer-L input makes every subsequent logit bit-identical | 2 of 2 refuted | Horizon is one token; cross-layer KV sharing bounds L; numerics must be pinned. |
| Writing through the target's projections is format-lossless for six subsystems | 2 of 2 refuted | Prefix caching, sliding-window trimming and FP8 KV each need a change; cache-safety section added. |
| No change to computed-token accounting is needed | 2 of 2 refuted | One verification cursor, per-block provenance, deferred caching, same-L co-scheduling. |
| Anchor hub gives O(N) onboarding with nothing existing changing | 2 of 2 refuted | Hub is raw anchor taps with private trunks; in-tree training now freezes the hub on onboarding, with a regression test. |
| Mode V is exact at 1.08x native with in-place rewrite and reused uniforms | 2 of 2 refuted | Shadow request, fresh uniform, stored draft distributions, 1.39x with native top layers. |

Open questions carried forward: whether the gate calibration curve is
informative, how many hub taps, whether L must be per-request, whether
capability (not just KL) composes across two hops, length
generalization of the residual hub, drift across repeated switches, and
what fraction of real traffic survives eligibility.
