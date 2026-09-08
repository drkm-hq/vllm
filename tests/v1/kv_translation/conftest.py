# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiny tokenizers and models trained/initialized locally: no downloads."""

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)

WORDS = " ".join(
    [
        "the kv cache in vllm is paged into blocks of sixteen tokens and prefix",
        "caching hashes each block by its token ids so a model switch breaks the",
        "cache because keys and values are model specific translation maps one",
        "residual stream into another across tokenizers positions and families",
        "with ridge probes rope stripping and byte span alignment 16 32 128 2048",
    ]
).split()


def synthetic_corpus(num_lines: int = 400) -> list[str]:
    lines = []
    for i in range(num_lines):
        start = (i * 7) % len(WORDS)
        chunk = WORDS[start : start + 12] + WORDS[: max(0, start + 12 - len(WORDS))]
        text = " ".join(chunk)
        if i % 3 == 0:
            text = text.capitalize() + "."
        if i % 5 == 0:
            text += " (Émigré café, naïve façade)"
        lines.append(text)
    return lines


def train_bytelevel_bpe(corpus: list[str], vocab_size: int = 600):
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=["<bos>", "<eos>"]
    )
    tok.train_from_iterator(corpus, trainer)
    tok.post_processor = None
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<bos>", eos_token="<eos>"
    )


def train_metaspace_unigram(corpus: list[str], vocab_size: int = 400):
    tok = Tokenizer(models.Unigram())
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size, special_tokens=["<s>", "</s>"], unk_token="<unk>"
    )
    tok.train_from_iterator(corpus, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<s>", eos_token="</s>"
    )


@pytest.fixture(scope="session")
def corpus() -> list[str]:
    return synthetic_corpus()


@pytest.fixture(scope="session")
def bpe_tokenizer(corpus):
    return train_bytelevel_bpe(corpus)


@pytest.fixture(scope="session")
def unigram_tokenizer(corpus):
    return train_metaspace_unigram(corpus)


@pytest.fixture(scope="session")
def tiny_llama(bpe_tokenizer):
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(bpe_tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        rope_theta=500_000.0,
        max_position_embeddings=256,
    )
    return LlamaForCausalLM(config).eval()


@pytest.fixture(scope="session")
def tiny_qwen3(unigram_tokenizer):
    torch.manual_seed(1)
    config = Qwen3Config(
        vocab_size=len(unigram_tokenizer),
        hidden_size=96,
        intermediate_size=192,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=24,
        rope_theta=1_000_000.0,
        max_position_embeddings=256,
    )
    return Qwen3ForCausalLM(config).eval()
