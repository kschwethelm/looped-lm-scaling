"""
Tests for batched generative evaluation (generate_multi).
Core property: batched generation must produce identical results to sequential
generation at temperature=0.

python -m pytest tests/test_batched_generation.py -v
"""

from dataclasses import dataclass

import torch

from nanochat.engine import Engine, KVCache


# -----------------------------------------------------------------------------
# Test fixtures


@dataclass
class MockConfig:
    n_kv_head: int = 4
    n_head: int = 4
    n_embd: int = 64
    n_layer: int = 2
    sequence_len: int = 128
    n_prelude: int = 1
    n_recur_block: int = 1
    n_coda: int = 1
    num_recur: int = 1


class DeterministicMockModel:
    """
    Mock model where logits deterministically predict (last_input_token + 1) % vocab_size.
    Different prompts produce different outputs, so row mix-ups or position bugs
    will cause mismatches between batched and sequential generation.
    """

    def __init__(self, vocab_size: int = 262):
        self.vocab_size = vocab_size
        self.config = MockConfig()
        self._device = torch.device("cpu")

    def get_device(self):
        return self._device

    def forward(self, ids, kv_cache=None, num_recur=None, **kwargs):
        B, T = ids.shape
        if kv_cache is not None:
            kv_cache.advance(T)
        # logit[v] = 100 if v == (input + 1) % vocab_size, else -100
        logits = torch.full((B, T, self.vocab_size), -100.0)
        for b in range(B):
            for t in range(T):
                next_tok = (ids[b, t].item() + 1) % self.vocab_size
                logits[b, t, next_tok] = 100.0
        return logits


class ByteTokenizer:
    """Tokens 0-255 are raw bytes, 256+ are special tokens."""

    def __init__(self):
        self._special_tokens = {
            "<|assistant_end|>": 260,
            "<|bos|>": 261,
        }
        self._bos = 261

    def encode_special(self, s):
        return self._special_tokens[s]

    def get_bos_token_id(self):
        return self._bos

    def encode(self, s, prepend=None):
        tokens = list(s.encode("utf-8"))
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, tokens):
        byte_tokens = [t for t in tokens if t < 256]
        return bytes(byte_tokens).decode("utf-8", errors="replace")


def _make_engine(vocab_size: int = 262) -> Engine:
    return Engine(DeterministicMockModel(vocab_size), ByteTokenizer())


# -----------------------------------------------------------------------------
# KVCache.prefill_row tests


def test_prefill_row_copies_data():
    """prefill_row copies batch=1 KV cache into the correct row of a batch cache."""
    num_heads, head_dim, num_layers = 4, 8, 2
    src = KVCache(batch_size=1, num_heads=num_heads, seq_len=32, head_dim=head_dim,
                  num_layers=num_layers, device="cpu", dtype=torch.float32)
    dst = KVCache(batch_size=3, num_heads=num_heads, seq_len=64, head_dim=head_dim,
                  num_layers=num_layers, device="cpu", dtype=torch.float32)

    # Write distinct data and advance source
    src.k_cache[:, 0, :10, :, :] = 1.0
    src.v_cache[:, 0, :10, :, :] = 2.0
    src.advance(10)

    dst.prefill_row(src, row_idx=1)

    # Row 1 should have the copied data
    assert (dst.k_cache[:, 1, :10, :, :] == 1.0).all()
    assert (dst.v_cache[:, 1, :10, :, :] == 2.0).all()
    assert dst.cache_seqlens[1].item() == 10
    # Other rows should be untouched
    assert (dst.k_cache[:, 0, :, :, :] == 0.0).all()
    assert (dst.k_cache[:, 2, :, :, :] == 0.0).all()
    assert dst.cache_seqlens[0].item() == 0
    assert dst.cache_seqlens[2].item() == 0


def test_prefill_row_different_lengths():
    """prefill_row handles rows with different sequence lengths."""
    num_heads, head_dim, num_layers = 4, 8, 2
    src = KVCache(batch_size=1, num_heads=num_heads, seq_len=64, head_dim=head_dim,
                  num_layers=num_layers, device="cpu", dtype=torch.float32)
    dst = KVCache(batch_size=2, num_heads=num_heads, seq_len=64, head_dim=head_dim,
                  num_layers=num_layers, device="cpu", dtype=torch.float32)

    # Prefill row 0 with length 10
    src.k_cache.fill_(1.0)
    src.advance(10)
    dst.prefill_row(src, row_idx=0)

    # Prefill row 1 with length 20
    src.reset()
    src.k_cache.fill_(3.0)
    src.advance(20)
    dst.prefill_row(src, row_idx=1)

    assert dst.cache_seqlens[0].item() == 10
    assert dst.cache_seqlens[1].item() == 20


# -----------------------------------------------------------------------------
# Batched vs sequential equivalence (the crucial test)


def test_batched_matches_sequential_temperature_zero():
    """
    Core correctness test: generate_multi with temperature=0 must produce
    identical results to calling generate_batch sequentially for each prompt.
    """
    engine = _make_engine()
    prompts = [
        [261, 10, 20, 30],          # 4 tokens
        [261, 100, 200],             # 3 tokens
        [261, 50],                   # 2 tokens
        [261, 10, 20, 30, 40, 50],  # 6 tokens
    ]
    max_tokens = 20
    gen_kwargs = dict(max_tokens=max_tokens, temperature=0.0, seed=42)

    # Sequential: one prompt at a time
    sequential_results = []
    for prompt in prompts:
        results, _ = engine.generate_batch(prompt, num_samples=1, **gen_kwargs)
        sequential_results.append(results[0])

    # Batched: all prompts at once
    batched_results, _ = engine.generate_multi(prompts, **gen_kwargs)

    for i, (seq, bat) in enumerate(zip(sequential_results, batched_results)):
        assert seq == bat, (
            f"Prompt {i}: sequential and batched outputs differ.\n"
            f"  Sequential: {seq}\n"
            f"  Batched:    {bat}"
        )


def test_batched_matches_sequential_various_lengths():
    """Test equivalence with a wider range of prompt lengths."""
    engine = _make_engine()
    prompts = [
        [261] + list(range(i, i + length))
        for i, length in [(0, 1), (10, 5), (50, 15), (80, 3), (120, 8)]
    ]
    gen_kwargs = dict(max_tokens=30, temperature=0.0, seed=42)

    sequential_results = []
    for prompt in prompts:
        results, _ = engine.generate_batch(prompt, num_samples=1, **gen_kwargs)
        sequential_results.append(results[0])

    batched_results, _ = engine.generate_multi(prompts, **gen_kwargs)

    for i, (seq, bat) in enumerate(zip(sequential_results, batched_results)):
        assert seq == bat, f"Mismatch at prompt {i}"


def test_batched_single_prompt_matches_sequential():
    """Batch of 1 degenerates correctly to sequential behavior."""
    engine = _make_engine()
    prompt = [261, 42, 100, 150]
    gen_kwargs = dict(max_tokens=15, temperature=0.0, seed=42)

    seq_results, seq_masks = engine.generate_batch(prompt, num_samples=1, **gen_kwargs)
    bat_results, bat_masks = engine.generate_multi([prompt], **gen_kwargs)

    assert seq_results[0] == bat_results[0]
    assert seq_masks[0] == bat_masks[0]


# -----------------------------------------------------------------------------
# Structural tests


def test_generate_multi_max_tokens():
    """Batched generation respects max_tokens."""
    engine = _make_engine()
    prompts = [[261, 10], [261, 20, 30]]

    for max_tokens in [1, 5, 20]:
        results, _ = engine.generate_multi(prompts, max_tokens=max_tokens, temperature=0.0)
        for i, result in enumerate(results):
            prompt_len = len(prompts[i])
            generated = len(result) - prompt_len
            assert generated <= max_tokens, (
                f"Prompt {i}: generated {generated} tokens, expected <= {max_tokens}"
            )


def test_generate_multi_result_count():
    """Returns exactly one result per prompt."""
    engine = _make_engine()
    for n in [1, 2, 4, 8]:
        prompts = [[261, i] for i in range(n)]
        results, masks = engine.generate_multi(prompts, max_tokens=5, temperature=0.0)
        assert len(results) == n
        assert len(masks) == n


def test_generate_multi_preserves_prompt_prefix():
    """Each result starts with its original prompt tokens."""
    engine = _make_engine()
    prompts = [[261, 10, 20], [261, 100, 200, 50]]
    results, _ = engine.generate_multi(prompts, max_tokens=10, temperature=0.0)

    for prompt, result in zip(prompts, results):
        assert result[:len(prompt)] == prompt, (
            f"Result doesn't start with prompt. Prompt: {prompt}, Result start: {result[:len(prompt)]}"
        )


# -----------------------------------------------------------------------------
# Early completion


def test_early_completion_one_prompt():
    """
    When one prompt finishes early (generates assistant_end), the other prompts
    continue generating. With DeterministicMockModel, token 259 → 260 (assistant_end).
    """
    engine = _make_engine()
    # Prompt A: ends with 259, so first generated token = 260 = assistant_end → immediate stop
    prompt_a = [261, 259]
    # Prompt B: ends with 10, generates 11, 12, 13, ... for many tokens
    prompt_b = [261, 10]

    results, _ = engine.generate_multi(
        [prompt_a, prompt_b], max_tokens=20, temperature=0.0,
    )

    # Prompt A should have no generated tokens (assistant_end is stripped)
    assert results[0] == prompt_a, f"Prompt A should stop immediately, got: {results[0]}"
    # Prompt B should have generated tokens
    assert len(results[1]) > len(prompt_b), "Prompt B should have generated tokens"


def test_early_completion_matches_sequential():
    """Early completion in batched mode produces same results as sequential."""
    engine = _make_engine()
    # Mix of prompts: one stops after 1 token (259→260=end), others keep going
    prompts = [
        [261, 259],      # stops immediately
        [261, 10, 20],   # keeps going
        [261, 50],        # keeps going
    ]
    gen_kwargs = dict(max_tokens=15, temperature=0.0, seed=42)

    sequential_results = []
    for prompt in prompts:
        results, _ = engine.generate_batch(prompt, num_samples=1, **gen_kwargs)
        sequential_results.append(results[0])

    batched_results, _ = engine.generate_multi(prompts, **gen_kwargs)

    for i, (seq, bat) in enumerate(zip(sequential_results, batched_results)):
        assert seq == bat, f"Mismatch at prompt {i} (early completion test)"
