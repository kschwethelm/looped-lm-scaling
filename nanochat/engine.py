"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

import torch
import torch.nn.functional as F


class KVCache:
    """
    KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API.

    Key differences from FA2-style cache:
    - Tensors are (B, T, H, D) not (B, H, T, D)
    - FA3 updates the cache in-place during flash_attn_with_kvcache
    - Position tracked per batch element via cache_seqlens tensor
    """

    def __init__(
        self,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        num_layers,
        device,
        dtype,
        num_recur=None,
        kv_budget=1,
    ):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        self.num_recur = num_recur  # Track num_recur used during cache creation
        self.kv_budget = kv_budget  # Fixed KV-cache budget for recurrences (default=1)
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(
            num_layers,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self.v_cache = torch.zeros(
            num_layers,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        # Current sequence length per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)

    def reset(self):
        """Reset cache to empty state."""
        self.cache_seqlens.zero_()

    def get_pos(self):
        """Get current position (assumes all batch elements at same position)."""
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        """
        Copy cached KV from another cache into this one.
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)

    def prefill_row(self, other, row_idx: int):
        """
        Copy a batch=1 KV cache into a specific row of this batch KV cache.
        Used for multi-prompt batched generation where each prompt is prefilled separately.
        """
        assert other.batch_size == 1, "Source cache must be batch_size=1"
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        other_pos = other.get_pos()
        self.k_cache[:, row_idx, :other_pos, :, :] = other.k_cache[:, 0, :other_pos, :, :]
        self.v_cache[:, row_idx, :other_pos, :, :] = other.v_cache[:, 0, :other_pos, :, :]
        self.cache_seqlens[row_idx] = other_pos


# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)


# -----------------------------------------------------------------------------


class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []  # Current token sequence for this row
        self.completed = False  # Whether this row has completed generation


class Engine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer  # needed for tool use

    def _get_special_tokens(self) -> tuple[int, int]:
        """Return (assistant_end, bos) special token IDs used to detect completion."""
        return (
            self.tokenizer.encode_special("<|assistant_end|>"),
            self.tokenizer.get_bos_token_id(),
        )

    def _get_kv_config(self, num_recur: int | None, kv_budget: int) -> tuple[int, dict]:
        """Return (cache_num_recur, kv_model_kwargs) for KVCache construction."""
        m = self.model.config
        cache_num_recur = num_recur if num_recur is not None else m.num_recur
        effective_num_layers = m.n_prelude + (m.n_recur_block * kv_budget) + m.n_coda
        return cache_num_recur, {
            "num_heads": m.n_kv_head,
            "head_dim": m.n_embd // m.n_head,
            "num_layers": effective_num_layers,
        }

    def _process_decode_step(
        self,
        row_states: list[RowState],
        sampled_tokens: list[int],
        special_tokens: tuple[int, int],
    ) -> tuple[list[int], list[int]]:
        """Process one decode step: append sampled tokens per row and flag completion."""
        assistant_end, bos = special_tokens
        token_column = list(sampled_tokens)
        token_masks = [1] * len(row_states)
        for state, next_token in zip(row_states, sampled_tokens):
            state.current_tokens.append(next_token)
            if next_token in (assistant_end, bos):
                state.completed = True
        return token_column, token_masks

    @torch.inference_mode()
    def generate(
        self,
        tokens,
        num_samples=1,
        max_tokens=None,
        temperature=1.0,
        top_k=None,
        seed=42,
        num_recur=None,
        kv_budget: int | None = None,
    ):
        """
        Generate tokens with KV caching and optional batch sampling.

        Args:
            kv_budget: KV-cache budget for recurrences. At iteration i, reads/writes
                cache entry i mod kv_budget. Defaults to num_recur (cache all recurrences,
                allowing attention across all loop states). Set to 1 to cache only the
                final recurrence (memory-efficient, matches training behavior).
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        effective_num_recur = num_recur if num_recur is not None else self.model.config.num_recur
        if kv_budget is None:
            kv_budget = effective_num_recur
        assert 1 <= kv_budget <= effective_num_recur, f"kv_budget must be in [1, {effective_num_recur}], got {kv_budget}"
        device = self.model.get_device()
        # NOTE: setting the dtype here and in this way is an ugly hack.
        # Currently the repo assumes that cuda -> bfloat16 and everything else -> float32.
        # We need to know the dtype here to call __init__ on KVCache and pre-allocate its tensors.
        # As a quick hack, we're making generate() function inherit and know about this repo-wise assumption.
        # I think there has to be a bigger refactor to deal with device/dtype tracking across the codebase.
        # In particular, the KVCache should allocate its tensors lazily
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        special_tokens = self._get_special_tokens()
        cache_num_recur, kv_model_kwargs = self._get_kv_config(num_recur, kv_budget)

        # 1) Run a batch 1 prefill of the prompt tokens
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            device=device,
            dtype=dtype,
            num_recur=cache_num_recur,
            kv_budget=kv_budget,
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill, num_recur=num_recur)
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            device=device,
            dtype=dtype,
            num_recur=cache_num_recur,
            kv_budget=kv_budget,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill  # no need to keep this memory around

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()
            token_column, token_masks = self._process_decode_step(row_states, sampled_tokens, special_tokens)

            yield token_column, token_masks
            num_generated += 1

            # Prepare logits for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self.model.forward(
                ids,
                kv_cache=kv_cache_decode,
                num_recur=num_recur,
            )
            logits = logits[:, -1, :]  # (B, vocab_size)

    @torch.inference_mode()
    def generate_multi(
        self,
        token_sequences: list[list[int]],
        max_tokens: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int = 42,
        num_recur: int | None = None,
        kv_budget: int | None = None,
    ):
        """
        Generate one completion for each of multiple different prompts in parallel.
        Sequential prefill (one prompt at a time) then batched autoregressive decode.

        Returns (results, masks) where each is a list of token sequences.
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        batch_size = len(token_sequences)
        assert batch_size > 0
        assert all(isinstance(ts, list) and isinstance(ts[0], int) for ts in token_sequences)
        effective_num_recur = num_recur if num_recur is not None else self.model.config.num_recur
        if kv_budget is None:
            kv_budget = effective_num_recur
        assert 1 <= kv_budget <= effective_num_recur, f"kv_budget must be in [1, {effective_num_recur}], got {kv_budget}"
        device = self.model.get_device()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        special_tokens = self._get_special_tokens()
        assistant_end, bos = special_tokens
        cache_num_recur, kv_model_kwargs = self._get_kv_config(num_recur, kv_budget)

        # 1) Sequential prefill: process each prompt individually, copy KV into shared decode cache
        max_prompt_len = max(len(ts) for ts in token_sequences)
        kv_length_hint = (max_prompt_len + max_tokens) if max_tokens is not None else self.model.config.sequence_len

        kv_cache_decode = KVCache(
            batch_size=batch_size,
            seq_len=kv_length_hint,
            device=device,
            dtype=dtype,
            num_recur=cache_num_recur,
            kv_budget=kv_budget,
            **kv_model_kwargs,
        )

        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=max_prompt_len,
            device=device,
            dtype=dtype,
            num_recur=cache_num_recur,
            kv_budget=kv_budget,
            **kv_model_kwargs,
        )

        all_logits = []
        for i, tokens in enumerate(token_sequences):
            kv_cache_prefill.reset()
            ids = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = self.model.forward(ids, kv_cache=kv_cache_prefill, num_recur=num_recur)
            all_logits.append(logits[:, -1, :])
            kv_cache_decode.prefill_row(kv_cache_prefill, i)

        del kv_cache_prefill
        logits = torch.cat(all_logits, dim=0)

        # 2) Initialize per-row state and result collectors
        row_states = [RowState(ts.copy()) for ts in token_sequences]
        results = [ts.copy() for ts in token_sequences]
        masks = [[0] * len(ts) for ts in token_sequences]
        completed = [False] * batch_size

        # 3) Batched autoregressive decode
        num_generated = 0
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(state.completed for state in row_states):
                break

            next_ids = sample_next_token(logits, rng, temperature, top_k)
            sampled_tokens = next_ids[:, 0].tolist()
            token_column, token_masks = self._process_decode_step(row_states, sampled_tokens, special_tokens)

            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token in (assistant_end, bos):
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            num_generated += 1

            if all(completed):
                break

            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self.model.forward(
                ids,
                kv_cache=kv_cache_decode,
                num_recur=num_recur,
            )
            logits = logits[:, -1, :]

        return results, masks

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token in (assistant_end, bos):
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks
