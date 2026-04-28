"""
Unified Flash Attention interface with automatic FA3/FA2/SDPA switching.

Three-tier fallback system:
1. Flash Attention 3 (Hopper+, sm90+) - H100, H200
2. Flash Attention 2 (Ampere/Ada, sm80-89) - A100, A10, RTX 3090, RTX 4090
3. PyTorch SDPA (older GPUs, MPS, CPU)

Exports `flash_attn` module that matches the FA3 API exactly.

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load FA3 on Hopper+ GPUs, FA2 on Ampere/Ada
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper+ GPU, sm90+)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major < 9:  # Hopper is sm90
            return None
        import os

        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel

        return get_kernel("varunneal/flash-attention-3").flash_attn_interface
    except Exception:
        return None


def _load_flash_attention_2():
    """Try to load Flash Attention 2 (requires Ampere/Ada GPU, sm80-89)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major < 8 or major >= 9:  # Ampere/Ada is sm80-89
            return None
        import flash_attn

        return flash_attn
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
_fa2 = _load_flash_attention_2()
HAS_FA3 = _fa3 is not None
HAS_FA2 = _fa2 is not None

# Print backend selection once at module initialization
if HAS_FA3:
    print("Flash Attention backend: FA3")
elif HAS_FA2:
    print("Flash Attention backend: FA2")
else:
    print("Flash Attention backend: SDPA (PyTorch fallback)")

# Override for testing: set to 'fa3', 'fa2', 'sdpa', or None (auto)
_override_impl = None


def _get_backend():
    """Determine which backend to use based on availability and override."""
    if _override_impl is not None:
        if _override_impl == "fa3":
            assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        elif _override_impl == "fa2":
            assert HAS_FA2, "Cannot override to FA2: not available on this hardware"
        return _override_impl

    # Auto selection
    if HAS_FA3:
        return "fa3"
    if HAS_FA2:
        return "fa2"
    return "sdpa"


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask
    device = q.device
    if Tq == Tk:
        # Causal + sliding window
        mask = torch.tril(torch.ones(Tq, Tk, device=device, dtype=torch.bool))
        if window > 0 and window < Tq:
            row_idx = torch.arange(Tq, device=device).unsqueeze(1)
            col_idx = torch.arange(Tk, device=device).unsqueeze(0)
            mask = mask & ((row_idx - col_idx) <= window)
    else:
        # Chunk inference: attend to prefix + causal within chunk
        prefix_len = Tk - Tq
        mask = torch.zeros(Tq, Tk, device=device, dtype=torch.bool)
        mask[:, :prefix_len] = True
        mask[:, prefix_len:] = torch.tril(torch.ones(Tq, Tq, device=device, dtype=torch.bool))

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    backend = _get_backend()

    if backend == "fa3":
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    if backend == "fa2":
        return _fa2.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None, causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3/FA2 update k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    backend = _get_backend()

    if backend == "fa3":
        return _fa3.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            causal=causal,
            window_size=window_size,
        )

    if backend == "fa2":
        return _fa2.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            causal=causal,
            window_size=window_size,
        )

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    uniform_seqlens = torch.all(cache_seqlens == cache_seqlens[0]).item()

    if uniform_seqlens:
        # Fast path: all batch elements at same position
        pos = cache_seqlens[0].item()
        if k is not None and v is not None:
            k_cache[:, pos : pos + T_new, :, :] = k
            v_cache[:, pos : pos + T_new, :, :] = v
        end_pos = pos + T_new
        k_full = k_cache[:, :end_pos, :, :]
        v_full = v_cache[:, :end_pos, :, :]
        q_sdpa = q.transpose(1, 2)
        k_sdpa = k_full.transpose(1, 2)
        v_sdpa = v_full.transpose(1, 2)
        enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
        y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)
        return y_sdpa.transpose(1, 2)

    # Non-uniform seqlens: process each batch element separately
    outputs = []
    for i in range(B):
        pos_i = cache_seqlens[i].item()
        if k is not None and v is not None:
            k_cache[i, pos_i : pos_i + T_new, :, :] = k[i]
            v_cache[i, pos_i : pos_i + T_new, :, :] = v[i]
        end_pos_i = pos_i + T_new
        q_i = q[i : i + 1].transpose(1, 2)
        k_i = k_cache[i : i + 1, :end_pos_i].transpose(1, 2)
        v_i = v_cache[i : i + 1, :end_pos_i].transpose(1, 2)
        enable_gqa = q_i.size(1) != k_i.size(1)
        y_i = _sdpa_attention(q_i, k_i, v_i, window_size, enable_gqa)
        outputs.append(y_i.transpose(1, 2))
    return torch.cat(outputs, dim=0)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
