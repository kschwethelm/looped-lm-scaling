"""
Test Flash Attention unified interface - verify FA3/FA2/SDPA produce identical results.

Run: python -m pytest tests/test_attention_fallback.py -v -s

Note on test structure:
    1. TestFlashAttnVsSDPA: Parameterized tests that compare Flash Attention (FA2/FA3)
       against SDPA. Tests are parameterized to run with both FA2 and FA3 (when available).
       FA3 requires Hopper GPU (sm90+), FA2 requires Ampere/Ada GPU (sm80-89).

    2. TestSDPAOnly: Tests that only exercise the SDPA fallback path. These can run
       on any device (CUDA, CPU, MPS) with the appropriate dtype for that device.

    3. TestOverrideMechanism: Tests that verify the implementation override system works.
"""

import pytest
import torch

import nanochat.flash_attention as fa_module
from nanochat.engine import KVCache
from nanochat.flash_attention import HAS_FA2, HAS_FA3, flash_attn


def set_impl(impl):
    """Set the implementation override ('fa3', 'fa2', 'sdpa', or None for auto)."""
    fa_module._override_impl = impl


def assert_close(t1, t2, name, atol=1e-2, rtol=1e-2):
    """Assert two tensors are close, with helpful error message."""
    max_diff = (t1 - t2).abs().max().item()
    mean_diff = (t1 - t2).abs().mean().item()
    assert torch.allclose(t1, t2, atol=atol, rtol=rtol), (
        f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
    )
    return max_diff, mean_diff


# =============================================================================
# Flash Attention vs SDPA comparison tests (parameterized for FA2 and FA3)
# =============================================================================
@pytest.mark.parametrize(
    "fa_impl",
    [
        pytest.param("fa3", marks=pytest.mark.skipif(not HAS_FA3, reason="FA3 required")),
        pytest.param("fa2", marks=pytest.mark.skipif(not HAS_FA2, reason="FA2 required")),
    ],
)
class TestFlashAttnVsSDPA:
    """Compare Flash Attention (FA2/FA3) and SDPA produce identical results."""

    DEVICE = "cuda"
    DTYPE = torch.bfloat16

    def run_both_impls(self, fn, fa_impl):
        """Run a function with both Flash Attention and SDPA, return both outputs."""
        set_impl(fa_impl)
        out_fa = fn()
        set_impl("sdpa")
        out_sdpa = fn()
        set_impl(None)  # reset
        return out_fa, out_sdpa

    def test_basic_causal(self, fa_impl):
        """Basic causal attention."""
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        y_fa, y_sdpa = self.run_both_impls(run, fa_impl)
        max_diff, mean_diff = assert_close(y_fa, y_sdpa, f"{fa_impl}_basic_causal")
        print(f"{fa_impl}_basic_causal: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_full_context(self, fa_impl):
        """Full context (window_size=-1)."""
        B, T, H, D = 2, 128, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))

        y_fa, y_sdpa = self.run_both_impls(run, fa_impl)
        max_diff, mean_diff = assert_close(y_fa, y_sdpa, f"{fa_impl}_full_context")
        print(f"{fa_impl}_full_context: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_sliding_window(self, fa_impl):
        """Sliding window attention."""
        B, T, H, D = 2, 128, 4, 32
        window = 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(window, 0))

        y_fa, y_sdpa = self.run_both_impls(run, fa_impl)
        max_diff, mean_diff = assert_close(y_fa, y_sdpa, f"{fa_impl}_sliding_window")
        print(f"{fa_impl}_sliding_window: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_gqa(self, fa_impl):
        """Group Query Attention (fewer KV heads than Q heads)."""
        B, T, D = 2, 64, 32
        n_heads = 8
        n_kv_heads = 2

        q = torch.randn(B, T, n_heads, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, n_kv_heads, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, n_kv_heads, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        y_fa, y_sdpa = self.run_both_impls(run, fa_impl)
        max_diff, mean_diff = assert_close(y_fa, y_sdpa, f"{fa_impl}_gqa")
        print(f"{fa_impl}_gqa: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_larger_model(self, fa_impl):
        """Larger dimensions closer to real model."""
        B, T, H, D = 4, 256, 12, 64
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))

        y_fa, y_sdpa = self.run_both_impls(run, fa_impl)
        max_diff, mean_diff = assert_close(y_fa, y_sdpa, f"{fa_impl}_larger_model")
        print(f"{fa_impl}_larger_model: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_kvcache_prefill(self, fa_impl):
        """Test prefill (inserting multiple tokens into empty cache)."""
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 16

        q = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            k_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            v_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            cache_seqlens = torch.zeros(B, dtype=torch.int32, device=self.DEVICE)
            return flash_attn.flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                k=k,
                v=v,
                cache_seqlens=cache_seqlens,
                causal=True,
                window_size=(T_max, 0),
            )

        y_fa, y_sdpa = self.run_both_impls(run, fa_impl)
        max_diff, mean_diff = assert_close(y_fa, y_sdpa, f"{fa_impl}_prefill")
        print(f"{fa_impl}_prefill: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_kvcache_single_token(self, fa_impl):
        """Test single token generation (cache already has content)."""
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 16

        k_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            k_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            v_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            k_cache[:, :T_prefill, :, :] = k_init
            v_cache[:, :T_prefill, :, :] = v_init
            cache_seqlens = torch.full((B,), T_prefill, dtype=torch.int32, device=self.DEVICE)
            return flash_attn.flash_attn_with_kvcache(
                q_single,
                k_cache,
                v_cache,
                k=k_single,
                v=v_single,
                cache_seqlens=cache_seqlens,
                causal=True,
                window_size=(T_max, 0),
            )

        y_fa, y_sdpa = self.run_both_impls(run, fa_impl)
        max_diff, mean_diff = assert_close(y_fa, y_sdpa, f"{fa_impl}_single_token")
        print(f"{fa_impl}_single_token: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_backward_gradients_match(self, fa_impl):
        """Verify gradients are similar between Flash Attention and SDPA."""
        B, T, H, D = 2, 32, 4, 16

        q_data = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_data = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_data = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            q = q_data.clone().requires_grad_(True)
            k = k_data.clone().requires_grad_(True)
            v = v_data.clone().requires_grad_(True)
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))
            loss = y.sum()
            loss.backward()
            return y.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()

        set_impl(fa_impl)
        y_fa, q_grad_fa, k_grad_fa, v_grad_fa = run()
        set_impl("sdpa")
        y_sdpa, q_grad_sdpa, k_grad_sdpa, v_grad_sdpa = run()
        set_impl(None)

        max_diff, mean_diff = assert_close(y_fa, y_sdpa, f"{fa_impl}_backward_output")
        print(f"{fa_impl}_backward_output: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        max_diff, mean_diff = assert_close(
            q_grad_fa, q_grad_sdpa, f"{fa_impl}_q_grad", atol=0.05, rtol=0.05
        )
        print(f"{fa_impl}_q_grad: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        max_diff, mean_diff = assert_close(
            k_grad_fa, k_grad_sdpa, f"{fa_impl}_k_grad", atol=0.05, rtol=0.05
        )
        print(f"{fa_impl}_k_grad: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        max_diff, mean_diff = assert_close(
            v_grad_fa, v_grad_sdpa, f"{fa_impl}_v_grad", atol=0.05, rtol=0.05
        )
        print(f"{fa_impl}_v_grad: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")


# =============================================================================
# SDPA-only tests (run on any device)
# =============================================================================
class TestSDPAOnly:
    """Test SDPA fallback works correctly. Runs on any device."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def test_basic_forward(self):
        """Test SDPA forward pass produces valid output."""
        set_impl("sdpa")
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        assert y.shape == (B, T, H, D)
        assert not torch.isnan(y).any(), "Output contains NaN"
        set_impl(None)

    def test_backward(self):
        """Test gradients flow through SDPA."""
        set_impl("sdpa")
        B, T, H, D = 2, 32, 4, 16
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))
        loss = y.sum()
        loss.backward()

        assert q.grad is not None, "No gradient for q"
        assert k.grad is not None, "No gradient for k"
        assert v.grad is not None, "No gradient for v"
        assert not torch.isnan(q.grad).any(), "NaN in q gradient"
        set_impl(None)

    def test_kvcache(self):
        """Test SDPA with KV cache."""
        set_impl("sdpa")
        B, T_max, H, D = 2, 64, 4, 32
        n_layers = 1

        cache = KVCache(
            batch_size=B,
            num_heads=H,
            seq_len=T_max,
            head_dim=D,
            num_layers=n_layers,
            device=self.DEVICE,
            dtype=self.DTYPE,
        )
        k_cache, v_cache = cache.get_layer_cache(0)

        # Prefill
        T_prefill = 16
        q = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y = flash_attn.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache.cache_seqlens,
            causal=True,
            window_size=(T_max, 0),
        )
        cache.advance(T_prefill)

        assert y.shape == (B, T_prefill, H, D)
        assert cache.get_pos() == T_prefill

        # Generate single token
        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y_single = flash_attn.flash_attn_with_kvcache(
            q_single,
            k_cache,
            v_cache,
            k=k_single,
            v=v_single,
            cache_seqlens=cache.cache_seqlens,
            causal=True,
            window_size=(T_max, 0),
        )
        cache.advance(1)

        assert y_single.shape == (B, 1, H, D)
        assert cache.get_pos() == T_prefill + 1
        set_impl(None)


# =============================================================================
# Override mechanism tests
# =============================================================================
class TestOverrideMechanism:
    """Test that the override mechanism works correctly."""

    @pytest.mark.skipif(not HAS_FA3, reason="FA3 required")
    def test_override_fa3(self):
        """Test that override='fa3' uses FA3."""
        set_impl("fa3")
        assert fa_module._get_backend() == "fa3"
        set_impl(None)

    @pytest.mark.skipif(not HAS_FA2, reason="FA2 required")
    def test_override_fa2(self):
        """Test that override='fa2' uses FA2."""
        set_impl("fa2")
        assert fa_module._get_backend() == "fa2"
        set_impl(None)

    def test_override_sdpa(self):
        """Test that override='sdpa' uses SDPA."""
        set_impl("sdpa")
        assert fa_module._get_backend() == "sdpa"
        set_impl(None)

    def test_override_auto(self):
        """Test that override=None uses auto-detection."""
        set_impl(None)
        expected = "fa3" if HAS_FA3 else ("fa2" if HAS_FA2 else "sdpa")
        assert fa_module._get_backend() == expected


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        major, minor = torch.cuda.get_device_capability()
        print(f"Compute capability: {major}.{minor}")
    print(f"HAS_FA3: {HAS_FA3}")
    print(f"HAS_FA2: {HAS_FA2}")
    print(f"Active backend: {fa_module._get_backend()}")
    print()

    pytest.main([__file__, "-v", "-s"])
