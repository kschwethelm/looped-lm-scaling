"""Tests for MuonH (Muon-Hyperball) optimizer."""

import torch
import pytest
from nanochat.optim import MuonAdamW


def make_muon_group(shapes: list[tuple[int, int]], lr: float = 0.02) -> tuple[list[torch.Tensor], list[dict]]:
    """Create params with grads and a muon param group."""
    params = [torch.randn(s) for s in shapes]
    for p in params:
        p.grad = torch.randn_like(p)
    groups = [dict(kind="muon", params=params, lr=lr, momentum=0.95, ns_steps=5)]
    return params, groups


class TestMuonH:
    def test_frobenius_norm_preserved(self):
        """Frobenius sphere constraint: ||W||_F must stay constant after updates."""
        torch._dynamo.config.disable = True
        params, groups = make_muon_group([(64, 256)] * 4)
        norms_before = [p.norm().item() for p in params]

        opt = MuonAdamW(groups)
        for _ in range(10):
            for p in params:
                p.grad = torch.randn_like(p)
            opt.step()

        for i, p in enumerate(params):
            assert p.norm().item() == pytest.approx(norms_before[i], abs=1e-4), \
                f"Param {i}: norm drifted from {norms_before[i]:.6f} to {p.norm().item():.6f}"

    def test_tall_and_wide_matrices(self):
        """Both Polar Express branches (tall: m>n, wide: m<=n) preserve norms."""
        torch._dynamo.config.disable = True
        # Separate groups required — Muon stacks params, needs same shape per group
        tall_params, tall_groups = make_muon_group([(256, 64)] * 2)
        wide_params, wide_groups = make_muon_group([(64, 256)] * 2)
        all_params = tall_params + wide_params
        norms_before = [p.norm().item() for p in all_params]

        opt = MuonAdamW(tall_groups + wide_groups)
        for _ in range(10):
            for p in all_params:
                p.grad = torch.randn_like(p)
            opt.step()

        for i, p in enumerate(all_params):
            assert p.norm().item() == pytest.approx(norms_before[i], abs=1e-4), \
                f"Param {i} ({p.shape}): norm drifted"

    def test_params_actually_change(self):
        """Params must not be frozen — updates should modify the weight matrices."""
        torch._dynamo.config.disable = True
        params, groups = make_muon_group([(32, 128)] * 2)
        values_before = [p.clone() for p in params]

        opt = MuonAdamW(groups)
        for _ in range(3):
            for p in params:
                p.grad = torch.randn_like(p)
            opt.step()

        for i, p in enumerate(params):
            assert not torch.allclose(p, values_before[i]), \
                f"Param {i} unchanged after 3 steps — optimizer not updating"

    def test_mixed_adamw_muon(self):
        """AdamW and MuonH groups coexist: AdamW params move freely, MuonH norms stay fixed."""
        torch._dynamo.config.disable = True
        muon_params = [torch.randn(64, 256) for _ in range(3)]
        adamw_param = torch.randn(100)
        for p in muon_params:
            p.grad = torch.randn_like(p)
        adamw_param.grad = torch.randn_like(adamw_param)

        muon_norms_before = [p.norm().item() for p in muon_params]
        adamw_values_before = adamw_param.clone()

        groups = [
            dict(kind="adamw", params=[adamw_param], lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
            dict(kind="muon", params=muon_params, lr=0.02, momentum=0.95, ns_steps=5),
        ]
        opt = MuonAdamW(groups)

        for _ in range(5):
            for p in muon_params:
                p.grad = torch.randn_like(p)
            adamw_param.grad = torch.randn_like(adamw_param)
            opt.step()

        # MuonH norms preserved
        for i, p in enumerate(muon_params):
            assert p.norm().item() == pytest.approx(muon_norms_before[i], abs=1e-4)

        # AdamW param actually updated
        assert not torch.allclose(adamw_param, adamw_values_before)
