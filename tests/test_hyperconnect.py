"""Sanity tests for the hyperconnect injection mode."""

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig


def make_cfg(**overrides) -> GPTConfig:
    defaults = dict(
        sequence_len=64, vocab_size=256, size=2,
        n_head=2, n_kv_head=2, n_embd=128,
        n_prelude=1, n_recur_block=1, n_coda=1,
        num_recur=3, bptt_k=3,
        input_injection="hyperconnect", num_lanes=2,
    )
    defaults.update(overrides)
    return GPTConfig(**defaults)


def build_model(cfg: GPTConfig) -> GPT:
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


class TestHyperconnect:
    def test_rejects_truncated_bptt(self):
        with pytest.raises(ValueError, match="full BPTT"):
            make_cfg(num_recur=4, bptt_k=2)

    def test_rejects_single_lane(self):
        with pytest.raises(ValueError, match="num_lanes"):
            make_cfg(num_lanes=1)

    def test_cyclic_init(self):
        """alpha_r = e_{r mod N}, M_r = I, beta_r = 1."""
        cfg = make_cfg(num_recur=4, bptt_k=4, num_lanes=2)
        model = build_model(cfg)
        N, R = cfg.num_lanes, cfg.num_recur
        expected_alpha = torch.zeros(R, N)
        for r in range(R):
            expected_alpha[r, r % N] = 1.0
        assert torch.allclose(model.lane_alpha, expected_alpha)
        for r in range(R):
            assert torch.allclose(model.lane_mixing[r], torch.eye(N))
        assert torch.allclose(model.lane_beta, torch.ones(R, N))

    def test_forward_shape_and_grad(self):
        cfg = make_cfg()
        model = build_model(cfg)
        idx = torch.randint(0, cfg.vocab_size, (2, 16))
        targets = torch.randint(0, cfg.vocab_size, (2, 16))
        loss = model(idx, targets=targets)
        loss.backward()
        # All lane params must get non-None gradients (full BPTT guaranteed)
        assert model.lane_alpha.grad is not None
        assert model.lane_mixing.grad is not None
        assert model.lane_beta.grad is not None
        # And at least one step must have nonzero grad on each
        assert model.lane_alpha.grad.abs().sum() > 0
        assert model.lane_mixing.grad.abs().sum() > 0
        assert model.lane_beta.grad.abs().sum() > 0
