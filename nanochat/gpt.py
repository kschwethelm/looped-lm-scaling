"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- RMSNorm with learnable scale
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration

Looped Transformer specifics:
- Three-stage architecture: prelude (n_prelude layers) → recur (n_recur_block layers, run num_recur times) → coda (n_coda layers)
- Input injection modes (input_injection config):
  - "inject_init_prelude": u = inject(concat(e, s)) with s initially from prelude output (default looped behavior)
  - "passthrough": u = s (pure recurrence, no inject layer), s initially from prelude output
  - "additive": u = s + e (parameter-free additive injection), s initially zeros so first iter u = e
  - "hyperconnect": maintain N parallel lane states; per-step unconstrained mixing
    (alpha_r aggregates lanes into block input, M_r mixes lanes, beta_r broadcasts block output).
    Cyclic init (alpha_r = e_{r mod N}, M_r = I, beta_r = 1). Sum-pool lanes before coda.
  - inject layer initialized as identity-like [I|0] so inject(concat(e,s)) ≈ e initially
- Truncated BPTT: gradients detached for all but last bptt_k recurrences to limit memory/compute
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn
from nanochat.optim import DistMuonAdamW, MuonAdamW


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32001
    size: int = 20  # Model size knob: model_dim = size * aspect_ratio
    n_head: int = 10  # number of query heads
    n_kv_head: int = 10  # number of key/value heads (GQA)
    n_embd: int = 1280

    # Looped Transformer config options
    n_prelude: int = 2
    n_recur_block: int = 4
    n_coda: int = 2

    # Looped Transformer training options
    num_recur: int = 4  # number of recurrences
    bptt_k: int = 4  # truncate backprop to last k recurrences
    # Input injection mode: controls how recurrent state is initialized and injected
    # - "inject_init_prelude": inject(concat(e, s)) with s initially from prelude output (default looped behavior)
    # - "passthrough": no injection, s passes through directly (pure recurrence, s initially from prelude)
    # - "additive": parameter-free additive injection s += e each recurrence (s initially zeros)
    # - "hyperconnect": N parallel lane states with per-step mixing (see class docstring)
    input_injection: Literal["inject_init_prelude", "passthrough", "additive", "hyperconnect"] = "inject_init_prelude"
    num_lanes: int = 2  # number of hyperconnection lanes (only used when input_injection == "hyperconnect")
    logit_softcap: float = 15.0  # smoothly cap logits to [-softcap, softcap] via tanh

    def __post_init__(self):
        valid_modes = {"inject_init_prelude", "passthrough", "additive", "hyperconnect"}
        if self.input_injection not in valid_modes:
            raise ValueError(f"input_injection must be one of {valid_modes}, got {self.input_injection}")
        if self.input_injection == "hyperconnect" and self.num_lanes < 2:
            raise ValueError(f"hyperconnect mode requires num_lanes >= 2, got {self.num_lanes}")
        # Truncated BPTT is incompatible with hyperconnect
        if self.input_injection == "hyperconnect" and self.bptt_k is not None and self.bptt_k < self.num_recur:
            raise ValueError(
                f"hyperconnect mode requires full BPTT (bptt_k == num_recur), "
                f"got bptt_k={self.bptt_k} < num_recur={self.num_recur}: "
                f"step-specific lane params would never receive gradients."
            )


def norm(x, eps: float = 1e-6):
    # Purely functional rmsnorm with no learnable params (used for QK norm)
    return F.rms_norm(x, (x.size(-1),), eps=eps)


class RMSNorm(nn.Module):
    """RMSNorm with learnable scale parameter."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps) * self.weight


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last dim into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache, layer_idx=None):
        B, T, C = x.size()
        # Fail fast: layer_idx required when using kv_cache
        assert (kv_cache is None) or (layer_idx is not None), "layer_idx required when kv_cache is provided"

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # Full causal attention (no sliding window)
        if kv_cache is None:
            # Training: causal attention
            y = flash_attn.flash_attn_func(q, k, v, causal=True)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                k=k,
                v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
            )

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.n1 = RMSNorm(config.n_embd)
        self.n3 = RMSNorm(config.n_embd)

    def forward(self, x, cos_sin, kv_cache, layer_idx=None):
        # NOTE: Depth-μP (HyperP, arXiv:2603.28743 §3.3) prescribes a residual multiplier
        # α_L = 1/√(2d) and MuonH LR ∝ 1/√d. We intentionally skip this because all models
        # use 20 effective layers — depth is constant so the factor is absorbed into the base LR.
        x = x + self.attn(self.n1(x), cos_sin, kv_cache, layer_idx)
        x = x + self.mlp(self.n3(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, pad_vocab_size_to: int = 64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config

        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(padded_vocab_size, config.n_embd),
                "prelude": nn.ModuleList([Block(config) for _ in range(config.n_prelude)]),
                "recur": nn.ModuleList([Block(config) for _ in range(config.n_recur_block)]),
                "coda": nn.ModuleList([Block(config) for _ in range(config.n_coda)]),
            }
        )
        # Input injection adapter (only needed for learned injection mode inject_init_prelude)
        if config.input_injection == "inject_init_prelude":
            self.inject = nn.Linear(2 * config.n_embd, config.n_embd, bias=False)
        # Hyperconnect lanes: per-step mixing parameters for N parallel state vectors
        if config.input_injection == "hyperconnect":
            N, R = config.num_lanes, config.num_recur
            self.lane_alpha = nn.Parameter(torch.empty(R, N))      # lane aggregation into block input
            self.lane_mixing = nn.Parameter(torch.empty(R, N, N))  # cross-lane mixing
            self.lane_beta = nn.Parameter(torch.empty(R, N))       # block-output broadcast to lanes
        # RMSNorm layers outside blocks
        self.norm_emb = RMSNorm(config.n_embd)  # after embedding
        self.norm_recur = RMSNorm(config.n_embd)  # at end of recurrent block (nc)
        self.norm_final = RMSNorm(config.n_embd)  # before lm_head
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10  # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)  # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     uniform, std=1/sqrt(n_embd) (MuonH needs nonzero norms)
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      uniform, std=1/sqrt(4*n_embd) (Kaiming fan_in; MuonH needs nonzero norms)
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5  # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in list(self.transformer.prelude) + list(self.transformer.recur) + list(self.transformer.coda):
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)  # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_proj.weight, -s, s)  # MuonH needs nonzero norms
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            s_proj = 3**0.5 * (4 * n_embd)**-0.5  # Kaiming fan_in for c_proj (input dim is 4*n_embd)
            torch.nn.init.uniform_(block.mlp.c_proj.weight, -s_proj, s_proj)  # MuonH needs nonzero norms

        # Initialize inject layer as identity-like: output = e (first half of concat(e, s))
        # This ensures gradients flow on the first forward pass
        # Weight shape is (n_embd, 2*n_embd), we want [I | 0] so inject(concat(e,s)) ≈ e
        # Only initialize if inject layer exists (inject_init_prelude only)
        if self.config.input_injection == "inject_init_prelude":
            with torch.no_grad():
                self.inject.weight.zero_()
                self.inject.weight[:, :n_embd].copy_(torch.eye(n_embd))

        # Hyperconnect lanes: cyclic init (HC paper). alpha_r = e_{r mod N}, M_r = I, beta_r = 1.
        # Cyclic asymmetry breaks lane symmetry; identity M + unit beta means step 0 is
        # equivalent to the plain looped forward, so training starts from a known-good point.
        if self.config.input_injection == "hyperconnect":
            N, R = self.config.num_lanes, self.config.num_recur
            self.lane_alpha.zero_()
            for r in range(R):
                self.lane_alpha[r, r % N] = 1.0
            self.lane_mixing.zero_()
            for r in range(R):
                for i in range(N):
                    self.lane_mixing[r, i, i] = 1.0
            self.lane_beta.fill_(1.0)

        # RMSNorm weights: initialize to ones
        for module in self.modules():
            if isinstance(module, RMSNorm):
                torch.nn.init.ones_(module.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()  # keep them in bfloat16
        cos, sin = (
            cos[None, :, None, :],
            sin[None, :, None, :],
        )  # add batch and head dims for later broadcasting
        return cos, sin

    def compile_blocks(self):
        """
        Compile individual transformer blocks for variable-length recurrence loops.

        Instead of torch.compile(model) which recompiles on every unique num_recur value
        (Python loop count changes trigger full retracing), this compiles each Block's
        forward method with dynamic=False. The Python loop in forward() runs eagerly,
        calling into compiled blocks each iteration — zero recompilations from varying
        loop count, while the heavy compute (attention + MLP) stays compiled.

        Compiles the forward *method* (not the module) to avoid wrapping blocks in
        OptimizedModule, which would add _orig_mod. prefix to state_dict keys and
        break checkpoint compatibility.
        """
        for section in ("prelude", "recur", "coda"):
            for block in self.transformer[section]:
                block.forward = torch.compile(block.forward, dynamic=False)

    def get_device(self):
        return self.transformer.wte.weight.device

    def _get_kv_layer_idx(
        self,
        section: str,
        block_idx: int,
        kv_budget: int | None,
        recur_iter: int | None = None,
    ) -> int | None:
        """
        Compute the KV cache layer index for a given section and block.

        The KV cache layout is:
            [prelude layers | recurrence slots | coda layers]
            [0..n_prelude)  | [n_prelude..n_prelude + kv_budget*n_recur_block) | [coda start..)

        Args:
            section: One of "prelude", "recur", or "coda"
            block_idx: Index within the section (0-indexed)
            kv_budget: Number of recurrence slots to store in cache (None means no cache)
            recur_iter: Current recurrence iteration (required for "recur" section)

        Returns:
            The layer index for the KV cache, or None if kv_budget is None
        """
        if kv_budget is None:
            return None

        if section == "prelude":
            return block_idx
        elif section == "recur":
            assert recur_iter is not None, "recur_iter required for recur section"
            # Circular indexing: slot = recur_iter % kv_budget
            slot = recur_iter % kv_budget
            return self.config.n_prelude + (slot * self.config.n_recur_block) + block_idx
        elif section == "coda":
            # Coda comes after all recurrence slots
            return self.config.n_prelude + (kv_budget * self.config.n_recur_block) + block_idx
        else:
            raise ValueError(f"Unknown section: {section}")

    def estimate_flops(self, num_recur=None):
        """
        Return the estimated FLOPs per token for the looped model (forward + backward).

        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * sequence_len accounts for key @ query matmul flops inside attention.
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)

        Truncated BPTT: when bptt_k < num_recur, early recurrences are detached and only
        incur forward cost (2x matmul, 4x attention), while the last bptt_k recurrences
        carry gradients and incur full fwd+bwd cost (6x matmul, 12x attention).

        Args:
            num_recur: Number of recurrences to assume. If None, uses num_recur.
        """
        if num_recur is None:
            num_recur = self.config.num_recur

        h, q, t = (
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
            self.config.sequence_len,
        )

        # Truncated BPTT splits recurrences into forward-only (detached) and full (fwd+bwd)
        bptt_k = self.config.bptt_k
        if bptt_k is not None and bptt_k < num_recur:
            fwd_only_recurs = num_recur - bptt_k
            full_recurs = bptt_k
        else:
            fwd_only_recurs = 0
            full_recurs = num_recur

        # 1. Count parameters by section
        prelude_params = sum(p.numel() for p in self.transformer.prelude.parameters())
        recur_params = sum(p.numel() for p in self.transformer.recur.parameters())
        coda_params = sum(p.numel() for p in self.transformer.coda.parameters())
        if self.config.input_injection == "inject_init_prelude":
            inject_params = sum(p.numel() for p in self.inject.parameters())
        elif self.config.input_injection == "hyperconnect":
            # Per-step lane mixing is O(N^2) scalars per step — model as "equivalent params"
            # scaled by embedding dim so matmul_flops weighting below matches actual cost.
            N = self.config.num_lanes
            inject_params = (N * N + 2 * N) * self.config.n_embd
        else:
            inject_params = 0
        lm_head_params = sum(p.numel() for p in self.lm_head.parameters())

        # 2. Matmul FLOPs weighted by usage
        # Non-recurrent parts: always full fwd+bwd (6x)
        # Recurrent parts: 2x (fwd only) for detached iters, 6x (fwd+bwd) for gradient iters
        matmul_flops = (
            6 * (prelude_params + coda_params + lm_head_params)
            + (2 * fwd_only_recurs + 6 * full_recurs) * recur_params
            + (2 * fwd_only_recurs + 6 * full_recurs) * inject_params
        )

        # 3. Attention FLOPs: full causal attention on all layers
        n_prelude = self.config.n_prelude
        n_recur_block = self.config.n_recur_block
        n_coda = self.config.n_coda
        attn_per_layer = h * q * t  # attention FLOPs per layer per token

        attn_flops = (
            12 * n_prelude * attn_per_layer  # prelude: 1x, full fwd+bwd
            + (4 * fwd_only_recurs + 12 * full_recurs) * n_recur_block * attn_per_layer  # recur
            + 12 * n_coda * attn_per_layer  # coda: 1x, full fwd+bwd
        )

        num_flops_per_token = matmul_flops + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.

        Key distinction for looped architecture:
        - recur_block: parameters executed num_recur times (reused)
        - prelude/coda/inject: parameters executed once per forward pass
        """
        # Count each group separately
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())

        # Split transformer matrices by execution pattern
        prelude = sum(p.numel() for p in self.transformer.prelude.parameters())
        recur_block = sum(p.numel() for p in self.transformer.recur.parameters())
        coda = sum(p.numel() for p in self.transformer.coda.parameters())
        if self.config.input_injection == "inject_init_prelude":
            inject = sum(p.numel() for p in self.inject.parameters())
        elif self.config.input_injection == "hyperconnect":
            inject = self.lane_alpha.numel() + self.lane_mixing.numel() + self.lane_beta.numel()
        else:
            inject = 0

        # Scalars: RMSNorm scale parameters
        scalars = (
            sum(p.numel() for p in self.norm_emb.parameters())
            + sum(p.numel() for p in self.norm_recur.parameters())
            + sum(p.numel() for p in self.norm_final.parameters())
        )

        total = wte + lm_head + prelude + recur_block + coda + inject + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"

        return {
            'wte': wte,
            'lm_head': lm_head,
            'prelude': prelude,
            'recur_block': recur_block,
            'coda': coda,
            'inject': inject,
            'scalars': scalars,
            'total': total,
        }

    def effective_params(self, num_recur=None, exclude_embedding=True):
        """
        Compute effective parameter count accounting for parameter reuse in recurrent block.

        The recurrent block's parameters are executed num_recur times, while prelude/coda
        run once. This gives a measure of "parameter uses" rather than unique parameters.

        Args:
            num_recur: Number of recurrences to assume. If None, uses num_recur.
            exclude_embedding: If True, excludes embedding and lm_head parameters from the count (common in scaling law analysis)   

        Returns:
            Effective parameter count (weighted by usage)
        """
        if num_recur is None:
            num_recur = self.config.num_recur

        counts = self.num_scaling_params()

        # Parameters that run once per forward pass
        once = counts['prelude'] + counts['coda'] + counts['scalars']
        if not exclude_embedding:
            once += counts['wte'] + counts['lm_head']

        # Parameters that run num_recur times per forward pass
        reused = counts['recur_block'] + counts['inject']

        effective = once + (reused * num_recur)
        return effective

    def _hyperconnect_combine(self, lanes: torch.Tensor, recur_idx: int) -> torch.Tensor:
        """Aggregate N lanes (N, B, T, d) → block input (B, T, d) via alpha_r."""
        alpha = self.lane_alpha[recur_idx]  # (N,)
        return torch.einsum("n,nbtd->btd", alpha, lanes)

    def _hyperconnect_step(self, lanes: torch.Tensor, s: torch.Tensor, recur_idx: int) -> torch.Tensor:
        """Update lanes: M_r mixes old lanes, beta_r broadcasts block output. Shapes: (N,B,T,d)."""
        M = self.lane_mixing[recur_idx]    # (N, N)
        beta = self.lane_beta[recur_idx]   # (N,)
        mixed = torch.einsum("ij,jbtd->ibtd", M, lanes)
        return mixed + beta[:, None, None, None] * s[None]

    def _state_transfer(self, e, s=None):
        """
        Handle state initialization and input injection based on input_injection mode.

        Combines state initialization and input injection into a single operation:
        - For "inject_init_prelude": returns inject(concat(e, s))
        - For "passthrough": returns s directly (no injection layer)
        - For "additive": returns s + e (parameter-free)

        Args:
            e: Prelude output (B, T, n_embd)
            s: Current recurrent state (B, T, n_embd), or None for initial recurrence

        Returns:
            Input to the recurrent block u (B, T, n_embd)
        """
        # Initialize state on first recurrence (when s is None)
        if s is None:
            if self.config.input_injection == "additive":
                # Parameter-free additive injection: start from zeros so first iter input = e
                s = torch.zeros_like(e)
            else:
                # Both "inject_init_prelude" and "passthrough" start with prelude output
                s = e

        # Apply injection mode
        if self.config.input_injection == "passthrough":
            return s
        elif self.config.input_injection == "additive":
            return s + e
        else:
            # "inject_init_prelude": learned injection layer
            return self.inject(torch.cat([e, s], dim=-1))

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        adam_betas=(0.8, 0.95),
        d_ref: int = 512,
    ):
        """
        Create combined MuonAdamW optimizer: Muon for 2D matrix params, AdamW for embeddings.
        """
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        # Collect all RMSNorm parameters (1D scale weights)
        norm_params = list(self.norm_emb.parameters()) + list(self.norm_recur.parameters()) + list(self.norm_final.parameters())
        for block in list(self.transformer.prelude) + list(self.transformer.recur) + list(self.transformer.coda):
            norm_params += list(block.n1.parameters()) + list(block.n3.parameters())
        norm_param_set = set(norm_params)

        # Matrix params: all block params except norms (include inject only for inject_init_prelude)
        inject_params_list = list(self.inject.parameters()) if self.config.input_injection == "inject_init_prelude" else []
        matrix_params = [
            p
            for p in (
                list(self.transformer.prelude.parameters())
                + list(self.transformer.recur.parameters())
                + list(self.transformer.coda.parameters())
                + inject_params_list
            )
            if p not in norm_param_set
        ]
        # Hyperconnect lane params: small per-step scalars, trained with AdamW (Muon does not apply).
        hyperconnect_params = (
            [self.lane_alpha, self.lane_mixing, self.lane_beta]
            if self.config.input_injection == "hyperconnect"
            else []
        )
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(norm_params) + len(hyperconnect_params)

        # Scale the LR for the AdamW parameters by ∝1/√(dmodel/d_ref) — muP width scaling
        # Ref: https://arxiv.org/abs/2203.03466 (Tensor Programs V, Yang et al. 2022)
        dmodel_lr_scale = (model_dim / d_ref) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/{d_ref}) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, norms)
            dict(kind="adamw", params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=norm_params, lr=0.005 * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
        if hyperconnect_params:
            param_groups.append(
                dict(kind="adamw", params=hyperconnect_params, lr=0.01 * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            )
        # MuonH groups (matrix params, grouped by shape for stacking)
        # MuonH constrains weights to a Frobenius sphere, making weight decay
        # a first-order no-op (arXiv:2603.28743). WD is omitted entirely.
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(
                dict(
                    kind="muon",
                    params=group_params,
                    lr=matrix_lr,
                    momentum=0.95,
                    ns_steps=5,
                )
            )

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _predict(self, s, cos_sin, kv_cache=None, kv_budget=None):
        """
        Run coda blocks + norm_final + lm_head to produce logits from recurrent state.

        Args:
            s: Recurrent state after norm_recur, shape (B, T, n_embd)
            cos_sin: Rotary embedding tuple for current positions
            kv_cache: Optional KV cache (entries overwritten in-place)
            kv_budget: KV budget (needed for layer index computation)

        Returns:
            logits: (B, T, vocab_size) in float32, with logit softcap applied
        """
        x = s
        for i, block in enumerate(self.transformer.coda):
            layer_idx = self._get_kv_layer_idx("coda", i, kv_budget)
            x = block(x, cos_sin, kv_cache, layer_idx)
        x = self.norm_final(x)

        softcap = self.config.logit_softcap
        logits = self.lm_head(x)  # (B, T, padded_vocab_size)
        logits = logits[..., : self.config.vocab_size]  # slice to remove padding
        logits = logits.float()  # fp32 for softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap)
        return logits

    def forward(
        self,
        idx,
        targets=None,
        kv_cache=None,
        loss_reduction="mean",
        num_recur=None,
    ):
        B, T = idx.size()
        if num_recur is None:
            num_recur = self.config.num_recur

        kv_budget = kv_cache.kv_budget if kv_cache is not None else None

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert self.cos.size(1) >= T, f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        if kv_cache is not None and T == 1 and not torch.all(kv_cache.cache_seqlens == kv_cache.cache_seqlens[0]):
            # Per-element rotary positions for batched decode with non-uniform prompt lengths
            # self.cos is (1, max_seq, 1, hd/2), index with (B,) positions → (B, 1, hd/2) → unsqueeze → (B, 1, 1, hd/2)
            # This broadcasts correctly with q/k shapes (B, T=1, H, hd)
            positions = kv_cache.cache_seqlens.long()
            cos_sin = (
                self.cos[0, positions].unsqueeze(1),
                self.sin[0, positions].unsqueeze(1),
            )
        else:
            T0 = 0 if kv_cache is None else kv_cache.get_pos()
            cos_sin = (
                self.cos[:, T0 : T0 + T],
                self.sin[:, T0 : T0 + T],
            )  # truncate cache to current sequence length

        # 1. Embedding + norm
        x = self.transformer.wte(idx)
        x = self.norm_emb(x)

        # 2. Prelude blocks (run once)
        for i, block in enumerate(self.transformer.prelude):
            layer_idx = self._get_kv_layer_idx("prelude", i, kv_budget)
            x = block(x, cos_sin, kv_cache, layer_idx)
        e = x  # prelude output, used for injection into each recurrence

        # 3. Recurrent block (run num_recur times)
        if self.config.input_injection == "hyperconnect":
            # N parallel lane states, all initialized to prelude output e (shape: (N, B, T, d))
            lanes = e.unsqueeze(0).expand(self.config.num_lanes, -1, -1, -1).contiguous()
            for i in range(num_recur):
                u = self._hyperconnect_combine(lanes, i)
                for j, block in enumerate(self.transformer.recur):
                    layer_idx = self._get_kv_layer_idx("recur", j, kv_budget, recur_iter=i)
                    u = block(u, cos_sin, kv_cache, layer_idx)
                s_step = self.norm_recur(u)
                lanes = self._hyperconnect_step(lanes, s_step, i)
                # Truncated BPTT: detach all lanes before the last bptt_k recurrences
                if self.config.bptt_k is not None and i < num_recur - self.config.bptt_k:
                    lanes = lanes.detach()
            s = lanes.sum(dim=0)  # sum-pool lanes for coda input
        else:
            s = None
            for i in range(num_recur):
                # State transfer: handles initialization (on i==0) and input injection
                u = self._state_transfer(e, s=s)
                # Run recur blocks with KV cache
                for j, block in enumerate(self.transformer.recur):
                    layer_idx = self._get_kv_layer_idx("recur", j, kv_budget, recur_iter=i)
                    u = block(u, cos_sin, kv_cache, layer_idx)
                s = self.norm_recur(u)  # nc: rescale at end of recurrent block
                # Truncated BPTT: detach gradients for recurrences before the last bptt_k
                # This limits gradient flow depth to bptt_k * n_recur_block layers through recurrence
                if self.config.bptt_k is not None and i < num_recur - self.config.bptt_k:
                    s = s.detach()

        # 5. Coda + norm_final + lm_head → logits (final prediction)
        logits = self._predict(s, cos_sin, kv_cache, kv_budget)

        # Advance KV cache position after all layers are done
        if kv_cache is not None:
            kv_cache.advance(T)

        if targets is not None:
            # training: given the targets, compute and return the loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return loss
        else:
            return logits
