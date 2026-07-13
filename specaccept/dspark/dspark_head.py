"""DSpark refinement head: a standalone residual refiner for a block of N
drafted subgoal latents.

Depends only on torch, so it ports unchanged to other planners. Takes a raw
draft block (B, N, D) and the conditioning latent (B, D), and returns a
refined block in the same native latent space:

    z_refined_i = z_draft_i + MLP([z_cond, z_draft_i, cond_prev, pos_emb_i])

Weights are shared across positions with a sinusoidal position embedding for
arbitrary-N generalization. The final linear is zero-initialized (adaLN-Zero
convention) so training starts as an identity pass-through. Inputs are per-dim
standardized and outputs inverted. mode='causal' uses the previously refined
latent as cond_prev (sequential unroll); mode='noncausal' uses the raw
previous draft plus a block summary, fully parallel. Trained only on real
frozen-drafter chains (see build_real_chains.py), never on ground-truth
prefixes or injected noise.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_pos_emb(positions: torch.Tensor, dim: int, max_period: int = 128) -> torch.Tensor:
    """positions: (...,) long/float -> (..., dim) sinusoidal embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=positions.device, dtype=torch.float32) / max(half, 1)
    )
    args = positions.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
    return emb


class DSparkHead(nn.Module):
    """First-order-conditioned residual refiner over a block of N draft latents.

    Args:
        dim: latent dimensionality D.
        mean, std: (D,) per-dim standardization stats (native <-> standardized).
        hidden: MLP hidden width.
        pos_dim: sinusoidal position-embedding width.
        mode: 'causal' or 'noncausal'.
        n_layers: number of hidden layers (>=1).
    """

    def __init__(self, dim: int, mean: torch.Tensor, std: torch.Tensor,
                 hidden: int = 256, pos_dim: int = 32, mode: str = "causal",
                 n_layers: int = 2):
        super().__init__()
        assert mode in ("causal", "noncausal"), mode
        self.dim = dim
        self.mode = mode
        self.pos_dim = pos_dim
        self.register_buffer("mean", mean.detach().clone().view(1, 1, dim))
        self.register_buffer("std", std.detach().clone().clamp_min(1e-6).view(1, 1, dim))

        # input: [z_cond_std, z_draft_i_std, cond_prev_std, pos_emb]  (3*D + pos_dim)
        in_dim = 3 * dim + pos_dim
        if mode == "noncausal":
            in_dim += dim  # + block-summary token (mean of full raw draft block, std space)

        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # adaLN-Zero style: zero the final projection -> starts as identity pass-through
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _std(self, z):   # native -> standardized
        return (z - self.mean) / self.std

    def _unstd(self, z):  # standardized -> native
        return z * self.std + self.mean

    def forward(self, draft: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """draft: (B, N, D) native; cond: (B, D) native. Returns (B, N, D) native."""
        B, N, D = draft.shape
        assert D == self.dim
        cond_std = self._std(cond.unsqueeze(1))            # (B,1,D)
        draft_std = self._std(draft)                       # (B,N,D)
        pos = torch.arange(N, device=draft.device)
        pos_emb = sinusoidal_pos_emb(pos, self.pos_dim).unsqueeze(0).expand(B, N, self.pos_dim)
        cond_rep = cond_std.expand(B, N, D)

        if self.mode == "noncausal":
            block_summary = draft_std.mean(dim=1, keepdim=True).expand(B, N, D)  # (B,N,D)
            # cond_prev = raw draft of previous position (z_cond for i=0); fully parallel
            prev = torch.cat([cond_std, draft_std[:, :-1]], dim=1)               # (B,N,D)
            feat = torch.cat([cond_rep, draft_std, prev, block_summary, pos_emb], dim=-1)
            resid = self.out(self.trunk(feat))
            refined_std = draft_std + resid
            return self._unstd(refined_std)

        # causal: sequential unroll, cond_prev = refined previous position
        refined_std_list = []
        prev = cond_std.squeeze(1)                          # (B,D) std
        for i in range(N):
            feat = torch.cat([cond_std.squeeze(1), draft_std[:, i], prev,
                              pos_emb[:, i]], dim=-1)        # (B, 3D+pos)
            resid = self.out(self.trunk(feat))              # (B,D)
            r = draft_std[:, i] + resid
            refined_std_list.append(r)
            prev = r                                        # feed own refined output
        refined_std = torch.stack(refined_std_list, dim=1)  # (B,N,D)
        return self._unstd(refined_std)


class ConfidenceHead(nn.Module):
    """Per-position acceptance confidence c_k for the DSpark speculative commit.

    Single-model (no ensemble) so it is deployable at the cost of one drafter
    sample. Predicts P(refined subgoal k is "good enough" = rel_err < tau_k) from
    (refined_k, draft_k, cond, residual_k, pos). At deploy time the cumulative
    product Π_{i<=k} c_i sets the commit depth k* = max{k : Π c_i > theta} -- the
    confidence-scheduled speculative-decoding rule adapted to continuous latents.
    """

    def __init__(self, dim: int, mean: torch.Tensor, std: torch.Tensor,
                 hidden: int = 128, pos_dim: int = 32, n_layers: int = 2):
        super().__init__()
        self.dim = dim
        self.pos_dim = pos_dim
        self.register_buffer("mean", mean.detach().clone().view(1, 1, dim))
        self.register_buffer("std", std.detach().clone().clamp_min(1e-6).view(1, 1, dim))
        in_dim = 4 * dim + pos_dim  # refined, draft, cond, residual(refined-draft), pos
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def _std(self, z):
        return (z - self.mean) / self.std

    def logits(self, refined, draft, cond):
        B, N, D = refined.shape
        r_std = self._std(refined); d_std = self._std(draft)
        c_std = self._std(cond.unsqueeze(1)).expand(B, N, D)
        resid = r_std - d_std
        pos = torch.arange(N, device=refined.device)
        pe = sinusoidal_pos_emb(pos, self.pos_dim).unsqueeze(0).expand(B, N, self.pos_dim)
        feat = torch.cat([r_std, d_std, c_std, resid, pe], dim=-1)
        return self.out(self.trunk(feat)).squeeze(-1)   # (B,N)

    def forward(self, refined, draft, cond, temperature=None):
        lg = self.logits(refined, draft, cond)
        if temperature is not None:
            t = temperature if torch.is_tensor(temperature) else torch.as_tensor(
                temperature, device=lg.device, dtype=lg.dtype)
            lg = lg / t.view(1, -1)                     # sequential temperature scaling (per-pos T)
        return torch.sigmoid(lg)                         # (B,N) in [0,1]


def commit_depth(conf: torch.Tensor, theta: float, max_commit: int) -> int:
    """conf: (N,) per-position confidence in [0,1]. Returns k* = max{k : Π_{i<k} c_i
    > theta}, clamped to [1, max_commit]. (cumprod is monotone-decreasing so the
    accepted set is a prefix.)"""
    cum = torch.cumprod(conf.clamp(0, 1), dim=0)
    k = int((cum > theta).sum().item())
    return max(1, min(k, max_commit))


class OracleFloorHead(nn.Module):
    """Probe-0 ceiling head: conditions on GROUND-TRUTH z_{i-1} (+ z_cond + full
    raw draft block) to predict z_i. This is the best-case any-conditioning
    ceiling -- it is NOT deployable (ground truth is unavailable at inference) and
    exists only to measure how much of the horizon error is recoverable in
    principle vs horizon-intrinsic. High capacity, direct prediction (not residual).
    """

    def __init__(self, dim: int, mean: torch.Tensor, std: torch.Tensor,
                 hidden: int = 512, pos_dim: int = 32, n_layers: int = 3):
        super().__init__()
        self.dim = dim
        self.pos_dim = pos_dim
        self.register_buffer("mean", mean.detach().clone().view(1, dim))
        self.register_buffer("std", std.detach().clone().clamp_min(1e-6).view(1, dim))
        # input: [z_cond_std, z_prevTRUE_std, full_block_std (N*D), pos_emb]
        # full-block width is resolved lazily on first forward via a LazyLinear.
        self.trunk = None
        self._hidden = hidden
        self._n_layers = n_layers
        self.out = None

    def _build(self, in_dim, device):
        hidden, n_layers = self._hidden, self._n_layers
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.trunk = nn.Sequential(*layers).to(device)
        self.out = nn.Linear(hidden, self.dim).to(device)
        for m in list(self.trunk.modules()) + [self.out]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _std(self, z):
        return (z - self.mean) / self.std

    def _unstd(self, z):
        return z * self.std + self.mean

    def forward(self, draft_block: torch.Tensor, cond: torch.Tensor,
                z_true_prev: torch.Tensor) -> torch.Tensor:
        """draft_block: (B,N,D) raw draft; cond: (B,D); z_true_prev: (B,N,D) where
        [:,i] = GROUND-TRUTH z_{i-1} (z_cond for i=0). Returns (B,N,D) predicted."""
        B, N, D = draft_block.shape
        cond_std = self._std(cond)                                    # (B,D)
        block_std = self._std(draft_block).reshape(B, N * D)          # (B,N*D)
        prev_std = self._std(z_true_prev)                             # (B,N,D)
        pos = torch.arange(N, device=draft_block.device)
        pos_emb = sinusoidal_pos_emb(pos, self.pos_dim)              # (N,pos)
        outs = []
        for i in range(N):
            feat = torch.cat([cond_std, prev_std[:, i], block_std,
                              pos_emb[i].unsqueeze(0).expand(B, self.pos_dim)], dim=-1)
            if self.trunk is None:
                self._build(feat.shape[-1], feat.device)
            outs.append(self._unstd(self.out(self.trunk(feat))))
        return torch.stack(outs, dim=1)                               # (B,N,D)
