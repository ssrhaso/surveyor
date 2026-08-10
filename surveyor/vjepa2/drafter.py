"""Token-grid GDM: the spec-accept drafter at V-JEPA 2 scale.

Same DiT-denoiser design as surveyor.drafter.GDM, drafting a block of n_future
SUBGOAL TOKEN GRIDS (each tokens_per_frame x D) rather than n_future 192-d
vectors, so drafted subgoals live in the exact space upstream's CEM scores
against. The diffusion process, blocks, and sampling loops are imported from the
validated LeWM-stack implementation instead of re-derived: TokenGDM keeps GDM's
forward contract (x_noised, cond, t, goal) so GaussianDiffusion works verbatim,
and only the sample shape changes from (B, N, D) to (B, N*T_tok, D).

Conditioning is the POOLED current-frame latent, plus the pooled goal when
goal_cond, so the drafter proposes where to go from a summary of where it is.
The full-resolution comparison happens in the planner's cost, not here.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

_LE_WM_ROOT = Path(__file__).resolve().parents[2]
if str(_LE_WM_ROOT) not in sys.path:
    sys.path.insert(0, str(_LE_WM_ROOT))

from surveyor.drafter import (  # noqa: E402  (validated implementations, single-sourced)
    DiTBlock,
    FinalLayer,
    GaussianDiffusion,
    timestep_embedding,
)


@dataclass
class TokenGDMConfig:
    latent_dim: int = 1408        # ViT-g embed dim
    tokens_per_frame: int = 256   # (256/16)^2 at crop 256
    n_future: int = 3             # N: subgoal block length (paper value)
    hidden: int = 512
    depth: int = 8
    heads: int = 8
    mlp_ratio: float = 4.0
    goal_cond: bool = True        # manipulation default (cube finding); flag off to ablate
    residual: bool = False        # v2: denoise (block - cond_tokens); the consumer adds
                                  # the current grid back, so static content is free
                                  # (v1 absolute drafting lost to no-op on DROID)


class TokenGDM(nn.Module):
    """DiT denoiser over a (n_future * tokens_per_frame)-token sequence.
    forward(x_noised (B, N*T_tok, D), cond (B, D), t (B,), goal (B, D)|None)
    -> predicted noise/v (B, N*T_tok, D)."""

    def __init__(self, cfg: TokenGDMConfig | None = None, **kw):
        super().__init__()
        if cfg is None:
            cfg = TokenGDMConfig(**kw)
        self.cfg = cfg
        D, H = cfg.latent_dim, cfg.hidden

        self.x_embed = nn.Linear(D, H)
        # factorized positions: block slot (which subgoal) + token index (where in frame)
        self.pos_block = nn.Parameter(torch.zeros(1, cfg.n_future, 1, H))
        self.pos_token = nn.Parameter(torch.zeros(1, 1, cfg.tokens_per_frame, H))
        self.cond_embed = nn.Linear(D, H)
        if cfg.goal_cond:
            self.goal_embed = nn.Linear(D, H)
        self.t_embed = nn.Sequential(nn.Linear(H, H), nn.SiLU(), nn.Linear(H, H))
        self.t_embed_dim = H

        self.blocks = nn.ModuleList(
            [DiTBlock(H, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth)]
        )
        self.final = FinalLayer(H, D)
        self._init_weights()

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic)
        nn.init.normal_(self.pos_block, std=0.02)
        nn.init.normal_(self.pos_token, std=0.02)
        for blk in self.blocks:
            nn.init.zeros_(blk.adaLN[-1].weight)
            nn.init.zeros_(blk.adaLN[-1].bias)
        nn.init.zeros_(self.final.adaLN[-1].weight)
        nn.init.zeros_(self.final.adaLN[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def forward(self, x_noised: torch.Tensor, cond: torch.Tensor, t: torch.Tensor,
                goal: torch.Tensor | None = None) -> torch.Tensor:
        B = x_noised.shape[0]
        N, T_tok = self.cfg.n_future, self.cfg.tokens_per_frame
        x = self.x_embed(x_noised).view(B, N, T_tok, -1)
        x = (x + self.pos_block + self.pos_token).flatten(1, 2)   # (B, N*T_tok, H)
        if cond.ndim == 3:
            cond = cond.reshape(B, -1)
        c = self.cond_embed(cond) + self.t_embed(timestep_embedding(t, self.t_embed_dim))
        if self.cfg.goal_cond:
            assert goal is not None, "goal_cond=True but no goal latent passed"
            if goal.ndim == 3:
                goal = goal.reshape(B, -1)
            c = c + self.goal_embed(goal)
        for blk in self.blocks:
            x = blk(x, c)
        return self.final(x, c)


class TokenGDMPlanner:
    """Bundles TokenGDM + diffusion + per-dim standardization stats (shared
    across tokens). sample_sequence: pooled cond in, token-grid block out.
    stat_a/b standardize the DENOISED QUANTITY (absolute grids, or residuals
    when cfg.residual); cond_stat_a/b standardize the conditioning inputs
    (frame stats; defaults to stat_a/b for v1 checkpoints)."""

    def __init__(self, model: TokenGDM, diffusion: GaussianDiffusion,
                 stat_a: torch.Tensor, stat_b: torch.Tensor, device="cpu",
                 cond_stat_a: torch.Tensor | None = None,
                 cond_stat_b: torch.Tensor | None = None):
        self.model = model.to(device).eval()
        self.diffusion = diffusion.to(device)
        self.stat_a = stat_a.to(device).view(1, 1, -1)   # (1, 1, D): mean
        self.stat_b = stat_b.to(device).view(1, 1, -1)   # (1, 1, D): std
        ca = stat_a if cond_stat_a is None else cond_stat_a
        cb = stat_b if cond_stat_b is None else cond_stat_b
        self.cond_a = ca.to(device).view(1, -1)
        self.cond_b = cb.to(device).view(1, -1)
        self.device = device
        self.cfg = model.cfg
        self.goal_cond = model.cfg.goal_cond

    def standardize(self, z: torch.Tensor) -> torch.Tensor:
        return (z.to(self.device) - self.stat_a.view(*([1] * (z.ndim - 1)), -1)) \
            / self.stat_b.view(*([1] * (z.ndim - 1)), -1)

    def unstandardize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.stat_b.view(*([1] * (z.ndim - 1)), -1) \
            + self.stat_a.view(*([1] * (z.ndim - 1)), -1)

    @torch.no_grad()
    def sample_sequence(self, z_cond_pooled: torch.Tensor, n_steps: int = 8,
                        generator: torch.Generator | None = None,
                        z_goal_pooled: torch.Tensor | None = None) -> torch.Tensor:
        """z_cond_pooled (B, D) native -> (B, n_future, tokens_per_frame, D)
        in native space: absolute grids, or RESIDUALS to be added to the
        conditioning grid when cfg.residual. n_steps = k (DDIM NFE)."""
        B = z_cond_pooled.shape[0]
        cond = (z_cond_pooled.to(self.device) - self.cond_a) / self.cond_b
        goal = ((z_goal_pooled.to(self.device) - self.cond_a) / self.cond_b
                if (self.goal_cond and z_goal_pooled is not None) else None)
        shape = (B, self.cfg.n_future * self.cfg.tokens_per_frame, self.cfg.latent_dim)
        seq = self.diffusion.ddim_sample(self.model, cond, shape, n_steps=n_steps,
                                         generator=generator, goal=goal)
        seq = seq.view(B, self.cfg.n_future, self.cfg.tokens_per_frame, self.cfg.latent_dim)
        return self.unstandardize(seq)


def save_token_gdm(path, model: TokenGDM, diffusion: GaussianDiffusion,
                   stat_a: torch.Tensor, stat_b: torch.Tensor,
                   cond_stat_a: torch.Tensor | None = None,
                   cond_stat_b: torch.Tensor | None = None,
                   extra: dict | None = None):
    torch.save({
        "model_state": model.state_dict(),
        "gdm_config": asdict(model.cfg),
        "diffusion": {"timesteps": diffusion.timesteps,
                      "beta_start": diffusion.beta_start,
                      "beta_end": diffusion.beta_end,
                      "schedule": diffusion.schedule,
                      "parameterization": diffusion.parameterization,
                      "min_snr_gamma": diffusion.min_snr_gamma,
                      "sampler": diffusion.sampler},
        "norm_stats": {"a": stat_a.cpu(), "b": stat_b.cpu(),
                       "cond_a": None if cond_stat_a is None else cond_stat_a.cpu(),
                       "cond_b": None if cond_stat_b is None else cond_stat_b.cpu()},
        "extra": extra or {},
    }, path)


def load_token_gdm(path, device="cpu") -> TokenGDMPlanner:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = TokenGDM(TokenGDMConfig(**ck["gdm_config"]))
    model.load_state_dict(ck["model_state"])
    diffusion = GaussianDiffusion(**ck["diffusion"], device=device)
    ns = ck["norm_stats"]
    return TokenGDMPlanner(model, diffusion, ns["a"], ns["b"], device=device,
                           cond_stat_a=ns.get("cond_a"), cond_stat_b=ns.get("cond_b"))
