"""GDM latent diffusion planner: model, diffusion process, and checkpoint
I/O. A conditional DDPM over length-N subgoal latent sequences (DiT
backbone, epsilon objective) operating on per-dim standardized latents with
the stats stored in the checkpoint and inverted on sampling. Never touches
the frozen LeWM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
from torch import nn
import torch.nn.functional as F


# Sinusoidal diffusion-timestep embedding
def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """t: (B,) int/float -> (B, dim) sinusoidal embedding (Vaswani/DDPM style)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:  # zero-pad if odd
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # x: (B,N,D); shift/scale: (B,D) -> broadcast over the N tokens
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning (Peebles & Xie, 2023)."""

    def __init__(self, hidden: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden), nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden),
        )
        # produces shift/scale/gate for both attn and mlp (6 * hidden)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    """adaLN-Zero final layer projecting hidden -> latent_dim (per token)."""

    def __init__(self, hidden: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden, out_dim)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


@dataclass
class GDMConfig:
    latent_dim: int = 192
    n_future: int = 3      # N: number of future subgoals predicted
    wg: int = 1            # context window (conditioning subgoals); paper uses WG=1
    hidden: int = 512
    depth: int = 11
    heads: int = 8
    mlp_ratio: float = 4.0
    goal_cond: bool = False   # ABLATION: also condition on the episode goal latent


class GDM(nn.Module):
    """DiT denoiser. Predicts the noise eps added to a length-N latent sequence,
    conditioned on the WG conditioning latent(s) and the diffusion timestep.

    forward(x_noised, cond, t):
      x_noised : (B, N, latent_dim)   noised target subgoal sequence
      cond     : (B, wg, latent_dim)  or (B, latent_dim) when wg==1  (standardized)
      t        : (B,)                 diffusion timestep indices
      returns  : (B, N, latent_dim)   predicted noise eps
    """

    def __init__(self, cfg: GDMConfig | None = None, **kw):
        super().__init__()
        if cfg is None:
            cfg = GDMConfig(**kw)
        self.cfg = cfg
        D, N, H = cfg.latent_dim, cfg.n_future, cfg.hidden

        self.x_embed = nn.Linear(D, H)
        self.pos_embed = nn.Parameter(torch.zeros(1, N, H))      # target positions
        self.cond_embed = nn.Linear(cfg.wg * D, H)               # conditioning latent(s)
        if cfg.goal_cond:
            self.goal_embed = nn.Linear(D, H)                    # optional goal latent
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
        nn.init.normal_(self.pos_embed, std=0.02)
        # adaLN-Zero: zero the modulation outputs so blocks start as identity
        for blk in self.blocks:
            nn.init.zeros_(blk.adaLN[-1].weight)
            nn.init.zeros_(blk.adaLN[-1].bias)
        nn.init.zeros_(self.final.adaLN[-1].weight)
        nn.init.zeros_(self.final.adaLN[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def forward(self, x_noised: torch.Tensor, cond: torch.Tensor, t: torch.Tensor,
                goal: torch.Tensor | None = None) -> torch.Tensor:
        if cond.ndim == 3:                      # (B, wg, D) -> (B, wg*D)
            cond = cond.reshape(cond.shape[0], -1)
        x = self.x_embed(x_noised) + self.pos_embed
        c = self.cond_embed(cond) + self.t_embed(timestep_embedding(t, self.t_embed_dim))
        if self.cfg.goal_cond:                  # optional goal conditioning (ablation)
            assert goal is not None, "goal_cond=True but no goal latent passed"
            if goal.ndim == 3:
                goal = goal.reshape(goal.shape[0], -1)
            c = c + self.goal_embed(goal)
        for blk in self.blocks:
            x = blk(x, c)
        return self.final(x, c)


# Gaussian diffusion (DDPM training / DDIM sampling)
def _cosine_alphas_cumprod(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal (2021) cosine bar-alpha schedule (float64).
    Returns a length-`timesteps` alphas_cumprod in (0,1], strictly decreasing.
    Less probability mass sits at extreme high-t than a linear beta schedule."""
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    f = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    acp = f / f[0]
    return acp[1:].clamp(1e-8, 1.0)


class GaussianDiffusion:
    """DDPM training with DDIM (eta=0) or ancestral DDPM sampling.

    Design knobs are flag-gated so an A/B comparison can flip exactly one:
      parameterization : "eps" (predict the noise; default) or "v" (predict
                         the velocity; Salimans & Ho 2022).
      schedule         : "linear" beta (default) or "cosine" bar-alpha.
      min_snr_gamma    : Min-SNR loss weighting (Hang et al. 2023) when > 0;
                         0 (default) is plain MSE in the prediction space.

    `pred_from_model` is the single place the parameterization is decoded, so
    training loss, sampling, and diagnostics stay consistent by construction.
    """

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 2e-2, schedule: str = "linear",
                 parameterization: str = "eps", min_snr_gamma: float = 0.0,
                 sampler: str = "ddim", device="cpu"):
        assert parameterization in ("eps", "v"), parameterization
        assert schedule in ("linear", "cosine"), schedule
        assert sampler in ("ddim", "ddpm"), sampler
        self.timesteps = timesteps
        self.schedule = schedule
        self.parameterization = parameterization
        self.min_snr_gamma = float(min_snr_gamma)
        self.sampler = sampler
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
            acp = torch.cumprod(1.0 - betas, dim=0)
        else:  # cosine
            acp = _cosine_alphas_cumprod(timesteps)
            acp_p = torch.cat([torch.ones(1, dtype=torch.float64), acp[:-1]])
            betas = (1.0 - acp / acp_p).clamp(0.0, 0.999)
        acp_prev = torch.cat([torch.ones(1, dtype=torch.float64), acp[:-1]])
        self.betas = betas.float().to(device)
        self.alphas = (1.0 - betas).float().to(device)               # per-step alpha
        self.alphas_cumprod = acp.float().to(device)
        self.alphas_cumprod_prev = acp_prev.float().to(device)       # for DDPM posterior
        self.sqrt_acp = torch.sqrt(acp).float().to(device)
        self.sqrt_one_minus_acp = torch.sqrt(1.0 - acp).float().to(device)
        self.device = device

    def to(self, device):
        for k in ("betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
                  "sqrt_acp", "sqrt_one_minus_acp"):
            setattr(self, k, getattr(self, k).to(device))
        self.device = device
        return self

    def _ab(self, t: torch.Tensor, ndim: int):
        """Broadcast (sqrt_acp[t], sqrt_one_minus_acp[t]) to ndim dims."""
        a = self.sqrt_acp[t].view(-1, *([1] * (ndim - 1)))
        b = self.sqrt_one_minus_acp[t].view(-1, *([1] * (ndim - 1)))
        return a, b

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(acp_t) x0 + sqrt(1-acp_t) eps. t: (B,)."""
        a, b = self._ab(t, x0.ndim)
        return a * x0 + b * noise

    def target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The regression target the model is trained to output at (x0,noise,t)."""
        if self.parameterization == "eps":
            return noise
        a, b = self._ab(t, x0.ndim)              # v = sqrt(acp)*eps - sqrt(1-acp)*x0
        return a * noise - b * x0

    def pred_from_model(self, model_out: torch.Tensor, x_t: torch.Tensor,
                        t: torch.Tensor):
        """Map a model output to (x0_hat, eps_hat), interpreting it per the
        active parameterization. This is the ONLY place eps/v is decoded, so the
        1/sqrt(acp) amplification is confined to the eps path: for v the x0 map
        uses bounded coefficients (sqrt(acp), sqrt(1-acp)) and cannot blow up."""
        a, b = self._ab(t, x_t.ndim)
        if self.parameterization == "eps":
            eps = model_out
            x0 = (x_t - b * eps) / a
        else:  # v
            x0 = a * x_t - b * model_out
            eps = b * x_t + a * model_out
        return x0, eps

    def p_losses(self, model: GDM, x0: torch.Tensor, cond: torch.Tensor,
                 t: torch.Tensor | None = None, noise: torch.Tensor | None = None,
                 goal: torch.Tensor | None = None):
        """One DDPM training loss on (x0, cond[, goal]). MSE in the prediction
        space (eps or v); Min-SNR-gamma reweighted when min_snr_gamma>0."""
        B = x0.shape[0]
        if t is None:
            t = torch.randint(0, self.timesteps, (B,), device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        model_out = model(x_t, cond, t, goal=goal)
        target = self.target(x0, noise, t)
        if self.min_snr_gamma > 0:
            snr = self.alphas_cumprod[t] / (1.0 - self.alphas_cumprod[t])  # (B,)
            w = torch.clamp(snr, max=self.min_snr_gamma)
            # convert the x0-space Min-SNR weight into the prediction space
            w = w / snr if self.parameterization == "eps" else w / (snr + 1.0)
            w = w.view(-1, *([1] * (x0.ndim - 1)))
            return (w * (model_out - target) ** 2).mean()
        return F.mse_loss(model_out, target)

    @torch.no_grad()
    def ddim_sample(self, model: GDM, cond: torch.Tensor, shape, n_steps: int = 50,
                    eta: float = 0.0, generator: torch.Generator | None = None,
                    goal: torch.Tensor | None = None,
                    noise_scale: float = 1.0) -> torch.Tensor:
        """Deterministic DDIM sampling (eta=0). cond: (B, wg, D) or (B,D).
        shape: (B, N, D). Returns x0 estimate in the DM's (standardized) space.
        noise_scale (gamma) scales every injected noise draw: with eta=0 the
        initial draw is the ONLY stochasticity, so gamma directly modulates the
        drafter's conditional spread at inference (gamma=0 -> deterministic
        mode-seeking sample; gamma=1 -> native; gamma>1 -> inflated spread)."""
        B = shape[0]
        device = cond.device
        x = noise_scale * torch.randn(shape, device=device, generator=generator)
        # evenly spaced subsequence of the full schedule, descending
        step_idx = torch.linspace(self.timesteps - 1, 0, n_steps, device=device).round().long()
        for i in range(n_steps):
            t = step_idx[i]
            t_batch = t.expand(B)
            model_out = model(x, cond, t_batch, goal=goal)
            x0_pred, eps = self.pred_from_model(model_out, x, t_batch)
            if i < n_steps - 1:
                acp_t = self.alphas_cumprod[t]
                acp_prev = self.alphas_cumprod[step_idx[i + 1]]
                sigma = eta * torch.sqrt((1 - acp_prev) / (1 - acp_t) * (1 - acp_t / acp_prev))
                dir_xt = torch.sqrt(1 - acp_prev - sigma ** 2) * eps
                x = torch.sqrt(acp_prev) * x0_pred + dir_xt
                if eta > 0:
                    x = x + sigma * noise_scale * torch.randn(shape, device=device, generator=generator)
            else:
                x = x0_pred
        return x

    @torch.no_grad()
    def ddpm_sample(self, model: GDM, cond: torch.Tensor, shape, n_steps=None,
                    generator: torch.Generator | None = None,
                    goal: torch.Tensor | None = None, clip_x0: bool = False,
                    noise_scale: float = 1.0) -> torch.Tensor:
        """Full DDPM ANCESTRAL sampling over all `timesteps` reverse steps (the LDP /
        Diffusion-Policy default; pair with a small T, e.g. 100). Stochastic: draws
        posterior noise at every step except t=0. Works for eps or v (both decoded via
        pred_from_model). `n_steps` is ignored (DDPM visits every timestep). clip_x0
        clamps the predicted x0 to [-1,1] each step (valid only for minmax-normalized
        data, matching Diffusion Policy). Returns x0 estimate in the DM's space."""
        B = shape[0]
        device = cond.device
        x = noise_scale * torch.randn(shape, device=device, generator=generator)
        for t in range(self.timesteps - 1, -1, -1):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            model_out = model(x, cond, t_batch, goal=goal)
            x0, _ = self.pred_from_model(model_out, x, t_batch)
            if clip_x0:
                x0 = x0.clamp(-1.0, 1.0)
            acp_t = self.alphas_cumprod[t]
            acp_prev = self.alphas_cumprod_prev[t]
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            # posterior q(x_{t-1}|x_t,x0) mean and variance (Ho et al. 2020)
            coef_x0 = torch.sqrt(acp_prev) * beta_t / (1.0 - acp_t)
            coef_xt = torch.sqrt(alpha_t) * (1.0 - acp_prev) / (1.0 - acp_t)
            mean = coef_x0 * x0 + coef_xt * x
            if t > 0:
                var = beta_t * (1.0 - acp_prev) / (1.0 - acp_t)
                noise = noise_scale * torch.randn(shape, device=device, generator=generator)
                x = mean + torch.sqrt(var.clamp_min(0.0)) * noise
            else:
                x = mean
        return x


# Standardization (native E-space <-> DM space) + checkpoint bundle
class GDMPlanner:
    """Bundles the trained GDM, the diffusion process, and the per-dim
    standardization stats. Converts native E-space latents to the DM's
    standardized space, samples, and inverts back to native E-space.

    `sample_next(z_cond_native, ...)` returns the IMMEDIATELY NEXT subgoal
    z_{m+1} in native E-space (||z||~13.9) for closed-loop replanning.
    """

    def __init__(self, model: GDM, diffusion: GaussianDiffusion,
                 stat_a: torch.Tensor, stat_b: torch.Tensor, device="cpu",
                 normalization: str = "standardize"):
        assert normalization in ("standardize", "minmax"), normalization
        self.model = model.to(device).eval()
        self.diffusion = diffusion.to(device)
        # standardize: (a,b)=(mean,std). minmax->[-1,1]: (a,b)=(min,max).
        self.stat_a = stat_a.to(device).view(1, -1)
        self.stat_b = stat_b.to(device).view(1, -1)
        self.normalization = normalization
        self.mean = self.stat_a          # back-compat aliases (standardize only)
        self.std = self.stat_b
        self.device = device
        self.cfg = model.cfg
        self.goal_cond = getattr(model.cfg, "goal_cond", False)

    def standardize(self, z_native: torch.Tensor) -> torch.Tensor:
        z = z_native.to(self.device)
        if self.normalization == "minmax":
            rng = (self.stat_b - self.stat_a).clamp_min(1e-6)
            return 2.0 * (z - self.stat_a) / rng - 1.0
        return (z - self.stat_a) / self.stat_b

    def unstandardize(self, z_std: torch.Tensor) -> torch.Tensor:
        if self.normalization == "minmax":
            return (z_std + 1.0) / 2.0 * (self.stat_b - self.stat_a) + self.stat_a
        return z_std * self.stat_b + self.stat_a

    @torch.no_grad()
    def sample_sequence(self, z_cond_native: torch.Tensor, n_steps: int = 50,
                        generator: torch.Generator | None = None,
                        z_goal_native: torch.Tensor | None = None) -> torch.Tensor:
        """z_cond_native: (B, D) current latent in native E-space.
        Returns (B, N, D) predicted future subgoals in native E-space.
        z_goal_native: (B, D) goal latent (native), used only if goal_cond."""
        B = z_cond_native.shape[0]
        cond = self.standardize(z_cond_native)                       # (B, D)
        goal = (self.standardize(z_goal_native)
                if (self.goal_cond and z_goal_native is not None) else None)
        shape = (B, self.cfg.n_future, self.cfg.latent_dim)
        # gamma dose-response knob: set `planner.noise_scale = gamma` (default 1.0
        # = native behavior, bit-identical to the pre-knob code path)
        gamma = float(getattr(self, "noise_scale", 1.0))
        if getattr(self.diffusion, "sampler", "ddim") == "ddpm":
            seq_std = self.diffusion.ddpm_sample(
                self.model, cond, shape, generator=generator, goal=goal,
                clip_x0=(self.normalization == "minmax"),            # clip valid iff [-1,1] data
                noise_scale=gamma)
        else:
            seq_std = self.diffusion.ddim_sample(self.model, cond, shape, n_steps=n_steps,
                                                 generator=generator, goal=goal,
                                                 noise_scale=gamma)
        return self.unstandardize(seq_std)                           # (B, N, D)

    @torch.no_grad()
    def sample_next(self, z_cond_native: torch.Tensor, n_steps: int = 50,
                    generator: torch.Generator | None = None,
                    z_goal_native: torch.Tensor | None = None) -> torch.Tensor:
        """Return the immediately-next subgoal z_{m+1}, native E-space (B, D)."""
        return self.sample_sequence(z_cond_native, n_steps=n_steps, generator=generator,
                                    z_goal_native=z_goal_native)[:, 0]


def save_gdm(path, model: GDM, diffusion: GaussianDiffusion, stat_a: torch.Tensor,
             stat_b: torch.Tensor, normalization: str = "standardize",
             extra: dict | None = None):
    """Persist everything needed to reconstruct a GDMPlanner. stat_a/stat_b are
    (mean,std) for standardize or (min,max) for minmax."""
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
                       "normalization": normalization},
        "extra": extra or {},
    }, path)


def load_gdm_planner(path, device="cpu") -> GDMPlanner:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = GDMConfig(**ckpt["gdm_config"])
    model = GDM(cfg)
    model.load_state_dict(ckpt["model_state"])
    d = ckpt["diffusion"]
    diffusion = GaussianDiffusion(
        timesteps=d["timesteps"], beta_start=d["beta_start"], beta_end=d["beta_end"],
        schedule=d.get("schedule", "linear"),                 # back-compat: old ckpts
        parameterization=d.get("parameterization", "eps"),    # were eps/linear/no-snr/ddim
        min_snr_gamma=d.get("min_snr_gamma", 0.0),
        sampler=d.get("sampler", "ddim"), device=device)
    ns = ckpt["norm_stats"]
    if "a" in ns:                                             # new format
        return GDMPlanner(model, diffusion, ns["a"], ns["b"], device=device,
                          normalization=ns.get("normalization", "standardize"))
    return GDMPlanner(model, diffusion, ns["mean"], ns["std"], device=device)  # old ckpts


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
