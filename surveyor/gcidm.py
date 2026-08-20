"""GC-IDM: goal-conditioned inverse dynamics model (arXiv 2605.08732).

Reimplementation of the amortised controller from "Latent Geometry Beyond
Search", built to the paper's stated spec and anchored against its published
cells before being quoted. It replaces search entirely: one MLP forward maps
(current latent, goal latent, remaining horizon) to the next action.

Spec, verbatim from the paper:
  * input  : z_t || z_goal, each 192-d from the FROZEN LeWM encoder
  * trunk  : 3-layer MLP, hidden 512, LayerNorm, GELU, dropout 0.1
  * horizon: h = min(steps_remaining, H_max) / H_max in [0,1], sinusoidal
             encoding (64-d), 2-layer MLP -> c, then AdaLN-Zero modulation
             x * (1 + gamma(c)) + beta(c) before the action head, gamma and
             beta zero-initialised (so the model starts horizon-agnostic)
  * loss   : MSE on actions, ||gcidm(z_t, z_{t+h}, h) - a_t||^2
  * data   : (t, h) sampled per example with h ~ U[1, H_max], triple
             (z_t, z_{t+h}, a_t); same offline demos that trained LeWM
  * H_max  : equals the evaluation budget T (50 in their protocol)
  * serving: closed loop, re-encode every step, one action per step
This lands at ~1.5M parameters, matching the paper's stated size.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from surveyor import encoder


def sinusoidal_embedding(h: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """Standard sinusoidal features for the scalar normalised horizon in [0,1].

    h is (B,); returns (B, dim). Scaled by 1000 before the frequency ladder so a
    quantity confined to the unit interval still spans the spectrum.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=h.device) / half
    )
    ang = (h.float() * 1000.0)[:, None] * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class GCIDM(nn.Module):
    """The paper's network: (z_t, z_goal) trunk, AdaLN-Zero horizon modulation.

    The horizon path is zero-initialised, so training starts from the
    horizon-agnostic model and learns the modulation from there.
    """
    def __init__(self, latent_dim: int = 192, action_dim: int = 2, hidden: int = 512,
                 depth: int = 3, dropout: float = 0.1, horizon_dim: int = 64,
                 h_max: int = 50):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.h_max = int(h_max)

        layers, d_in = [], 2 * latent_dim
        for _ in range(depth):
            layers += [nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(),
                       nn.Dropout(dropout)]
            d_in = hidden
        self.trunk = nn.Sequential(*layers)

        self.horizon_mlp = nn.Sequential(
            nn.Linear(horizon_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
        )
        # AdaLN-Zero: one projection to (gamma, beta), zero-initialised so the
        # network begins as the horizon-agnostic model and learns modulation.
        self.adaln = nn.Linear(hidden, 2 * hidden)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)

        self.head = nn.Linear(hidden, action_dim)
        self.horizon_dim = horizon_dim

        # Paper Appendix F: "Weights are initialized with Kaiming normal; the
        # output head uses a small initialization (sigma=0.01) to start near
        # zero." The AdaLN projection stays zero-init (that is its own rule).
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        for m in self.horizon_mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, z_t: torch.Tensor, z_goal: torch.Tensor,
                h_norm: torch.Tensor) -> torch.Tensor:
        """z_t, z_goal: (B, latent_dim). h_norm: (B,) already in [0,1]."""
        x = self.trunk(torch.cat([z_t, z_goal], dim=-1))
        c = self.horizon_mlp(sinusoidal_embedding(h_norm, self.horizon_dim))
        gamma, beta = self.adaln(c).chunk(2, dim=-1)
        return self.head(x * (1.0 + gamma) + beta)

    def normalise_horizon(self, steps_remaining) -> torch.Tensor:
        """h = min(steps_remaining, H_max) / H_max, the paper's conditioning."""
        sr = torch.as_tensor(steps_remaining, dtype=torch.float32)
        return torch.clamp(sr, max=float(self.h_max)) / float(self.h_max)


def count_params(m: nn.Module) -> int:
    """Total number of parameters in `m`."""
    return sum(p.numel() for p in m.parameters())


def save_gcidm(path, model: GCIDM, action_scaler=None, meta=None):
    """Write weights, architecture config and the action scaler to `path`."""
    torch.save({
        "state": model.state_dict(),
        "cfg": {"latent_dim": model.latent_dim, "action_dim": model.action_dim,
                "hidden": model.head.in_features, "h_max": model.h_max,
                "horizon_dim": model.horizon_dim},
        # actions train in standardised space, so the scaler must travel with
        # the weights or served actions land in the wrong units
        "action_scaler": ({"mean": action_scaler.mean_.tolist(),
                           "scale": action_scaler.scale_.tolist()}
                          if action_scaler is not None else None),
        "meta": meta or {},
    }, path)


def load_gcidm(path, device="cuda"):
    """Load a `save_gcidm` checkpoint -> (frozen model, action scaler, meta)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = GCIDM(latent_dim=cfg["latent_dim"], action_dim=cfg["action_dim"],
                  hidden=cfg["hidden"], h_max=cfg["h_max"],
                  horizon_dim=cfg.get("horizon_dim", 64))
    model.load_state_dict(ck["state"])
    model.to(device).eval()
    model.requires_grad_(False)
    return model, ck.get("action_scaler"), ck.get("meta", {})


class GCIDMPolicy:
    """Closed-loop GC-IDM controller: no solver, one forward pass per step.

    Mirrors swm's BasePolicy contract (set_env / get_action) without subclassing
    it. Frames and goals are encoded by the same path every other arm uses
    (encoder.encode_frames), so nothing privileged enters: pixels in, action out.
    """

    def __init__(self, model: GCIDM, lewm, *, budget: int, device="cuda",
                 action_scaler=None):
        self.model = model
        self.lewm = lewm
        self.device = device
        self.budget = int(budget)
        self.action_scaler = action_scaler
        self.type = "gcidm"
        self.env = None
        self._t = None            # per-env steps taken
        self.n_calls = 0          # forward passes = decisions, for the cost column

    def set_env(self, env):
        """Bind the vector env and reset the per-env step counters."""
        self.env = env
        n = getattr(env, "num_envs", 1)
        self._t = np.zeros(n, dtype=np.int64)

    @staticmethod
    def _latest(arr):
        """(N,[T,]H,W,3) -> (N,H,W,3); the eval harness may carry a history dim."""
        a = np.asarray(arr)
        return a[:, -1] if a.ndim == 5 else a

    @torch.no_grad()
    def get_action(self, info_dict, **kwargs):
        """Encode frames and goals, then emit one action per env.

        The only clock the controller reads is steps remaining, clamped at H_max
        and normalised exactly as in training.
        """
        n = self.env.num_envs
        frames = self._latest(info_dict["pixels"])
        goals = self._latest(info_dict["goal"])

        z_t = encoder.encode_frames(self.lewm, frames, device=self.device)
        z_g = encoder.encode_frames(self.lewm, goals, device=self.device)

        # the controller's only clock: steps left, clamped at H_max, as trained
        remaining = np.maximum(self.budget - self._t, 0)
        h = self.model.normalise_horizon(remaining).to(self.device)

        a = self.model(z_t.to(self.device), z_g.to(self.device), h)
        self.n_calls += n
        a = a.float().cpu().numpy()

        if self.action_scaler is not None:
            a = a * np.asarray(self.action_scaler["scale"]) + np.asarray(
                self.action_scaler["mean"])

        self._t += 1
        return a.reshape(*self.env.action_space.shape).astype(np.float32)
