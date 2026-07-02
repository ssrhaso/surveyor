"""TASK B - injection wrapper + FF-JEPA policy (no training).

Three pieces that sit on top of the FROZEN LeWM + the existing swm planning stack:

* SubgoalCostModel - thin wrapper over LeWM. Its get_cost IGNORES the goal image
  and instead reads a per-env subgoal latent from info_dict['subgoal_emb'],
  using it as the goal latent for LeWM.criterion. Reduces EXACTLY to flat-LeWM
  cost when subgoal_emb == E(goal_image).

* FFJEPAPolicy(WorldModelPolicy) - at each 25-step replan boundary, advances each
  env's subgoal index and injects the per-env subgoal latent into info_dict so
  the existing policy-slice + solver-expand carries it to get_cost aligned per-env.
  The subgoal SOURCE is pluggable (oracle now, GDM in Task C).

* OracleSubgoalSource / build_oracle_table - the oracle source: the TRUE demo
  latents at start, start+H, ..., start+goal_offset (the goal frame), encoded with
  the frozen E. subgoal[1] == the baseline's goal latent (short-horizon).

Representation contract: the injected latent must be the IDENTICAL object E(goal_image)
produces (same encode path, dtype, device, native ||z||~13.9). No renorm/projection.
build_oracle_table uses ffjepa.lewm_io.encode_frames, which is verified to match the
harness goal-encode path to ~1e-6.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ffjepa import lewm_io


# Cost model: swap the goal-latent source, keep LeWM's rollout + criterion math
class SubgoalCostModel(nn.Module):
    """Wrap a frozen LeWM; cost = terminal L2^2 of the predictor rollout to a
    per-env subgoal latent carried in info_dict['subgoal_emb'].

    The class is an nn.Module so CEMSolver can read `next(model.parameters())`
    for dtype/device. It owns no parameters of its own (LeWM stays frozen).
    """

    SUBGOAL_KEY = "subgoal_emb"

    def __init__(self, lewm: nn.Module):
        super().__init__()
        self.lewm = lewm  # frozen

    @property
    def device(self):
        return next(self.lewm.parameters()).device

    @property
    def _dtype(self):
        return next(self.lewm.parameters()).dtype

    @staticmethod
    def _terminal_l2sq(predicted_emb: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Terminal L2^2 of each candidate's final predicted latent vs subgoal z.

        Matches LeWM.criterion's math exactly: F.mse_loss(pred[...,-1:,:],
        goal[...,-1:,:], reduction='none').sum over (T=1, D) == sum_d (pred-z)^2.
        predicted_emb: (B,S,L,D); z: (B,D) -> cost (B,S).
        """
        pred_terminal = predicted_emb[..., -1, :]          # (B,S,D)
        return ((pred_terminal - z.unsqueeze(1)) ** 2).sum(dim=-1)  # (B,S)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """info_dict values are (B, S, ...) as expanded by CEMSolver. Reads the
        per-env subgoal latent, rolls out the frozen predictor under the
        candidates (reusing LeWM.rollout), and returns terminal L2^2 to the
        subgoal. IGNORES info_dict['goal'] (the goal image).

        Cost is computed directly (not via LeWM.criterion) so behavior is
        independent of swm-version criterion-shape conventions; the math is
        identical to LeWM.criterion (terminal L2^2). MUTATES info_dict (like
        LeWM.get_cost) so rollout's 'emb' cache persists across CEM iterations.
        """
        assert self.SUBGOAL_KEY in info_dict, (
            f"info_dict missing '{self.SUBGOAL_KEY}'; FFJEPAPolicy must inject it"
        )
        z = info_dict[self.SUBGOAL_KEY]
        if not torch.is_tensor(z):
            z = torch.as_tensor(z)
        z = z.to(device=self.device, dtype=self._dtype)
        # (B,S,192) expanded by solver -> take one sample (all identical); (B,192) as-is
        if z.ndim == 3:
            z = z[:, 0]
        elif z.ndim != 2:
            z = z.reshape(z.shape[0], -1)

        info_dict = self.lewm.rollout(info_dict, action_candidates)
        return self._terminal_l2sq(info_dict["predicted_emb"], z)


# Subgoal sources (pluggable)
# Common API: current(sg_steps, obs_latent=None, replan_idx=None) -> (n_envs, 192).
# `needs_obs` tells FFJEPAPolicy whether to encode the current frame at replan and
# pass it as obs_latent (closed-loop). The oracle ignores obs_latent (precomputed).
class OracleSubgoalSource:
    """Per-env table of TRUE demo subgoal latents. `current(sg_steps)` returns
    (n_envs, 192) where row i = subgoal[clamp(sg_steps[i], 0, K_i-1)]."""

    needs_obs = False
    needs_goal = False

    def __init__(self, table: list[torch.Tensor], device="cpu"):
        # table[i]: (K_i, 192) latents at start, start+H, ..., goal frame
        self.table = [t.to(device) for t in table]
        self.device = device
        self.n_envs = len(table)
        self.dim = table[0].shape[1]

    def current(self, sg_steps, obs_latent=None, replan_idx=None, goal_latent=None) -> torch.Tensor:
        rows = []
        for i in range(self.n_envs):
            k = int(np.clip(sg_steps[i], 0, self.table[i].shape[0] - 1))
            rows.append(self.table[i][k])
        return torch.stack(rows, dim=0)  # (n_envs, 192)


class GDMSubgoalSource:
    """GOAL-FREE subgoal source: a trained GDM latent diffusion planner.

    At each replan boundary the policy encodes the CURRENT env frame (the achieved
    state) with the frozen E and hands it here as `obs_latent`. We condition the
    DM on it, sample, and select the IMMEDIATELY NEXT subgoal z_{m+1} (paper
    fig.1 / SS9.5). The result is in native E-space (||z||~13.9; the planner
    inverts its standardization), so it drops straight into SubgoalCostModel.

    Subgoals are sampled only at replan and cached per-env; `current` returns the
    cache between replans (the cost model is only queried at replan anyway).
    Sampling is deterministic given a seeded generator.
    """

    needs_obs = True

    def __init__(self, planner, n_envs, dim=192, device="cpu", n_steps=50, seed=42):
        self.planner = planner
        self.device = device
        self.n_envs = n_envs
        self.dim = dim
        self.n_steps = n_steps
        self.needs_goal = getattr(planner, "goal_cond", False)  # ablation: goal-cond GDM
        self._cache = torch.zeros(n_envs, dim, device=device)
        self._gen = torch.Generator(device=planner.device)
        self._gen.manual_seed(int(seed))

    @torch.no_grad()
    def current(self, sg_steps, obs_latent=None, replan_idx=None, goal_latent=None) -> torch.Tensor:
        if replan_idx is not None and len(replan_idx) > 0 and obs_latent is not None:
            z_cond = obs_latent.to(self.planner.device)            # (R, dim) native E-space
            z_goal = (goal_latent.to(self.planner.device)
                      if (self.needs_goal and goal_latent is not None) else None)
            z_next = self.planner.sample_next(z_cond, n_steps=self.n_steps,
                                              generator=self._gen,
                                              z_goal_native=z_goal)  # (R, dim) native E-space
            z_next = z_next.to(self.device)
            for r, i in enumerate(replan_idx):
                self._cache[i] = z_next[r]
        return self._cache.clone()


@torch.no_grad()
def build_oracle_table(h5_path, model, episodes_idx, start_steps, goal_offset,
                       stride=25, device="cpu", batch_size=256):
    """Encode demo subgoal frames for each eval episode at rows
    start, start+stride, ..., start+goal_offset (the goal frame), per env.

    Returns list of (K_i, 192) tensors with K_i = goal_offset//stride + 1.
    Uses lewm_io.encode_frames (matches the harness goal-encode path).
    """
    import h5py

    n_sg = goal_offset // stride + 1
    with h5py.File(h5_path, "r") as f:
        ep_off = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
        pixels = f["pixels"]
        # gather all rows (per env: n_sg frames), clamped to last valid demo frame
        per_env_rows = []
        flat_rows = []
        for ep, start in zip(episodes_idx, start_steps):
            base = int(ep_off[ep]); last = base + int(ep_len[ep]) - 1
            rows = [min(base + int(start) + k * stride, last) for k in range(n_sg)]
            per_env_rows.append(rows)
            flat_rows.extend(rows)
        flat_rows = np.array(flat_rows)
        # h5 fancy indexing needs increasing order -> sort, encode, scatter back
        order = np.argsort(flat_rows, kind="stable")
        sorted_rows = flat_rows[order]
        lat_sorted = lewm_io.encode_frames(model, pixels[sorted_rows], device=device,
                                           batch_size=batch_size)
        lat = torch.empty_like(lat_sorted)
        lat[order] = lat_sorted  # undo the sort
    table = []
    j = 0
    for rows in per_env_rows:
        table.append(lat[j:j + len(rows)].clone())
        j += len(rows)
    return table


# Policy: advance subgoal at each replan boundary, inject into info_dict
class FFJEPAPolicy:
    """Mixin-style wrapper requiring WorldModelPolicy as base. Constructed by
    `make_ffjepa_policy` so the swm base class is resolved at call time (the box
    has it pip-installed; CPU uses the source checkout)."""


def make_ffjepa_policy(base_cls):
    """Return an FFJEPAPolicy class subclassing `base_cls` (WorldModelPolicy)."""

    class _FFJEPAPolicy(base_cls):
        def __init__(self, *, cost_model, subgoal_source, **wmp_kwargs):
            # base WorldModelPolicy.__init__(solver, config, process, transform, ...)
            super().__init__(**wmp_kwargs)
            self.type = "ffjepa"
            self.cost_model = cost_model
            self.subgoal_source = subgoal_source
            self._sg_step = None  # per-env subgoal index, init in set_env

        def set_env(self, env):
            super().set_env(env)
            n = getattr(env, "num_envs", 1)
            self._sg_step = np.zeros(n, dtype=np.int64)

        def reset_subgoals(self):
            if self._sg_step is not None:
                self._sg_step[:] = 0

        def get_action(self, info_dict, **kwargs):
            assert hasattr(self, "env"), "Environment not set for the policy"
            n = self.env.num_envs
            # mirror base replan detection (dataset eval: mode='wait', no needs_flush)
            terminated = info_dict.get("terminated")
            dead = (np.asarray(terminated, dtype=bool)
                    if terminated is not None else np.zeros(n, dtype=bool))
            replan = [i for i in range(n)
                      if len(self._action_buffer[i]) == 0 and not dead[i]]
            for i in replan:
                self._sg_step[i] += 1  # 0 -> 1 on first replan => target subgoal[1]

            # closed-loop sources (GDM) need the CURRENT achieved frame latent at
            # replan: encode the raw env frame with the frozen E (== goal-encode
            # path, native E-space). Oracle sources set needs_obs=False.
            obs_latent = None
            goal_latent = None
            if getattr(self.subgoal_source, "needs_obs", False) and replan:
                frames = np.asarray(info_dict["pixels"])[replan]  # (R,[T,]H,W,3) uint8
                if frames.ndim == 5:                              # history dim -> take latest
                    frames = frames[:, -1]
                obs_latent = lewm_io.encode_frames(
                    self.cost_model.lewm, frames, device=self.cost_model.device)
                # goal-cond ablation: also encode the raw goal image (same E-path)
                if getattr(self.subgoal_source, "needs_goal", False) and "goal" in info_dict:
                    gframes = np.asarray(info_dict["goal"])[replan]
                    if gframes.ndim == 5:
                        gframes = gframes[:, -1]
                    goal_latent = lewm_io.encode_frames(
                        self.cost_model.lewm, gframes, device=self.cost_model.device)

            z = self.subgoal_source.current(self._sg_step, obs_latent=obs_latent,
                                            replan_idx=replan, goal_latent=goal_latent)  # (n,192)
            info_dict = {**info_dict, SubgoalCostModel.SUBGOAL_KEY: z}
            return super().get_action(info_dict, **kwargs)

    return _FFJEPAPolicy
