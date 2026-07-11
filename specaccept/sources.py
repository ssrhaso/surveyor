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
build_oracle_table uses specaccept.encoder.encode_frames, which is verified to match the
harness goal-encode path to ~1e-6.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from specaccept import encoder


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

    def __init__(self, planner, n_envs, dim=192, device="cpu", n_steps=50, seed=42,
                 record=False):
        self.planner = planner
        self.device = device
        self.n_envs = n_envs
        self.dim = dim
        self.n_steps = n_steps
        self.needs_goal = getattr(planner, "goal_cond", False)  # ablation: goal-cond GDM
        self._cache = torch.zeros(n_envs, dim, device=device)
        self._gen = torch.Generator(device=planner.device)
        self._gen.manual_seed(int(seed))
        self.record = record   # keep per-replan (z_cond, z_next) for failure anatomy
        self.trace = []

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
            if self.record:
                self.trace.append({"replan_idx": np.asarray(replan_idx),
                                   "z_cond": z_cond.detach().float().cpu(),
                                   "z_next": z_next.detach().float().cpu()})
            for r, i in enumerate(replan_idx):
                self._cache[i] = z_next[r]
        return self._cache.clone()


class RegressorSubgoalSource:
    """Deterministic z_cond -> m+1 subgoal source (train_regressor.py ckpt).

    The diffusion-vs-regression closed-loop arm: offline this architecture
    dominated the diffusion drafter on latent fidelity; the spread thesis
    (validated on the refiner, -11pp) predicts its zero conditional spread
    makes a WORSE CEM target. Same current() contract as GDMSubgoalSource;
    re-anchors every replan."""

    needs_obs = True
    needs_goal = False

    def __init__(self, ckpt_path, n_envs, device="cpu", record=False):
        from specaccept.train_regressor import BlockRegressor
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model = BlockRegressor(ck["dim"], ck["n_future"], ck["hid"])
        self.model.load_state_dict(ck["state"])
        self.model.to(device).eval()
        self.mean = ck["mean"].to(device)
        self.std = ck["std"].to(device)
        self.dim = int(ck["dim"])
        self.device = device
        self.n_envs = n_envs
        self._cache = torch.zeros(n_envs, self.dim, device=device)
        self.record = record
        self.trace = []

    @torch.no_grad()
    def current(self, sg_steps, obs_latent=None, replan_idx=None, goal_latent=None) -> torch.Tensor:
        if replan_idx is not None and len(replan_idx) > 0 and obs_latent is not None:
            z_cond = obs_latent.to(self.device)
            z_std = (z_cond - self.mean) / self.std
            block = self.model(z_std)                       # (R, N, D) standardized
            z_next = block[:, 0] * self.std + self.mean     # m+1, native E-space
            if self.record:
                self.trace.append({"replan_idx": np.asarray(replan_idx),
                                   "z_cond": z_cond.detach().float().cpu(),
                                   "z_next": z_next.detach().float().cpu()})
            for r, i in enumerate(replan_idx):
                self._cache[i] = z_next[r]
        return self._cache.clone()


class SpecAcceptSubgoalSource:
    """Speculative subgoal consumption with REALITY as the verifier (the DSpark
    draft-verify-accept skeleton with the learned confidence head replaced by
    the achieved latent; DIRECTION_dspark_gdm.md Phase B).

    At each replan boundary the policy hands over the CURRENT achieved latent.
    Verify it against the subgoal we were just driving toward: within `tau`
    (relative L2) -> consume the NEXT position of the block drafted at the last
    re-anchor (no diffusion call); outside `tau`, or block exhausted -> re-draft
    the whole N-block from the achieved state. Re-anchoring is never sacrificed
    when reality diverges; diffusion is only skipped while reality confirms the
    plan. Offline calibration (probe_dspark_gates A2): s5 tau=0.20 / s25
    tau=0.40 give ~1.9x fewer diffusion calls at ~zero staleness penalty.

    Goal-conditioned drafters (Reacher: planner.goal_cond=True) are supported:
    needs_goal mirrors the planner flag and re-drafts pass the per-env goal
    latent through to sample_sequence. Verification itself stays goal-free -
    reality is compared against the subgoal just pursued, same as ever."""

    needs_obs = True

    def __init__(self, planner, n_envs, device="cpu", n_steps=50, seed=42,
                 tau=0.2, record=False):
        self.planner = planner
        self.needs_goal = getattr(planner, "goal_cond", False)
        self.N = int(planner.cfg.n_future)
        self.dim = int(planner.cfg.latent_dim)
        self.device = device
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.tau = float(tau)
        self._queue = [None] * n_envs           # (N, dim) block from the last re-anchor
        self._ptr = np.zeros(n_envs, dtype=int)  # next block position to serve
        self._target = [None] * n_envs           # latent currently being driven toward
        self._cache = torch.zeros(n_envs, self.dim, device=device)
        self._gen = torch.Generator(device=planner.device)
        self._gen.manual_seed(int(seed))
        self.record = record
        self.trace = []
        self.n_redraft = 0   # diffusion calls
        self.n_advance = 0   # positions served from the queue (calls skipped)
        self.n_reject = 0    # verification failures (subset of redrafts)

    @torch.no_grad()
    def current(self, sg_steps, obs_latent=None, replan_idx=None, goal_latent=None) -> torch.Tensor:
        if replan_idx is not None and len(replan_idx) > 0 and obs_latent is not None:
            z_now = obs_latent.to(self.device)               # (R, dim) achieved latents
            need_rows, need_envs = [], []
            for r, i in enumerate(replan_idx):
                q, tgt = self._queue[i], self._target[i]
                accept = False
                if q is not None and tgt is not None:
                    rel = float((z_now[r] - tgt).norm()
                                / z_now[r].norm().clamp_min(1e-8))
                    verified = rel <= self.tau
                    if not verified:
                        self.n_reject += 1
                    accept = verified and self._ptr[i] < self.N
                if accept:
                    nxt = self._queue[i][self._ptr[i]]
                    self._ptr[i] += 1
                    self._target[i] = nxt
                    self._cache[i] = nxt
                    self.n_advance += 1
                else:
                    need_rows.append(r)
                    need_envs.append(i)
            if need_rows:
                z_cond = z_now[need_rows].to(self.planner.device)
                z_goal = (goal_latent[need_rows].to(self.planner.device)
                          if (self.needs_goal and goal_latent is not None) else None)
                blocks = self.planner.sample_sequence(z_cond, n_steps=self.n_steps,
                                                      generator=self._gen,
                                                      z_goal_native=z_goal)  # (R', N, dim)
                blocks = blocks.to(self.device)
                if self.record:
                    self.trace.append({"replan_idx": np.asarray(need_envs),
                                       "z_cond": z_cond.detach().float().cpu(),
                                       "block": blocks.detach().float().cpu()})
                for j, i in enumerate(need_envs):
                    self._queue[i] = blocks[j]
                    self._target[i] = blocks[j][0]
                    self._cache[i] = blocks[j][0]
                    self._ptr[i] = 1
                    self.n_redraft += 1
        return self._cache.clone()


def load_dspark_heads(path, device):
    """Load a train_dspark_head.py checkpoint -> (DSparkHead, ConfidenceHead, ckpt)."""
    from specaccept.dspark.dspark_head import DSparkHead, ConfidenceHead
    ck = torch.load(path, map_location="cpu", weights_only=False)
    D = ck["dim"]
    head = DSparkHead(D, ck["mean"], ck["std"], mode=ck.get("mode", "causal"))
    head.load_state_dict(ck["head_state"]); head.to(device).eval()
    conf = ConfidenceHead(D, ck["mean"], ck["std"])
    conf.load_state_dict(ck["conf_state"]); conf.to(device).eval()
    return head, conf, ck


class DSparkSubgoalSource:
    """The DSpark mechanism for FF-JEPA: confidence-scheduled speculative decoding
    with semi-autoregressive refinement, over the frozen GDM's drafted subgoal block.

    At a replan boundary, if the per-env committed block is exhausted, it RE-DRAFTS:
    the frozen GDM samples the N-block from the CURRENT achieved latent (re-anchor),
    the DSparkHead refines it (semi-AR residual), the ConfidenceHead scores each
    refined position, and the commit depth is set speculatively to
    k* = max{k : Π_{i<=k} c_i > theta} (adaptive) or a fixed k. Those k* subgoals are
    committed to a per-env queue and consumed one per replan (every `stride` steps),
    so the expensive diffusion re-draft runs only every k* replans instead of every
    replan -- decoupling subgoal-resample cadence from CEM's action-replan cadence
    (the "fewer replans at equal fidelity" systems win). Reduces to GDMSubgoalSource
    when commit='fixed' with fixed_k=1.

    Same `current(...)` contract as GDMSubgoalSource; standalone apart from the frozen
    GDM planner + the loaded heads (no coupling to CEM internals).
    """

    needs_obs = True
    needs_goal = False

    def __init__(self, gdm_planner, dspark_path, n_envs, device="cpu", n_steps=50,
                 seed=42, commit="adaptive", theta=0.5, fixed_k=None, use_sts=True,
                 chain_n=None, refine=True):
        from specaccept.dspark.dspark_head import commit_depth
        self._commit_depth = commit_depth
        self.planner = gdm_planner
        self.native_n = gdm_planner.cfg.n_future
        # block_n = length of the (possibly AR-chained) drafted block to work on
        self.block_n = int(chain_n) if chain_n else self.native_n
        self.refine = refine
        if refine:
            self.head, self.conf, ck = load_dspark_heads(dspark_path, device)
            self.sts_temp = ck["sts_temp"].to(device) if (use_sts and "sts_temp" in ck) else None
        else:  # raw open-loop: no refinement, no confidence (commit must be 'fixed')
            self.head = self.conf = self.sts_temp = None
            assert commit == "fixed", "refine=False requires commit='fixed' (no confidence head)"
        self.N = self.block_n
        self.dim = gdm_planner.cfg.latent_dim
        self.device = device
        self.n_steps = n_steps
        self.commit = commit                        # 'adaptive' | 'fixed'
        self.theta = float(theta)
        self.max_commit = self.block_n
        self.fixed_k = min(fixed_k or self.block_n, self.block_n)
        self.n_envs = n_envs
        self._queue = [None] * n_envs               # per-env (k*, dim) committed refined subgoals
        self._ptr = np.zeros(n_envs, dtype=int)
        self._cache = torch.zeros(n_envs, self.dim, device=device)
        self._gen = torch.Generator(device=gdm_planner.device)
        self._gen.manual_seed(int(seed))
        # telemetry
        self.n_redraft = 0
        self.n_advance = 0
        self.commit_depths = []

    @torch.no_grad()
    def _draft_block(self, zc):
        """Draft a length-block_n block, AR-chaining the native-N drafter (re-condition
        on its own last latent each hop) when block_n > native_n."""
        if self.block_n <= self.native_n:
            return self.planner.sample_sequence(zc, n_steps=self.n_steps,
                                                generator=self._gen)[:, :self.block_n]
        hops = (self.block_n + self.native_n - 1) // self.native_n
        blocks, cond = [], zc
        for _ in range(hops):
            blk = self.planner.sample_sequence(cond, n_steps=self.n_steps, generator=self._gen)
            blocks.append(blk); cond = blk[:, -1]
        return torch.cat(blocks, dim=1)[:, :self.block_n]

    @torch.no_grad()
    def current(self, sg_steps, obs_latent=None, replan_idx=None, goal_latent=None) -> torch.Tensor:
        if replan_idx is not None and len(replan_idx) > 0 and obs_latent is not None:
            replan_idx = list(replan_idx)
            # replan envs whose committed block is empty/exhausted -> need a re-draft
            need = [r for r, i in enumerate(replan_idx)
                    if self._queue[i] is None or self._ptr[i] >= self._queue[i].shape[0]]
            if need:
                zc = obs_latent[need].to(self.planner.device)                  # (Rn, D) native
                block = self._draft_block(zc)                                  # (Rn, block_n, D)
                if self.refine:
                    refined = self.head(block, zc)                             # (Rn, block_n, D)
                    cvals = self.conf(refined, block, zc, temperature=self.sts_temp)
                else:
                    refined, cvals = block, None                              # raw block, no confidence
                for j, r in enumerate(need):
                    i = replan_idx[r]
                    if self.commit == "adaptive":
                        k = self._commit_depth(cvals[j], self.theta, self.max_commit)
                    else:
                        k = self.fixed_k
                    self._queue[i] = refined[j, :k].to(self.device).clone()
                    self._ptr[i] = 0
                    self.n_redraft += 1
                    self.commit_depths.append(int(k))
            # consume one committed subgoal per replan env (clamp at last if exhausted)
            for r, i in enumerate(replan_idx):
                q = self._queue[i]
                k = min(self._ptr[i], q.shape[0] - 1)
                self._cache[i] = q[k]
                self._ptr[i] += 1
                self.n_advance += 1
        return self._cache.clone()


@torch.no_grad()
def build_oracle_table(h5_path, model, episodes_idx, start_steps, goal_offset,
                       stride=25, device="cpu", batch_size=256):
    """Encode demo subgoal frames for each eval episode at rows
    start, start+stride, ..., start+goal_offset (the goal frame), per env.

    Returns list of (K_i, 192) tensors with K_i = goal_offset//stride + 1.
    Uses encoder.encode_frames (matches the harness goal-encode path).
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
        lat_sorted = encoder.encode_frames(model, pixels[sorted_rows], device=device,
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
        def __init__(self, *, cost_model, subgoal_source, time_instrument=False,
                    dump_frames=False, **wmp_kwargs):
            # base WorldModelPolicy.__init__(solver, config, process, transform, ...)
            super().__init__(**wmp_kwargs)
            self.type = "ffjepa"
            self.cost_model = cost_model
            self.subgoal_source = subgoal_source
            self._sg_step = None  # per-env subgoal index, init in set_env
            # optional wall-clock instrumentation (additive; zero overhead when off):
            # t_drafter = frame-encode + subgoal_source.current(...); t_cem = the CEM
            # solve inside super().get_action(...). CUDA ops are async, so each
            # boundary is torch.cuda.synchronize()'d before the timestamp is taken.
            self.time_instrument = time_instrument
            self.t_drafter = 0.0
            self.t_cem = 0.0
            self._timed_steps = 0
            # optional per-env frame capture for the random-init visual sanity
            # check (additive; a few numpy copies/step, only when opted in): the
            # FIRST info_dict['pixels']/['goal'] this policy ever sees per env is
            # the raw reset frame (get_action is called BEFORE env.step at t=0),
            # and _frame_last is refreshed every call so the final overwrite --
            # the call where info_dict['terminated'][i] first flips True -- holds
            # the true terminal frame (env freezes afterward under reset_mode='wait').
            self.dump_frames = dump_frames
            self._frame_start = None
            self._frame_goal = None
            self._frame_last = None
            self._captured_start = None

        def set_env(self, env):
            super().set_env(env)
            n = getattr(env, "num_envs", 1)
            self._sg_step = np.zeros(n, dtype=np.int64)
            if self.dump_frames:
                self._frame_start = [None] * n
                self._frame_goal = [None] * n
                self._frame_last = [None] * n
                self._captured_start = np.zeros(n, dtype=bool)

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

            if self.dump_frames:
                pixels_all = np.asarray(info_dict["pixels"])
                goal_all = np.asarray(info_dict["goal"]) if "goal" in info_dict else None
                for i in range(n):
                    frame = pixels_all[i]
                    if frame.ndim == 4:  # history dim -> take latest
                        frame = frame[-1]
                    self._frame_last[i] = np.array(frame, copy=True)
                    if not self._captured_start[i]:
                        self._frame_start[i] = np.array(frame, copy=True)
                        if goal_all is not None:
                            g = goal_all[i]
                            if g.ndim == 4:
                                g = g[-1]
                            self._frame_goal[i] = np.array(g, copy=True)
                        self._captured_start[i] = True

            # closed-loop sources (GDM) need the CURRENT achieved frame latent at
            # replan: encode the raw env frame with the frozen E (== goal-encode
            # path, native E-space). Oracle sources set needs_obs=False.
            if not self.time_instrument:
                obs_latent = None
                goal_latent = None
                if getattr(self.subgoal_source, "needs_obs", False) and replan:
                    frames = np.asarray(info_dict["pixels"])[replan]  # (R,[T,]H,W,3) uint8
                    if frames.ndim == 5:                              # history dim -> take latest
                        frames = frames[:, -1]
                    obs_latent = encoder.encode_frames(
                        self.cost_model.lewm, frames, device=self.cost_model.device)
                    # goal-cond ablation: also encode the raw goal image (same E-path)
                    if getattr(self.subgoal_source, "needs_goal", False) and "goal" in info_dict:
                        gframes = np.asarray(info_dict["goal"])[replan]
                        if gframes.ndim == 5:
                            gframes = gframes[:, -1]
                        goal_latent = encoder.encode_frames(
                            self.cost_model.lewm, gframes, device=self.cost_model.device)

                z = self.subgoal_source.current(self._sg_step, obs_latent=obs_latent,
                                                replan_idx=replan, goal_latent=goal_latent)  # (n,192)
                info_dict = {**info_dict, SubgoalCostModel.SUBGOAL_KEY: z}
                return super().get_action(info_dict, **kwargs)

            # timed path (--time-instrument): identical logic, wrapped with
            # perf_counter() at the drafter/CEM boundary, CUDA-synced first.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t0 = time.perf_counter()

            obs_latent = None
            goal_latent = None
            if getattr(self.subgoal_source, "needs_obs", False) and replan:
                frames = np.asarray(info_dict["pixels"])[replan]
                if frames.ndim == 5:
                    frames = frames[:, -1]
                obs_latent = encoder.encode_frames(
                    self.cost_model.lewm, frames, device=self.cost_model.device)
                if getattr(self.subgoal_source, "needs_goal", False) and "goal" in info_dict:
                    gframes = np.asarray(info_dict["goal"])[replan]
                    if gframes.ndim == 5:
                        gframes = gframes[:, -1]
                    goal_latent = encoder.encode_frames(
                        self.cost_model.lewm, gframes, device=self.cost_model.device)

            z = self.subgoal_source.current(self._sg_step, obs_latent=obs_latent,
                                            replan_idx=replan, goal_latent=goal_latent)
            info_dict = {**info_dict, SubgoalCostModel.SUBGOAL_KEY: z}

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t1 = time.perf_counter()
            self.t_drafter += _t1 - _t0

            result = super().get_action(info_dict, **kwargs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t2 = time.perf_counter()
            self.t_cem += _t2 - _t1
            self._timed_steps += 1
            return result

    return _FFJEPAPolicy
