"""Subgoal injection layer: cost model, subgoal sources, and the FF-JEPA policy.

SubgoalCostModel scores rollouts against info_dict['subgoal_emb'] instead of the
goal image, reducing to the flat LeWM cost when the two are equal. FFJEPAPolicy
injects a per-env subgoal at every replan boundary from a pluggable source
(oracle, GDM, regressor, Surveyor, DSpark, lerp, horizon-gated, c*-retire).
Injected latents must be native encoder outputs; nothing is renormalized.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from surveyor import encoder


# Cost model: swap the goal-latent source, keep LeWM's rollout + criterion math
class SubgoalCostModel(nn.Module):
    """Terminal L2^2 from the frozen predictor rollout to info_dict['subgoal_emb'].

    An nn.Module only so CEMSolver can read parameter dtype/device; owns no
    parameters itself.
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
        """(B,S,L,D) x (B,D) -> (B,S). Math-identical to LeWM.criterion."""
        pred_terminal = predicted_emb[..., -1, :]          # (B,S,D)
        return ((pred_terminal - z.unsqueeze(1)) ** 2).sum(dim=-1)  # (B,S)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Terminal L2^2 to the subgoal latent, ignoring info_dict['goal'].

        Mutates info_dict, as LeWM.get_cost does, so the 'emb' cache survives
        across CEM iterations.
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
# `needs_obs` asks FFJEPAPolicy to encode the current frame at replan and pass it
# as obs_latent (closed-loop); the oracle is precomputed and ignores it.
class OracleSubgoalSource:
    """Per-env table of TRUE demo subgoal latents.

    Row i of current(sg_steps) is subgoal[clamp(sg_steps[i], 0, K_i-1)]."""

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


class VerifiedOracleSource:
    """TRUE demo waypoints consumed the way Surveyor consumes drafts.

    Advances to the next waypoint only when the ACHIEVED latent verifies
    against the current one (rel L2 <= tau), re-anchored at every replan, so
    this is the clean subgoal-serving ceiling. OracleSubgoalSource instead
    advances on a schedule regardless of achievement and goes stale, making its
    SR a contaminated lower bound. ptr starts at 1: table[0] is the start frame,
    not a target."""

    needs_obs = True
    needs_goal = False

    def __init__(self, table: list[torch.Tensor], device="cpu", tau: float = 0.2):
        self.table = [t.to(device) for t in table]
        self.device = device
        self.n_envs = len(table)
        self.dim = table[0].shape[1]
        self.tau = float(tau)
        self._ptr = np.ones(self.n_envs, dtype=int)
        self.n_advance = 0
        self.n_hold = 0

    @torch.no_grad()
    def current(self, sg_steps, obs_latent=None, replan_idx=None, goal_latent=None) -> torch.Tensor:
        if replan_idx is not None and len(replan_idx) > 0 and obs_latent is not None:
            z_now = obs_latent.to(self.device)
            for r, i in enumerate(replan_idx):
                K = self.table[i].shape[0]
                advanced = False
                while self._ptr[i] < K - 1:
                    tgt = self.table[i][self._ptr[i]]
                    rel = float((z_now[r] - tgt).norm() / z_now[r].norm().clamp_min(1e-8))
                    if rel <= self.tau:
                        self._ptr[i] += 1          # reached -> next waypoint
                        advanced = True
                    else:
                        break
                if advanced:
                    self.n_advance += 1
                else:
                    self.n_hold += 1
        rows = [self.table[i][min(self._ptr[i], self.table[i].shape[0] - 1)]
                for i in range(self.n_envs)]
        return torch.stack(rows, dim=0)


class GDMSubgoalSource:
    """Every-step drafting: condition the GDM planner on the achieved latent at
    each replan and sample the next subgoal z_{m+1} in native encoder space.
    Cached per env, deterministic given the seeded generator.
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
    """Deterministic z_cond -> m+1 subgoal source (train_regressor.py checkpoint).

    The regression arm of the diffusion-vs-regression comparison. Same current()
    contract as GDMSubgoalSource, re-anchored at every replan."""

    needs_obs = True
    needs_goal = False

    def __init__(self, ckpt_path, n_envs, device="cpu", record=False):
        from surveyor.train_regressor import BlockRegressor
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


class SurveyorSource:
    """Speculative subgoal consumption with reality as the verifier.

    At each replan the achieved latent is checked against the subgoal last
    pursued. Within tau (relative L2) the next pre-drafted block position is
    served with no diffusion call; on rejection or block exhaustion the N-block
    is re-drafted from the achieved state. Goal-conditioned drafters work here,
    but verification is goal-free. goal_gate=True serves a waypoint only if it
    reduces latent distance to the goal and substitutes the goal latent
    otherwise, an arrival filter for goal-near tasks."""

    needs_obs = True

    def __init__(self, planner, n_envs, device="cpu", n_steps=50, seed=42,
                 tau=0.2, record=False, goal_gate=False,
                 readout=None, readout_tau=None, draft_noise=0.0,
                 random_reject=None):
        # readout: optional gap-maximizing verification lens (learn_readout.py).
        # Verification-side ONLY, so drafting/cost/planner stay in native space:
        # accept iff ||r(z_now) - r(tgt)|| <= readout_tau (unit-norm distance).
        self.readout = readout
        self.readout_tau = None if readout_tau is None else float(readout_tau)
        self.planner = planner
        self.goal_gate = bool(goal_gate)
        self.needs_goal = getattr(planner, "goal_cond", False) or self.goal_gate
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
        # calibration event log (prereg/2026-07-31_certification.md M1), record-gated:
        # one (env, rel, accepted, z_achieved, target) per VERIFICATION event, so
        # accepts are logged too (the draft-only trace cannot reconstruct the
        # achieved latent of an accepted replan).
        self.cal = []
        # corruption sweep (prereg/2026-07-31_certification.md): displace every drafted
        # waypoint w by draft_noise * ||w|| along a random unit direction, at
        # EVERY draft including re-drafts (deployed semantics). 0.0 = off.
        # Verification and traces see the corrupted, served block.
        self.draft_noise = float(draft_noise)
        # matched-rate random-rejection control (prereg/2026-08-01_random_rejection.md):
        # reject every verification event with this probability, drawn from a
        # DEDICATED coin generator so the draft-sampling stream matches a normal
        # spec run at the same seed.
        self.random_reject = None if random_reject is None else float(random_reject)
        self._coin_gen = torch.Generator()
        self._coin_gen.manual_seed(int(seed) + 777001)
        self._gated = np.zeros(n_envs, dtype=bool)  # env is serving the goal, not a waypoint
        self.n_redraft = 0   # diffusion calls
        self.n_advance = 0   # positions served from the queue (calls skipped)
        self.n_reject = 0    # verification failures (subset of redrafts)
        self.n_gate = 0      # goal-progress failures: replans served with the raw goal
        # per-replan event log for filmstrips: exactly ONE (env, kind, rel) per
        # env per current() call, so the k-th event of env i aligns with the
        # k-th strip frame the policy captured for env i.
        self.events = []

    @torch.no_grad()
    def current(self, sg_steps, obs_latent=None, replan_idx=None, goal_latent=None) -> torch.Tensor:
        if replan_idx is not None and len(replan_idx) > 0 and obs_latent is not None:
            z_now = obs_latent.to(self.device)               # (R, dim) achieved latents
            z_goal_rows = (goal_latent.to(self.device)
                           if (self.goal_gate and goal_latent is not None) else None)

            def _prog(cand, r):
                # goal-progress test on the metric CEM plans with (latent L2)
                return float((cand - z_goal_rows[r]).norm()) \
                    < float((z_now[r] - z_goal_rows[r]).norm())

            need_rows, need_envs = [], []
            for r, i in enumerate(replan_idx):
                q, tgt = self._queue[i], self._target[i]

                if z_goal_rows is not None and self._gated[i] and q is not None:
                    # goal-fallback: consume-and-discard through the stale block
                    # (zero diffusion), resuming waypoints on the first pass
                    if self._ptr[i] < self.N:
                        cand = q[self._ptr[i]]
                        self._ptr[i] += 1
                        if _prog(cand, r):
                            self._gated[i] = False
                            self._target[i] = cand
                            self._cache[i] = cand
                            self.n_advance += 1
                            self.events.append((int(i), "advance", float("nan")))
                        else:
                            self._cache[i] = z_goal_rows[r]
                            self.n_gate += 1
                            self.events.append((int(i), "gate", float("nan")))
                        continue
                    need_rows.append(r)   # block exhausted -> fresh draft
                    need_envs.append(i)
                    self.events.append((int(i), "redraft", float("nan")))
                    continue

                accept = False
                rel = float("nan")
                if q is not None and tgt is not None:
                    if self.readout is not None:
                        d = float((self.readout(z_now[r]) - self.readout(tgt)).norm())
                        verified = d <= self.readout_tau
                        rel = d
                    else:
                        rel = float((z_now[r] - tgt).norm()
                                    / z_now[r].norm().clamp_min(1e-8))
                        verified = rel <= self.tau
                    if self.random_reject is not None:
                        # coin control: the coin decides, not the latent test.
                        # rel is still computed to keep telemetry comparable
                        verified = bool(float(torch.rand(
                            (), generator=self._coin_gen)) >= self.random_reject)
                    if self.record:
                        self.cal.append((int(i), float(rel), bool(verified),
                                         z_now[r].detach().float().cpu(),
                                         tgt.detach().float().cpu()))
                    if not verified:
                        self.n_reject += 1
                    accept = verified and self._ptr[i] < self.N
                if accept:
                    nxt = self._queue[i][self._ptr[i]]
                    self._ptr[i] += 1
                    if z_goal_rows is not None and not _prog(nxt, r):
                        self._gated[i] = True
                        self._target[i] = None
                        self._cache[i] = z_goal_rows[r]
                        self.n_gate += 1
                        self.events.append((int(i), "gate", float(rel)))
                    else:
                        self._target[i] = nxt
                        self._cache[i] = nxt
                        self.n_advance += 1
                        self.events.append((int(i), "advance", float(rel)))
                else:
                    need_rows.append(r)
                    need_envs.append(i)
                    self.events.append((int(i), "redraft", float(rel)))
            if need_rows:
                z_cond = z_now[need_rows].to(self.planner.device)
                z_goal = (goal_latent[need_rows].to(self.planner.device)
                          if (self.needs_goal and goal_latent is not None
                              and getattr(self.planner, "goal_cond", False)) else None)
                blocks = self.planner.sample_sequence(z_cond, n_steps=self.n_steps,
                                                      generator=self._gen,
                                                      z_goal_native=z_goal)  # (R', N, dim)
                if self.draft_noise > 0:
                    u = torch.randn(blocks.shape, generator=self._gen,
                                    device=blocks.device, dtype=blocks.dtype)
                    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    blocks = blocks + (self.draft_noise
                                       * blocks.norm(dim=-1, keepdim=True) * u)
                blocks = blocks.to(self.device)
                if self.record:
                    self.trace.append({"replan_idx": np.asarray(need_envs),
                                       "z_cond": z_cond.detach().float().cpu(),
                                       "block": blocks.detach().float().cpu()})
                for j, i in enumerate(need_envs):
                    rr = need_rows[j]
                    self._queue[i] = blocks[j]
                    self._ptr[i] = 1
                    self.n_redraft += 1
                    if z_goal_rows is not None and not _prog(blocks[j][0], rr):
                        self._gated[i] = True
                        self._target[i] = None
                        self._cache[i] = z_goal_rows[rr]
                        self.n_gate += 1
                    else:
                        self._gated[i] = False
                        self._target[i] = blocks[j][0]
                        self._cache[i] = blocks[j][0]
        return self._cache.clone()


class LerpSubgoalSource:
    """Straight-line control: the subgoal is a fixed fraction along the
    current-to-goal latent segment, re-anchored each replan. frac=1.0
    degenerates to flat goal planning. A diagnostic arm, not a method."""

    needs_obs = True
    needs_goal = True

    def __init__(self, n_envs, device="cpu", frac=0.5):
        self.device = device
        self.n_envs = n_envs
        self.frac = float(frac)
        self._cache = None  # lazy: dim taken from the first observed latent

    @torch.no_grad()
    def current(self, sg_steps, obs_latent=None, replan_idx=None, goal_latent=None) -> torch.Tensor:
        if self._cache is None:
            ref = obs_latent if obs_latent is not None else goal_latent
            if ref is None:
                raise RuntimeError("LerpSubgoalSource needs obs/goal latents on first call")
            self._cache = torch.zeros(self.n_envs, ref.shape[-1], device=self.device)
        if replan_idx is not None and len(replan_idx) > 0 \
                and obs_latent is not None and goal_latent is not None:
            z_now = obs_latent.to(self.device)
            z_goal = goal_latent.to(self.device)
            tgt = z_now + self.frac * (z_goal - z_now)
            for r, i in enumerate(replan_idx):
                self._cache[i] = tgt[r]
        return self._cache.clone()


@torch.no_grad()
def cem_flat_cstar(lewm, cost_model, z0, z_goal, cem, adim=2):
    """One flat CEM plan per row -> predicted terminal rel to the goal.

    The exact replan-0 call the flat policy would execute, with emb pre-injected
    so the rollout skips the pixel encode. `adim` is the env action dim."""
    import stable_worldmodel as swm
    from gymnasium.spaces import Box

    n = z0.shape[0]
    dev = cost_model.device
    config = swm.PlanConfig(horizon=cem["horizon"],
                            receding_horizon=cem["horizon"],
                            action_block=cem["action_block"])
    # SDPA kernel limit: batch x num_samples rollouts per solve step. Stay inside
    # the validated 128x300 envelope (128x600 and 256x300 both raise CUDA
    # invalid-configuration); CEMSolver chunks n_envs by batch_size internally.
    batch = max(1, min(n, 38400 // int(cem["num_samples"])))
    solver = swm.solver.CEMSolver(model=cost_model, batch_size=batch,
                                  num_samples=cem["num_samples"],
                                  var_scale=cem["var_scale"],
                                  n_steps=cem["n_steps"],
                                  topk=cem["topk"], device=dev,
                                  seed=cem["seed"])
    solver.configure(action_space=Box(low=-1.0, high=1.0, shape=(n, adim),
                                      dtype=np.float32),
                     n_envs=n, config=config)
    z0d, zgd = z0.to(dev), z_goal.to(dev)
    info = {"pixels": torch.zeros(n, 1, 1, 1, 1, device=dev),
            "emb": z0d[:, None, :],
            SubgoalCostModel.SUBGOAL_KEY: zgd}
    plan = solver.solve(info)["actions"].to(dev)       # (n, H, block*adim)
    roll = {"pixels": torch.zeros(n, 1, 1, 1, 1, 1, device=dev),
            "emb": z0d[:, None, None, :]}
    roll = lewm.rollout(roll, plan.unsqueeze(1))       # S=1 candidate
    z_hat = roll["predicted_emb"][:, 0, -1, :]
    return ((z_hat - zgd).norm(dim=-1)
            / z_hat.norm(dim=-1).clamp_min(1e-8)).cpu().numpy()


class CstarRetireSource:
    """Routed Surveyor: draft only while the planner certifies the goal is
    out of reach.

    Every unretired env re-reads c* = rel(z_hat_H, z_goal) from one flat CEM plan
    at each replan. The first time c* <= tau the drafter RETIRES, one-way, and
    the goal latent is served thereafter with zero diffusion. This replaces the
    latent-distance goal_gate, which saturates at range, and the first-replan
    check doubles as the episode router's fire test, so one threshold (the
    verifier's tau) covers every scope. Costs one extra batched CEM solve per
    replan on unretired envs."""

    needs_obs = True
    needs_goal = True

    def __init__(self, planner, lewm, n_envs, device="cpu", n_steps=50, seed=42,
                 tau=0.2, horizon=2, action_block=5, num_samples=300,
                 cem_steps=30, topk=30, var_scale=1.0, adim=2, record=False):
        self.spec = SurveyorSource(planner, n_envs, device=device,
                                            n_steps=n_steps, seed=seed, tau=tau,
                                            record=record)  # latent goal_gate OFF by design
        self.lewm = lewm
        self.cost_model = SubgoalCostModel(lewm)
        self.dim = int(planner.cfg.latent_dim)
        self.device = device
        self.n_envs = n_envs
        self.tau = float(tau)
        self.adim = int(adim)
        self.cem = dict(horizon=int(horizon), action_block=int(action_block),
                        num_samples=int(num_samples), n_steps=int(cem_steps),
                        topk=int(topk), var_scale=float(var_scale), seed=int(seed))
        self._retired = np.zeros(n_envs, dtype=bool)
        self._seen = np.zeros(n_envs, dtype=bool)
        self._replans = np.zeros(n_envs, dtype=int)
        self.retire_replan = np.full(n_envs, -1, dtype=int)
        self.c_first = np.full(n_envs, np.nan)   # first-replan c* == router fire test
        self.c_last = np.full(n_envs, np.nan)    # most recent c* per env
        self._cache = torch.zeros(n_envs, self.dim, device=device)
        self.record = record
        self.trace = {"c_first": self.c_first, "retire_replan": self.retire_replan,
                      "spec": self.spec.trace}

    @torch.no_grad()
    def current(self, sg_steps, obs_latent=None, replan_idx=None, goal_latent=None) -> torch.Tensor:
        if replan_idx is not None and len(replan_idx) > 0 \
                and obs_latent is not None and goal_latent is not None:
            replan_idx = list(replan_idx)
            z_goal = goal_latent.to(self.device)

            # per-replan retirement check, batched over unretired rows
            live = [r for r, i in enumerate(replan_idx) if not self._retired[i]]
            if live:
                cs = cem_flat_cstar(self.lewm, self.cost_model,
                                    obs_latent[live], goal_latent[live],
                                    self.cem, adim=self.adim)
                for j, r in enumerate(live):
                    i = replan_idx[r]
                    self.c_last[i] = cs[j]
                    if not self._seen[i]:
                        self.c_first[i] = cs[j]
                        self._seen[i] = True
                    if cs[j] <= self.tau:
                        self._retired[i] = True
                        self.retire_replan[i] = self._replans[i]

            # retired envs: the goal latent, every replan, zero drafter calls
            for r, i in enumerate(replan_idx):
                if self._retired[i]:
                    self._cache[i] = z_goal[r]
                self._replans[i] += 1

            # live envs: plain Surveyor (rows re-aligned)
            sub = [(r, i) for r, i in enumerate(replan_idx) if not self._retired[i]]
            if sub:
                rows = [r for r, _ in sub]
                envs = [i for _, i in sub]
                out = self.spec.current(sg_steps, obs_latent=obs_latent[rows],
                                        replan_idx=envs,
                                        goal_latent=goal_latent[rows])
                for i in envs:
                    self._cache[i] = out[i]
        return self._cache.clone()


class HorizonGatedSource:
    """Episode-level planner-reachability gate (gate v3): route flat vs spec.

    At each env's first replan, one flat CEM plan toward the goal latent gives
    c*, the predicted terminal discrepancy of the plan the policy would execute.
    c* <= tau serves the goal latent for the whole episode (zero drafter calls);
    otherwise the env is delegated to an internal SurveyorSource. One
    decision per env, never revisited. c* predicts per-episode flat success
    (AUC 0.85 to 0.92), so fire-rate tracks flat-solvability instead of
    vanishing with horizon.
    """

    needs_obs = True
    needs_goal = True

    def __init__(self, planner, lewm, n_envs, device="cpu", n_steps=50, seed=42,
                 tau=0.2, horizon=2, action_block=5, num_samples=300,
                 cem_steps=30, topk=30, var_scale=1.0, record=False):
        self.spec = SurveyorSource(planner, n_envs, device=device,
                                            n_steps=n_steps, seed=seed, tau=tau,
                                            record=record)
        self.lewm = lewm
        self.cost_model = SubgoalCostModel(lewm)
        self.dim = int(planner.cfg.latent_dim)
        self.device = device
        self.n_envs = n_envs
        self.tau = float(tau)
        self.cem = dict(horizon=int(horizon), action_block=int(action_block),
                        num_samples=int(num_samples), n_steps=int(cem_steps),
                        topk=int(topk), var_scale=float(var_scale), seed=int(seed))
        self._decided = np.zeros(n_envs, dtype=bool)
        self._fired = np.zeros(n_envs, dtype=bool)
        self._cache = torch.zeros(n_envs, self.dim, device=device)
        self.c_star = np.full(n_envs, np.nan)
        self.record = record
        self.trace = {"c_star": self.c_star, "fired": self._fired,
                      "spec": self.spec.trace}

    @torch.no_grad()
    def _flat_c_star(self, z0, z_goal):
        """One flat CEM plan per row -> predicted terminal rel to the goal."""
        return cem_flat_cstar(self.lewm, self.cost_model, z0, z_goal,
                              self.cem, adim=2)

    @torch.no_grad()
    def current(self, sg_steps, obs_latent=None, replan_idx=None, goal_latent=None) -> torch.Tensor:
        if replan_idx is not None and len(replan_idx) > 0 \
                and obs_latent is not None and goal_latent is not None:
            replan_idx = list(replan_idx)
            z_goal = goal_latent.to(self.device)

            # episode-start gate: decide once per env, one batched solve
            new_rows = [r for r, i in enumerate(replan_idx) if not self._decided[i]]
            if new_rows:
                cs = self._flat_c_star(obs_latent[new_rows], goal_latent[new_rows])
                for j, r in enumerate(new_rows):
                    i = replan_idx[r]
                    self.c_star[i] = cs[j]
                    self._fired[i] = cs[j] <= self.tau
                    self._decided[i] = True

            # fired envs: the goal latent, every replan, zero drafter calls
            for r, i in enumerate(replan_idx):
                if self._fired[i]:
                    self._cache[i] = z_goal[r]

            # unfired envs: whole episode on Surveyor (rows re-aligned)
            sub = [(r, i) for r, i in enumerate(replan_idx) if not self._fired[i]]
            if sub:
                rows = [r for r, _ in sub]
                envs = [i for _, i in sub]
                out = self.spec.current(sg_steps, obs_latent=obs_latent[rows],
                                        replan_idx=envs,
                                        goal_latent=goal_latent[rows])
                for i in envs:
                    self._cache[i] = out[i]
        return self._cache.clone()


def load_dspark_heads(path, device):
    """Load a train_dspark_head.py checkpoint -> (DSparkHead, ConfidenceHead, ckpt)."""
    from surveyor.dspark.dspark_head import DSparkHead, ConfidenceHead
    ck = torch.load(path, map_location="cpu", weights_only=False)
    D = ck["dim"]
    head = DSparkHead(D, ck["mean"], ck["std"], mode=ck.get("mode", "causal"))
    head.load_state_dict(ck["head_state"]); head.to(device).eval()
    conf = ConfidenceHead(D, ck["mean"], ck["std"])
    conf.load_state_dict(ck["conf_state"]); conf.to(device).eval()
    return head, conf, ck


class DSparkSubgoalSource:
    """Confidence-scheduled commitment with semi-autoregressive refinement over
    the frozen GDM's drafted block.

    The confidence head sets the commit depth (adaptive or fixed) and committed
    subgoals are consumed one per replan, so re-drafting runs once per depth.
    Reduces to GDMSubgoalSource at fixed depth 1.
    """

    needs_obs = True
    needs_goal = False

    def __init__(self, gdm_planner, dspark_path, n_envs, device="cpu", n_steps=50,
                 seed=42, commit="adaptive", theta=0.5, fixed_k=None, use_sts=True,
                 chain_n=None, refine=True):
        from surveyor.dspark.dspark_head import commit_depth
        self._commit_depth = commit_depth
        self.planner = gdm_planner
        # Shadow the class attribute per planner, exactly as GDMSubgoalSource
        # does. PushT's drafter is goal-free and this stays False, leaving the
        # goal-free path byte-identical; Reacher's is goal-conditioned, where
        # the drafter asserts if no goal latent reaches it.
        self.needs_goal = getattr(gdm_planner, "goal_cond", False)
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
    def _draft_block(self, zc, z_goal=None):
        """Draft a length-block_n block. When block_n > native_n, AR-chain the
        native-N drafter by re-conditioning on its own last latent each hop.

        z_goal is the episode goal, so it is held fixed across hops while the
        conditioning latent advances; it is ignored unless the planner is
        goal-conditioned (drafter.sample_sequence gates on its own goal_cond)."""
        if self.block_n <= self.native_n:
            return self.planner.sample_sequence(zc, n_steps=self.n_steps,
                                                generator=self._gen,
                                                z_goal_native=z_goal)[:, :self.block_n]
        hops = (self.block_n + self.native_n - 1) // self.native_n
        blocks, cond = [], zc
        for _ in range(hops):
            blk = self.planner.sample_sequence(cond, n_steps=self.n_steps,
                                               generator=self._gen,
                                               z_goal_native=z_goal)
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
                # goal_latent is (R, D) aligned with replan_idx, as in
                # GDMSubgoalSource, so `need` (positions into replan_idx)
                # selects each env's own goal.
                z_goal = (goal_latent[need].to(self.planner.device)
                          if (self.needs_goal and goal_latent is not None) else None)
                block = self._draft_block(zc, z_goal)                          # (Rn, block_n, D)
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
    """Encode demo subgoal frames at start, start+stride, ..., start+goal_offset.

    Returns one (K_i, 192) tensor per episode, via the harness goal-encode path."""
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
    """Mixin-style wrapper over WorldModelPolicy. Built by `make_ffjepa_policy`
    so the swm base class resolves at call time (pip-installed in the environment, source
    checkout on CPU)."""


def make_ffjepa_policy(base_cls):
    """Return an FFJEPAPolicy class subclassing `base_cls` (WorldModelPolicy)."""

    class _FFJEPAPolicy(base_cls):
        def __init__(self, *, cost_model, subgoal_source, time_instrument=False,
                    dump_frames=False, dump_strip=0, **wmp_kwargs):
            # base WorldModelPolicy.__init__(solver, config, process, transform, ...)
            super().__init__(**wmp_kwargs)
            self.type = "ffjepa"
            self.cost_model = cost_model
            self.subgoal_source = subgoal_source
            self._sg_step = None  # per-env subgoal index, init in set_env
            # optional wall-clock instrumentation, zero overhead when off:
            # t_drafter = frame-encode + subgoal_source.current(), t_cem = the CEM
            # solve inside super().get_action(). CUDA ops are async, so every
            # boundary is synchronized before the timestamp is taken.
            self.time_instrument = time_instrument
            self.t_drafter = 0.0
            self.t_cem = 0.0
            self._timed_steps = 0
            # optional per-env frame capture for the random-init visual sanity
            # check, a few numpy copies per step when opted in. The FIRST
            # info_dict['pixels']/['goal'] per env is the raw reset frame, since
            # get_action runs before env.step at t=0. _frame_last is refreshed
            # every call, so its final value is the true terminal frame: the env
            # freezes once terminated flips under reset_mode='wait'.
            self.dump_frames = dump_frames
            self._frame_start = None
            self._frame_goal = None
            self._frame_last = None
            self._captured_start = None
            # per-replan frame capture for qualitative filmstrips. dump_strip =
            # M > 0 captures, for envs 0..M-1, the raw frame at EVERY replan
            # boundary (the achieved state the verifier sees), tagged with the
            # replan ordinal. The k-th strip frame of env i matches the k-th
            # current() event of env i: both append exactly once per replan.
            self.dump_strip = int(dump_strip)
            self._strip = None

        def set_env(self, env):
            super().set_env(env)
            n = getattr(env, "num_envs", 1)
            self._sg_step = np.zeros(n, dtype=np.int64)
            if self.dump_frames:
                self._frame_start = [None] * n
                self._frame_goal = [None] * n
                self._frame_last = [None] * n
                self._captured_start = np.zeros(n, dtype=bool)
            if self.dump_strip:
                self._strip = [[] for _ in range(min(self.dump_strip, n))]

        def reset_subgoals(self):
            if self._sg_step is not None:
                self._sg_step[:] = 0

        def _frame_kwargs(self, frames, gframes):
            """Raw replan frames for sources that certify OUTSIDE the LeWM latent
            space (surveyor.paired). Opt-in via `wants_frames`, so every other
            source keeps its call signature."""
            if not getattr(self.subgoal_source, "wants_frames", False):
                return {}
            return {"frames": frames, "goal_frames": gframes}

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

            if self._strip is not None and replan:
                pixels_all = np.asarray(info_dict["pixels"])
                for i in replan:
                    if i < len(self._strip):
                        fr = pixels_all[i]
                        if fr.ndim == 4:      # history dim -> take latest
                            fr = fr[-1]
                        self._strip[i].append(
                            (int(self._sg_step[i]), np.array(fr, copy=True)))

            # closed-loop sources need the achieved frame latent at replan: encode
            # the raw env frame with the frozen E, the same path the goal takes.
            # Oracle sources are precomputed and set needs_obs=False.
            if not self.time_instrument:
                obs_latent = None
                goal_latent = None
                frames = gframes = None
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
                                                replan_idx=replan, goal_latent=goal_latent,
                                                **self._frame_kwargs(frames, gframes))  # (n,192)
                info_dict = {**info_dict, SubgoalCostModel.SUBGOAL_KEY: z}
                return super().get_action(info_dict, **kwargs)

            # timed path (--time-instrument): identical logic, wrapped with
            # perf_counter() at the drafter/CEM boundary, CUDA-synced first.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t0 = time.perf_counter()

            obs_latent = None
            goal_latent = None
            frames = gframes = None
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
                                            replan_idx=replan, goal_latent=goal_latent,
                                            **self._frame_kwargs(frames, gframes))
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
