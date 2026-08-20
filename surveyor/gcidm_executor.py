"""Certified speculative execution: drafted waypoints tracked by GC-IDM.

SURVEYOR's consumption layer with the CEM solver replaced by the amortised
GC-IDM controller (gcidm.py): the drafter proposes a block of N waypoint
latents, GC-IDM tracks the current waypoint at one MLP forward per step, and at
every subgoal boundary (S env steps) the accept rule verifies the achieved
latent against the waypoint just pursued: within tau the next pre-drafted
waypoint is served free, otherwise the block is re-drafted from reality.

Mechanism bet (pre-registered as P-EXEC-1, passed):
GC-IDM degrades with goal distance (its paper's Table 2) while the drafter
manufactures near goals; a 10-step hop sits mid-distribution for the H_max=50
checkpoint. The accept rule never inspects the executor, so tau, k and S carry
unchanged, and no CEM runs in the bare arm.

cstar_route=True adds the certified scope (P-EXEC-6): at each env's first
boundary one flat CEM probe reads c* = rel(z_hat_H, z_goal), the planner's own
certificate at the same tau; c* <= tau routes the episode to plain GC-IDM
(goal served directly, zero drafter calls). Drafting envs carry an arrival
gate at every boundary: rel(z, z_goal) <= tau retires the drafter one-way
(Cube's gate, no new constant). Cost: one batched CEM solve per episode;
execution stays fully amortised.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from surveyor import encoder


class SurveyorGCIDMPolicy:
    """Mirrors GCIDMPolicy's swm contract (set_env / get_action) and
    SurveyorSource's accept semantics (queue / ptr / target /
    redraft-on-reject-or-exhaustion), minus the solver. Counter names match the
    source so harvesters read this arm unchanged."""

    def __init__(self, gcidm_model, planner, lewm, *, sg_steps=10, tau=0.20,
                 n_steps=3, seed=42, device="cuda", action_scaler=None,
                 budget=None, cstar_route=False, cem=None, adim=2,
                 goal_gate=False):
        self.model = gcidm_model
        self.planner = planner
        self.lewm = lewm
        self.device = device
        self.sg_steps = int(sg_steps)
        self.tau = float(tau)
        self.n_steps = int(n_steps)
        self.N = int(planner.cfg.n_future)
        self.dim = int(planner.cfg.latent_dim)
        self.goal_cond = getattr(planner, "goal_cond", False)
        self.cstar_route = bool(cstar_route)
        # goal_gate=True enables the arrival gate WITHOUT the c* route (no CEM
        # probe, no cost model): the drafting-env gate below fires on either
        # flag. Cube's confirmed CEM arm is gated, so its executor twin must be.
        self.goal_gate = bool(goal_gate)
        # goal frames must be encoded if the drafter is goal-conditioned OR the
        # certificate / arrival gate needs the goal latent
        self.needs_goal = self.goal_cond or self.cstar_route or self.goal_gate
        self.budget = None if budget is None else int(budget)
        self.action_scaler = action_scaler
        self._gen = torch.Generator(device=planner.device)
        self._gen.manual_seed(int(seed))
        self.type = "gcidm+surveyor"
        self.env = None
        if self.cstar_route:
            if cem is None or self.budget is None:
                raise ValueError("cstar_route needs cem=dict(...) and budget=")
            from surveyor.sources import SubgoalCostModel
            self.cost_model = SubgoalCostModel(lewm)
            self.cem = dict(cem)
            self.adim = int(adim)
        self.n_redraft = 0   # diffusion calls
        self.n_advance = 0   # positions served from the queue (calls skipped)
        self.n_reject = 0    # verification failures (subset of redrafts)
        self.n_calls = 0     # executor forwards, for the cost column
        self.n_routed = 0    # envs routed direct at the first boundary (c*)
        self.n_arrive = 0    # drafting envs retired by the arrival gate
        self.t_draft = 0.0   # wall-clock in the drafter (s)
        self.t_exec = 0.0    # wall-clock in encoder+executor (s)
        self.t_probe = 0.0   # wall-clock in the c* probe (s)
        self.c_first = None  # per-env first-boundary c*, for the router audit
        self.events = []     # (env, kind, rel) per boundary, filmstrip contract

    def set_env(self, env):
        """Bind the vector env and allocate the per-env block, pointer and route state."""
        self.env = env
        n = getattr(env, "num_envs", 1)
        self._t = np.zeros(n, dtype=np.int64)
        self._queue = [None] * n
        self._ptr = np.zeros(n, dtype=int)
        self._target = torch.zeros(n, self.dim, device=self.device)
        self._has_target = np.zeros(n, dtype=bool)
        self._direct = np.zeros(n, dtype=bool)   # serving the goal, not waypoints
        self.c_first = np.full(n, np.nan)

    @staticmethod
    def _latest(arr):
        """(N,[T,]H,W,3) -> (N,H,W,3); the eval harness may carry a history dim."""
        a = np.asarray(arr)
        return a[:, -1] if a.ndim == 5 else a

    @torch.no_grad()
    def _draft(self, rows, z_now, z_goal):
        t0 = time.perf_counter()
        z_cond = z_now[rows].to(self.planner.device)
        zg = (z_goal[rows].to(self.planner.device)
              if (self.goal_cond and z_goal is not None) else None)
        blocks = self.planner.sample_sequence(z_cond, n_steps=self.n_steps,
                                              generator=self._gen,
                                              z_goal_native=zg)  # (R, N, dim)
        blocks = blocks.to(self.device)
        for j, i in enumerate(rows):
            self._queue[i] = blocks[j]
            self._ptr[i] = 1
            self._target[i] = blocks[j][0]
            self._has_target[i] = True
            self.n_redraft += 1
        self.t_draft += time.perf_counter() - t0

    @torch.no_grad()
    def get_action(self, info_dict, **kwargs):
        """Serve one GC-IDM action per env, verifying and redrafting at boundaries.

        Between boundaries this is plain GC-IDM tracking the current waypoint. On
        a boundary the achieved latent is verified against that waypoint, and the
        block is served on or redrafted. With `cstar_route` the first step also
        runs the flat CEM probe that decides the episode's scope.
        """
        t0 = time.perf_counter()
        n = self.env.num_envs
        frames = self._latest(info_dict["pixels"])
        z_t = encoder.encode_frames(self.lewm, frames, device=self.device).to(self.device)
        z_goal = None
        if self.needs_goal:
            goals = self._latest(info_dict["goal"])
            z_goal = encoder.encode_frames(self.lewm, goals, device=self.device).to(self.device)
        self.t_exec += time.perf_counter() - t0

        # certificate, episode scope: one batched flat-CEM probe at t=0 only
        if self.cstar_route and int(self._t[0]) == 0:
            tp = time.perf_counter()
            from surveyor.sources import cem_flat_cstar
            cs = cem_flat_cstar(self.lewm, self.cost_model, z_t, z_goal,
                                self.cem, adim=self.adim)
            cs = np.asarray(cs, dtype=np.float64).reshape(-1)
            self.c_first[:] = cs
            self._direct = cs <= self.tau
            self.n_routed = int(self._direct.sum())
            self.t_probe += time.perf_counter() - tp

        boundary = (self._t % self.sg_steps == 0)
        drafting = ~self._direct
        need = list(np.nonzero(boundary & drafting & ~self._has_target)[0])
        for i in np.nonzero(boundary & drafting & self._has_target)[0]:
            # arrival gate (certificate's replan scope, zero new constants):
            # verified arrival at the FINAL goal retires the drafter one-way
            if self.cstar_route or self.goal_gate:
                relg = float((z_t[i] - z_goal[i]).norm()
                             / z_t[i].norm().clamp_min(1e-8))
                if relg <= self.tau:
                    self._direct[i] = True
                    self.n_arrive += 1
                    self.events.append((int(i), "gate", relg))
                    continue
            w = self._target[i]
            rel = float((z_t[i] - w).norm() / z_t[i].norm().clamp_min(1e-8))
            if rel <= self.tau and self._ptr[i] < self.N:
                self._target[i] = self._queue[i][self._ptr[i]]
                self._ptr[i] += 1
                self.n_advance += 1
                self.events.append((int(i), "advance", rel))
            else:
                if rel > self.tau:
                    self.n_reject += 1
                need.append(int(i))
                self.events.append((int(i), "redraft", rel))
        if need:
            self._draft(need, z_t, z_goal)

        # executor clock: drafting envs count steps to the next boundary (the
        # pursued waypoint is that many steps out); direct envs run plain
        # GC-IDM semantics, remaining episode budget clamped at H_max
        t1 = time.perf_counter()
        rem = self.sg_steps - (self._t % self.sg_steps)
        target = self._target
        if self._direct.any():
            target = self._target.clone()
            d = np.nonzero(self._direct)[0]
            if self.budget is not None:
                rem = rem.copy()
                rem[d] = np.maximum(self.budget - self._t[d], 1)
            target[d] = z_goal[d]
        h = self.model.normalise_horizon(rem).to(self.device)
        a = self.model(z_t, target, h)
        self.n_calls += n
        a = a.float().cpu().numpy()
        if self.action_scaler is not None:
            a = a * np.asarray(self.action_scaler["scale"]) + np.asarray(
                self.action_scaler["mean"])
        self.t_exec += time.perf_counter() - t1
        self._t += 1
        return a.reshape(*self.env.action_space.shape).astype(np.float32)
