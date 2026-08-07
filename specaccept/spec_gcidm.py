"""Certified speculative execution: drafted waypoints tracked by GC-IDM.

SURVEYOR's consumption layer with the CEM solver replaced by the amortised
GC-IDM controller (gcidm.py): the drafter proposes a block of N waypoint
latents, GC-IDM tracks the current waypoint at one MLP forward per step, and at
every subgoal boundary (S env steps) the accept rule verifies the achieved
latent against the waypoint just pursued -- within tau the next pre-drafted
waypoint is served free, otherwise the block is re-drafted from reality.

Mechanism bet (prereg/2026-08-07_composite_and_executor.md, P-EXEC-1): GC-IDM
degrades with goal DISTANCE (its paper's Table 2), and the drafter's job is
manufacturing NEAR goals -- a 10-step hop sits mid-distribution for the
H_max=50 checkpoint (h = 10/50). The accept rule never inspects the executor,
so tau, k and S all carry unchanged; no CEM runs anywhere in this arm.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from specaccept import encoder


class SpecGCIDMPolicy:
    """Mirrors GCIDMPolicy's swm contract (set_env / get_action) and
    SpecAcceptSubgoalSource's accept semantics (queue / ptr / target /
    redraft-on-reject-or-exhaustion), minus the solver. Counter names match the
    source so harvesters read this arm unchanged."""

    def __init__(self, gcidm_model, planner, lewm, *, sg_steps=10, tau=0.20,
                 n_steps=3, seed=42, device="cuda", action_scaler=None):
        self.model = gcidm_model
        self.planner = planner
        self.lewm = lewm
        self.device = device
        self.sg_steps = int(sg_steps)
        self.tau = float(tau)
        self.n_steps = int(n_steps)
        self.N = int(planner.cfg.n_future)
        self.dim = int(planner.cfg.latent_dim)
        self.needs_goal = getattr(planner, "goal_cond", False)
        self.action_scaler = action_scaler
        self._gen = torch.Generator(device=planner.device)
        self._gen.manual_seed(int(seed))
        self.type = "specgcidm"
        self.env = None
        self.n_redraft = 0   # diffusion calls
        self.n_advance = 0   # positions served from the queue (calls skipped)
        self.n_reject = 0    # verification failures (subset of redrafts)
        self.n_calls = 0     # executor forwards, for the cost column
        self.t_draft = 0.0   # wall-clock in the drafter (s)
        self.t_exec = 0.0    # wall-clock in encoder+executor (s)
        self.events = []     # (env, kind, rel) per boundary, filmstrip contract

    def set_env(self, env):
        self.env = env
        n = getattr(env, "num_envs", 1)
        self._t = np.zeros(n, dtype=np.int64)
        self._queue = [None] * n
        self._ptr = np.zeros(n, dtype=int)
        self._target = torch.zeros(n, self.dim, device=self.device)
        self._has_target = np.zeros(n, dtype=bool)

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
              if (self.needs_goal and z_goal is not None) else None)
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
        t0 = time.perf_counter()
        n = self.env.num_envs
        frames = self._latest(info_dict["pixels"])
        z_t = encoder.encode_frames(self.lewm, frames, device=self.device).to(self.device)
        z_goal = None
        if self.needs_goal:
            goals = self._latest(info_dict["goal"])
            z_goal = encoder.encode_frames(self.lewm, goals, device=self.device).to(self.device)
        self.t_exec += time.perf_counter() - t0

        boundary = (self._t % self.sg_steps == 0)
        need = list(np.nonzero(boundary & ~self._has_target)[0])  # first draft
        for i in np.nonzero(boundary & self._has_target)[0]:
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

        # the executor's clock: the pursued waypoint is `rem` steps away, the
        # same "goal is h steps out" semantics GC-IDM trained on.
        t1 = time.perf_counter()
        rem = self.sg_steps - (self._t % self.sg_steps)
        h = self.model.normalise_horizon(rem).to(self.device)
        a = self.model(z_t, self._target, h)
        self.n_calls += n
        a = a.float().cpu().numpy()
        if self.action_scaler is not None:
            a = a * np.asarray(self.action_scaler["scale"]) + np.asarray(
                self.action_scaler["mean"])
        self.t_exec += time.perf_counter() - t1
        self._t += 1
        return a.reshape(*self.env.action_space.shape).astype(np.float32)
