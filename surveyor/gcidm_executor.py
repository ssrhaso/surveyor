"""Drafted waypoints tracked by GC-IDM.

SURVEYOR's consumption layer with the CEM solver replaced by the amortised
GC-IDM controller (gcidm.py): the drafter proposes a block of N waypoint
latents, GC-IDM tracks the current waypoint at one MLP forward per step, and at
every subgoal boundary (S env steps) the accept rule verifies the achieved
latent against the waypoint just pursued: within tau the next pre-drafted
waypoint is served free, otherwise the block is re-drafted from reality.

Mechanism: GC-IDM degrades with goal distance
(its paper's Table 2) while the drafter manufactures near goals, and a 10-step
hop sits mid-distribution for the H_max=50 checkpoint. The accept rule never
inspects the executor, so tau, k and S carry unchanged and no CEM runs in the
bare arm.

cstar_route=True adds the certified scope: at each env's first
boundary one flat CEM probe reads c* = rel(z_hat_H, z_goal), the planner's own
certificate at the same tau, and c* <= tau routes the episode to plain GC-IDM
(goal served directly, zero drafter calls). Drafting envs carry an arrival gate
at every boundary: rel(z, z_goal) <= tau retires the drafter one-way (Cube's
gate, no new constant). Cost is one batched CEM solve per episode; execution
stays fully amortised.
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
                 goal_gate=False, random_reject=None, paired_encoder=None,
                 imagine=False, action_proc=None, imagine_block=5,
                 probe_solver="cem", retire_tau=None):
        self.model = gcidm_model
        # Two-Room: drafts live in the 576-d [lewm | dino] space; the accept
        # test reads the DINOv2 half, the executor the LeWM half
        self.penc = paired_encoder
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
        # arbiter settings (patch P, 2026-09-17): which planner computes the
        # reachability probe, "cem" (the shared flat-CEM probe; the arrival gate
        # retires) or "match" (the executor's own forecast, _own_probe, routes
        # and retires), and the tolerance of the retirement test, which is also
        # the routing test at the first boundary (retire_tau = tau by default)
        self.probe_solver = str(probe_solver)
        self.retire_tau = float(tau if retire_tau is None else retire_tau)
        # goal_gate=True enables the arrival gate WITHOUT the c* route (no CEM
        # probe, no cost model); the gate below fires on either flag. Cube's
        # confirmed CEM arm is gated, so its executor twin must be.
        self.goal_gate = bool(goal_gate)
        # goal frames must be encoded if the drafter is goal-conditioned OR the
        # certificate / arrival gate needs the goal latent
        self.needs_goal = self.goal_cond or self.cstar_route or self.goal_gate
        self.budget = None if budget is None else int(budget)
        self.action_scaler = action_scaler
        self._gen = torch.Generator(device=planner.device)
        self._gen.manual_seed(int(seed))
        # coin control (rate-matched random rejector): when set, the coin
        # decides serve-or-redraft at every boundary and the latent test is
        # only logged, exactly as SurveyorSource does over CEM
        self.random_reject = None if random_reject is None else float(random_reject)
        self._coin_gen = torch.Generator()
        self._coin_gen.manual_seed(int(seed) + 777001)
        self.type = "gcidm+surveyor"
        self.env = None
        if self.cstar_route:
            if cem is None or self.budget is None:
                raise ValueError("cstar_route needs cem=dict(...) and budget=")
            from surveyor.sources import SubgoalCostModel
            self.cost_model = SubgoalCostModel(lewm)
            self.cem = dict(cem)
            self.adim = int(adim)
            self.probe_block = int(self.cem["action_block"])
            print(f"[gcidm+surveyor] c* probe solver="
                  f"{'gcidm' if self.probe_solver == 'match' else 'cem'} "
                  f"retire_tau={self.retire_tau:g} (verifier tau={self.tau:g})")
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
        # imagination-check control: the boundary tests read the predictor's
        # forecast (last boundary latent + the S executed actions) instead of
        # the achieved latent; action_proc maps env units into the world
        # model's standardised action space (the eval's StandardScaler)
        self.imagine = bool(imagine)
        self.action_proc = action_proc
        self.imagine_block = int(imagine_block)
        self.imag_log = []
        if self.imagine:
            assert paired_encoder is None, "imagination check is not wired for paired drafts"
            assert self.sg_steps % self.imagine_block == 0, (self.sg_steps, self.imagine_block)

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
        self._z_b = None     # latent at the last boundary (imagination check)
        self._abuf = []      # env-unit actions executed since that boundary

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

    def _exec_target(self):
        """Waypoint the executor tracks: the LeWM half when drafts are paired."""
        if self.penc is None:
            return self._target
        from surveyor.paired import split_paired
        return split_paired(self._target)[0]

    def _verify_target(self, i):
        """Waypoint the accept test reads: the DINOv2 half when drafts are paired."""
        if self.penc is None:
            return self._target[i]
        from surveyor.paired import split_paired
        return split_paired(self._target[i])[1]

    @torch.no_grad()
    def _imagined(self, v_t):
        """Forecast of this boundary's latent from the last boundary's latent over
        the S executed actions; logs (imagined, achieved) rel per drafting env."""
        from surveyor.sources import imagine_terminal
        a = np.stack(self._abuf, axis=1).astype(np.float32)     # (n, S, adim) env units
        n, S, d = a.shape
        if self.action_proc is not None:
            a = self.action_proc.transform(a.reshape(n * S, d)).reshape(n, S, d)
        b = self.imagine_block
        plan = torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32).reshape(n, S // b, b * d)
        zhat = imagine_terminal(self.lewm, self._z_b, plan).to(v_t.device)
        for i in range(n):
            if self._has_target[i] and not self._direct[i]:
                w = self._verify_target(i)
                self.imag_log.append((
                    float((zhat[i] - w).norm() / zhat[i].norm().clamp_min(1e-8)),
                    float((v_t[i] - w).norm() / v_t[i].norm().clamp_min(1e-8))))
        return zhat

    @torch.no_grad()
    def _probe(self, z_t, z_goal, rows):
        """Reachability probe c* for `rows`: the shared flat-CEM plan (probe_solver
        "cem") or the executor's own forecast (probe_solver "match")."""
        rows = np.asarray(list(rows), dtype=np.int64)
        if self.probe_solver == "match":
            return self._own_probe(z_t, z_goal, rows)
        from surveyor.sources import cem_flat_cstar
        idx = torch.as_tensor(rows, device=z_t.device)
        cs = cem_flat_cstar(self.lewm, self.cost_model, z_t[idx], z_goal[idx],
                            self.cem, adim=self.adim)
        return np.asarray(cs, dtype=np.float64).reshape(-1)

    @torch.no_grad()
    def _own_probe(self, z_t, z_goal, rows):
        """The executor's own reachability probe: the frozen predictor's forecast of
        GC-IDM's closed-loop trajectory toward the goal over its remaining budget
        (clamped at H_max), the policy re-queried at every raw step of each predictor
        block from the forecast latent, read as rel(z_hat_T, z_goal). No search: one
        policy forward per raw step and one predictor step per block, batched."""
        from surveyor.sources import imagine_terminal
        rows = np.asarray(list(rows), dtype=np.int64)
        idx = torch.as_tensor(rows, device=z_t.device)
        z = z_t[idx].to(self.device).clone()
        zg = z_goal[idx].to(self.device)
        if self.budget is not None:
            left = self.budget - self._t[rows]
        else:
            left = np.full(len(rows), self.model.h_max)
        left = np.clip(np.asarray(left, dtype=np.int64), 1, int(self.model.h_max))
        blk = int(self.probe_block)
        n_blocks = int(np.ceil(left.max() / blk))
        for b in range(n_blocks):
            live = np.nonzero(left > b * blk)[0]
            if len(live) == 0:
                break
            live_t = torch.as_tensor(live, device=z.device)
            zl, zgl = z[live_t], zg[live_t]
            acts = []
            for s in range(blk):
                rem = np.maximum(left[live] - b * blk - s, 1)
                h = self.model.normalise_horizon(rem).to(self.device)
                a = self.model(zl, zgl, h).float().cpu().numpy()
                if self.action_scaler is not None:
                    a = a * np.asarray(self.action_scaler["scale"]) + np.asarray(
                        self.action_scaler["mean"])
                acts.append(np.asarray(a, dtype=np.float32).reshape(len(live), -1))
            a = np.stack(acts, axis=1).astype(np.float32)       # (m, blk, adim) env units
            m, S, d = a.shape
            if self.action_proc is not None:
                a = self.action_proc.transform(a.reshape(m * S, d)).reshape(m, S, d)
            plan = torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32).reshape(m, 1, S * d)
            z[live_t] = imagine_terminal(self.lewm, zl, plan).to(z.device)
        return ((z - zg).norm(dim=-1) / z.norm(dim=-1).clamp_min(1e-8)).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def get_action(self, info_dict, **kwargs):
        """Serve one GC-IDM action per env, verifying and redrafting at boundaries.

        Between boundaries this is plain GC-IDM tracking the current waypoint; on
        a boundary the achieved latent is verified against that waypoint and the
        block is served on or redrafted. With `cstar_route` the first step also
        runs the flat CEM probe that decides the episode's scope. With a
        `paired_encoder` (Two-Room) the drafter works in the 576-d paired space:
        the executor and the probe read the LeWM half, the accept test and the
        arrival gate read the DINOv2 half.
        """
        t0 = time.perf_counter()
        n = self.env.num_envs
        frames = self._latest(info_dict["pixels"])
        z_goal = v_goal = zg_cond = None
        if self.penc is None:
            z_t = encoder.encode_frames(self.lewm, frames, device=self.device).to(self.device)
            v_t, zc_cond = z_t, z_t
            if self.needs_goal:
                goals = self._latest(info_dict["goal"])
                z_goal = encoder.encode_frames(self.lewm, goals, device=self.device).to(self.device)
                v_goal, zg_cond = z_goal, z_goal
        else:
            from surveyor.paired import split_paired
            zc_cond = self.penc.encode(frames).to(self.device)          # (n, 576)
            z_t, v_t = split_paired(zc_cond)
            if self.needs_goal:
                goals = self._latest(info_dict["goal"])
                zg_cond = self.penc.encode(goals).to(self.device)
                z_goal, v_goal = split_paired(zg_cond)
        self.t_exec += time.perf_counter() - t0

        # certificate, episode scope: one batched flat-CEM probe at t=0 only
        if self.cstar_route and int(self._t[0]) == 0:
            tp = time.perf_counter()
            cs = self._probe(z_t, z_goal, np.arange(n))
            self.c_first[:] = cs
            self._direct = cs <= self.retire_tau
            self.n_routed = int(self._direct.sum())
            self.t_probe += time.perf_counter() - tp

        # imagination check: at a boundary the forecast stands in for the
        # achieved latent in the accept test and the arrival gate
        v_ver = v_t
        if (self.imagine and self._z_b is not None and len(self._abuf) == self.sg_steps
                and int(self._t[0]) % self.sg_steps == 0):
            v_ver = self._imagined(v_t)
        boundary = (self._t % self.sg_steps == 0)
        drafting = ~self._direct
        need = list(np.nonzero(boundary & drafting & ~self._has_target)[0])
        cand = np.nonzero(boundary & drafting & self._has_target)[0]
        # probe by the executor: its own forecast, re-read at every boundary on
        # the drafting envs, replaces the arrival gate as the retirement test
        probe_cs = None
        if self.cstar_route and self.probe_solver == "match" and len(cand):
            tp = time.perf_counter()
            probe_cs = dict(zip(cand.tolist(), self._own_probe(z_t, z_goal, cand).tolist()))
            self.t_probe += time.perf_counter() - tp
        for i in cand:
            # retirement test (replan scope), one-way at retire_tau: the executor's
            # own forecast when the probe is matched, else verified arrival at the
            # FINAL goal (the arrival gate, no new constants)
            if self.cstar_route or self.goal_gate:
                if probe_cs is not None:
                    relg = float(probe_cs[int(i)])
                else:
                    relg = float((v_ver[i] - v_goal[i]).norm()
                                 / v_ver[i].norm().clamp_min(1e-8))
                if relg <= self.retire_tau:
                    self._direct[i] = True
                    self.n_arrive += 1
                    self.events.append((int(i), "gate", relg))
                    continue
            w = self._verify_target(i)
            rel = float((v_ver[i] - w).norm() / v_ver[i].norm().clamp_min(1e-8))
            verified = rel <= self.tau
            if self.random_reject is not None:
                verified = bool(float(torch.rand(
                    (), generator=self._coin_gen)) >= self.random_reject)
            if verified and self._ptr[i] < self.N:
                self._target[i] = self._queue[i][self._ptr[i]]
                self._ptr[i] += 1
                self.n_advance += 1
                self.events.append((int(i), "advance", rel))
            else:
                if not verified:
                    self.n_reject += 1
                need.append(int(i))
                self.events.append((int(i), "redraft", rel))
        if need:
            self._draft(need, zc_cond, zg_cond)

        # executor clock: drafting envs count steps to the next boundary (the
        # pursued waypoint is that many steps out); direct envs run plain
        # GC-IDM semantics, remaining episode budget clamped at H_max
        t1 = time.perf_counter()
        rem = self.sg_steps - (self._t % self.sg_steps)
        target = self._exec_target()
        if self._direct.any():
            target = target.clone()
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
        if self.imagine:
            if int(self._t[0]) % self.sg_steps == 0:
                self._z_b = z_t.detach().clone()
                self._abuf = []
            self._abuf.append(np.asarray(a, dtype=np.float32).reshape(n, -1).copy())
        self._t += 1
        return a.reshape(*self.env.action_space.shape).astype(np.float32)
