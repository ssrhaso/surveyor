"""LeFlow (arXiv 2608.24855) as an executor, alone and under the consumption layer.

LeFlow keeps LeWM frozen and replaces CEM with a learned latent path flow plus
an inverse-dynamics decoder: given (z_start, z_goal) it samples latent paths of
H latent steps, decodes each into H x action_block raw actions, reranks by the
LeWM-rolled final latent's distance to the goal, and executes the best. The
released checkpoints are private, so the planner is trained with the authors'
public code (external/LeFlow/train_latent_planner.py) on the frozen LeWM
weights: their defaults, except max_train_batches=2000 per epoch, seed 42 and
an episode split that holds out the evaluation episodes (written by
reproduce/data_prep/make_split.py: PushT drops the episodes listed in
episodes/pusht_c2222.source_episodes*.json, Reacher keeps episodes < 8000).
The payload it writes, leflow/<env>/latent_planner.pt, is what
`LatentPlannerRuntime.from_checkpoint` reads; leflow/<env>/action_stats.json
holds the mean and unbiased std of the dataset's action column (non-NaN rows).

Its encoder is LeWM's own (`external/LeFlow/latent_planner.py::load_lewm` built
from the same weights.pt) and encodes bit-identically to `encoder.encode_frames`
(checked), so drafter waypoints are valid LeFlow goals as they are.

Two policies mirror gcidm.GCIDMPolicy / gcidm_executor.SurveyorGCIDMPolicy:
  LeFlowPolicy          plan toward the goal image every `receding_steps` raw
                        steps (their protocol: H=5 latent steps x action_block
                        5 = 25 raw actions, receding 5 -> replan every 25)
  SurveyorLeFlowPolicy  the drafter's block is consumed under the accept rule at
                        every S-step boundary; between boundaries LeFlow tracks
                        the current waypoint (a plan per boundary, S raw actions
                        of it executed). Counters match the GC-IDM executor so
                        the harvester reads call_ratio unchanged.
LeFlow's inverse dynamics emits actions normalised by the training set's mean
and std (external/LeFlow/utils.py::get_column_normalizer, unbiased std over
non-NaN rows); `leflow/<env>/action_stats.json` stores them and the policies
un-normalise with the same numbers.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from surveyor import encoder

LEFLOW_DIR = Path(__file__).resolve().parents[1] / "external" / "LeFlow"


def _latent_planner():
    if str(LEFLOW_DIR) not in sys.path:
        sys.path.insert(0, str(LEFLOW_DIR))
    import latent_planner  # noqa: WPS433 (the authors' module, imported in place)
    return latent_planner


def load_leflow(ckpt, device="cuda"):
    """The authors' runtime from their checkpoint payload (flow + inverse dynamics
    + the LeWM referenced inside it), frozen, on `device`."""
    lp = _latent_planner()
    rt = lp.LatentPlannerRuntime.from_checkpoint(ckpt, device=device)
    rt.eval()
    for p in rt.parameters():
        p.requires_grad_(False)
    return rt


def load_action_stats(path):
    """{'mean': [...], 'std': [...]} written by leflow_train.sbatch."""
    with open(path) as fh:
        d = json.load(fh)
    return {"mean": np.asarray(d["mean"], dtype=np.float32),
            "std": np.asarray(d["std"], dtype=np.float32)}


class LeFlowPlanner:
    """One LeFlow plan toward a LATENT goal: sample, decode, rerank, return the
    best candidate's raw-action sequence in the normalised action space."""

    def __init__(self, runtime, *, horizon=5, num_samples=64, flow_steps=16,
                 history_size=3, seed=42):
        self.rt = runtime
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.flow_steps = int(flow_steps)
        self.history_size = int(history_size)
        self.gen = torch.Generator(device=runtime.device).manual_seed(int(seed))
        self.n_plans = 0
        self.t_plan = 0.0

    @property
    def steps_per_plan(self):
        return self.horizon * int(self.rt.action_block)

    @torch.no_grad()
    def plan(self, z_start, z_goal):
        """(b, d), (b, d) -> (b, H * action_block, adim) normalised actions."""
        t0 = time.perf_counter()
        rt = self.rt
        b = z_start.shape[0]
        paths = rt.sample_paths(z_start, z_goal, horizon=self.horizon,
                                num_samples=self.num_samples, flow_steps=self.flow_steps,
                                generator=self.gen)                       # (b, S, H+1, d)
        actions = rt.decode_actions(paths)                                 # (b, S, H, ab*adim)
        final = rt.rollout_final_latent(z_start, actions, self.history_size)  # (b, S, d)
        cost = ((final - z_goal[:, None]) ** 2).mean(-1)                   # rollout_goal score
        best = cost.argmin(dim=1)
        a = actions[torch.arange(b, device=actions.device), best]          # (b, H, ab*adim)
        self.n_plans += b
        self.t_plan += time.perf_counter() - t0
        return a.reshape(b, self.steps_per_plan, -1)


def _latest(arr):
    a = np.asarray(arr)
    return a[:, -1] if a.ndim == 5 else a


class LeFlowPolicy:
    """LeFlow alone: replan toward the goal image every `receding_steps` raw steps."""

    def __init__(self, planner: LeFlowPlanner, lewm, *, budget, action_stats,
                 receding_steps=25, device="cuda"):
        self.planner = planner
        self.lewm = lewm
        self.budget = int(budget)
        self.stats = action_stats
        self.receding = int(receding_steps)
        assert self.receding <= planner.steps_per_plan, "receding_steps exceeds one plan"
        self.device = device
        self.type = "leflow"
        self.env = None
        self.n_calls = 0          # raw actions served (decisions)
        self.t_exec = 0.0
        self._t = 0
        self._queue = None
        self._qpos = 0

    def set_env(self, env):
        self.env = env
        self._t = 0
        self._queue = None
        self._qpos = 0

    def _unnorm(self, a):
        return a * self.stats["std"] + self.stats["mean"]

    @torch.no_grad()
    def get_action(self, info_dict, **kwargs):
        t0 = time.perf_counter()
        n = self.env.num_envs
        if self._queue is None or self._t % self.receding == 0:
            frames = _latest(info_dict["pixels"])
            goals = _latest(info_dict["goal"])
            z_t = encoder.encode_frames(self.lewm, frames, device=self.device).to(self.device)
            z_g = encoder.encode_frames(self.lewm, goals, device=self.device).to(self.device)
            a = self.planner.plan(z_t, z_g).float().cpu().numpy()
            self._queue = self._unnorm(a)
            self._qpos = 0
        act = self._queue[:, self._qpos]
        self._qpos += 1
        self._t += 1
        self.n_calls += n
        self.t_exec += time.perf_counter() - t0
        return act.reshape(*self.env.action_space.shape).astype(np.float32)


class SurveyorLeFlowPolicy:
    """Drafted waypoints tracked by LeFlow under the accept rule.

    Same queue / ptr / target semantics and counters as SurveyorGCIDMPolicy:
    at every boundary (S raw steps) the achieved latent is verified against the
    waypoint just pursued (rel L2 <= tau, or the coin when random_reject is
    set): within tau the next pre-drafted waypoint is served free, otherwise the
    block is redrafted from reality. After the boundary LeFlow plans one path
    toward each env's current waypoint and the next S raw actions of it are
    executed. S must not exceed one plan (H x action_block raw steps).

    cstar_route=True adds the certified scope exactly as the GC-IDM executor
    does: one flat-CEM c* probe at t=0 routes an episode to LeFlow alone (the
    goal latent is planned toward at every boundary, zero drafter calls) when
    c* <= tau; drafting envs carry the tau arrival gate. goal_gate=True keeps
    the gate without the probe. A paired_encoder (Two-Room) puts drafting and
    verification in the 576-d paired space: LeFlow and the probe read the LeWM
    half, the accept test and the gate read the DINOv2 half.
    """

    def __init__(self, planner: LeFlowPlanner, drafter, lewm, *, sg_steps=10, tau=0.20,
                 n_steps=3, seed=42, device="cuda", action_stats=None, budget=None,
                 random_reject=None, cstar_route=False, cem=None, adim=2,
                 goal_gate=False, paired_encoder=None,
                 imagine=False, action_proc=None, imagine_block=5):
        self.planner = planner
        self.drafter = drafter
        self.lewm = lewm
        self.device = device
        self.sg_steps = int(sg_steps)
        assert self.sg_steps <= planner.steps_per_plan, "sg_steps exceeds one LeFlow plan"
        self.tau = float(tau)
        self.n_steps = int(n_steps)
        self.N = int(drafter.cfg.n_future)
        self.dim = int(drafter.cfg.latent_dim)
        self.goal_cond = getattr(drafter, "goal_cond", False)
        self.cstar_route = bool(cstar_route)
        self.goal_gate = bool(goal_gate)
        self.needs_goal = self.goal_cond or self.cstar_route or self.goal_gate
        self.penc = paired_encoder
        self.budget = None if budget is None else int(budget)
        self.stats = action_stats
        self._gen = torch.Generator(device=drafter.device)
        self._gen.manual_seed(int(seed))
        self.random_reject = None if random_reject is None else float(random_reject)
        self._coin_gen = torch.Generator()
        self._coin_gen.manual_seed(int(seed) + 777001)
        self.type = "leflow+surveyor"
        self.env = None
        if self.cstar_route:
            if cem is None:
                raise ValueError("cstar_route needs cem=dict(...)")
            from surveyor.sources import SubgoalCostModel
            self.cost_model = SubgoalCostModel(lewm)
            self.cem = dict(cem)
            self.adim = int(adim)
        self.n_calls = 0      # raw actions served
        self.n_redraft = 0    # diffusion calls
        self.n_advance = 0    # waypoints served free
        self.n_reject = 0     # verification failures
        self.n_routed = 0     # envs routed to LeFlow alone at t=0 (c*)
        self.n_arrive = 0     # drafting envs retired by the arrival gate
        self.t_draft = 0.0
        self.t_exec = 0.0
        self.t_probe = 0.0
        self.c_first = None
        self.events = []
        # imagination-check control (see SurveyorGCIDMPolicy)
        self.imagine = bool(imagine)
        self.action_proc = action_proc
        self.imagine_block = int(imagine_block)
        self.imag_log = []
        if self.imagine:
            assert paired_encoder is None, "imagination check is not wired for paired drafts"
            assert self.sg_steps % self.imagine_block == 0, (self.sg_steps, self.imagine_block)

    def set_env(self, env):
        self.env = env
        n = env.num_envs
        self._t = 0
        self._queue = [None] * n
        self._ptr = np.zeros(n, dtype=np.int64)
        self._target = torch.zeros(n, self.dim, device=self.device)
        self._has_target = np.zeros(n, dtype=bool)
        self._direct = np.zeros(n, dtype=bool)
        self.c_first = np.full(n, np.nan)
        self._aqueue = None
        self._qpos = 0
        self._z_b = None     # latent at the last boundary (imagination check)

    @torch.no_grad()
    def _draft(self, rows, z_now, z_goal):
        t0 = time.perf_counter()
        z_cond = z_now[rows].to(self.drafter.device)
        zg = (z_goal[rows].to(self.drafter.device) if (self.goal_cond and z_goal is not None) else None)
        blocks = self.drafter.sample_sequence(z_cond, n_steps=self.n_steps, generator=self._gen,
                                              z_goal_native=zg)              # (R, N, dim)
        blocks = blocks.to(self.device)
        for j, i in enumerate(rows):
            self._queue[i] = blocks[j]
            self._ptr[i] = 1
            self._target[i] = blocks[j][0]
            self._has_target[i] = True
            self.n_redraft += 1
        self.t_draft += time.perf_counter() - t0

    def _unnorm(self, a):
        return a * self.stats["std"] + self.stats["mean"]

    def _exec_target(self):
        if self.penc is None:
            return self._target
        from surveyor.paired import split_paired
        return split_paired(self._target)[0]

    def _verify_target(self, i):
        if self.penc is None:
            return self._target[i]
        from surveyor.paired import split_paired
        return split_paired(self._target[i])[1]

    @torch.no_grad()
    def _imagined(self, v_t):
        """Forecast of this boundary's latent from the last boundary's latent over
        the S executed LeFlow actions; logs (imagined, achieved) rel per env."""
        from surveyor.sources import imagine_terminal
        a = np.asarray(self._aqueue[:, :self.sg_steps], dtype=np.float32)   # (n, S, adim) env units
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
    def get_action(self, info_dict, **kwargs):
        t0 = time.perf_counter()
        n = self.env.num_envs
        boundary = (self._t % self.sg_steps == 0)
        if boundary:
            frames = _latest(info_dict["pixels"])
            z_goal = v_goal = zg_cond = None
            if self.penc is None:
                z_t = encoder.encode_frames(self.lewm, frames, device=self.device).to(self.device)
                v_t, zc_cond = z_t, z_t
                if self.needs_goal:
                    goals = _latest(info_dict["goal"])
                    z_goal = encoder.encode_frames(self.lewm, goals, device=self.device).to(self.device)
                    v_goal, zg_cond = z_goal, z_goal
            else:
                from surveyor.paired import split_paired
                zc_cond = self.penc.encode(frames).to(self.device)
                z_t, v_t = split_paired(zc_cond)
                if self.needs_goal:
                    goals = _latest(info_dict["goal"])
                    zg_cond = self.penc.encode(goals).to(self.device)
                    z_goal, v_goal = split_paired(zg_cond)
            # imagination check: the forecast stands in for the achieved latent
            v_ver = v_t
            if self.imagine and self._z_b is not None and self._aqueue is not None:
                v_ver = self._imagined(v_t)
            if self.cstar_route and self._t == 0:
                tp = time.perf_counter()
                from surveyor.sources import cem_flat_cstar
                cs = cem_flat_cstar(self.lewm, self.cost_model, z_t, z_goal, self.cem, adim=self.adim)
                cs = np.asarray(cs, dtype=np.float64).reshape(-1)
                self.c_first[:] = cs
                self._direct = cs <= self.tau
                self.n_routed = int(self._direct.sum())
                self.t_probe += time.perf_counter() - tp
            drafting = ~self._direct
            need = list(np.nonzero(drafting & ~self._has_target)[0])
            for i in np.nonzero(drafting & self._has_target)[0]:
                if self.cstar_route or self.goal_gate:
                    relg = float((v_ver[i] - v_goal[i]).norm() / v_ver[i].norm().clamp_min(1e-8))
                    if relg <= self.tau:
                        self._direct[i] = True
                        self.n_arrive += 1
                        self.events.append((int(i), "gate", relg))
                        continue
                w = self._verify_target(i)
                rel = float((v_ver[i] - w).norm() / v_ver[i].norm().clamp_min(1e-8))
                verified = rel <= self.tau
                if self.random_reject is not None:
                    verified = bool(float(torch.rand((), generator=self._coin_gen)) >= self.random_reject)
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
            # one LeFlow plan per env toward its current waypoint (or the goal
            # for routed / arrived envs: LeFlow alone, replanned every S steps)
            target = self._exec_target()
            if self._direct.any():
                target = target.clone()
                d = np.nonzero(self._direct)[0]
                target[d] = z_goal[d]
            a = self.planner.plan(z_t, target).float().cpu().numpy()
            self._aqueue = self._unnorm(a)
            self._qpos = 0
            if self.imagine:
                self._z_b = z_t.detach().clone()
        act = self._aqueue[:, self._qpos]
        self._qpos += 1
        self._t += 1
        self.n_calls += n
        self.t_exec += time.perf_counter() - t0
        return act.reshape(*self.env.action_space.shape).astype(np.float32)
