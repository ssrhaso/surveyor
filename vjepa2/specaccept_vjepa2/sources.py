"""Subgoal sources ported to token-grid targets: spec-accept + c*-retire.

Faithful ports of specaccept.sources.SpecAcceptSubgoalSource and
CstarRetireSource with two substrate changes and nothing else:
  * served targets are token grids (T_tok, D) consumed by upstream cem() as
    `goal_frame`;
  * verification / goal-gate / retirement read pooled vectors (rel L2,
    tau=0.20 frozen); see the package docstring's space contract.

Drafter protocol (dependency-injected so sources stay pure torch):
  draft(z_now_tokens (R, T_tok, D), goal_tokens (R, T_tok, D) | None, k)
      -> (R, N, T_tok, D)
TokenGDMPlanner is adapted via GDMDraft (pools internally); LerpBlockDrafter
is the data-free decomposition control (and frac=1.0 recovers flat exactly,
the validated instrumented-flat tautology from the timing grid).

c* is injected as cstar_fn(z_tokens (1,T_tok,D), pose (1,1,7),
goal_tokens (1,T_tok,D)) -> float, normally partial(planner.cstar, wm).
"""

from __future__ import annotations

import numpy as np
import torch

from .wm import pool


class LerpBlockDrafter:
    """Straight-line block: subgoal j = z_now + (j+1)/(N+1) * (goal - z_now),
    token-wise. The decomposition-tax control; needs no training data."""

    def __init__(self, n_future: int = 3):
        self.n_future = int(n_future)

    @torch.no_grad()
    def draft(self, z_now_tokens: torch.Tensor, goal_tokens: torch.Tensor | None,
              k: int = 0) -> torch.Tensor:
        assert goal_tokens is not None, "LerpBlockDrafter needs goal tokens"
        N = self.n_future
        fracs = torch.arange(1, N + 1, device=z_now_tokens.device,
                             dtype=z_now_tokens.dtype) / (N + 1)
        delta = (goal_tokens - z_now_tokens).unsqueeze(1)          # (R, 1, T_tok, D)
        return z_now_tokens.unsqueeze(1) + fracs.view(1, N, 1, 1) * delta


class GDMDraft:
    """TokenGDMPlanner -> drafter protocol (pools the conditioning inputs)."""

    def __init__(self, planner, seed: int = 42):
        self.planner = planner
        self.n_future = int(planner.cfg.n_future)
        self._gen = torch.Generator(device=planner.device)
        self._gen.manual_seed(int(seed))

    @torch.no_grad()
    def draft(self, z_now_tokens: torch.Tensor, goal_tokens: torch.Tensor | None,
              k: int = 8) -> torch.Tensor:
        cond = pool(z_now_tokens)
        goal = pool(goal_tokens) if (self.planner.goal_cond and goal_tokens is not None) else None
        out = self.planner.sample_sequence(cond, n_steps=k, generator=self._gen,
                                           z_goal_pooled=goal)
        if getattr(self.planner.cfg, "residual", False):
            out = out + z_now_tokens.to(out.device).unsqueeze(1)   # add current grid back
        return out


class SpecAcceptTokenSource:
    """Speculative subgoal consumption with reality as the verifier (token
    substrate). At each replan the achieved pooled latent is checked against
    the pooled target last pursued: within tau the next pre-drafted grid is
    served with no drafter call; on rejection or block exhaustion the block
    is re-drafted from the achieved state. goal_gate is the cube-style
    arrival filter (pooled-space progress test), off by default."""

    def __init__(self, drafter, n_envs: int = 1, device: str = "cpu",
                 tau: float = 0.2, k: int = 8, goal_gate: bool = False,
                 record: bool = False):
        self.drafter = drafter
        self.N = int(drafter.n_future)
        self.n_envs = n_envs
        self.device = device
        self.tau = float(tau)
        self.k = int(k)
        self.goal_gate = bool(goal_gate)
        self._queue = [None] * n_envs            # (N, T_tok, D) drafted block
        self._ptr = np.zeros(n_envs, dtype=int)
        self._target = [None] * n_envs           # pooled (D,) latent being pursued
        self._cache = [None] * n_envs            # (T_tok, D) target served last
        self._gated = np.zeros(n_envs, dtype=bool)
        self.record = record
        self.trace = []
        self.n_redraft = 0
        self.n_advance = 0
        self.n_reject = 0
        self.n_gate = 0

    def _serve(self, i: int, grid: torch.Tensor, target_pooled: torch.Tensor | None):
        self._cache[i] = grid
        self._target[i] = target_pooled

    @torch.no_grad()
    def current(self, obs_tokens: torch.Tensor, replan_idx, goal_tokens: torch.Tensor):
        """obs_tokens (n_envs, T_tok, D) achieved latents for the rows in
        replan_idx (others ignored); goal_tokens (n_envs, T_tok, D).
        Returns list of (T_tok, D) targets, one per env."""
        replan_idx = list(replan_idx)
        if replan_idx:
            z_now = obs_tokens.to(self.device)
            p_now = pool(z_now)                                   # (n_envs, D)
            p_goal = pool(goal_tokens.to(self.device))

            def _prog(cand_grid, i):
                cand = pool(cand_grid)
                return float((cand - p_goal[i]).norm()) < float((p_now[i] - p_goal[i]).norm())

            need = []
            for i in replan_idx:
                q, tgt = self._queue[i], self._target[i]

                if self.goal_gate and self._gated[i] and q is not None:
                    if self._ptr[i] < self.N:
                        cand = q[self._ptr[i]]
                        self._ptr[i] += 1
                        if _prog(cand, i):
                            self._gated[i] = False
                            self._serve(i, cand, pool(cand))
                            self.n_advance += 1
                        else:
                            self._serve(i, goal_tokens[i], None)
                            self.n_gate += 1
                        continue
                    need.append(i)
                    continue

                accept = False
                if q is not None and tgt is not None:
                    rel = float((p_now[i] - tgt).norm() / p_now[i].norm().clamp_min(1e-8))
                    verified = rel <= self.tau
                    if not verified:
                        self.n_reject += 1
                    accept = verified and self._ptr[i] < self.N
                if accept:
                    nxt = self._queue[i][self._ptr[i]]
                    self._ptr[i] += 1
                    if self.goal_gate and not _prog(nxt, i):
                        self._gated[i] = True
                        self._serve(i, goal_tokens[i], None)
                        self.n_gate += 1
                    else:
                        self._serve(i, nxt, pool(nxt))
                        self.n_advance += 1
                else:
                    need.append(i)

            if need:
                rows = torch.stack([z_now[i] for i in need])
                goals = torch.stack([goal_tokens[i] for i in need]).to(self.device)
                blocks = self.drafter.draft(rows, goals, k=self.k)   # (R, N, T_tok, D)
                if self.record:
                    self.trace.append({"replan_idx": np.asarray(need),
                                       "cond_pooled": pool(rows).float().cpu(),
                                       "block_pooled": pool(blocks).float().cpu()})
                for j, i in enumerate(need):
                    self._queue[i] = blocks[j]
                    self._ptr[i] = 1
                    self.n_redraft += 1
                    first = blocks[j][0]
                    if self.goal_gate and not _prog(first, i):
                        self._gated[i] = True
                        self._serve(i, goal_tokens[i], None)
                        self.n_gate += 1
                    else:
                        self._gated[i] = False
                        self._serve(i, first, pool(first))
        return [self._cache[i] for i in range(self.n_envs)]


class CstarRetireTokenSource:
    """Unified certified spec-accept: draft only while the planner certifies
    the goal is out of reach. Each unretired env re-reads c* (one flat CEM
    solve toward the goal tokens) at every replan; the first time c* <= tau
    the drafter RETIRES one-way and the goal tokens are served thereafter.
    The first-replan c* doubles as the episode router's fire test."""

    def __init__(self, drafter, cstar_fn, n_envs: int = 1, device: str = "cpu",
                 tau: float = 0.2, k: int = 8, record: bool = False):
        self.spec = SpecAcceptTokenSource(drafter, n_envs=n_envs, device=device,
                                          tau=tau, k=k, goal_gate=False,
                                          record=record)
        self.cstar_fn = cstar_fn
        self.n_envs = n_envs
        self.device = device
        self.tau = float(tau)
        self._retired = np.zeros(n_envs, dtype=bool)
        self._seen = np.zeros(n_envs, dtype=bool)
        self._replans = np.zeros(n_envs, dtype=int)
        self.retire_replan = np.full(n_envs, -1, dtype=int)
        self.c_first = np.full(n_envs, np.nan)   # == the router fire test
        self.c_last = np.full(n_envs, np.nan)

    @torch.no_grad()
    def current(self, obs_tokens: torch.Tensor, poses: torch.Tensor, replan_idx,
                goal_tokens: torch.Tensor):
        """obs_tokens (n_envs, T_tok, D), poses (n_envs, 1, 7),
        replan_idx: envs at a replan boundary, goal_tokens (n_envs, T_tok, D).
        Returns list of (T_tok, D) targets, one per env."""
        replan_idx = list(replan_idx)
        for i in replan_idx:
            if not self._retired[i]:
                c, _ = self.cstar_fn(obs_tokens[i:i + 1], poses[i:i + 1],
                                     goal_tokens[i:i + 1])
                self.c_last[i] = c
                if not self._seen[i]:
                    self.c_first[i] = c
                    self._seen[i] = True
                if c <= self.tau:
                    self._retired[i] = True
                    self.retire_replan[i] = self._replans[i]
            self._replans[i] += 1

        live = [i for i in replan_idx if not self._retired[i]]
        out = self.spec.current(obs_tokens, live, goal_tokens) if live \
            else [self.spec._cache[i] for i in range(self.n_envs)]
        return [goal_tokens[i] if self._retired[i] else out[i]
                for i in range(self.n_envs)]

    def stats(self) -> str:
        fired = int(np.nansum(self.c_first <= self.tau))
        return (f"[unified] tau={self.tau} retired={int(self._retired.sum())}/{self.n_envs} "
                f"(fired-at-first-replan={fired}) "
                f"c*_first p50={np.nanmedian(self.c_first):.3f} | "
                f"spec: re-drafts={self.spec.n_redraft} advances={self.spec.n_advance} "
                f"rejects={self.spec.n_reject}")
