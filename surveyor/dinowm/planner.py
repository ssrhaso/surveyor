"""Surveyor on the DINO-WM stack (PushT).

SurveyorMPCPlanner subclasses their MPCPlanner with an identical outer loop,
except each MPC iteration serves a drafted subgoal latent (a (P+1, D) grid of
P DINOv2 patch tokens plus one proprio token) as the sub-planner's goal
instead of the final goal, verifying against reality at every replan boundary.
The CEM sub-planner, world model, and evaluator are untouched; cem.py has a
5-line guarded branch accepting pre-encoded latent goals.

Verification metric, which must match the pre-registered gap space
of the DINO-WM instrument: rel L2 between pooled visual tokens,
the mean over patch tokens only, excluding proprio as the gap statistic does.

Arrival gate (added 2026-07-28): without it spec rose then declined while flat
plateaued, the overshoot tax also diagnosed on Cube and short-horizon Reacher
(worth +32pp there): the drafter keeps proposing onward waypoints after the
agent has arrived. LeWM PushT hid this because its episodes end at the goal;
DINO-WM takes the goal from mid-trajectory (goal_source=dset), so arrival is
not terminal and the gate is required. Semantics are one-way and match
CstarRetireSource: once the achieved latent verifies against the final goal,
the env retires and serves the goal latent at zero drafter cost. Off by
default, so every previously recorded arm reproduces exactly.
"""
import os
import sys
from pathlib import Path

# This module is imported from inside the DINO-WM tree, whose own utils and
# planning packages shadow ours, so the repo root goes on sys.path explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from planning.mpc import MPCPlanner
from utils import move_to_device, slice_trajdict_with_t
from surveyor.vjepa2.drafter import load_token_gdm


def pooled_visual(grid):
    """(B, P+1, D) -> (B, D): mean over the P visual tokens only (the last
    token is proprio; the gap statistic space excludes it). P is derived from
    the grid, NOT hardcoded: pusht runs at img_size 196 -> 196 patches."""
    return grid[:, :-1].mean(dim=1)


def pooled_all(grid):
    """(B, P+1, D) -> (B, D): mean over all tokens (drafter cond convention)."""
    return grid.mean(dim=1)


class SurveyorGridSource:
    """Per-env verify/advance/re-draft state machine over drafted grid blocks.
    LeWM-faithful semantics: draft a block of N subgoal grids; at each replan
    verify the pursued target in pooled-visual rel L2 (<= tau = arrived ->
    advance); on rejection or block exhaustion re-draft from the current
    state. Serves the currently-pursued grid.

    With goal_gate=True an env that has verifiably ARRIVED at the final goal
    retires and serves the goal thereafter, which is what stops the drafter
    walking the agent back off the target."""

    def __init__(self, planner, n_envs, tau, k, device, seed=0,
                 readout=None, readout_tau=None, goal_gate=False,
                 arrive_tau=None, snap_fn=None):
        self.planner = planner
        self.n = n_envs
        self.tau = float(tau)
        self.k = int(k)
        self.device = device
        # optional lens: verify in the learned unit-norm readout space at its
        # derived tau (absolute L2 there, matching learn_readout's gap_of)
        self.readout = readout
        self.readout_tau = None if readout_tau is None else float(readout_tau)
        self.goal_gate = bool(goal_gate)
        # arrival is judged by the SAME test as waypoint arrival unless told
        # otherwise, so the gate introduces no new tuned constant
        self.arrive_tau = float(arrive_tau) if arrive_tau is not None else None
        # optional draft-and-snap (autopsy D2): replace every drafted grid
        # with its nearest REAL cached grid before serving/verifying
        self.snap_fn = snap_fn
        self.gen = torch.Generator(device=str(device)).manual_seed(seed)
        self.blocks = [None] * n_envs      # (N, P+1, D) each
        self.ptr = [0] * n_envs
        self.retired = [False] * n_envs
        self.re_drafts = 0
        self.advances = 0
        self.rejects = 0
        self.gated = 0
        self.steps = 0

    def _verified(self, z_a, z_b):
        """Is z_a at z_b, in the pre-registered verification space?"""
        if self.readout is not None:
            with torch.no_grad():
                d = float((self.readout(pooled_visual(z_a))
                           - self.readout(pooled_visual(z_b))).norm())
            return d <= self.readout_tau, d
        rel = float((pooled_visual(z_a) - pooled_visual(z_b)).norm()
                    / pooled_visual(z_b).norm().clamp_min(1e-8))
        return rel <= self.tau, rel

    def _arrived(self, z_a, z_b):
        """Arrival test for the gate. Same metric; arrive_tau only if given."""
        if self.arrive_tau is None:
            return self._verified(z_a, z_b)[0]
        rel = float((pooled_visual(z_a) - pooled_visual(z_b)).norm()
                    / pooled_visual(z_b).norm().clamp_min(1e-8))
        return rel <= self.arrive_tau

    def _draft(self, idxs, z_now, z_goal):
        cond = pooled_all(z_now[idxs])
        goal = pooled_all(z_goal[idxs])
        blocks = self.planner.sample_sequence(
            cond, n_steps=self.k, generator=self.gen, z_goal_pooled=goal)
        if self.snap_fn is not None:
            blocks = self.snap_fn(blocks)
        for j, i in enumerate(idxs):
            self.blocks[i] = blocks[j]
            self.ptr[i] = 0
        self.re_drafts += len(idxs)

    def step(self, z_now, z_goal):
        """z_now, z_goal: (B, P+1, D) native. Returns targets (B, P+1, D)."""
        self.steps += self.n

        # --- arrival gate: retire envs that have verifiably reached the goal
        if self.goal_gate:
            for i in range(self.n):
                if not self.retired[i] and self._arrived(z_now[i].unsqueeze(0),
                                                         z_goal[i].unsqueeze(0)):
                    self.retired[i] = True
        live = [i for i in range(self.n) if not self.retired[i]]

        need = [i for i in live if self.blocks[i] is None]
        if need:
            self._draft(need, z_now, z_goal)
        redraft = []
        for i in live:
            tgt = self.blocks[i][self.ptr[i]].unsqueeze(0)
            zn = z_now[i].unsqueeze(0)
            verified, _ = self._verified(zn, tgt)
            if verified:                              # arrived at this subgoal
                self.advances += 1
                if self.ptr[i] + 1 < self.blocks[i].shape[0]:
                    self.ptr[i] += 1
                else:
                    redraft.append(i)                 # block exhausted
            else:
                self.rejects += 1
                redraft.append(i)                     # reality check failed
        if redraft:
            self._draft(redraft, z_now, z_goal)

        out = []
        for i in range(self.n):
            if self.retired[i]:
                self.gated += 1
                out.append(z_goal[i])                 # hold the goal, no drafting
            else:
                out.append(self.blocks[i][self.ptr[i]])
        return torch.stack(out)

    def stats(self):
        cr = self.re_drafts / max(self.steps, 1)
        g = (f" gated_serves={self.gated} retired={sum(self.retired)}/{self.n}"
             if self.goal_gate else "")
        return (f"[surveyor] tau={self.tau} k={self.k} re-drafts={self.re_drafts} "
                f"advances={self.advances} rejects={self.rejects} "
                f"call_ratio={cr:.3f} (every-step=1.000){g}")


class SurveyorMPCPlanner(MPCPlanner):
    """MPCPlanner with per-iteration drafted-subgoal serving + verification.
    Extra cfg keys: gdm_ckpt, accept_tau, gdm_steps, spec_seed, goal_gate."""

    def __init__(self, *args, gdm_ckpt=None, accept_tau=0.121, gdm_steps=8,
                 spec_seed=0, readout_ckpt=None, readout_tau=None,
                 goal_gate=False, arrive_tau=None, spec_serve="draft",
                 snap_dir=None, snap_max=6000, **kwargs):
        super().__init__(*args, **kwargs)
        assert gdm_ckpt, "SurveyorMPCPlanner needs gdm_ckpt"
        self.gdm = load_token_gdm(gdm_ckpt, device=str(self.device))
        self.accept_tau = float(accept_tau)
        self.gdm_steps = int(gdm_steps)
        self.spec_seed = int(spec_seed)
        self.goal_gate = bool(goal_gate)
        self.arrive_tau = arrive_tau
        self.readout = None
        self.readout_tau = None
        if readout_ckpt:
            from surveyor.probes.learn_readout import Readout
            ck = torch.load(readout_ckpt, map_location="cpu", weights_only=False)
            self.readout = Readout(ck["dim"], ck["out_dim"],
                                   attentive=ck["attentive"]).to(self.device)
            self.readout.load_state_dict(ck["state"])
            self.readout.eval()
            if readout_tau is None and ck.get("tau_derived") is None:
                raise ValueError(
                    f"{readout_ckpt} has tau_derived=None (its gap never opened); "
                    "pass readout_tau explicitly rather than inventing one")
            self.readout_tau = float(readout_tau if readout_tau is not None
                                     else ck["tau_derived"])
            print(f"[spec] lens={readout_ckpt} lens_tau={self.readout_tau:.3f}")
        # ---- autopsy serve modes (pre-registered under AUTOPSY REGISTRATION):
        # draft = production path | goal = D1 tautology (serve the goal grid
        # through the same latent-goal branch) | snap = D2 draft-and-snap
        # (every drafted grid replaced by its nearest REAL cached train grid,
        # matched in pooled-visual rel L2 -- the TwoRoom snap-bank mechanism,
        # no new constant)
        self.spec_serve = str(spec_serve)
        assert self.spec_serve in ("draft", "goal", "snap"), self.spec_serve
        self.bank_grids = None
        self.bank_pooled = None
        self.bank_norms = None
        if self.spec_serve == "snap" or snap_dir:
            assert snap_dir, "spec_serve=snap needs snap_dir"
            self._load_snap_bank(snap_dir, int(snap_max))
        print(f"[spec] drafter={gdm_ckpt} tau={self.accept_tau} k={self.gdm_steps} "
              f"goal_gate={self.goal_gate} serve_mode={self.spec_serve}")

    def _load_snap_bank(self, snap_dir, snap_max):
        import glob
        files = sorted(glob.glob(os.path.join(snap_dir, "grid_pusht_train_*.npy")))
        assert files, f"no train grids under {snap_dir}"
        grids = []
        for f in files:
            a = np.load(f)                                # (T, P+1, D) fp16
            grids.append(torch.from_numpy(a[::2].copy())) # stride-2 subsample
        bank = torch.cat(grids)
        if bank.shape[0] > snap_max:
            sel = torch.linspace(0, bank.shape[0] - 1, snap_max).long()
            bank = bank[sel]
        self.bank_grids = bank.half().to(self.device)     # (K, P+1, D)
        pv = self.bank_grids[:, :-1].float().mean(dim=1)  # (K, D)
        self.bank_pooled = pv
        self.bank_norms = pv.norm(dim=-1).clamp_min(1e-8)
        print(f"[snap] bank {bank.shape[0]} real grids from "
              f"{len(files)} train episodes ({snap_dir})", flush=True)

    def _snap_grids(self, grids):
        """(..., P+1, D) drafted -> nearest REAL bank grid, matched in
        pooled-visual rel L2 (distance normalized by the bank grid's norm)."""
        shape = grids.shape
        flat = grids.reshape(-1, shape[-2], shape[-1])
        pv = flat[:, :-1].float().mean(dim=1)             # (M, D)
        d = torch.cdist(pv, self.bank_pooled) / self.bank_norms.unsqueeze(0)
        idx = d.argmin(dim=1)
        return self.bank_grids[idx].to(grids.dtype).reshape(shape)

    def _diag(self, tag, tgt, z_now, z_goal):
        """Per-iteration serving diagnostics: where does the served target sit
        (pooled space), and how far off the real-grid manifold is it (token
        space)? Print-only; no effect on planning."""
        with torch.no_grad():
            pt, pn, pg = pooled_visual(tgt), pooled_visual(z_now), pooled_visual(z_goal)
            rel_now = ((pt - pn).norm(dim=-1)
                       / pn.norm(dim=-1).clamp_min(1e-8)).mean()
            rel_goal = ((pt - pg).norm(dim=-1)
                        / pg.norm(dim=-1).clamp_min(1e-8)).mean()
            line = (f"[diag] serve={tag} rel_to_now={float(rel_now):.3f} "
                    f"rel_to_goal={float(rel_goal):.3f}")
            if self.bank_pooled is not None:
                pv = tgt[:, :-1].float().mean(dim=1)
                d = (torch.cdist(pv, self.bank_pooled)
                     / self.bank_norms.unsqueeze(0))
                idx = d.argmin(dim=1)
                near = self.bank_grids[idx].float()
                tok_mse = ((tgt[:, :-1].float() - near[:, :-1]) ** 2).mean()
                line += (f" tok_mse_nearest={float(tok_mse):.5f} "
                         f"pooled_rel_nearest={float(d.min(dim=1).values.mean()):.3f}")
        print(line, flush=True)

    def _grid(self, obs):
        """obs dict (B, 1, ...) numpy/tensor -> (B, P+1, D) native latents via
        the model's own encode path + normalized proprio token."""
        trans = move_to_device(self.preprocessor.transform_obs(obs), self.device)
        with torch.no_grad():
            z = self.wm.encode_obs(trans)
        vis = z["visual"][:, -1]                       # (B, P, D)
        pro2 = self.preprocessor.normalize_proprios(
            torch.as_tensor(np.asarray(obs["proprio"])).float())[:, -1].to(self.device)
        ptok = torch.zeros(vis.shape[0], 1, vis.shape[-1], device=self.device)
        ptok[:, 0, :pro2.shape[-1]] = pro2
        return torch.cat([vis, ptok], dim=1)

    # kept under the old name for any caller that still uses it
    _grid257 = _grid

    def _latent_goal(self, tgt, z_goal_pro):
        """(B, P+1, D) drafted grid -> the latent-goal dict cem.py accepts.
        Spec alters only the VISUAL target; the proprio target is the final
        goal's own embedding, identical to what flat serves. (The drafted
        proprio token gets ~4/75k of the training loss - it is untrained -
        and the first smoke showed serving it steers the agent into garbage.
        Verification was always visual-only, matching the gap space.)"""
        z_vis = tgt[:, :-1].unsqueeze(1)               # (B, 1, P, D)
        return {"z_visual": z_vis, "z_proprio": z_goal_pro}

    def plan(self, obs_0, obs_g, actions=None):
        n_evals = obs_0["visual"].shape[0]
        self.is_success = np.zeros(n_evals, dtype=bool)
        self.action_len = np.full(n_evals, np.inf)
        init_obs_0, init_state_0 = self.evaluator.get_init_cond()

        source = SurveyorGridSource(self.gdm, n_evals, self.accept_tau,
                                      self.gdm_steps, self.device,
                                      seed=self.spec_seed, readout=self.readout,
                                      readout_tau=self.readout_tau,
                                      goal_gate=self.goal_gate,
                                      arrive_tau=self.arrive_tau,
                                      snap_fn=(self._snap_grids
                                               if self.spec_serve == "snap"
                                               else None))
        z_goal = self._grid(obs_g)
        trans_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
        with torch.no_grad():
            z_goal_pro = self.wm.encode_obs(trans_g)["proprio"][:, -1:]  # (B, 1, d)

        cur_obs_0 = obs_0
        memo_actions = None
        while not np.all(self.is_success) and self.iter < self.max_iter:
            z_now = self._grid(cur_obs_0)
            if self.spec_serve == "goal":
                # D1 tautology: the goal grid (real encoded tokens) through
                # the SAME latent-goal branch; must reproduce flat if the
                # hand-off is sound
                tgt = z_goal
            else:
                tgt = source.step(z_now, z_goal)
            self._diag(self.spec_serve, tgt, z_now, z_goal)
            goal_latent = self._latent_goal(tgt, z_goal_pro)

            self.sub_planner.logging_prefix = f"plan_{self.iter}"
            actions, _ = self.sub_planner.plan(
                obs_0=cur_obs_0, obs_g=goal_latent, actions=memo_actions)
            taken_actions = actions.detach()[:, : self.n_taken_actions]
            self._apply_success_mask(taken_actions)
            memo_actions = actions.detach()[:, self.n_taken_actions:]
            self.planned_actions.append(taken_actions)

            print(f"MPC iter {self.iter} Eval ------- ")
            action_so_far = torch.cat(self.planned_actions, dim=1)
            self.evaluator.assign_init_cond(obs_0=init_obs_0, state_0=init_state_0)
            logs, successes, e_obses, e_states = self.evaluator.eval_actions(
                action_so_far, self.action_len,
                filename=f"plan{self.iter}", save_video=True)
            new_successes = successes & ~self.is_success
            self.is_success = self.is_success | successes
            self.action_len[new_successes] = (self.iter + 1) * self.n_taken_actions

            print("self.is_success: ", self.is_success)
            if self.spec_serve != "goal":
                print(source.stats(), flush=True)
            logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
            logs.update({"step": self.iter + 1})
            self.wandb_run.log(logs)
            self.dump_logs(logs)

            e_final_obs = slice_trajdict_with_t(e_obses, start_idx=-1)
            cur_obs_0 = e_final_obs
            e_final_state = e_states[:, -1]
            self.evaluator.assign_init_cond(obs_0=e_final_obs, state_0=e_final_state)
            self.iter += 1
            self.sub_planner.logging_prefix = f"plan_{self.iter}"

        if self.spec_serve != "goal":
            print(source.stats())
        planned_actions = torch.cat(self.planned_actions, dim=1)
        self.evaluator.assign_init_cond(obs_0=init_obs_0, state_0=init_state_0)
        return planned_actions, self.action_len
