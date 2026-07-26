"""Certified spec-accept on the DINO-WM stack (PushT).

SpecMPCPlanner subclasses their MPCPlanner: identical outer loop, but each
MPC iteration serves a DRAFTED subgoal latent (a (257,384) grid: 256 DINOv2
patch tokens + 1 proprio token) as the sub-planner's goal instead of the
final goal, with reality-verification at every replan boundary. The CEM
sub-planner, world model, and evaluator are untouched (cem.py has a 5-line
guarded branch accepting pre-encoded latent goals).

Verification metric (MUST match the pre-registered gap space, docs/
dinowm_prereg.md): rel L2 between POOLED VISUAL tokens (mean over the 256
patch tokens only - the proprio token is excluded, as in the gap statistic).
Frozen prediction: tau=0.121 (in-gap) certifies; tau=0.20 (above gap) is
degenerate-accept.
"""
import sys

sys.path.insert(0, "/lustre/home/ha676/le-wm/vjepa2")
sys.path.insert(0, "/lustre/home/ha676/le-wm")

import numpy as np
import torch
from einops import repeat

from planning.mpc import MPCPlanner
from utils import move_to_device, slice_trajdict_with_t
from specaccept_vjepa2.drafter import load_token_gdm


def pooled_visual(grid257):
    """(B, 257, D) -> (B, D): mean over the 256 visual tokens only."""
    return grid257[:, :256].mean(dim=1)


def pooled_all(grid257):
    """(B, 257, D) -> (B, D): mean over all 257 (drafter cond convention)."""
    return grid257.mean(dim=1)


class SpecAcceptGridSource:
    """Per-env verify/advance/re-draft state machine over drafted grid blocks.
    LeWM-faithful semantics: draft a block of N subgoal grids; at each replan
    verify the pursued target in pooled-visual rel L2 (<= tau = arrived ->
    advance); on rejection or block exhaustion re-draft from the current
    state. Serves the currently-pursued grid."""

    def __init__(self, planner, n_envs, tau, k, device, seed=0):
        self.planner = planner
        self.n = n_envs
        self.tau = float(tau)
        self.k = int(k)
        self.device = device
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.blocks = [None] * n_envs      # (N, 257, D) each
        self.ptr = [0] * n_envs
        self.re_drafts = 0
        self.advances = 0
        self.rejects = 0
        self.steps = 0

    def _draft(self, idxs, z_now, z_goal):
        cond = pooled_all(z_now[idxs])
        goal = pooled_all(z_goal[idxs])
        blocks = self.planner.sample_sequence(
            cond, n_steps=self.k, generator=self.gen, z_goal_pooled=goal)
        for j, i in enumerate(idxs):
            self.blocks[i] = blocks[j]
            self.ptr[i] = 0
        self.re_drafts += len(idxs)

    def step(self, z_now, z_goal):
        """z_now, z_goal: (B, 257, D) native. Returns targets (B, 257, D)."""
        self.steps += self.n
        need = [i for i in range(self.n) if self.blocks[i] is None]
        if need:
            self._draft(need, z_now, z_goal)
        redraft = []
        for i in range(self.n):
            tgt = self.blocks[i][self.ptr[i]].unsqueeze(0)
            zn = z_now[i].unsqueeze(0)
            rel = float((pooled_visual(zn) - pooled_visual(tgt)).norm()
                        / pooled_visual(tgt).norm().clamp_min(1e-8))
            if rel <= self.tau:                       # arrived at this subgoal
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
        return torch.stack([self.blocks[i][self.ptr[i]] for i in range(self.n)])

    def stats(self):
        cr = self.re_drafts / max(self.steps, 1)
        return (f"[specaccept] tau={self.tau} k={self.k} re-drafts={self.re_drafts} "
                f"advances={self.advances} rejects={self.rejects} "
                f"call_ratio={cr:.3f} (every-step=1.000)")


class SpecMPCPlanner(MPCPlanner):
    """MPCPlanner with per-iteration drafted-subgoal serving + verification.
    Extra cfg keys: gdm_ckpt, accept_tau, gdm_steps, spec_seed."""

    def __init__(self, *args, gdm_ckpt=None, accept_tau=0.121, gdm_steps=8,
                 spec_seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        assert gdm_ckpt, "SpecMPCPlanner needs gdm_ckpt"
        self.gdm = load_token_gdm(gdm_ckpt, device=str(self.device))
        self.accept_tau = float(accept_tau)
        self.gdm_steps = int(gdm_steps)
        self.spec_seed = int(spec_seed)
        print(f"[spec] drafter={gdm_ckpt} tau={self.accept_tau} k={self.gdm_steps}")

    def _grid257(self, obs):
        """obs dict (B, 1, ...) numpy/tensor -> (B, 257, D) native latents via
        the model's own encode path + normalized 2-d proprio token."""
        trans = move_to_device(self.preprocessor.transform_obs(obs), self.device)
        with torch.no_grad():
            z = self.wm.encode_obs(trans)
        vis = z["visual"][:, -1]                       # (B, 256, D)
        pro2 = self.preprocessor.normalize_proprios(
            torch.as_tensor(np.asarray(obs["proprio"])).float())[:, -1].to(self.device)
        ptok = torch.zeros(vis.shape[0], 1, vis.shape[-1], device=self.device)
        ptok[:, 0, :pro2.shape[-1]] = pro2
        return torch.cat([vis, ptok], dim=1)

    def _latent_goal(self, tgt):
        """(B, 257, D) drafted grid -> the latent-goal dict cem.py accepts."""
        z_vis = tgt[:, :256].unsqueeze(1)              # (B, 1, 256, D)
        pro2 = tgt[:, 256, :2].unsqueeze(1)            # (B, 1, 2)
        with torch.no_grad():
            z_pro = self.wm.encode_proprio(pro2)       # (B, 1, d_pro)
        return {"z_visual": z_vis, "z_proprio": z_pro}

    def plan(self, obs_0, obs_g, actions=None):
        n_evals = obs_0["visual"].shape[0]
        self.is_success = np.zeros(n_evals, dtype=bool)
        self.action_len = np.full(n_evals, np.inf)
        init_obs_0, init_state_0 = self.evaluator.get_init_cond()

        source = SpecAcceptGridSource(self.gdm, n_evals, self.accept_tau,
                                      self.gdm_steps, self.device,
                                      seed=self.spec_seed)
        z_goal = self._grid257(obs_g)

        cur_obs_0 = obs_0
        memo_actions = None
        while not np.all(self.is_success) and self.iter < self.max_iter:
            z_now = self._grid257(cur_obs_0)
            tgt = source.step(z_now, z_goal)
            goal_latent = self._latent_goal(tgt)

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

        print(source.stats())
        planned_actions = torch.cat(self.planned_actions, dim=1)
        self.evaluator.assign_init_cond(obs_0=init_obs_0, state_0=init_state_0)
        return planned_actions, self.action_len
