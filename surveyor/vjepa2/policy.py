"""Closed-loop certified Surveyor policy for a single V-JEPA 2-AC agent.

The upstream deployment loop (energy notebook / DROID protocol) is: encode
current frame -> CEM toward goal tokens -> execute first action -> repeat. This
policy keeps that loop verbatim and changes ONE thing: the target handed to CEM
comes from the certified source, meaning goal tokens once retired and drafted
subgoal tokens while the certificate says the goal is out of reach. With
use_certificate=False and a frac=1.0 lerp drafter it reduces exactly to upstream
flat planning, which is the anchor arm.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import torch

from . import planner as _planner
from .sources import CstarRetireTokenSource, SurveyorTokenSource
from .wm import VJEPA2WM


class CertifiedSurveyorPolicy:
    def __init__(self, wm: VJEPA2WM, drafter, goal_tokens: torch.Tensor,
                 tau: float = 0.2, k: int = 8, cem_args: dict | None = None,
                 cstar_cem_args: dict | None = None,
                 use_certificate: bool = True, record: bool = False):
        """goal_tokens (T_tok, D). cstar_cem_args defaults to cem_args (the
        certificate reads the exact plan the flat policy would execute)."""
        self.wm = wm
        self.cem_args = cem_args
        self.goal_tokens = goal_tokens.unsqueeze(0)      # (1, T_tok, D)
        cstar_fn = partial(_planner.cstar, wm,
                           cem_args=(cstar_cem_args or cem_args))
        if use_certificate:
            self.source = CstarRetireTokenSource(drafter, cstar_fn, n_envs=1,
                                                 device=wm.device, tau=tau, k=k,
                                                 record=record)
        else:
            self.source = SurveyorTokenSource(drafter, n_envs=1,
                                                device=wm.device, tau=tau, k=k,
                                                record=record)
        self.replans = 0

    @torch.no_grad()
    def act(self, frame: np.ndarray, pose: torch.Tensor,
            close_gripper: int | None = None) -> torch.Tensor:
        """frame (H, W, C) uint8 current observation; pose (7,) current EE
        pose -> (rollout, 7) action trajectory (execute the first action,
        then call act again: receding horizon 1, upstream convention)."""
        z_now = self.wm.encode_frames(frame[None])       # (1, T_tok, D)
        pose = torch.as_tensor(pose, dtype=torch.float32).view(1, 1, 7).to(z_now.device)
        if isinstance(self.source, CstarRetireTokenSource):
            targets = self.source.current(z_now, pose, [0], self.goal_tokens)
        else:
            targets = self.source.current(z_now, [0], self.goal_tokens)
        target = targets[0].unsqueeze(0).to(z_now.device)  # (1, T_tok, D)
        self.replans += 1
        plan = _planner.flat_plan(self.wm, z_now, pose, target,
                                  cem_args=self.cem_args,
                                  close_gripper=close_gripper)
        return plan[0]

    def stats(self) -> str:
        return self.source.stats() if hasattr(self.source, "stats") else (
            f"[spec] re-drafts={self.source.n_redraft} "
            f"advances={self.source.n_advance} rejects={self.source.n_reject}")
