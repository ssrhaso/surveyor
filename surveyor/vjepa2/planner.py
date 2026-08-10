"""Flat planning + the c* certificate on V-JEPA 2-AC.

flat_plan() is a thin pass-through to upstream's cem() with the target token
grid as `goal_frame`. Serving a drafted subgoal instead of the goal is the whole
cost-model swap; the planner itself is untouched. cstar() ports
surveyor.sources.cem_flat_cstar: one flat CEM solve toward the goal tokens,
the returned plan re-rolled once, and the relative pooled distance from the
predicted terminal to the goal read off as the certificate.
"""

from __future__ import annotations

import torch

from .wm import VJEPA2WM, ensure_upstream_on_path, pool

# Notebook defaults; probe/battery scripts override via --cem-* flags. rollout
# is the lookahead H (frames), maxnorm the per-step EE displacement cap.
DEFAULT_CEM = dict(
    rollout=2,
    samples=200,
    topk=10,
    cem_steps=5,
    momentum_mean=0.15,
    momentum_mean_gripper=0.15,
    momentum_std=0.75,
    momentum_std_gripper=0.15,
    maxnorm=0.075,
    verbose=False,
)


@torch.no_grad()
def flat_plan(wm: VJEPA2WM, z_now: torch.Tensor, pose_now: torch.Tensor,
              target_tokens: torch.Tensor, cem_args: dict | None = None,
              close_gripper: int | None = None) -> torch.Tensor:
    """One CEM solve toward an arbitrary target token grid.
    z_now (1, T_tok, D), pose_now (1, 1, 7), target_tokens (1, T_tok, D)
    -> action trajectory (1, rollout, 7). Upstream cem() verbatim."""
    ensure_upstream_on_path()
    from utils.mpc_utils import cem

    args = dict(DEFAULT_CEM, **(cem_args or {}))

    def step_predictor(reps, actions, poses):
        return wm.step(reps, actions, poses)

    return cem(
        context_frame=z_now.unsqueeze(1),
        context_pose=pose_now,
        goal_frame=target_tokens.unsqueeze(1),
        world_model=step_predictor,
        close_gripper=close_gripper,
        **args,
    )


@torch.no_grad()
def cstar(wm: VJEPA2WM, z_now: torch.Tensor, pose_now: torch.Tensor,
          goal_tokens: torch.Tensor, cem_args: dict | None = None,
          metric: str = "pooled") -> tuple[float, torch.Tensor]:
    """The planner's reachability certificate: solve flat toward the goal,
    roll the plan the policy would execute, and measure the predicted
    terminal against the goal. metric="pooled" (default, the LeWM-stack
    instrument): rel = ||pool(z_hat_T) - pool(z_goal)|| / ||pool(z_hat_T)||.
    metric="token": mean L1 over tokens+dims, upstream's own CEM energy
    (c*-v2 candidate: pooling may destroy the spatial signal on DROID).
    z_now (1, T_tok, D), pose_now (1, 1, 7), goal_tokens (1, T_tok, D)."""
    plan = flat_plan(wm, z_now, pose_now, goal_tokens, cem_args=cem_args)
    z_hat = wm.rollout_plan(z_now, pose_now, plan.to(z_now.device))     # (1, T_tok, D)
    if metric == "token":
        c = float((z_hat - goal_tokens.to(z_hat.device)).abs().mean())
    else:
        p_hat, p_goal = pool(z_hat), pool(goal_tokens.to(z_hat.device))
        c = float((p_hat - p_goal).norm() / p_hat.norm().clamp_min(1e-8))
    return c, plan
