"""Certified Surveyor transplanted onto V-JEPA 2-AC.

The method is the paper's unified rule, unchanged: draft subgoals only while the
planner certifies the goal is out of reach (c* > tau), with the first-replan c*
check acting as the episode router and the per-replan check as one-way
retirement. Only the substrate changes:

  latent        LeWM 192-d vector      -> V-JEPA 2 token grid (T_tok, D)
  world model   lewm.rollout           -> VisionTransformerPredictorAC (frame-causal AR)
  flat planner  swm CEMSolver          -> upstream notebooks/utils/mpc_utils.cem, UNCHANGED
  cost          terminal L2^2 to z     -> upstream L1 to target tokens, UNCHANGED

Space contract (the one transplant decision):
  * Targets handed to the planner are TOKEN GRIDS, passed as `goal_frame` to
    the unmodified upstream CEM. Serving the true goal tokens is therefore
    bit-identical to Meta's flat planning protocol, preserving exactly the
    reduction property the LeWM SubgoalCostModel had by construction.
  * Verification, c*, retirement, and routing read MEAN-POOLED vectors under
    relative L2, the paper's instrument. tau=0.20 is the frozen transfer
    hypothesis; re-derive it from the new setting's criterion floor only if
    the pre-registered transfer fails.

Anchor rule: reproduce upstream's flat CEM behavior on the bundled Franka
trajectory before enabling any drafting. probe_offline.py runs that anchor plus
the c* landscape without a robot or simulator.

Upstream is a shallow clone of github.com/facebookresearch/vjepa2 pinned at
204698b45b37 (2026-03-23), imported lazily and never edited in place; point
VJEPA2_UPSTREAM at it, or leave it at vjepa2/upstream in the repo root.

    wm.py                     encoder + AC predictor, frame-causal step/rollout
    planner.py                pass-through to upstream cem(); c* certificate
    drafter.py                token-grid DiT drafter, reusing the LeWM-stack
                              diffusion machinery verbatim
    sources.py                accept rule and c* retirement over token grids
    policy.py                 upstream's deploy loop with the target swapped
    train_drafter.py          block pair builder and training loop
    encode_episodes.py        episodes to token grids
    fetch_droid.py            DROID episodes to the encoder's input format
    calibrate_droid_pairs.py  verifier calibration against recorded proprio
    probe_offline.py          anchor, c* ladder, criterion floor, plumbing
    probe_*.py                the remaining offline probes
    smoke_test.py             CPU, tiny tensors, no weights
"""

from .drafter import TokenGDM, TokenGDMConfig, TokenGDMPlanner, load_token_gdm, save_token_gdm
from .sources import (
    CstarRetireTokenSource,
    GDMDraft,
    LerpBlockDrafter,
    SurveyorTokenSource,
    pool,
)

__all__ = [
    "TokenGDM", "TokenGDMConfig", "TokenGDMPlanner", "load_token_gdm", "save_token_gdm",
    "CstarRetireTokenSource", "GDMDraft", "LerpBlockDrafter", "SurveyorTokenSource", "pool",
]
