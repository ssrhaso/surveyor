"""Certified spec-accept transplanted onto V-JEPA 2-AC.

The method is the paper's unified rule, unchanged: draft subgoals only while
the planner certifies the goal is out of reach (c* > tau); the first-replan
c* check is the episode router, the per-replan check is one-way retirement.
What changes is the substrate:

  latent        LeWM 192-d vector      -> V-JEPA 2 token grid (T_tok, D)
  world model   lewm.rollout           -> VisionTransformerPredictorAC (frame-causal AR)
  flat planner  swm CEMSolver          -> upstream notebooks/utils/mpc_utils.cem, UNCHANGED
  cost          terminal L2^2 to z     -> upstream L1 to target tokens, UNCHANGED

Space contract (the one transplant decision):
  * Targets handed to the planner are TOKEN GRIDS, passed as `goal_frame` to
    the unmodified upstream CEM. Serving the true goal tokens is therefore
    bit-identical to Meta's flat planning protocol; the reduction property
    the LeWM SubgoalCostModel had by construction is preserved exactly.
  * Verification, c*, retirement, and routing read MEAN-POOLED vectors with
    relative L2, the paper's instrument. tau=0.20 is the frozen transfer
    hypothesis; re-derive from the new setting's criterion floor only if the
    pre-registered transfer fails.

Anchor rule (cube lesson): reproduce upstream's flat CEM behavior on the
bundled Franka trajectory before enabling any drafting. probe_offline.py runs
that anchor plus the c* landscape without a robot or simulator.
"""

from .drafter import TokenGDM, TokenGDMConfig, TokenGDMPlanner, load_token_gdm, save_token_gdm
from .sources import (
    CstarRetireTokenSource,
    GDMDraft,
    LerpBlockDrafter,
    SpecAcceptTokenSource,
    pool,
)

__all__ = [
    "TokenGDM", "TokenGDMConfig", "TokenGDMPlanner", "load_token_gdm", "save_token_gdm",
    "CstarRetireTokenSource", "GDMDraft", "LerpBlockDrafter", "SpecAcceptTokenSource", "pool",
]
