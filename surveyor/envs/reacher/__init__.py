"""Reacher (DMControl qpos_match), where drafting must be goal-conditioned.

Run as modules, e.g. `python -m surveyor.envs.reacher.eval --subgoal surveyor`.

  eval                  evaluation driver, success on the env's qpos_match rule
  build_subgoals        dense stride-1 subgoal latents for hindsight goals
  build_populations     fixed horizon population shared across the sweep
  make_holdout_subset   holdout-only h5, index-compatible with the full file
"""
