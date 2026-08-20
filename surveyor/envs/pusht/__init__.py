"""PushT: the primary environment, and the pattern the other drivers follow.

Run as modules, e.g. `python -m surveyor.envs.pusht.eval --subgoal surveyor`.

  eval                 evaluation driver (FFJEPAPolicy + SubgoalCostModel)
  build_subgoals       dense subgoal latents from successful expert episodes
  build_populations    the fixed horizon populations, one set reused at every t
  build_valsplit       disjoint validation population, where tau and k are set
  build_val_episodes   held-out episodes for the DSpark closed-loop eval
  extract_subset       compact subset h5 for hosts without the full dataset
"""
