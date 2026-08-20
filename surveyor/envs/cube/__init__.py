"""OGBench-Cube (single): hindsight goal-reaching on the frozen cube encoder.

Run as modules, e.g. `python -m surveyor.envs.cube.eval --subgoal surveyor`.

  eval                 evaluation driver, success on the env's 0.04 m rule
  build_subgoals       dense stride-1 subgoal latents (every demo succeeds)
  build_populations    fixed horizon population, non-vacuous by construction
"""
