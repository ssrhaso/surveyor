"""TwoRoom, whose own latent metric is too degenerate to verify in.

Planning stays in LeWM latents while the accept rule is transplanted into a
frozen DINOv2 half (surveyor/paired.py), so this package also carries the
probes that derive that verifier's constants offline.

Run as modules, e.g. `python -m surveyor.envs.tworoom.eval --subgoal surveyor`.

  eval                    evaluation driver, success on the env's own flag
  build_subgoals          subgoal latents under the per-episode target rule
  probe_dino_gap          the verification gap in the paired verifier's space
  probe_k                 k for the paired drafter, read where it verifies
  probe_bok               k for best-of-k paired drafting
  probe_lens_floor        criterion floor of the lens space
  probe_paired_drafter    offline QC for the paired 576-d drafter
  probe_eval_population   is the population at each horizon a real task?
"""
