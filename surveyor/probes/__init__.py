"""Offline measurement probes (no closed-loop env rollouts).

Run as modules, e.g. `python -m surveyor.probes.probe_suffix_decay`.

  gap_stat                  the verification gap: preflight applicability test
  calibrate_verifier        accept statistic vs ground-truth state (M1)
  learn_readout             gap-maximizing readout: derive the metric, not tau
  probe_suffix_decay        drafter rel_err by block position (anchor protocol)
  probe_stride_drafters     anchor protocol across stride checkpoints
  probe_stride_displacement latent displacement vs the encoder noise floor
  probe_cost_snr            CEM cost-signal SNR at fine vs coarse stride
  probe_dspark_gates        offline acceptance simulator (tau / call-ratio design)
  probe_failure_anatomy     classify closed-loop failures from --dump-traces
  probe_floor               encoder resolution floor vs sampler dispersion
  probe_stride_saturation   where an encoder saturates as stride grows
  probe_cube_displacement   Cube start-to-goal displacement, their protocol vs ours
  probe_tworoom_protocol_diff  why the Two-Room random floor reads above zero
  horizon_gate              gate v1: encoder distance as a horizon signal (killed)
  horizon_gate_cem          gate v2: planner c* as a horizon signal (killed)
  gate_validity             post-mortem: routing signals vs recorded outcomes
  route_horizon             gate v4: offline ternary router (flat2/flat5/spec)

The killed gates are kept because the paper reports them as negative results.
"""
