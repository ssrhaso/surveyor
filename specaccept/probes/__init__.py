"""Offline measurement probes (no closed-loop env rollouts; see the env eval drivers for those).

Run as modules, e.g. `python -m specaccept.probes.probe_suffix_decay ...`. Each file's
docstring carries its exact invocation and the results doc it feeds.

  probe_suffix_decay        the anchor protocol: drafter rel_err by block position
                            (0.1593/0.2288/0.3179 reproduction standard)
  probe_stride_drafters     anchor protocol across all stride checkpoints
  probe_stride_displacement Stage-0 CPU pre-gate: latent displacement vs noise floor
  probe_cost_snr            CEM cost-signal SNR at fine vs coarse stride
  probe_dspark_gates        A2 acceptance simulator (spec-accept tau/call_ratio design)
  probe_failure_anatomy     classify residual closed-loop failures from --dump-traces
"""
