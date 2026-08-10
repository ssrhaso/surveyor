# specaccept

Hierarchical latent planning on the frozen LeWM JEPA encoder: a diffusion
subgoal drafter (GDM) at a matched subgoal scale, consumed via reality-verified
speculative consumption (spec-accept). FF-JEPA is the reproduced substrate and
baseline; the two contributions are the subgoal-scale law and spec-accept.

## Layout

    encoder.py            frozen LeWM encoder IO (load, encode_frames)
    drafter.py            GDM: DiT eps-prediction diffusion drafter + GDMPlanner
    sources.py            subgoal sources: oracle / GDM / regressor / spec-accept, CEM cost model
    train_drafter.py      drafter training (pair build, normalization, loop)
    train_regressor.py    deterministic regressor baseline (negative result)
    diag_gdm.py           offline drafter fidelity diagnostic (no CEM)
    diag_parity.py        box parity diagnostic (encode + cost paths)
    test_gdm_cpu.py       CPU smoke test for the drafter stack
    envs/
      pusht/              eval driver, dense subgoal builder, subset tools
      reacher/            eval driver, dense subgoal builder
      tworoom/            eval driver, dense subgoal builder
    probes/               offline measurement probes (see probes/__init__.py)
    dspark/               literal DSpark port, kept as the negative-result baseline

## Entry points

    python -m surveyor.train_drafter --subgoals <dense.pt> --out <ckpt.pt> ...
    python -m surveyor.envs.pusht.eval --subgoal {baseline,oracle,gdm,dspark,specaccept,regressor} ...
    python -m surveyor.envs.reacher.eval --subgoal {baseline,oracle,gdm,specaccept} ...
    python -m surveyor.probes.probe_suffix_decay --gdm-ckpt <ckpt.pt> ...

Batch drivers (SLURM for Isambard, no-SLURM pool for shared MIG boxes) live in
../batch.
