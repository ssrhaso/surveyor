"""SURVEYOR: certified speculative plan consumption for latent world models.

Speculative Use of Reality-Verified Execution, Yielding On-demand Redrafts. A
diffusion drafter proposes a block of N subgoal latents once; at every replan
boundary the achieved latent is verified against the waypoint just pursued, and
within a relative distance tau the next pre-drafted waypoint is served free,
otherwise the block is redrafted from reality. There is nothing to train: tau
comes from the encoder's criterion floor and k from the sampler's convergence
point, both measured offline.

The substrate is a frozen LeWM JEPA encoder with a CEM controller; the same rule
is served through a GC-IDM executor, a DINO-WM encoder and V-JEPA 2 without
modification.

    encoder.py           frozen LeWM encoder IO (load, encode_frames)
    drafter.py           GDM: DiT eps-prediction diffusion drafter, GDMPlanner
    sources.py           cost model, subgoal sources including the accept rule
                         and the c* router, and the serving policy
    paired.py            the accept rule verified in a second frozen encoder
    gcidm.py             GC-IDM, the concurrent amortized baseline
    gcidm_executor.py    the accept rule with GC-IDM as the executor
    arms.py              released arm names and their development aliases
    train_drafter.py     drafter training (pair build, normalization, loop)
    train_gcidm.py       GC-IDM training
    train_regressor.py   deterministic drafter, kept as a negative result
    diag_gdm.py          offline drafter fidelity diagnostic (no CEM)
    diag_parity.py       machine-parity diagnostic (encode and cost paths)
    render_strips.py     filmstrips of recorded rollouts, for figures
    test_gdm_cpu.py      CPU smoke test for the drafter stack

    envs/                per-environment eval drivers and dataset builders
    probes/              offline measurement probes (see probes/__init__.py)
    dspark/              literal DSpark port, kept as a negative result
    dinowm/              the DINO-WM transplant (second architecture)
    vjepa2/              the V-JEPA 2 transplant (foundation scale)

Entry points, one per line of the README's reproduction table:

    python -m surveyor.train_drafter --subgoals <dense.pt> --out <ckpt.pt>
    python -m surveyor.envs.pusht.eval --subgoal surveyor --gdm-ckpt <ckpt.pt>
    python -m surveyor.probes.calibrate_verifier --help
"""
