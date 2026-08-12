"""CPU test for blind fixed-depth commitment: does --commit-k actually engage?

This exists because of a real failure. Job 2331836 swept `d in {1,2,3}` by
passing `--commit fixed --commit-k D` alongside `--subgoal specaccept`, but
those flags are read only by DSparkSubgoalSource, which is built only under
`--subgoal dspark`. They were silently ignored: all fifteen tasks ran the plain
accept rule and reported call ratios of .605/.608/.611 where depth 1 must be
1.000 and depth 3 about 1/3. Fifteen tasks were three mislabelled replicas of
one configuration, and nothing failed loudly. It was caught only by noticing
that the ratio did not move across settings.

The invariant that would have caught it immediately: under fixed-depth
commitment, one draft serves exactly d boundaries, so

    redraft / advance  ==  1 / d      and      mean_commit_depth == d

exactly, with no dependence on the environment, the drafter's quality, or the
episode population. That is asserted here on synthetic tensors, so it runs
without a GPU, the frozen LeWM encoder, an h5, or stable_worldmodel, and it
covers the accounting for both env drivers, which construct this same
env-agnostic source with the same arguments.

See prereg/2026-08-11_rate_transfer.md, Extension A and "voided run".
"""

from __future__ import annotations

import math

import numpy as np
import torch

from surveyor.drafter import GDM, GDMConfig, GaussianDiffusion, GDMPlanner
from surveyor.sources import DSparkSubgoalSource

D = 192
N = 3
DEV = "cpu"


def _synth_latents(n, seed=0):
    """n latents at the native scale: random directions x sqrt(D) (||z||~13.9)."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, D, generator=g)
    return z / z.norm(dim=1, keepdim=True) * math.sqrt(D)


def _planner(seed=11):
    z = _synth_latents(2000, seed=seed)
    cfg = GDMConfig(latent_dim=D, n_future=N, wg=1, hidden=64, depth=2, heads=4)
    return GDMPlanner(GDM(cfg), GaussianDiffusion(timesteps=100, device=DEV),
                      z.mean(0), z.std(0).clamp_min(1e-6), device=DEV)


def _drive(src, n_envs, n_boundaries, seed=12):
    """Run n_boundaries replan boundaries with every env replanning each time,
    which is how both env drivers call the source at a fixed receding horizon."""
    idx = list(range(n_envs))
    for b in range(n_boundaries):
        obs = _synth_latents(n_envs, seed=seed + b)
        src.current(np.full(n_envs, b, dtype=np.int64), obs_latent=obs, replan_idx=idx)


def test_fixed_depth_sets_the_call_ratio():
    """The engagement invariant: redraft/advance == 1/d and mean depth == d,
    for every depth in 1..N. Depth 1 is every-step drafting (ratio 1.000);
    depth N is full-block blind commitment (ratio 1/N)."""
    n_envs, n_bounds = 4, 12
    ratios = {}
    for d in (1, 2, 3):
        src = DSparkSubgoalSource(_planner(), None, n_envs=n_envs, device=DEV,
                                  n_steps=5, seed=42, commit="fixed",
                                  fixed_k=d, refine=False)
        _drive(src, n_envs, n_bounds)
        ratio = src.n_redraft / max(src.n_advance, 1)
        depth = float(np.mean(src.commit_depths))
        assert src.n_advance == n_envs * n_bounds, \
            f"d={d}: advanced {src.n_advance}, expected {n_envs * n_bounds}"
        assert abs(depth - d) < 1e-9, f"d={d}: mean_commit_depth={depth}, expected {d}"
        assert abs(ratio - 1.0 / d) < 1e-9, \
            f"d={d}: redraft/advance={ratio:.4f}, expected {1.0 / d:.4f}"
        ratios[d] = ratio

    # The specific signature of the voided run: every depth reporting the SAME
    # ratio. Assert the knob separates the settings, not merely that each is
    # self-consistent.
    assert ratios[1] > ratios[2] > ratios[3], f"ratio not monotone in depth: {ratios}"
    assert ratios[1] - ratios[3] >= 0.30, \
        f"depths 1 and 3 differ by only {ratios[1] - ratios[3]:.3f}; knob is inert"
    print(f"[commit] fixed-depth ratios {ratios[1]:.3f}/{ratios[2]:.3f}/{ratios[3]:.3f} "
          f"== 1/1, 1/2, 1/3 exactly; depths engage and separate")


def test_depth_one_reduces_to_every_step_drafting():
    """The source's own docstring claims depth 1 reduces to GDMSubgoalSource.
    At depth 1 every boundary re-drafts, so redrafts == advances exactly."""
    n_envs, n_bounds = 3, 8
    src = DSparkSubgoalSource(_planner(), None, n_envs=n_envs, device=DEV,
                              n_steps=5, seed=7, commit="fixed", fixed_k=1,
                              refine=False)
    _drive(src, n_envs, n_bounds)
    assert src.n_redraft == src.n_advance == n_envs * n_bounds
    print(f"[commit] depth 1 == every-step drafting: "
          f"{src.n_redraft} redrafts / {src.n_advance} advances, ratio 1.000")


def test_fixed_k_is_clamped_to_the_block_and_refine_requires_fixed():
    """Two guards worth pinning. A depth above the drafted block length clamps
    to the block rather than silently over-serving, so `--commit-k 99` cannot
    fake a cheaper ratio than 1/N. And refine=False demands commit='fixed',
    since without the confidence head there is nothing to set an adaptive
    depth from; that assertion is what makes --no-refine an honest
    blind-commitment arm rather than a partially-disabled adaptive one."""
    src = DSparkSubgoalSource(_planner(), None, n_envs=2, device=DEV, n_steps=5,
                              seed=3, commit="fixed", fixed_k=99, refine=False)
    assert src.fixed_k == N, f"fixed_k={src.fixed_k} not clamped to block length {N}"
    _drive(src, 2, 6)
    assert abs(src.n_redraft / src.n_advance - 1.0 / N) < 1e-9

    raised = False
    try:
        DSparkSubgoalSource(_planner(), None, n_envs=2, device=DEV, n_steps=5,
                            seed=3, commit="adaptive", refine=False)
    except AssertionError:
        raised = True
    assert raised, "refine=False with commit='adaptive' must fail loudly"
    print(f"[commit] over-large depth clamps to N={N}; adaptive+no-refine rejected")


def test_partial_replan_keeps_each_env_on_its_own_block():
    """Envs replan on independent schedules. A block committed for one env must
    not be consumed or re-drafted by another, or the per-env depth accounting
    (and so the call ratio) would be wrong in exactly the way that is hardest
    to see in aggregate telemetry."""
    n_envs, d = 4, 3
    src = DSparkSubgoalSource(_planner(), None, n_envs=n_envs, device=DEV,
                              n_steps=5, seed=21, commit="fixed", fixed_k=d,
                              refine=False)
    # env 0 and 2 replan three times; env 1 and 3 never do
    for b in range(3):
        obs = _synth_latents(2, seed=30 + b)
        src.current(np.full(n_envs, b, dtype=np.int64), obs_latent=obs, replan_idx=[0, 2])

    assert src.n_advance == 6, f"advanced {src.n_advance}, expected 6 (2 envs x 3)"
    assert src.n_redraft == 2, f"re-drafted {src.n_redraft}, expected 2 (one block each)"
    cache = src.current(np.zeros(n_envs, dtype=np.int64), obs_latent=None, replan_idx=[])
    assert torch.count_nonzero(cache[1]) == 0 and torch.count_nonzero(cache[3]) == 0, \
        "an env that never replanned received a subgoal"
    print(f"[commit] partial replan: {src.n_redraft} blocks serve {src.n_advance} "
          f"boundaries across 2 of 4 envs; idle envs untouched")
