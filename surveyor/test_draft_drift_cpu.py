"""CPU engagement gate for autocorrelated draft corruption (--draft-noise-rho).

The autocorrelated-divergence pre-registration fixes this criterion in the
document, before any cell:

    at sigma=0.2, rho=0.99 must produce a materially higher cosine similarity
    between consecutive drafts' injected displacement than rho=0
    (target: >0.9 against ~0 for white noise). If the two are
    indistinguishable the flag is inert, and NO array is submitted.

That is measured here directly, on the realised displacement directions rather
than inferred from SR, so an inert flag cannot reach a cluster array. It is the
same class of failure as job 2331836, which returned fifteen mislabelled
replicas of one configuration because nobody checked that the knob moved.

Synthetic tensors, no GPU/LeWM/h5/stable_worldmodel.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from surveyor.drafter import GDM, GDMConfig, GaussianDiffusion, GDMPlanner
from surveyor.sources import SurveyorSource

D = 192
N = 3
DEV = "cpu"
SIGMA = 0.2          # the prereg's smoke sigma
COS_BAR = 0.9        # the prereg's bar for rho=0.99
WHITE_BAR = 0.25     # "~0" for an uncorrelated draw in D=192


def _synth_latents(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, D, generator=g)
    return z / z.norm(dim=1, keepdim=True) * math.sqrt(D)


def _planner(seed=11):
    z = _synth_latents(2000, seed=seed)
    cfg = GDMConfig(latent_dim=D, n_future=N, wg=1, hidden=64, depth=2, heads=4)
    return GDMPlanner(GDM(cfg), GaussianDiffusion(timesteps=100, device=DEV),
                      z.mean(0), z.std(0).clamp_min(1e-6), device=DEV)


def _mean_consecutive_cos(rho, n_envs=4, n_drafts=12, seed=42):
    """Drive the source so every env re-drafts at every boundary (tau=0 rejects
    everything), then read the mean cos between each env's consecutive
    displacement directions."""
    src = SurveyorSource(_planner(), n_envs=n_envs, device=DEV, n_steps=5,
                         seed=seed, tau=0.0, draft_noise=SIGMA,
                         draft_noise_rho=rho)
    idx = list(range(n_envs))
    for b in range(n_drafts):
        obs = _synth_latents(n_envs, seed=500 + b)
        src.current(np.full(n_envs, b, dtype=np.int64), obs_latent=obs, replan_idx=idx)

    assert src.drift_log, "no corruption was injected at all"
    per_env = {i: [] for i in range(n_envs)}
    for rows, dirs in src.drift_log:
        for j, e in enumerate(rows):
            per_env[int(e)].append(dirs[j])
    cos = []
    for e, seq in per_env.items():
        for a, b_ in zip(seq[:-1], seq[1:]):
            cos.append(float(torch.dot(a, b_) / (a.norm() * b_.norm() + 1e-12)))
    assert cos, "fewer than two drafts per env; cannot measure autocorrelation"
    return float(np.mean(cos)), len(cos)


def test_rho_engages_the_drift():
    """The gate. rho=0.99 must hold its direction across re-drafts; rho=0 must
    not. Equal values mean the flag is inert."""
    c0, n0 = _mean_consecutive_cos(0.0)
    c99, n99 = _mean_consecutive_cos(0.99)

    assert abs(c0) < WHITE_BAR, \
        f"rho=0 should be uncorrelated, got mean cos={c0:.3f} over {n0} pairs"
    assert c99 > COS_BAR, \
        f"rho=0.99 should persist, got mean cos={c99:.3f} over {n99} pairs (bar {COS_BAR})"
    assert c99 - c0 > 0.5, f"flag inert: rho=0 -> {c0:.3f}, rho=0.99 -> {c99:.3f}"
    print(f"[drift] ENGAGEMENT PASS: mean consecutive cos "
          f"rho=0 -> {c0:+.3f}, rho=0.99 -> {c99:+.3f} "
          f"(bars: <{WHITE_BAR}, >{COS_BAR})")


def test_drift_is_monotone_in_rho():
    """P-DRIFT-3 is a dose-response claim, so the knob itself must be monotone
    in rho before any SR is read. If the mechanism were not ordered, an ordered
    SR result could not be attributed to it."""
    cs = [(_mean_consecutive_cos(r)[0], r) for r in (0.0, 0.9, 0.99)]
    vals = [c for c, _ in cs]
    assert vals[0] < vals[1] < vals[2], f"cos not monotone in rho: {cs}"
    print("[drift] monotone in rho: " +
          ", ".join(f"rho={r} -> {c:+.3f}" for c, r in cs))


def test_corruption_is_shared_across_the_block():
    """The chosen reading: one direction per ENV, shared across the N waypoints
    of a block, at every rho including 0. This is what makes rho the only
    quantity varying across arms, so the dose-response is unconfounded.
    Recorded in the prereg Outcome before any cell ran."""
    for rho in (0.0, 0.99):
        src = SurveyorSource(_planner(), n_envs=2, device=DEV, n_steps=5, seed=1,
                             tau=0.0, draft_noise=SIGMA, draft_noise_rho=rho)
        obs = _synth_latents(2, seed=77)
        src.current(np.zeros(2, dtype=np.int64), obs_latent=obs, replan_idx=[0, 1])
        rows, dirs = src.drift_log[0]
        assert dirs.shape == (2, D), \
            f"rho={rho}: expected one direction per env, got {tuple(dirs.shape)}"
        assert abs(float(dirs[0].norm()) - 1.0) < 1e-5, "direction not unit-norm"
    print(f"[drift] one unit direction per env, shared over N={N} waypoints, at every rho")


def test_zero_sigma_leaves_drafts_untouched():
    """Regression guard: every banked cell in the paper runs draft_noise=0, and
    must be unaffected by any of this. With sigma=0 no corruption is injected
    and no drift state is created at all."""
    src = SurveyorSource(_planner(), n_envs=3, device=DEV, n_steps=5, seed=9,
                         tau=0.0, draft_noise=0.0, draft_noise_rho=0.99)
    obs = _synth_latents(3, seed=31)
    src.current(np.zeros(3, dtype=np.int64), obs_latent=obs, replan_idx=[0, 1, 2])
    assert src.drift_log == [], "corruption injected at sigma=0"
    assert src._drift is None, "drift state allocated at sigma=0"
    print("[drift] sigma=0 untouched: no injection, no drift state (banked cells safe)")
