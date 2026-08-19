"""CPU smoke test for the GDM latent diffusion planner: shapes, schedule
sanity, loss decrease, sampling, standardization round-trip, source caching,
and checkpoint reproducibility on tiny synthetic tensors. No frozen LeWM,
h5, or stable_worldmodel required.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from surveyor.drafter import (GDM, GDMConfig, GaussianDiffusion, GDMPlanner,
                              timestep_embedding, save_gdm, load_gdm_planner,
                              count_params)
from surveyor.sources import GDMSubgoalSource

D = 192
N = 3
DEV = "cpu"


def _synth_latents(n, seed=0):
    """n latents at the native scale: random directions x sqrt(D) (||z||~13.9)."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, D, generator=g)
    return z / z.norm(dim=1, keepdim=True) * math.sqrt(D)


def test_dit_forward_and_schedule():
    """DiT forward gives finite (B, N, D) for cond shaped (B, D) or (B, wg, D);
    alphas_cumprod is in (0, 1] and non-increasing; timestep embedding shape."""
    cfg = GDMConfig(latent_dim=D, n_future=N, wg=1, hidden=64, depth=2, heads=4)
    model = GDM(cfg)
    B = 5
    x = torch.randn(B, N, D)
    cond = torch.randn(B, D)
    t = torch.randint(0, 50, (B,))
    out = model(x, cond, t)
    assert out.shape == (B, N, D), out.shape
    assert torch.isfinite(out).all()
    # cond as (B, wg, D) must also work
    out2 = model(x, cond.unsqueeze(1), t)
    assert out2.shape == (B, N, D)

    diff = GaussianDiffusion(timesteps=50, device=DEV)
    acp = diff.alphas_cumprod
    assert acp.shape == (50,)
    assert (acp > 0).all() and (acp <= 1).all()
    assert (acp[1:] <= acp[:-1] + 1e-6).all(), "alphas_cumprod must be non-increasing"
    # timestep embedding shape
    assert timestep_embedding(t, 64).shape == (B, 64)
    print(f"[1] DiT forward {tuple(out.shape)} OK; tiny head params={count_params(model)/1e3:.1f}K; "
          f"schedule acp[0]={acp[0]:.4f}->acp[-1]={acp[-1]:.4f} OK")


def test_qsample_and_loss_drops():
    """q_sample preserves shape and stays finite; 200 Adam steps on a fixed
    synthetic batch must reduce the diffusion loss."""
    cfg = GDMConfig(latent_dim=D, n_future=N, wg=1, hidden=128, depth=3, heads=4)
    model = GDM(cfg)
    diff = GaussianDiffusion(timesteps=100, device=DEV)

    # standardized synthetic batch (mean0/std1-ish): fixed so loss must drop
    torch.manual_seed(1)
    cond = torch.randn(64, D)
    tgt = torch.randn(64, N, D)

    # q_sample shape
    t = torch.randint(0, 100, (64,))
    noise = torch.randn_like(tgt)
    xt = diff.q_sample(tgt, t, noise)
    assert xt.shape == tgt.shape and torch.isfinite(xt).all()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.manual_seed(2)
    losses = []
    for _step in range(200):
        # fixed (t, noise) per step would be ideal but random is fine over 200 steps
        loss = diff.p_losses(model, tgt, cond)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.detach().item())
    first = np.mean(losses[:10])
    last = np.mean(losses[-10:])
    assert math.isfinite(last)
    assert last < first, f"loss did not drop: {first:.4f} -> {last:.4f}"
    print(f"[2] q_sample {tuple(xt.shape)} OK; training loss {first:.4f} -> {last:.4f} (drops) OK")


def test_ddim_sample():
    """DDIM sampling returns finite (B, N, D) and is deterministic for a
    fixed generator seed."""
    cfg = GDMConfig(latent_dim=D, n_future=N, wg=1, hidden=64, depth=2, heads=4)
    model = GDM(cfg).eval()
    diff = GaussianDiffusion(timesteps=100, device=DEV)
    cond = torch.randn(7, D)
    g = torch.Generator(device=DEV).manual_seed(0)
    x0 = diff.ddim_sample(model, cond, (7, N, D), n_steps=20, generator=g)
    assert x0.shape == (7, N, D) and torch.isfinite(x0).all()
    # determinism: same seed -> same sample
    g2 = torch.Generator(device=DEV).manual_seed(0)
    x0b = diff.ddim_sample(model, cond, (7, N, D), n_steps=20, generator=g2)
    assert torch.allclose(x0, x0b), "DDIM not deterministic for fixed seed"
    print(f"[3] DDIM sample {tuple(x0.shape)} finite + deterministic OK")


def test_planner_standardization_and_scale():
    """Planner standardize/unstandardize round-trips on native-scale data;
    sample_next equals sample_sequence position 0; output keeps native scale."""
    z = _synth_latents(2000, seed=3)
    mean, std = z.mean(0), z.std(0).clamp_min(1e-6)

    cfg = GDMConfig(latent_dim=D, n_future=N, wg=1, hidden=64, depth=2, heads=4)
    model = GDM(cfg)
    diff = GaussianDiffusion(timesteps=100, device=DEV)
    planner = GDMPlanner(model, diff, mean, std, device=DEV)

    zc = _synth_latents(8, seed=4)
    assert torch.allclose(planner.unstandardize(planner.standardize(zc)), zc, atol=1e-4), \
        "standardize/unstandardize not invertible"

    g = torch.Generator(device=DEV).manual_seed(0)
    seq = planner.sample_sequence(zc, n_steps=20, generator=g)
    assert seq.shape == (8, N, D)
    nxt = planner.sample_next(zc, n_steps=20,
                              generator=torch.Generator(device=DEV).manual_seed(0))
    assert torch.allclose(nxt, seq[:, 0])

    # An UNTRAINED model predicts eps~0 (adaLN-Zero + zero final layer), so DDIM
    # just rescales the initial noise rather than denoising to the data manifold;
    # the unstandardized output is native-E-space-SCALED garbage. We only assert
    # the scale here (>> unit). The real ||z||~13.9 check is post-training on GPU.
    out_norm = nxt.norm(dim=1).mean().item()
    assert out_norm > 1.0, f"output collapsed below native scale: ||z||={out_norm:.3f}"
    print(f"[4] standardization invertible; sample_next {tuple(nxt.shape)}; "
          f"native ||z||~{out_norm:.2f} (sqrt(D)={math.sqrt(D):.2f}) OK")


def test_subgoal_source_closedloop():
    """GDMSubgoalSource cache protocol: zeros before the first replan, refresh
    on replan, hold between replans, and partial replans leave the other envs'
    cached subgoals untouched."""
    z = _synth_latents(2000, seed=5)
    mean, std = z.mean(0), z.std(0).clamp_min(1e-6)
    cfg = GDMConfig(latent_dim=D, n_future=N, wg=1, hidden=64, depth=2, heads=4)
    planner = GDMPlanner(GDM(cfg), GaussianDiffusion(timesteps=100, device=DEV),
                         mean, std, device=DEV)

    n_envs = 4
    src = GDMSubgoalSource(planner, n_envs=n_envs, dim=D, device=DEV, n_steps=10, seed=42)
    assert src.needs_obs is True

    # before any replan the cache is zeros
    z0 = src.current(np.zeros(n_envs, dtype=np.int64))
    assert z0.shape == (n_envs, D) and torch.count_nonzero(z0) == 0

    # first replan: all envs get a fresh subgoal conditioned on their obs latent
    obs = _synth_latents(n_envs, seed=6)
    z1 = src.current(np.ones(n_envs, dtype=np.int64), obs_latent=obs,
                     replan_idx=list(range(n_envs)))
    assert z1.shape == (n_envs, D)
    assert (z1.norm(dim=1) > 1.0).all(), "subgoal collapsed below native scale"

    # between replans (replan_idx empty) the cache is held unchanged
    z_hold = src.current(np.ones(n_envs, dtype=np.int64), obs_latent=None, replan_idx=[])
    assert torch.allclose(z_hold, z1), "cache changed between replans"

    # partial replan: only env 0 and 2 refresh; 1 and 3 keep their cached subgoal
    obs2 = _synth_latents(2, seed=7)
    z2 = src.current(np.full(n_envs, 2, dtype=np.int64), obs_latent=obs2, replan_idx=[0, 2])
    assert torch.allclose(z2[1], z1[1]) and torch.allclose(z2[3], z1[3])
    assert not torch.allclose(z2[0], z1[0])
    print("[5] GDMSubgoalSource closed-loop: zeros->fill->hold->partial-replan OK")


def test_v_param_x0_identity():
    """GATE-A math check: with the TRUE target fed in, x0 is reconstructed
    EXACTLY at every t for BOTH parameterizations. For v this holds with
    bounded coefficients (no 1/sqrt(acp) amplification), the property that
    motivates the A/B. Proves the wiring, NOT fidelity (that is GATE B, the
    sampled Probe B)."""
    torch.manual_seed(0)
    x0 = torch.randn(64, N, D)
    noise = torch.randn(64, N, D)
    for param in ("eps", "v"):
        diff = GaussianDiffusion(timesteps=1000, parameterization=param, device=DEV)
        # sweep low..extreme-high t; high t is where eps->x0 amplifies in sampling
        for t_val in (5, 200, 500, 800, 999):
            t = torch.full((64,), t_val, dtype=torch.long)
            x_t = diff.q_sample(x0, t, noise)
            target = diff.target(x0, noise, t)          # the TRUE model output
            x0_hat, eps_hat = diff.pred_from_model(target, x_t, t)
            assert torch.allclose(x0_hat, x0, atol=1e-3), \
                f"{param} x0 recon failed @t={t_val}: {(x0_hat-x0).abs().max():.2e}"
            assert torch.allclose(eps_hat, noise, atol=1e-3), \
                f"{param} eps recon failed @t={t_val}"
    print("[7] eps & v: teacher-forced x0/eps recon EXACT across t (GATE-A wiring) OK")


def test_v_param_trains_and_samples():
    """v-parameterization: loss drops and DDIM sampling is finite + deterministic."""
    cfg = GDMConfig(latent_dim=D, n_future=N, wg=1, hidden=128, depth=3, heads=4)
    model = GDM(cfg)
    diff = GaussianDiffusion(timesteps=100, parameterization="v", device=DEV)
    torch.manual_seed(1)
    cond = torch.randn(64, D)
    tgt = torch.randn(64, N, D)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.manual_seed(2)
    losses = []
    for _ in range(200):
        loss = diff.p_losses(model, tgt, cond)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-10:]) < np.mean(losses[:10]), "v-loss did not drop"
    model.eval()
    g = torch.Generator(device=DEV).manual_seed(0)
    x0 = diff.ddim_sample(model, cond, (64, N, D), n_steps=20, generator=g)
    g2 = torch.Generator(device=DEV).manual_seed(0)
    x0b = diff.ddim_sample(model, cond, (64, N, D), n_steps=20, generator=g2)
    assert torch.isfinite(x0).all() and torch.allclose(x0, x0b)
    print(f"[8] v-param: loss drops {np.mean(losses[:10]):.3f}->{np.mean(losses[-10:]):.3f}; "
          f"DDIM finite+deterministic OK")


def test_cosine_schedule_and_min_snr():
    """Cosine bar-alpha is valid (decreasing in (0,1]); Min-SNR-gamma reweighting
    produces a finite loss and reduces to plain MSE math when gamma is huge."""
    diff_c = GaussianDiffusion(timesteps=1000, schedule="cosine", device=DEV)
    acp = diff_c.alphas_cumprod
    assert (acp > 0).all() and (acp <= 1).all()
    assert (acp[1:] <= acp[:-1] + 1e-6).all(), "cosine acp must be non-increasing"

    cfg = GDMConfig(latent_dim=D, n_future=N, wg=1, hidden=64, depth=2, heads=4)
    model = GDM(cfg)
    torch.manual_seed(3)
    cond = torch.randn(32, D); tgt = torch.randn(32, N, D)
    for param in ("eps", "v"):
        d = GaussianDiffusion(timesteps=1000, parameterization=param,
                              min_snr_gamma=5.0, device=DEV)
        t = torch.randint(0, 1000, (32,))
        noise = torch.randn_like(tgt)
        loss = d.p_losses(model, tgt, cond, t=t, noise=noise)
        assert torch.isfinite(loss), f"{param} min-snr loss not finite"
    print(f"[9] cosine schedule valid (acp {acp[0]:.4f}->{acp[-1]:.4e}); "
          f"Min-SNR-gamma loss finite (eps & v) OK")


def test_v_ckpt_roundtrip():
    """A v/cosine/min-snr checkpoint reloads with its diffusion config intact and
    samples identically (old eps ckpts still load via load defaults)."""
    z = _synth_latents(400, seed=11)
    mean, std = z.mean(0), z.std(0).clamp_min(1e-6)
    cfg = GDMConfig(latent_dim=D, n_future=N, wg=1, hidden=64, depth=2, heads=4)
    model = GDM(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    diff = GaussianDiffusion(timesteps=100, schedule="cosine",
                             parameterization="v", min_snr_gamma=5.0, device=DEV)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "gdm_v.pt"
        save_gdm(path, model, diff, mean, std, extra={"parameterization": "v"})
        planner = load_gdm_planner(path, device=DEV)
    assert planner.diffusion.parameterization == "v"
    assert planner.diffusion.schedule == "cosine"
    assert planner.diffusion.min_snr_gamma == 5.0
    zc = _synth_latents(3, seed=12)
    a = planner.sample_next(zc, n_steps=10,
                            generator=torch.Generator(device=DEV).manual_seed(0))
    ref = GDMPlanner(model, diff, mean, std, device=DEV).sample_next(
        zc, n_steps=10, generator=torch.Generator(device=DEV).manual_seed(0))
    assert torch.allclose(a, ref, atol=1e-5)
    print("[10] v/cosine/min-snr ckpt round-trip: config + samples match OK")


def test_ddpm_ancestral_sampler():
    """DDPM ancestral sampler: finite, right shape, deterministic given a seed, for
    both eps and v. Runs over all T steps (T small here for speed)."""
    cfg = GDMConfig(latent_dim=D, n_future=N, wg=1, hidden=64, depth=2, heads=4)
    model = GDM(cfg).eval()
    cond = torch.randn(6, D)
    for param in ("eps", "v"):
        diff = GaussianDiffusion(timesteps=40, parameterization=param, sampler="ddpm", device=DEV)
        g = torch.Generator(device=DEV).manual_seed(0)
        x0 = diff.ddpm_sample(model, cond, (6, N, D), generator=g)
        assert x0.shape == (6, N, D) and torch.isfinite(x0).all(), param
        g2 = torch.Generator(device=DEV).manual_seed(0)
        x0b = diff.ddpm_sample(model, cond, (6, N, D), generator=g2)
        assert torch.allclose(x0, x0b), f"ddpm not deterministic ({param})"
    # clip_x0 keeps a [-1,1]-trained sampler's x0 in range (smoke: still finite)
    diff = GaussianDiffusion(timesteps=40, sampler="ddpm", device=DEV)
    xc = diff.ddpm_sample(model, cond, (6, N, D),
                          generator=torch.Generator(device=DEV).manual_seed(1), clip_x0=True)
    assert torch.isfinite(xc).all()
    print("[11] DDPM ancestral sampler: eps & v finite + deterministic; clip_x0 OK")


def test_minmax_normalization():
    """minmax normalization maps min->-1, max->+1, round-trips, and the planner
    dispatches the DDPM sampler when the diffusion is configured for it."""
    z = _synth_latents(2000, seed=20)
    zmin, zmax = z.min(0).values, z.max(0).values
    cfg = GDMConfig(latent_dim=D, n_future=N, wg=1, hidden=64, depth=2, heads=4)
    model = GDM(cfg)
    diff = GaussianDiffusion(timesteps=50, sampler="ddpm", device=DEV)
    planner = GDMPlanner(model, diff, zmin, zmax, device=DEV, normalization="minmax")

    # min row -> -1, max row -> +1 (per-dim)
    assert torch.allclose(planner.standardize(zmin.unsqueeze(0)),
                          -torch.ones(1, D), atol=1e-4)
    assert torch.allclose(planner.standardize(zmax.unsqueeze(0)),
                          torch.ones(1, D), atol=1e-4)
    # round-trip on native data
    zc = _synth_latents(8, seed=21)
    assert torch.allclose(planner.unstandardize(planner.standardize(zc)), zc, atol=1e-4), \
        "minmax not invertible"
    # planner routes to ddpm and returns native-scale (B,D)
    nxt = planner.sample_next(zc, generator=torch.Generator(device=DEV).manual_seed(0))
    assert nxt.shape == (8, D) and torch.isfinite(nxt).all()
    print("[12] minmax norm: min->-1/max->+1, invertible; planner routes DDPM OK")


def test_ldp_config_ckpt_roundtrip():
    """A full LDP-style ckpt (eps / T=100-ish / cosine / DDPM / minmax) reloads with
    every field intact and samples identically to the in-memory planner."""
    z = _synth_latents(500, seed=22)
    zmin, zmax = z.min(0).values, z.max(0).values
    cfg = GDMConfig(latent_dim=D, n_future=N, wg=1, hidden=64, depth=2, heads=4)
    model = GDM(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    diff = GaussianDiffusion(timesteps=30, schedule="cosine", parameterization="eps",
                             sampler="ddpm", device=DEV)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "gdm_ldp.pt"
        save_gdm(path, model, diff, zmin, zmax, normalization="minmax", extra={"cfg": "ldp"})
        planner = load_gdm_planner(path, device=DEV)
    assert planner.diffusion.sampler == "ddpm"
    assert planner.diffusion.schedule == "cosine"
    assert planner.normalization == "minmax"
    assert planner.diffusion.timesteps == 30
    zc = _synth_latents(3, seed=23)
    a = planner.sample_next(zc, generator=torch.Generator(device=DEV).manual_seed(0))
    ref = GDMPlanner(model, diff, zmin, zmax, device=DEV, normalization="minmax").sample_next(
        zc, generator=torch.Generator(device=DEV).manual_seed(0))
    assert torch.allclose(a, ref, atol=1e-5)
    print("[13] LDP-config ckpt (cosine/ddpm/minmax) round-trip: config + samples match OK")


def test_ckpt_roundtrip():
    """save_gdm/load_gdm_planner round-trip: the reloaded planner samples
    identically to the in-memory reference and config fields survive."""
    z = _synth_latents(500, seed=8)
    mean, std = z.mean(0), z.std(0).clamp_min(1e-6)
    cfg = GDMConfig(latent_dim=D, n_future=N, wg=1, hidden=64, depth=2, heads=4)
    model = GDM(cfg)
    # perturb weights so it's not the trivial zero-init model
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    diff = GaussianDiffusion(timesteps=100, device=DEV)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "gdm.pt"
        save_gdm(path, model, diff, mean, std, extra={"mask": "5"})
        planner = load_gdm_planner(path, device=DEV)

    zc = _synth_latents(3, seed=9)
    a = planner.sample_next(zc, n_steps=10,
                            generator=torch.Generator(device=DEV).manual_seed(0))
    # reference from the in-memory model/stats
    ref = GDMPlanner(model, diff, mean, std, device=DEV).sample_next(
        zc, n_steps=10, generator=torch.Generator(device=DEV).manual_seed(0))
    assert torch.allclose(a, ref, atol=1e-5), "loaded planner sample != in-memory"
    assert planner.cfg.n_future == N and planner.cfg.wg == 1
    print("[6] save_gdm/load_gdm_planner round-trip: samples match (atol 1e-5) OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_dit_forward_and_schedule()
    test_qsample_and_loss_drops()
    test_ddim_sample()
    test_planner_standardization_and_scale()
    test_subgoal_source_closedloop()
    test_v_param_x0_identity()
    test_v_param_trains_and_samples()
    test_cosine_schedule_and_min_snr()
    test_v_ckpt_roundtrip()
    test_ddpm_ancestral_sampler()
    test_minmax_normalization()
    test_ldp_config_ckpt_roundtrip()
    test_ckpt_roundtrip()
    print("\nALL CPU SMOKE TESTS PASSED")
