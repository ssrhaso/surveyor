"""CPU smoke test: every package code path on tiny tensors, no upstream
clone imports, no pretrained weights, seconds to run.

  python -m specaccept.vjepa2.smoke_test
"""

from __future__ import annotations

import numpy as np
import torch

from .drafter import GaussianDiffusion, TokenGDM, TokenGDMConfig, TokenGDMPlanner
from .sources import CstarRetireTokenSource, GDMDraft, LerpBlockDrafter, SpecAcceptTokenSource
from .wm import pool

D, T_TOK, N = 32, 16, 3
torch.manual_seed(0)


def test_drafter():
    cfg = TokenGDMConfig(latent_dim=D, tokens_per_frame=T_TOK, n_future=N,
                         hidden=64, depth=2, heads=4, goal_cond=True)
    model = TokenGDM(cfg)
    diff = GaussianDiffusion(timesteps=50, schedule="cosine", parameterization="v",
                             min_snr_gamma=5.0, device="cpu")
    x0 = torch.randn(4, N * T_TOK, D)
    cond = torch.randn(4, D)
    goal = torch.randn(4, D)
    loss = diff.p_losses(model, x0, cond, goal=goal)
    assert loss.ndim == 0 and torch.isfinite(loss), loss
    loss.backward()

    planner = TokenGDMPlanner(model, diff, x0.reshape(-1, D).mean(0),
                              x0.reshape(-1, D).std(0), device="cpu")
    block = planner.sample_sequence(cond, n_steps=4, z_goal_pooled=goal)
    assert block.shape == (4, N, T_TOK, D), block.shape
    print(f"[drafter] p_losses={float(loss):.4f}, sample {tuple(block.shape)} OK")


def test_lerp_drafter():
    z = torch.zeros(2, T_TOK, D)
    g = torch.ones(2, T_TOK, D)
    blk = LerpBlockDrafter(n_future=N).draft(z, g)
    assert blk.shape == (2, N, T_TOK, D)
    fr = [float(blk[0, j, 0, 0]) for j in range(N)]
    assert np.allclose(fr, [(j + 1) / (N + 1) for j in range(N)]), fr
    print(f"[lerp] fractions {[f'{f:.2f}' for f in fr]} OK")


def test_spec_source():
    goal = torch.randn(1, T_TOK, D)
    src = SpecAcceptTokenSource(LerpBlockDrafter(n_future=N), n_envs=1,
                                tau=0.2, k=0)
    z = torch.randn(1, T_TOK, D)
    t1 = src.current(z, [0], goal)                    # first call -> redraft
    assert src.n_redraft == 1 and t1[0].shape == (T_TOK, D)
    # achieved == pursued target -> verify passes -> advance, no redraft
    achieved = t1[0].unsqueeze(0)
    src.current(achieved, [0], goal)
    assert src.n_advance == 1 and src.n_redraft == 1
    # achieved far from target -> reject -> redraft
    src.current(achieved + 10.0, [0], goal)
    assert src.n_reject == 1 and src.n_redraft == 2
    print(f"[spec] redraft/advance/reject = "
          f"{src.n_redraft}/{src.n_advance}/{src.n_reject} OK")


def test_retire_source():
    goal = torch.randn(1, T_TOK, D)
    cs = iter([0.9, 0.5, 0.15, 0.1])                  # falls through tau=0.2

    def fake_cstar(z, pose, g):
        return next(cs), None

    src = CstarRetireTokenSource(LerpBlockDrafter(n_future=N), fake_cstar,
                                 n_envs=1, tau=0.2, k=0)
    z = torch.randn(1, T_TOK, D)
    pose = torch.zeros(1, 1, 7)
    for step in range(4):
        t = src.current(z, pose, [0], goal)
    assert src.c_first[0] == 0.9 and not (src.c_first[0] <= 0.2), "router fire test"
    assert src._retired[0] and src.retire_replan[0] == 2, src.retire_replan
    assert torch.equal(t[0], goal[0]), "retired env must serve goal tokens"
    print(f"[retire] {src.stats()} OK")


def test_gdm_draft_adapter():
    cfg = TokenGDMConfig(latent_dim=D, tokens_per_frame=T_TOK, n_future=N,
                         hidden=64, depth=2, heads=4, goal_cond=True)
    model = TokenGDM(cfg)
    diff = GaussianDiffusion(timesteps=50, device="cpu")
    planner = TokenGDMPlanner(model, diff, torch.zeros(D), torch.ones(D))
    drafter = GDMDraft(planner, seed=1)
    blk = drafter.draft(torch.randn(2, T_TOK, D), torch.randn(2, T_TOK, D), k=3)
    assert blk.shape == (2, N, T_TOK, D)
    print(f"[gdm-draft] {tuple(blk.shape)} OK")


def test_residual_draft():
    cfg = TokenGDMConfig(latent_dim=D, tokens_per_frame=T_TOK, n_future=N,
                         hidden=64, depth=2, heads=4, goal_cond=True, residual=True)
    model = TokenGDM(cfg)
    diff = GaussianDiffusion(timesteps=50, device="cpu")
    planner = TokenGDMPlanner(model, diff, torch.zeros(D), torch.ones(D),
                              cond_stat_a=torch.zeros(D), cond_stat_b=torch.ones(D))
    z = torch.randn(2, T_TOK, D) * 100.0   # large grid: add-back must dominate output
    blk = GDMDraft(planner, seed=1).draft(z, torch.randn(2, T_TOK, D), k=2)
    assert blk.shape == (2, N, T_TOK, D)
    rel = float((blk - z.unsqueeze(1)).norm() / z.norm())
    assert rel < 0.5, f"residual add-back missing (rel dev {rel})"
    print(f"[residual] draft = cond grid + O(1) residual (rel dev {rel:.4f}) OK")


def test_pair_dataset(tmp="_smoke_lat.npz"):
    import os

    from .train_drafter import PairDataset, stats_from
    np.savez(tmp, tokens=np.random.randn(30, T_TOK, D).astype(np.float16))
    ds = PairDataset([tmp], stride=5, n_future=N)
    assert len(ds) == 30 - N * 5, len(ds)
    c, b, g = ds[0]
    assert c.shape == (T_TOK, D) and b.shape == (N, T_TOK, D) and g.shape == (T_TOK, D)
    c2, b2, g2 = ds[len(ds) - 1]
    assert torch.equal(pool(g), pool(g2)), "goal = last frame everywhere"
    sa, sb = stats_from([tmp], every=2)
    assert sa.shape == (D,) and sb.shape == (D,) and (sb > 0).all()
    os.remove(tmp)
    print(f"[pairs] {len(ds)} lazy pairs, block {tuple(b.shape)}, stats {tuple(sa.shape)} OK")


if __name__ == "__main__":
    test_drafter()
    test_lerp_drafter()
    test_spec_source()
    test_retire_source()
    test_gdm_draft_adapter()
    test_residual_draft()
    test_pair_dataset()
    print("\nALL SMOKE TESTS PASSED")
