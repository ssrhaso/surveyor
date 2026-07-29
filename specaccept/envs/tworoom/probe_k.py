"""Derive k (DDIM sampling steps) for the TwoRoom paired drafter.

The TwoRoom row currently runs the 50-step default, the one constant in it
that is not derivation-backed. This probe applies probe_floor.py's section-C
rule verbatim to the paired drafter, measured in the VERIFICATION space (the
DINOv2 half, where tau lives):

  * dispersion per k: spread of the position-1 waypoint across n_draws
    repeated samples of the same conditioning (rel to the cond norm);
  * bias per k: rel L2 between the mean waypoint at k and at k_ref=50.

DERIVATION RULE (same as pusht 3 / reacher 8 / cube 3): k* = the smallest k
whose bias-to-k50 p50 is below the criterion floor p50 (0.098 here, from the
gap probe) -- i.e. the sampler's remaining bias is below what the verifier
can resolve, so more steps cannot change any accept decision. Deriving k
changes NFE/draft, not SR; a confirm run at k* completes the faithful row.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from specaccept.drafter import load_gdm_planner
from specaccept.paired import DINO_DIM, LEWM_DIM


KS = (2, 3, 4, 6, 8, 12, 25, 50)


def parse_args():
    p = argparse.ArgumentParser(description="TwoRoom paired-drafter k derivation")
    p.add_argument("--subgoals", default="subgoals_tworoom_s10_paired.pt")
    p.add_argument("--ckpt", default="gdm_tworoom_s10_gc_paired_e400v.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=256, help="conditioning points")
    p.add_argument("--n-draws", type=int, default=8)
    p.add_argument("--floor", type=float, default=0.098,
                   help="criterion floor p50 in the verification space "
                        "(docs/tworoom_paired_prereg.md, gap probe 2300010)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def pct(x):
    x = np.asarray(x)
    return {q: float(np.percentile(x, q)) for q in (10, 50, 90)}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    blob = torch.load(args.subgoals, map_location="cpu", weights_only=False)
    lat = blob["latents"].float()
    lengths = blob["lengths"].numpy()
    offsets = blob["offsets"].numpy()
    pl = load_gdm_planner(args.ckpt, device=args.device)
    N = pl.cfg.n_future
    assert int(pl.cfg.latent_dim) == LEWM_DIM + DINO_DIM
    print(f"[gdm] {args.ckpt}: N={N} goal_cond={pl.goal_cond}")

    conds, goals = [], []
    ok = np.where(lengths >= N + 2)[0]
    for _ in range(args.n):
        e = int(rng.choice(ok))
        off, L = int(offsets[e]), int(lengths[e])
        m = int(rng.integers(0, L - N - 1))
        conds.append(lat[off + m])
        goals.append(lat[off + L - 1])
    z_cond = torch.stack(conds).to(args.device)
    z_goal = torch.stack(goals).to(args.device) if pl.goal_cond else None
    d_cond_norm = z_cond[:, LEWM_DIM:].norm(dim=-1).clamp_min(1e-8)

    disp, mean_by_k = {}, {}
    with torch.no_grad():
        for k in KS:
            gen = torch.Generator(device=pl.device)
            gen.manual_seed(args.seed + k)
            draws = []
            for _ in range(args.n_draws):
                blk = pl.sample_sequence(z_cond, n_steps=k, generator=gen,
                                         z_goal_native=z_goal)
                draws.append(blk[:, 0, LEWM_DIM:].detach())     # served waypoint, dino half
            W = torch.stack(draws, 0)                           # (R, M, 384)
            mu = W.mean(dim=0)                                  # (M, 384)
            d = ((W - mu.unsqueeze(0)).norm(dim=-1) / d_cond_norm.unsqueeze(0))
            disp[k] = pct(d.flatten().cpu().numpy())
            mean_by_k[k] = mu
            print(f"[disp] k={k:>3d} p10/p50/p90 = "
                  f"{disp[k][10]:.4f}/{disp[k][50]:.4f}/{disp[k][90]:.4f}")

    ref = max(KS)
    print(f"\n[bias vs k={ref}] (verification-space rel L2 of mean waypoints)")
    bias = {}
    for k in KS:
        if k == ref:
            continue
        b = ((mean_by_k[k] - mean_by_k[ref]).norm(dim=-1)
             / mean_by_k[ref].norm(dim=-1).clamp_min(1e-8)).cpu().numpy()
        bias[k] = pct(b)
        print(f"[bias] k={k:>3d} p10/p50/p90 = "
              f"{bias[k][10]:.4f}/{bias[k][50]:.4f}/{bias[k][90]:.4f}")

    kstar = ref
    for k in KS[:-1]:
        if bias[k][50] < args.floor:
            kstar = k
            break
    print(f"\n==== verdict ====")
    print(f"criterion floor p50 = {args.floor}")
    print(f"derived k* = {kstar}  (smallest k with bias-to-k{ref} p50 below the floor;"
          f" more steps cannot change an accept decision)")
    for k in KS:
        tag = "dispersion above floor" if disp[k][50] > args.floor else "resolvable"
        print(f"  k={k:>3d}: dispersion p50 {disp[k][50]:.4f} ({tag})")


if __name__ == "__main__":
    main()
