"""Offline QC for the paired 576-d TwoRoom drafter.

The closed-loop battery reported verification distances pinned at rel L2 ~1.00
(raw) and ~1.42 (lens, where 1.414 is two unit vectors at a random angle),
i.e. the achieved state and the drafted waypoint were unrelated at EVERY
replan. That is either a drafter that never learned the DINOv2 half or a
serving-side fault, and the two are distinguishable offline in seconds.

Conditions the drafter on real subgoal latents from the training file and
compares its block against the true future, reporting the halves SEPARATELY
(LeWM 192 vs DINOv2 384) plus the norms of each, since a scale mismatch
between the halves is the most likely way a joint drafter silently drops one.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from surveyor.drafter import load_gdm_planner
from surveyor.paired import DINO_DIM, LEWM_DIM


def parse_args():
    p = argparse.ArgumentParser(description="paired TwoRoom drafter QC")
    p.add_argument("--subgoals", default="subgoals_tworoom_s10_paired.pt")
    p.add_argument("--ckpt", default="gdm_tworoom_s10_gc_paired.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=256, help="conditioning latents sampled")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def rep(tag, x):
    x = np.asarray(x)
    print("  %-28s p10/p50/p90 = %.4f / %.4f / %.4f"
          % (tag, np.percentile(x, 10), np.percentile(x, 50), np.percentile(x, 90)))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    blob = torch.load(args.subgoals, map_location="cpu", weights_only=False)
    lat = blob["latents"].float()
    lengths = blob["lengths"].numpy()
    offsets = blob["offsets"].numpy()
    print(f"[data] {args.subgoals}: latents {tuple(lat.shape)} "
          f"latent_dim={blob.get('latent_dim')} paired={blob.get('paired') is not None}")

    pl = load_gdm_planner(args.ckpt, device=args.device)
    N = pl.cfg.n_future
    print(f"[gdm] latent_dim={pl.cfg.latent_dim} N={N} goal_cond={pl.goal_cond} "
          f"norm={pl.normalization}")

    # sample (cond, true block, goal) triples from episodes long enough for N hops
    conds, trues, goals = [], [], []
    ok = np.where(lengths >= N + 2)[0]
    for _ in range(args.n):
        e = int(rng.choice(ok))
        off, L = int(offsets[e]), int(lengths[e])
        m = int(rng.integers(0, L - N - 1))
        conds.append(lat[off + m])
        trues.append(lat[off + m + 1: off + m + 1 + N])
        goals.append(lat[off + L - 1])
    z_cond = torch.stack(conds).to(args.device)
    z_true = torch.stack(trues).to(args.device)
    z_goal = torch.stack(goals).to(args.device)

    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    with torch.no_grad():
        blk = pl.sample_sequence(z_cond, n_steps=args.steps, generator=gen,
                                 z_goal_native=z_goal if pl.goal_cond else None)
    print(f"[draft] block {tuple(blk.shape)}")

    is_paired = int(pl.cfg.latent_dim) == LEWM_DIM + DINO_DIM

    def half(t, which):
        if not is_paired:
            return t
        return t[..., :LEWM_DIM] if which == "lewm" else t[..., LEWM_DIM:]

    halves = (("lewm", LEWM_DIM), ("dino", DINO_DIM)) if is_paired else (("all", pl.cfg.latent_dim),)

    print("\n=== NORMS (drafted vs true) ===")
    for which, _d in halves:
        dn = half(blk[:, 0], which).norm(dim=-1).cpu().numpy()
        tn = half(z_true[:, 0], which).norm(dim=-1).cpu().numpy()
        cn = half(z_cond, which).norm(dim=-1).cpu().numpy()
        print(f"  {which}: drafted {dn.mean():.3f}  true {tn.mean():.3f}  "
              f"cond(real) {cn.mean():.3f}  ratio drafted/true {dn.mean()/max(tn.mean(),1e-8):.3f}")

    print("\n=== REL L2 of drafted vs TRUE future, per half, per block position ===")
    for pos in range(N):
        for which, _d in halves:
            d = half(blk[:, pos], which)
            t = half(z_true[:, pos], which)
            rel = ((d - t).norm(dim=-1) / t.norm(dim=-1).clamp_min(1e-8)).cpu().numpy()
            rep(f"m+{pos+1} {which}", rel)

    print("\n=== BASELINE: rel L2 of the CONDITIONING frame vs the true future ===")
    print("    (a drafter must beat this to be adding anything)")
    for pos in range(N):
        for which, _d in halves:
            c = half(z_cond, which)
            t = half(z_true[:, pos], which)
            rel = ((c - t).norm(dim=-1) / t.norm(dim=-1).clamp_min(1e-8)).cpu().numpy()
            rep(f"m+{pos+1} {which} no-op", rel)

    print("\n=== what the closed loop actually compares ===")
    print("    achieved DINOv2 (a real frame) vs drafted DINOv2 half.")
    print("    If the drafted dino norm is far off the true norm, rel pins near")
    print("    1.0 regardless of tau and every waypoint is rejected.")


if __name__ == "__main__":
    main()
