"""Offline k derivation for best-of-k paired drafting (TwoRoom).

Best-of-k samples k candidate blocks per re-draft and serves the one scoring
best in the DINOv2 half (specaccept/paired.py). k must be derived OFFLINE --
the project rule is that no constant is tuned against a closed-loop number --
so this probe measures, on held-out conditioning points from the training
file, how the selected candidate's score improves with k and where it
saturates.

DERIVATION RULE, fixed before reading any numbers:
    k* = the smallest k in {2, 4, 8, 16} whose median selected score captures
    at least 80 percent of the k=16 gain over k=1 (per scoring rule).
The 0.80 cutoff is a chosen constant, same epistemic status as the stride
saturation cutoff; the defensible output is the saturation CURVE, and k* is
just where we stop paying diffusion calls for it.

Also reported, per k: the selected block's rel L2 against the TRUE future in
BOTH halves. Selection happens in the DINOv2 half only, so the LeWM column is
the check that picking by DINO score does not hand the planner a worse
waypoint (if it does, best-of-k is dead offline and never goes closed-loop).

Candidates are nested (the k=4 pool is the first 4 of the k=16 pool), so the
curves are monotone by construction and the comparison across k is paired.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from specaccept.drafter import load_gdm_planner
from specaccept.paired import DINO_DIM, LEWM_DIM


KS = (1, 2, 4, 8, 16)


def parse_args():
    p = argparse.ArgumentParser(description="best-of-k offline derivation (TwoRoom)")
    p.add_argument("--subgoals", default="subgoals_tworoom_s10_paired.pt")
    p.add_argument("--ckpt", default="gdm_tworoom_s10_gc_paired_e400v.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=256, help="conditioning points")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    blob = torch.load(args.subgoals, map_location="cpu", weights_only=False)
    lat = blob["latents"].float()
    lengths = blob["lengths"].numpy()
    offsets = blob["offsets"].numpy()
    print(f"[data] {args.subgoals}: latents {tuple(lat.shape)}")

    pl = load_gdm_planner(args.ckpt, device=args.device)
    N = pl.cfg.n_future
    assert int(pl.cfg.latent_dim) == LEWM_DIM + DINO_DIM, "paired drafter required"
    print(f"[gdm] {args.ckpt}: N={N} goal_cond={pl.goal_cond}")

    conds, trues, goals = [], [], []
    ok = np.where(lengths >= N + 2)[0]
    for _ in range(args.n):
        e = int(rng.choice(ok))
        off, L = int(offsets[e]), int(lengths[e])
        m = int(rng.integers(0, L - N - 1))
        conds.append(lat[off + m])
        trues.append(lat[off + m + 1: off + m + 1 + N])
        goals.append(lat[off + L - 1])
    z_cond = torch.stack(conds).to(args.device)                     # (n, 576)
    z_true = torch.stack(trues).to(args.device)                     # (n, N, 576)
    z_goal = torch.stack(goals).to(args.device)                     # (n, 576)

    kmax = max(KS)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    with torch.no_grad():
        cand = pl.sample_sequence(
            z_cond.repeat_interleave(kmax, dim=0), n_steps=args.steps,
            generator=gen,
            z_goal_native=(z_goal.repeat_interleave(kmax, dim=0)
                           if pl.goal_cond else None))
    cand = cand.reshape(args.n, kmax, N, -1)                        # (n, K, N, 576)
    d_cand = cand[..., LEWM_DIM:]                                   # (n, K, N, 384)
    d_goal = z_goal[:, LEWM_DIM:]
    d_now = z_cond[:, LEWM_DIM:]

    def rell2(a, b):
        return (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-8)

    scores = {
        "goal": rell2(d_cand[:, :, -1], d_goal[:, None]),           # (n, K)
        "feas": rell2(d_cand[:, :, 0], d_now[:, None]),
    }

    for rule, s in scores.items():
        print(f"\n=== rule '{rule}' ===")
        med = {}
        for k in KS:
            pick = s[:, :k].argmin(dim=1)                           # (n,)
            sel = cand[torch.arange(args.n, device=cand.device), pick]  # (n, N, 576)
            med[k] = float(s[:, :k].min(dim=1).values.median())
            fid_d = [float(rell2(sel[:, p, LEWM_DIM:], z_true[:, p, LEWM_DIM:])
                           .median()) for p in range(N)]
            fid_l = [float(rell2(sel[:, p, :LEWM_DIM], z_true[:, p, :LEWM_DIM])
                           .median()) for p in range(N)]
            print(f"  k={k:2d}  sel score p50 {med[k]:.4f} | "
                  f"dino rel-l2 vs true {['%.3f' % v for v in fid_d]} | "
                  f"lewm {['%.3f' % v for v in fid_l]}")
        gain16 = med[1] - med[16]
        kstar = 16
        for k in (2, 4, 8):
            if gain16 > 0 and (med[1] - med[k]) >= 0.80 * gain16:
                kstar = k
                break
        print(f"  k=16 gain over k=1: {gain16:.4f}  ->  derived k* = {kstar}"
              + ("  (NO GAIN: best-of-k has nothing to select on; "
                 "do not run it closed-loop)" if gain16 <= 1e-4 else ""))


if __name__ == "__main__":
    main()
