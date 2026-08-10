"""Offline block-quality battery for stride-S drafters: per-position sampled
fidelity (suffix decay) plus across-seed conditional spread in one pass.

Collapsed spread is flagged here before any closed-loop spend, and windows reuse
the anchor's episodes-file convention so strides stay directly comparable.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from surveyor import encoder
from surveyor.diag_gdm import metrics, show
from surveyor.drafter import load_gdm_planner


def parse_args():
    p = argparse.ArgumentParser(description="stride-S drafter battery: suffix decay + cond spread")
    p.add_argument("--gdm-ckpt", required=True)
    p.add_argument("--stride", type=int, required=True, help="S: env-steps between subgoal positions")
    p.add_argument("--h5", required=True)
    p.add_argument("--episodes-file", required=True)
    p.add_argument("--source", choices=["pretrained", "local"], default="local")
    p.add_argument("--encoder-id", default="quentinll/lewm-pusht")
    p.add_argument("--local-dir", default=None)
    p.add_argument("--swm-src", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-probe", type=int, default=256)
    p.add_argument("--n-seeds", type=int, default=8, help="sampler seeds for conditional spread")
    p.add_argument("--gdm-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json-out", default=None)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    planner = load_gdm_planner(args.gdm_ckpt, device=args.device)
    N = planner.cfg.n_future
    print(f"[gdm] ckpt={args.gdm_ckpt} N={N} T={planner.diffusion.timesteps} "
          f"sampler={planner.diffusion.sampler} horizon_frames="
          f"{planner.extra.get('subgoal_horizon_frames', '?') if hasattr(planner, 'extra') else '?'}")

    with open(args.episodes_file) as f:
        payload = json.load(f)
    pairs = payload["episodes"][:args.n_probe]
    print(f"[episodes-file] {args.episodes_file}: {len(pairs)} pairs "
          f"(goal_offset={payload.get('goal_offset')})")

    model = encoder.load_lewm(source=args.source, encoder_id=args.encoder_id,
                              local_dir=args.local_dir, swm_src=args.swm_src, device=args.device)

    import h5py
    with h5py.File(args.h5, "r") as f:
        ep_off = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
        pixels = f["pixels"]
        cond_rows, target_rows = [], [[] for _ in range(N)]
        for ep, st in pairs:
            base = int(ep_off[ep]); last = base + int(ep_len[ep]) - 1
            cond_rows.append(base + int(st))
            for k in range(N):
                row = base + int(st) + (k + 1) * args.stride
                assert row <= last, (f"episode {ep} too short for m+{k+1} at stride "
                                     f"{args.stride} (row {row} > last {last})")
                target_rows[k].append(row)
        all_rows = np.array(cond_rows + [r for tk in target_rows for r in tk])
        order = np.argsort(all_rows, kind="stable")
        lat = encoder.encode_frames(model, pixels[all_rows[order]], device=args.device)
        out = torch.empty_like(lat)
        out[order] = lat

    M = len(pairs)
    z_cond = out[:M].to(args.device)
    z_true = [out[M + k * M: M + (k + 1) * M].to(args.device) for k in range(N)]

    # K seeded samples of the full block for the SAME conditions
    preds = []
    for s in range(args.n_seeds):
        gen = torch.Generator(device=planner.device).manual_seed(args.seed + s)
        preds.append(planner.sample_sequence(z_cond, n_steps=args.gdm_steps, generator=gen))
    P = torch.stack(preds, dim=1)  # (M, K, N, D)

    print(f"\n[STRIDE-{args.stride} DRAFTER BATTERY] n={M}, K={args.n_seeds} seeds, "
          f"real sampling, per-position truth at +(k+1)*{args.stride} steps")
    results = {}
    for k in range(N):
        m = metrics(P[:, 0, k], z_true[k], z_cond)   # seed-0 = the suffix-decay readout
        spread = P[:, :, k].std(dim=1).mean().item()  # across-seed, same input
        m["cond_spread"] = spread
        show(f"position m+{k+1} (+{(k+1)*args.stride} steps)", m)
        print(f"  cond_spread(K={args.n_seeds}) = {spread:.4f}   "
              f"(refiner that hurt SR: 0.37-0.42 at S=25; healthy raw ~>0.1 and grows with k)")
        results[k + 1] = {key: float(v) for key, v in m.items()}

    print("\n[read-off] S=25 arm must reproduce the anchor triplet "
          "0.1593/0.2288/0.3179 (rel_err m+1/2/3). For finer S: rel_err must sit "
          "clearly under noop_err (degeneracy) and cond_spread must NOT collapse "
          "vs the S=25 reference.")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"args": vars(args), "n": M, "per_position": results}, fh, indent=2)
        print(f"[save] {args.json_out}")


if __name__ == "__main__":
    main()
