"""TASK A (Reacher variant) - subgoal-dataset builder.

Reacher analog of build_subgoals.py / build_subgoals_tworoom.py. One structural
difference from both: the LeWM Reacher dataset (quentinll/lewm-reacher,
dmc/reacher_random.h5) is RANDOM-POLICY data - there are no "successful demos"
to filter, and no canonical target to filter against. Every episode is a valid
random walk through (shoulder, wrist) space, so the success filter is dropped
entirely and every considered episode is kept.

That same fact is why the Reacher drafter must be GOAL-CONDITIONED
(train_gdm --goal-cond --goal-rule window): a goal-free drafter trained on
random walks predicts undirected drift (the TwoRoom lesson). The hindsight
window goal (z at m+G frames) is drawn from these dense latents at training
time, so this builder just encodes EVERY frame (--stride 1) of every episode.

Examples:
  # GH200 full run (frozen reacher encoder downloaded from HF):
  python -m ffjepa.build_subgoals_reacher --source local --local-dir encoder_reacher \
      --h5 /scratch/u6ko/hasoshu.u6ko/data/reacher/reacher.h5 \
      --out subgoals_reacher_dense.pt --stride 1 --device cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import h5py

from ffjepa import lewm_io


def parse_args():
    p = argparse.ArgumentParser(description="FF-JEPA Task A (Reacher): subgoal-dataset builder")
    p.add_argument("--h5", required=True, help="reacher h5 dataset path")
    p.add_argument("--out", required=True, help="output .pt path")
    p.add_argument("--source", choices=["pretrained", "local"], default="local")
    p.add_argument("--encoder-id", default="quentinll/lewm-reacher")
    p.add_argument("--local-dir", default="encoder_reacher",
                   help="dir with config.json+weights.pt (source=local)")
    p.add_argument("--swm-src", default=None, help="stable-worldmodel checkout (CPU import)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=1,
                   help="frame subsample stride; 1 (dense) so train_gdm --subgoal-step S "
                        "can carve any stride from one file (the PushT stride-sweep scheme)")
    p.add_argument("--max-episodes", type=int, default=None, help="limit #episodes considered")
    p.add_argument("--sample-mode", choices=["head", "spread", "random"], default="head")
    p.add_argument("--sample-seed", type=int, default=0)
    return p.parse_args()


def select_episodes(n_total, max_episodes, mode, seed):
    if max_episodes is None or max_episodes >= n_total:
        return np.arange(n_total)
    if mode == "head":
        return np.arange(max_episodes)
    if mode == "spread":
        return np.unique(np.linspace(0, n_total - 1, max_episodes).astype(int))
    if mode == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n_total, size=max_episodes, replace=False))
    raise ValueError(mode)


def main():
    args = parse_args()
    t0 = time.time()
    print(f"[cfg] {vars(args)}")

    model = lewm_io.load_lewm(
        source=args.source, encoder_id=args.encoder_id,
        local_dir=args.local_dir, swm_src=args.swm_src, device=args.device,
    )
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[model] frozen LeWM loaded: {nparams/1e6:.2f}M params, device={args.device}, "
          f"training={model.training}")

    with h5py.File(args.h5, "r") as f:
        ep_off = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
        pixels = f["pixels"]
        n_total = len(ep_off)
        ep_ids = select_episodes(n_total, args.max_episodes, args.sample_mode, args.sample_seed)
        print(f"[data] {args.h5}: {n_total} episodes total; considering {len(ep_ids)} "
              f"(mode={args.sample_mode}); ep_len min={ep_len.min()} max={ep_len.max()} "
              f"mean={ep_len.mean():.1f}")

        # no success filter (random-policy data: every episode kept)
        kept_rows_per_ep = []
        kept_ids, kept_len = [], []
        for e in ep_ids:
            off, L = int(ep_off[e]), int(ep_len[e])
            rows = np.arange(off, off + L, args.stride)
            kept_ids.append(int(e))
            kept_len.append(L)
            kept_rows_per_ep.append(rows)
        n_kept = len(kept_ids)
        lengths = np.array([len(r) for r in kept_rows_per_ep], dtype=np.int64)
        n_total_rows = int(lengths.sum())
        print(f"[subsample] stride={args.stride}: {n_total_rows} frames to encode; "
              f"n_sg/episode min={lengths.min()} max={lengths.max()} mean={lengths.mean():.2f}")

        # encode, one contiguous h5 slice per episode (fast; avoids scattered
        # point-selection - see build_subgoals.py)
        lat_chunks = []
        done = 0
        for ep_i, rows in enumerate(kept_rows_per_ep):
            lo, hi = int(rows[0]), int(rows[-1]) + 1
            span = pixels[lo:hi]
            sel = span[rows - lo]
            z = lewm_io.encode_frames(model, sel, device=args.device, batch_size=args.batch_size)
            lat_chunks.append(z)
            done += len(rows)
            if ep_i % 200 == 0:
                print(f"  encoded {done}/{n_total_rows} frames over {ep_i+1} eps "
                      f"({time.time()-t0:.0f}s)", flush=True)

    latents = torch.cat(lat_chunks, 0).contiguous().float()
    assert latents.shape[0] == n_total_rows
    offsets = np.zeros(n_kept, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)[:-1]

    norms = latents.norm(dim=1)
    norm_stats = dict(mean=float(norms.mean()), std=float(norms.std()),
                      min=float(norms.min()), max=float(norms.max()))

    D = latents.shape[1]
    out = {
        "latents": latents,                                   # (total_nsg, D) float32
        "lengths": torch.from_numpy(lengths),                 # (K,) n_sg per kept episode
        "offsets": torch.from_numpy(offsets),                 # (K,) start idx into latents
        "episode_idx": torch.tensor(kept_ids, dtype=torch.long),
        "ep_len": torch.tensor(kept_len, dtype=torch.long),
        "in_5deg": torch.ones(n_kept, dtype=torch.bool),      # no success tiers for Reacher;
                                                              # kept for train_gdm.py --mask compat
        "stride": args.stride,
        "latent_dim": int(D),
        "criterion": {"mode": "reacher_random_no_filter"},
        "encoder": {"source": args.source, "encoder_id": args.encoder_id,
                    "local_dir": args.local_dir},
        "counts": {"episodes_considered": int(len(ep_ids)),
                   "total_episodes": int(n_total),
                   "kept_headline": int(n_kept),
                   "dropped": 0},
        "norm_stats": norm_stats,
        "sampling": {"max_episodes": args.max_episodes, "mode": args.sample_mode,
                     "seed": args.sample_seed},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    sz = Path(args.out).stat().st_size / 1e6
    print(f"[save] {args.out}  ({sz:.1f} MB)")
    print(f"[norm] ||z|| mean={norm_stats['mean']:.2f} std={norm_stats['std']:.2f} "
          f"min={norm_stats['min']:.2f} max={norm_stats['max']:.2f}  (sqrt({D})={np.sqrt(D):.2f})")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
