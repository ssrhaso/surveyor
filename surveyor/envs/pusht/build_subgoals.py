"""Subgoal-dataset builder for PushT (offline).

Builds per-episode stride-H subgoal-latent sequences from successful expert
episodes: filter against the canonical target (block-only criterion, with the
strict 5 deg subset flagged per episode), subsample at stride H, encode with the
frozen encoder, and pack into one .pt with metadata.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import h5py

from surveyor import encoder


def parse_args():
    p = argparse.ArgumentParser(description="FF-JEPA Task A: subgoal-dataset builder")
    # data / output
    p.add_argument("--h5", required=True, help="path to pusht_expert_train.h5")
    p.add_argument("--out", required=True, help="output .pt path")
    # model
    p.add_argument("--source", choices=["pretrained", "local"], default="pretrained")
    p.add_argument("--encoder-id", default="quentinll/lewm-pusht")
    p.add_argument("--local-dir", default=None, help="dir with config.json+weights.pt (source=local)")
    p.add_argument("--swm-src", default=None, help="stable-worldmodel checkout (CPU import)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=256)
    # filter / subsample knobs (exposed per spec)
    p.add_argument("--stride", type=int, default=25, help="H: subgoal stride (env steps)")
    p.add_argument("--angle-headline", type=float, default=20.0, help="keep tolerance (deg)")
    p.add_argument("--angle-strict", type=float, default=5.0, help="recorded subset tolerance (deg)")
    p.add_argument("--pos-thresh", type=float, default=encoder.POS_THRESH)
    # sampling (for CPU testing on a handful of episodes)
    p.add_argument("--max-episodes", type=int, default=None, help="limit #episodes considered")
    p.add_argument("--sample-mode", choices=["head", "spread", "random"], default="head")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--dino-pair", action="store_true",
                   help="also encode every subgoal frame with frozen DINOv2 and "
                        "store the concatenation [lewm(192) | dino(384)], as the "
                        "TwoRoom builder does (surveyor.paired). Control leg: "
                        "does DINOv2-space verification also work on an env whose "
                        "OWN space passes the gap probe?")
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
    model = encoder.load_lewm(
        source=args.source, encoder_id=args.encoder_id,
        local_dir=args.local_dir, swm_src=args.swm_src, device=args.device,
    )
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[model] frozen LeWM loaded: {nparams/1e6:.2f}M params, device={args.device}, "
          f"training={model.training}")

    dino = None
    if args.dino_pair:
        from surveyor import paired
        dino = paired.load_dinov2(device=args.device)
        print(f"[model] frozen DINOv2 loaded for the verifier half "
              f"({sum(p.numel() for p in dino.parameters())/1e6:.2f}M params)")

    with h5py.File(args.h5, "r") as f:
        ep_off = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
        state = f["state"]
        pixels = f["pixels"]
        n_total = len(ep_off)
        ep_ids = select_episodes(n_total, args.max_episodes, args.sample_mode, args.sample_seed)
        print(f"[data] {n_total} episodes total; considering {len(ep_ids)} "
              f"(mode={args.sample_mode})")

        # Pass 1: success filter on FINAL state (no pixel reads)
        kept_ids, kept_len, in_5deg = [], [], []
        kept_rows_per_ep = []           # stride-H absolute row indices per kept ep
        n_drop = 0
        for e in ep_ids:
            off, L = int(ep_off[e]), int(ep_len[e])
            final = state[off + L - 1].astype(np.float64)
            target = encoder.canonical_target_for(final)
            ok20 = encoder.eval_state_tol(target, final, args.angle_headline, args.pos_thresh)
            if not ok20:
                n_drop += 1
                continue
            ok5 = encoder.eval_state_tol(target, final, args.angle_strict, args.pos_thresh)
            rows = np.arange(off, off + L, args.stride)
            kept_ids.append(int(e))
            kept_len.append(L)
            in_5deg.append(ok5)
            kept_rows_per_ep.append(rows)

        n_kept = len(kept_ids)
        if n_kept == 0:
            raise RuntimeError("no episodes passed the success filter")
        lengths = np.array([len(r) for r in kept_rows_per_ep], dtype=np.int64)  # n_sg per ep
        all_rows = np.concatenate(kept_rows_per_ep)  # strictly increasing (episodes ordered)
        assert np.all(np.diff(all_rows) > 0), "row indices not strictly increasing"
        print(f"[filter] kept@{args.angle_headline:g}deg={n_kept}  "
              f"kept@{args.angle_strict:g}deg={int(np.sum(in_5deg))}  dropped={n_drop}  "
              f"({100*n_kept/len(ep_ids):.1f}% kept)")
        print(f"[subsample] stride={args.stride}: {int(all_rows.size)} subgoal frames to encode; "
              f"n_sg/episode min={lengths.min()} max={lengths.max()} mean={lengths.mean():.2f}")

        # Pass 2: encode kept stride-H frames. Each episode is read as one
        # CONTIGUOUS h5 slice and subsampled in RAM rather than fancy-indexed by
        # scattered rows, because h5py point-selection over a list is far slower
        # than a slice on a network filesystem and dominates wall-clock for the
        # dense stride-1 build. kept_rows_per_ep stays in kept-episode order, so
        # lengths and offsets remain valid.
        lat_chunks = []
        bs = args.batch_size
        n_total_rows = int(all_rows.size)
        done = 0
        for ep_i, rows in enumerate(kept_rows_per_ep):
            lo, hi = int(rows[0]), int(rows[-1]) + 1
            span = pixels[lo:hi]                       # one contiguous read (fast)
            sel = span[rows - lo]                      # stride-H subsample in RAM
            z = encoder.encode_frames(model, sel, device=args.device, batch_size=bs)
            if dino is not None:
                from surveyor import paired
                zd = paired.encode_frames_dino(dino, sel, device=args.device,
                                               batch_size=bs)
                z = torch.cat([z, zd], dim=-1)     # [lewm(192) | dino(384)]
            lat_chunks.append(z)
            done += len(rows)
            if ep_i % 200 == 0:
                print(f"  encoded {done}/{n_total_rows} frames over {ep_i+1} eps "
                      f"({time.time()-t0:.0f}s)")
        latents = torch.cat(lat_chunks, 0).contiguous().float()  # (total_nsg, 192)
        assert latents.shape[0] == int(lengths.sum())

    offsets = np.zeros(n_kept, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)[:-1]

    norms = latents.norm(dim=1)
    norm_stats = dict(mean=float(norms.mean()), std=float(norms.std()),
                      min=float(norms.min()), max=float(norms.max()))

    out = {
        "latents": latents,                                   # (total_nsg, 192) float32
        "lengths": torch.from_numpy(lengths),                 # (K,) n_sg per kept episode
        "offsets": torch.from_numpy(offsets),                 # (K,) start idx into latents
        "episode_idx": torch.tensor(kept_ids, dtype=torch.long),
        "ep_len": torch.tensor(kept_len, dtype=torch.long),
        "in_5deg": torch.tensor(in_5deg, dtype=torch.bool),   # subset flag (subset of kept)
        "stride": args.stride,
        "latent_dim": int(latents.shape[1]),
        "paired": ({"lewm_dim": 192, "dino_dim": 384, "verify_space": "dino_pooled",
                    "encode_fn": "surveyor.paired.encode_frames_dino"}
                   if args.dino_pair else None),
        "criterion": {
            "pos_thresh": args.pos_thresh,
            "angle_deg_headline": args.angle_headline,
            "angle_deg_strict": args.angle_strict,
            "mode": "block_only_canonical",
            "target_block_xy": encoder.TARGET_BLOCK_XY.tolist(),
            "target_block_angle": encoder.TARGET_BLOCK_ANGLE,
        },
        "encoder": {"source": args.source, "encoder_id": args.encoder_id,
                    "local_dir": args.local_dir},
        "counts": {"episodes_considered": int(len(ep_ids)),
                   "total_episodes": int(n_total),
                   "kept_headline": int(n_kept),
                   "kept_strict": int(np.sum(in_5deg)),
                   "dropped": int(n_drop)},
        "norm_stats": norm_stats,
        "sampling": {"max_episodes": args.max_episodes, "mode": args.sample_mode,
                     "seed": args.sample_seed},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    sz = Path(args.out).stat().st_size / 1e6
    print(f"[save] {args.out}  ({sz:.1f} MB)")
    print(f"[norm] ||z|| mean={norm_stats['mean']:.2f} std={norm_stats['std']:.2f} "
          f"min={norm_stats['min']:.2f} max={norm_stats['max']:.2f}  (sqrt(192)={np.sqrt(192):.2f})")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
