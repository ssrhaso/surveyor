"""Subgoal-dataset builder for TwoRoom.

TwoRoom analog of the PushT builder (success filter, stride subsample, encode,
pack) with its own success rule: the target is sampled per episode and recorded
in the h5, so success is the final-row distance_to_target below the env
threshold (encoder.tworoom_success). Accepts multiple --h5 files, concatenated
with per-episode source tags. --require-cross-room drops episodes where agent
and target start in the same room, roughly half the collected episodes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
try:  # tworoom.h5 pixels are blosc-compressed; this registers the filter
    import hdf5plugin  # noqa: F401
except ImportError:
    pass
import h5py

from specaccept import encoder


def parse_args():
    p = argparse.ArgumentParser(description="FF-JEPA Task A (TwoRoom): subgoal-dataset builder")
    p.add_argument("--h5", required=True, nargs="+", help="one or more TwoRoom h5 dataset paths")
    p.add_argument("--out", required=True, help="output .pt path")
    p.add_argument("--source", choices=["pretrained", "local"], default="local")
    p.add_argument("--encoder-id", default="quentinll/lewm-tworooms")
    p.add_argument("--local-dir", default="encoder_tworoom",
                   help="dir with config.json+weights.pt (source=local)")
    p.add_argument("--swm-src", default=None, help="stable-worldmodel checkout (CPU import)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=25, help="H: subgoal stride (env steps)")
    p.add_argument("--pos-thresh", type=float, default=encoder.TWOROOM_POS_THRESH)
    p.add_argument("--max-episodes", type=int, default=None,
                   help="limit #episodes considered PER FILE")
    p.add_argument("--sample-mode", choices=["head", "spread", "random"], default="head")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--require-cross-room", action="store_true", default=True,
                   help="drop episodes where agent and target start in the SAME room "
                        "(no door-crossing needed); the env's own target-sampling "
                        "constraint that enforces this is disabled by default, so "
                        "~48%% of collected episodes are a trivial same-room task, "
                        "not the intended door-crossing one")
    p.add_argument("--no-require-cross-room", dest="require_cross_room", action="store_false")
    p.add_argument("--no-success-filter", dest="success_filter", action="store_false",
                   default=True,
                   help="keep episodes whose demo never reached its target. The "
                        "anchor ablation (2304691) measured the success filter as "
                        "selecting a strictly harder EVAL population, and the "
                        "anchored protocol (goal_proprio = reach where the demo "
                        "ended) makes failed demos a well-posed training signal: "
                        "conditioned on the demo's end frame, predict the states "
                        "along the way. Roughly doubles the pair pool.")
    p.add_argument("--wall-center", type=float, default=112.0)
    p.add_argument("--dino-pair", action="store_true",
                   help="also encode every subgoal frame with frozen DINOv2 and "
                        "store the concatenation [lewm(192) | dino(384)]. The "
                        "drafter then emits both halves of each waypoint from one "
                        "sample, so the half served to the planner and the half "
                        "the verifier certifies describe the SAME imagined future "
                        "(specaccept.paired). Needed because LeWM's TwoRoom latent "
                        "space cannot support verification at any threshold.")
    return p.parse_args()


def is_cross_room(state, goal_state, wall_center):
    return (state[0] < wall_center) != (goal_state[0] < wall_center)


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
        from specaccept import paired
        dino = paired.load_dinov2(device=args.device)
        print(f"[model] frozen DINOv2 loaded for the verifier half "
              f"({sum(p.numel() for p in dino.parameters())/1e6:.2f}M params)")

    lat_chunks = []
    lengths_all, kept_ids_all, kept_len_all, source_file_all = [], [], [], []
    n_kept_total = 0
    n_drop_total = 0
    n_considered_total = 0

    for file_i, h5_path in enumerate(args.h5):
        with h5py.File(h5_path, "r") as f:
            ep_off = f["ep_offset"][:]
            ep_len = f["ep_len"][:]
            dist_to_target = f["distance_to_target"]
            state = f["state"]
            goal_state = f["goal_state"]
            pixels = f["pixels"]
            n_total = len(ep_off)
            ep_ids = select_episodes(n_total, args.max_episodes, args.sample_mode, args.sample_seed)
            print(f"[data] {h5_path}: {n_total} episodes total; considering {len(ep_ids)} "
                  f"(mode={args.sample_mode})")

            # Pass 1: success filter (final distance-to-target) plus the optional
            # cross-room filter, which keeps only the intended task of agent and
            # target starting in different rooms (see --require-cross-room help)
            kept_rows_per_ep = []
            n_drop = 0
            n_drop_same_room = 0
            for e in ep_ids:
                off, L = int(ep_off[e]), int(ep_len[e])
                if args.require_cross_room and not is_cross_room(
                        state[off], goal_state[off], args.wall_center):
                    n_drop_same_room += 1
                    continue
                if args.success_filter:
                    final_dist = float(dist_to_target[off + L - 1])
                    if not encoder.tworoom_success(final_dist, args.pos_thresh):
                        n_drop += 1
                        continue
                rows = np.arange(off, off + L, args.stride)
                kept_ids_all.append(int(e))
                kept_len_all.append(L)
                source_file_all.append(file_i)
                kept_rows_per_ep.append(rows)

            n_kept_file = len(kept_rows_per_ep)
            if n_kept_file == 0:
                print(f"[warn] no episodes passed the success filter in {h5_path}")
            n_kept_total += n_kept_file
            n_drop_total += n_drop
            n_considered_total += len(ep_ids)
            print(f"[filter] {h5_path}: kept={n_kept_file} dropped_same_room={n_drop_same_room} "
                  f"dropped_unsuccessful={n_drop} "
                  f"({100*n_kept_file/max(len(ep_ids),1):.1f}% kept)")

            # Pass 2: encode kept stride-H frames, one contiguous h5 slice per
            # episode, avoiding slow scattered point-selection (build_subgoals.py)
            n_rows_file = sum(len(r) for r in kept_rows_per_ep)
            done = 0
            for ep_i, rows in enumerate(kept_rows_per_ep):
                lo, hi = int(rows[0]), int(rows[-1]) + 1
                span = pixels[lo:hi]
                sel = span[rows - lo]
                z = encoder.encode_frames(model, sel, device=args.device, batch_size=args.batch_size)
                if dino is not None:
                    from specaccept import paired
                    zd = paired.encode_frames_dino(dino, sel, device=args.device,
                                                   batch_size=args.batch_size)
                    z = torch.cat([z, zd], dim=-1)     # [lewm(192) | dino(384)]
                lat_chunks.append(z)
                lengths_all.append(len(rows))
                done += len(rows)
                if ep_i % 200 == 0:
                    print(f"  encoded {done}/{n_rows_file} frames over {ep_i+1} eps "
                          f"({time.time()-t0:.0f}s)")

    if n_kept_total == 0:
        raise RuntimeError("no episodes passed the success filter across all files")

    latents = torch.cat(lat_chunks, 0).contiguous().float()
    lengths = np.array(lengths_all, dtype=np.int64)
    assert latents.shape[0] == int(lengths.sum())
    offsets = np.zeros(n_kept_total, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)[:-1]

    norms = latents.norm(dim=1)
    norm_stats = dict(mean=float(norms.mean()), std=float(norms.std()),
                      min=float(norms.min()), max=float(norms.max()))

    print(f"[filter] TOTAL kept={n_kept_total} dropped={n_drop_total} "
          f"({100*n_kept_total/max(n_considered_total,1):.1f}% kept across "
          f"{len(args.h5)} file(s))")
    print(f"[subsample] stride={args.stride}: n_sg/episode min={lengths.min()} "
          f"max={lengths.max()} mean={lengths.mean():.2f}")

    out = {
        "latents": latents,                                       # (total_nsg, 192) float32
        "lengths": torch.from_numpy(lengths),                     # (K,) n_sg per kept episode
        "offsets": torch.from_numpy(offsets),                     # (K,) start idx into latents
        "episode_idx": torch.tensor(kept_ids_all, dtype=torch.long),
        "ep_len": torch.tensor(kept_len_all, dtype=torch.long),
        "source_file": torch.tensor(source_file_all, dtype=torch.long),  # index into args.h5
        "source_files": list(args.h5),
        "in_5deg": torch.ones(n_kept_total, dtype=torch.bool),     # no stricter tier for TwoRoom;
                                                                    # kept for train_gdm.py --mask compat
        "stride": args.stride,
        "latent_dim": int(latents.shape[1]),
        "paired": ({"lewm_dim": 192, "dino_dim": 384, "verify_space": "dino_pooled",
                    "encode_fn": "specaccept.paired.encode_frames_dino"}
                   if args.dino_pair else None),
        "criterion": {
            "pos_thresh": args.pos_thresh,
            "mode": "tworoom_native_dist_to_target",
            "require_cross_room": args.require_cross_room,
            "success_filter": args.success_filter,
            "wall_center": args.wall_center,
        },
        "encoder": {"source": args.source, "encoder_id": args.encoder_id,
                    "local_dir": args.local_dir},
        "counts": {"episodes_considered": int(n_considered_total),
                   "kept": int(n_kept_total),
                   "dropped": int(n_drop_total)},
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
