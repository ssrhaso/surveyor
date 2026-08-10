"""Extract a small self-contained subset h5 for specific eval conditions,
for hosts without the full dataset. Episodes are copied and remapped to a
compact index (schema-identical, drop-in --h5 replacement), and one
episodes-file JSON per goal-offset is written for --episodes-file.
"""

from __future__ import annotations

import argparse
import json

import hdf5plugin  # noqa: F401; registers the dynamically-loaded filter plugin
                    # this h5's `pixels` dataset uses (~10x compression); without
                    # this import, reading pixels fails with "Can't open directory
                    # (/usr/local/lib/plugin)". Other scripts don't need this
                    # explicit import because they always import stable_worldmodel
                    # first, which imports hdf5plugin as a side effect.
import h5py
import numpy as np

from surveyor.envs.pusht.eval import sample_long, success_mask


def parse_args():
    p = argparse.ArgumentParser(description="Extract a small subset h5 for transfer")
    p.add_argument("--h5", required=True, help="the full source h5 (must actually work, e.g. ofs-v01)")
    p.add_argument("--out", required=True, help="output subset h5 path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-filter", choices=["none", "success20", "success5"], default="success5")
    p.add_argument("--goal-offsets", type=int, nargs="+", default=[75, 150],
                   help="one episodes-file will be written per goal-offset")
    p.add_argument("--num-eval", type=int, default=256)
    p.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    return p.parse_args()


def main():
    args = parse_args()
    ep_mask = None
    if args.eval_filter != "none":
        ang = 20.0 if args.eval_filter == "success20" else 5.0
        ep_mask = success_mask(args.h5, ang)
        print(f"[filter] {args.eval_filter}: {int(ep_mask.sum())} episodes eligible")

    conditions = {}
    needed_eps = set()
    for goff in args.goal_offsets:
        eps, starts = sample_long(args.h5, args.num_eval, goff, args.seed, ep_mask=ep_mask)
        conditions[goff] = (eps, starts)
        needed_eps.update(int(e) for e in eps)
        print(f"[goal_offset={goff}] {len(eps)} episodes drawn, "
              f"{len(set(int(e) for e in eps))} distinct")

    needed_eps = sorted(needed_eps)
    print(f"[union] {len(needed_eps)} distinct episodes needed across all conditions")

    comp = None if args.compression == "none" else args.compression
    with h5py.File(args.h5, "r") as fin:
        ep_off = fin["ep_offset"][:]
        ep_len = fin["ep_len"][:]
        state_dim = fin["state"].shape[1]
        proprio_dim = fin["proprio"].shape[1]
        action_dim = fin["action"].shape[1]
        img_shape = fin["pixels"].shape[1:]
        total_rows = int(sum(int(ep_len[e]) for e in needed_eps))
        print(f"[size] {total_rows} total rows to copy "
              f"({total_rows * np.prod(img_shape) / 1e9:.2f} GB raw pixels before compression)")

        with h5py.File(args.out, "w") as fout:
            d_pixels = fout.create_dataset("pixels", shape=(total_rows, *img_shape), dtype="uint8",
                                           chunks=(min(100, total_rows), *img_shape), compression=comp)
            d_action = fout.create_dataset("action", shape=(total_rows, action_dim), dtype="float32")
            d_proprio = fout.create_dataset("proprio", shape=(total_rows, proprio_dim), dtype="float32")
            d_state = fout.create_dataset("state", shape=(total_rows, state_dim), dtype="float32")
            d_epidx = fout.create_dataset("episode_idx", shape=(total_rows,), dtype="int64")
            d_stepidx = fout.create_dataset("step_idx", shape=(total_rows,), dtype="int64")

            new_ep_off = np.zeros(len(needed_eps), dtype=np.int64)
            new_ep_len = np.zeros(len(needed_eps), dtype=np.int32)
            orig_to_new = {}
            cursor = 0
            for new_idx, e in enumerate(needed_eps):
                lo, hi = int(ep_off[e]), int(ep_off[e]) + int(ep_len[e])
                n = hi - lo
                d_pixels[cursor:cursor + n] = fin["pixels"][lo:hi]
                d_action[cursor:cursor + n] = fin["action"][lo:hi]
                d_proprio[cursor:cursor + n] = fin["proprio"][lo:hi]
                d_state[cursor:cursor + n] = fin["state"][lo:hi]
                d_epidx[cursor:cursor + n] = new_idx
                d_stepidx[cursor:cursor + n] = np.arange(n, dtype=np.int64)
                new_ep_off[new_idx] = cursor
                new_ep_len[new_idx] = n
                orig_to_new[e] = new_idx
                cursor += n
                if new_idx % 50 == 0:
                    print(f"  [{new_idx}/{len(needed_eps)}] episode {e} -> {new_idx}, "
                          f"{n} rows, cursor={cursor}/{total_rows}")

            fout.create_dataset("ep_offset", data=new_ep_off)
            fout.create_dataset("ep_len", data=new_ep_len)

    print(f"[done] wrote {args.out}")

    for goff, (eps, starts) in conditions.items():
        remapped = [[orig_to_new[int(e)], int(s)] for e, s in zip(eps, starts)]
        outpath = args.out.rsplit(".", 1)[0] + f".episodes{goff}.json"
        with open(outpath, "w") as f:
            json.dump({"goal_offset": goff, "seed": args.seed, "eval_filter": args.eval_filter,
                      "episodes": remapped}, f)
        print(f"[done] wrote {outpath} ({len(remapped)} entries)")


if __name__ == "__main__":
    main()
