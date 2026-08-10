"""Fixed-population horizon episode file for OGBench-Cube.

Uses hindsight goals at start + t (small offsets under the final-frame
convention are vacuous: the cube is already at its target) and one fixed
(episode, start) set reused at every offset so horizon is the only
variable. A non-vacuity filter requires the cube to move by more than
--min-disp between start and goal at every evaluated offset. NOTE: the
filter over-selects hard episodes at small offsets; the certified cube
protocol uses per-seed sampling at offset 150 instead (see cube eval).
"""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Cube fixed-population horizon episodes")
    p.add_argument("--h5", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--offsets", type=int, nargs="+", default=[25, 50, 75, 100, 150])
    p.add_argument("--min-disp", type=float, default=0.08,
                   help="min cube displacement (m) start->goal at EVERY offset "
                        "(2x the 0.04 success threshold => task is non-trivial)")
    p.add_argument("--episode-min", type=int, default=8000,
                   help="holdout guard: eval draws from episodes >= this")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    max_off = max(a.offsets)
    with h5py.File(a.h5, "r") as f:
        ep_off, ep_len = f["ep_offset"][:], f["ep_len"][:]
        blk = f["privileged_block_0_pos"]
        n_eps = len(ep_off)
        eligible = np.arange(a.episode_min, n_eps)
        print(f"[data] {n_eps} episodes; {len(eligible)} eligible (>= {a.episode_min})")

        pairs, tried, rejected = [], 0, 0
        while len(pairs) < a.n and tried < a.n * 400:
            tried += 1
            e = int(rng.choice(eligible))
            o, L = int(ep_off[e]), int(ep_len[e])
            if L - 1 - max_off <= 0:
                continue
            s = int(rng.integers(0, L - 1 - max_off))
            z0 = blk[o + s]
            # non-vacuity at EVERY offset: the cube must actually move by min_disp
            if all(np.linalg.norm(blk[o + s + t] - z0) > a.min_disp for t in a.offsets):
                pairs.append([e, s])
            else:
                rejected += 1
        print(f"[filter] {len(pairs)} pairs kept, {rejected} rejected as vacuous "
              f"(cube moved <= {a.min_disp} m at some offset), {tried} tried")
        if len(pairs) < a.n:
            raise SystemExit(f"only {len(pairs)}/{a.n} non-vacuous pairs found; "
                             f"lower --min-disp or --offsets")

        # report the realised displacement per offset (the sanity read)
        for t in a.offsets:
            d = [np.linalg.norm(blk[int(ep_off[e]) + s + t] - blk[int(ep_off[e]) + s])
                 for e, s in pairs]
            print(f"  t={t:3d}: median start->goal cube displacement = {np.median(d):.4f} m "
                  f"(min {np.min(d):.4f})")

    payload = {"episodes": pairs, "max_offset": max_off, "seed": a.seed,
               "episode_min": a.episode_min, "min_disp": a.min_disp,
               "offsets": a.offsets,
               "note": "goal = frame at start+t (NOT the final frame): cube episodes are "
                       "multi-segment pick-and-place, so final-frame goals are vacuous."}
    with open(a.out, "w") as fh:
        json.dump(payload, fh)
    print(f"[save] {a.out}: {len(pairs)} fixed pairs, reused at every offset")


if __name__ == "__main__":
    main()
