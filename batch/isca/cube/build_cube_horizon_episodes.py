"""Fixed-population horizon episode file for OGBench-Cube.

Two protocol bugs this fixes (both found 2026-07-14 by the vacuous-cell diagnostic):

 1. Cube expert episodes are NOT single goal-directed trajectories: the collector
    repeatedly re-samples a new target within one episode (data_collection mode), so
    the expert performs a SEQUENCE of pick-and-places. The PushT convention
    (goal = final frame, start = ep_len-1-t) is therefore meaningless here -- for some
    offsets the cube at the start frame is coincidentally already where it ends up, so
    the cell is VACUOUS (100% of episodes already solved at t=25 and t=100; every arm
    trivially scores 100%). We instead take goal = frame at start + t (a genuine
    t-step-ahead hindsight goal, the Reacher convention).

 2. Independent per-offset sampling gives a DIFFERENT population at every offset (the
    PushT t75-vs-t150 lesson). We emit ONE fixed set of (episode, start) pairs, valid
    for the largest offset, reused at every t -- so horizon is the only variable.

Non-vacuity filter: require the cube to actually move by > --min-disp between the start
and the goal at EVERY offset evaluated, so no cell degenerates to "already solved".

Usage:
  python batch/isca/cube/build_cube_horizon_episodes.py --h5 <cube.h5> \
      --out cube_horizon.ep.json --n 128 --offsets 25 50 75 100 150 --min-disp 0.08
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
