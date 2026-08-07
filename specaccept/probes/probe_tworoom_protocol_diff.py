#!/usr/bin/env python3
"""Two-Room random-floor discrepancy: why do we read 22.53% where LeWM reads 0?

LeWM Fig. 6 reports Random = 0 on Two-Room. Our measured random floor at what we
call their t=25 protocol is 22.53% (job 2329041, 12 seeds, 173/768). Both cannot
be right, and we separately claim to reproduce their published Two-Room number
under their config (85.33 vs their 87), so this questions an anchor we rely on.

Offline: no policy, no model, just the cached columns.

CANDIDATE MECHANISM, same shape as Cube's P-DISP-1: under LeWM's config the
goal is `goal_proprio` (the AGENT's position at the goal frame) and the env
terminates at dist(agent, target) < 16 px, so the task is "be where you were
25 steps later" and any start whose expert displacement is already under 16 px
is solved by standing still.

This probe reports, for a range of goal offsets, the distribution of
||proprio[start+offset] - proprio[start]|| and the fraction below the success
radius, over exactly the episode population our random-floor job used and over
the unrestricted population LeWM would draw from.

  python -m specaccept.probes.probe_tworoom_protocol_diff --h5 <tworoom.h5>
"""
from __future__ import annotations

import argparse
import json

import numpy as np

TWOROOM_RADIUS_PX = 16.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", required=True)
    p.add_argument("--offsets", type=int, nargs="*", default=[25, 40, 50, 75])
    p.add_argument("--radius", type=float, default=TWOROOM_RADIUS_PX)
    p.add_argument("--episode-min", type=int, default=4000,
                   help="the floor our random-floor job used (--episode-min 4000)")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    import h5py

    with h5py.File(args.h5, "r") as f:
        cols = {k: f[k].shape for k in f.keys()}
        print("[columns]")
        for k, s in cols.items():
            print(f"   {k:28s} {s}")
        ep_len = f["ep_len"][:].astype(np.int64)
        ep_off = f["ep_offset"][:].astype(np.int64)
        # goal_proprio at row r is by construction proprio at the goal row, so
        # proprio alone measures the displacement
        proprio = f["proprio"][:].astype(np.float64)
        has_state = "state" in f
        state = f["state"][:].astype(np.float64) if has_state else None

    print(f"\n[data] {len(ep_len)} episodes, ep_len "
          f"min/median/max = {ep_len.min()}/{int(np.median(ep_len))}/{ep_len.max()}")
    print(f"[cols] proprio dim = {proprio.shape[1]}"
          + (f", state dim = {state.shape[1]}" if has_state else ", state ABSENT"))
    if has_state:
        # LeWM sets _set_state from `proprio`; our driver sets it from `state`.
        # If they differ in width, the two protocols are seeding different things.
        print("[diff] LeWM config/eval/tworoom.yaml sets _set_state <- proprio; "
              "our driver sets _set_state <- state.")
        print(f"[diff] proprio[:1] = {proprio[0]}")
        print(f"[diff] state[:1]   = {state[0]}")

    results = {"columns": {k: list(v) for k, v in cols.items()},
               "radius_px": args.radius, "by_offset": {}}

    print(f"\n{'offset':>7s} {'population':>26s} {'n':>8s} {'med(px)':>9s} "
          f"{'p90(px)':>9s} {'<16px':>8s}")
    print("-" * 74)
    for off in args.offsets:
        for label, lo in (("all episodes (LeWM draws)", 0),
                          (f"episodes >= {args.episode_min} (ours)", args.episode_min)):
            eps = np.nonzero(ep_len > off + 1)[0]
            eps = eps[eps >= lo]
            if len(eps) == 0:
                continue
            # every valid start within each episode, matching a uniform draw over
            # rows rather than one start per episode
            starts, ids = [], []
            for e in eps:
                m = int(ep_len[e]) - off - 1
                starts.append(np.arange(m + 1))
                ids.append(np.full(m + 1, e))
            starts = np.concatenate(starts)
            ids = np.concatenate(ids)
            a = ep_off[ids] + starts
            d = np.linalg.norm(proprio[a + off, :2] - proprio[a, :2], axis=1)
            frac = float((d < args.radius).mean())
            print(f"{off:7d} {label:>26s} {len(d):8d} {np.median(d):9.2f} "
                  f"{np.percentile(d, 90):9.2f} {100*frac:7.2f}%")
            results["by_offset"].setdefault(str(off), {})[label] = {
                "n_pairs": int(len(d)), "median_px": float(np.median(d)),
                "p90_px": float(np.percentile(d, 90)),
                "frac_below_radius": frac,
            }

    # ---- which GOAL is the task? The decisive fork. Whichever column feeds
    # _set_goal_state IS the task: LeWM's released config feeds goal_proprio
    # (agent position at the goal frame), our driver's default feeds goal_state
    # (the episode's fixed target). Only one can be behind their Random = 0.
    off = args.offsets[0]
    eps = np.nonzero(ep_len > off + 1)[0]
    starts, ids = [], []
    for e in eps:
        m = int(ep_len[e]) - off - 1
        starts.append(np.arange(m + 1))
        ids.append(np.full(m + 1, e))
    starts = np.concatenate(starts)
    ids = np.concatenate(ids)
    a = ep_off[ids] + starts
    with h5py.File(args.h5, "r") as f:
        goal_state = f["goal_state"][:].astype(np.float64)
        pos_target = f["pos_target"][:].astype(np.float64) if "pos_target" in f else None
    same = (pos_target is not None and np.allclose(goal_state, pos_target))
    print(f"\n[goal columns] goal_state == pos_target : {same}")
    print(f"\n{'goal definition':>52s} {'n':>9s} {'med(px)':>9s} {'p90(px)':>9s} {'<16px':>8s}")
    print("-" * 92)
    variants = {
        "goal_proprio: agent position at start+%d (LeWM cfg)" % off:
            np.linalg.norm(proprio[a + off, :2] - proprio[a, :2], axis=1),
        "goal_state: the episode's FIXED target":
            np.linalg.norm(goal_state[a, :2] - proprio[a, :2], axis=1),
    }
    for name, d in variants.items():
        frac = float((d < args.radius).mean())
        print(f"{name:>52s} {len(d):9d} {np.median(d):9.2f} "
              f"{np.percentile(d, 90):9.2f} {100*frac:7.2f}%")
        results.setdefault("goal_variants", {})[name] = {
            "n_pairs": int(len(d)), "median_px": float(np.median(d)),
            "p90_px": float(np.percentile(d, 90)), "frac_below_radius": frac}

    print(f"\n[read] 'frac < {args.radius:g}px' is the share of (episode, start) pairs whose "
          f"goal is already satisfied at t=0,\n       i.e. solved by standing still. That is "
          f"the ceiling a random policy inherits for free.")
    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"[json] -> {args.out}")


if __name__ == "__main__":
    main()
