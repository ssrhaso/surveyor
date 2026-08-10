"""Is the TwoRoom eval population at each horizon a real task?

The cube lesson: a cell can be VACUOUS (the start already satisfies the success
criterion, so doing nothing scores) or STARVED (the budget is below what the
task physically needs). Both produce numbers that look like findings. This
reconstructs the exact populations the driver evaluates, using the driver's own
samplers, and reports for each horizon:

  start->goal distance   what the agent actually has to travel
  vacuous fraction       start already within the success radius of the goal
  steps needed           distance / agent speed, versus the eval budget
  cross-wall fraction    whether the start and goal are in different rooms,
                         i.e. whether the door has to be used at all

No pixels are read, so this costs seconds.
"""

from __future__ import annotations

import argparse

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass
import h5py
import numpy as np

from surveyor import encoder
from surveyor.envs.tworoom.eval import (
    cross_room_mask, sample_long, sample_short, tworoom_success_mask,
)

AGENT_SPEED = 5.0


def parse_args():
    p = argparse.ArgumentParser(description="TwoRoom eval-population sanity")
    p.add_argument("--h5", required=True)
    p.add_argument("--offsets", type=int, nargs="+", default=[25, 75])
    p.add_argument("--num-eval", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--episode-min", type=int, default=4000)
    p.add_argument("--pos-thresh", type=float, default=encoder.TWOROOM_POS_THRESH)
    p.add_argument("--wall-center", type=float, default=112.0)
    return p.parse_args()


def main():
    args = parse_args()
    ep_mask = tworoom_success_mask(args.h5, args.pos_thresh)
    ep_mask = ep_mask & cross_room_mask(args.h5, args.wall_center)
    with h5py.File(args.h5, "r") as f:
        n_eps = len(f["ep_len"])
        ep_off = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
        state = f["state"][:]
    rng_mask = np.zeros(n_eps, dtype=bool)
    rng_mask[args.episode_min:] = True
    ep_mask = ep_mask & rng_mask
    print(f"[pop] {int(ep_mask.sum())} episodes eligible "
          f"(success AND cross-room AND index >= {args.episode_min})")

    for off in args.offsets:
        budget = 2 * off
        d_all, vac_all, cross_all = [], [], []
        for seed in args.seeds:
            # the driver uses sample_long whenever mode=long or start=final;
            # every cell in this battery ran with --start final
            eps_idx, starts = sample_long(args.h5, args.num_eval, off, seed,
                                          ep_mask=ep_mask)
            for e, s in zip(eps_idx, starts):
                base, L = int(ep_off[e]), int(ep_len[e])
                i0 = base + int(s)
                ig = base + min(int(s) + off, L - 1)
                p0, pg = state[i0], state[ig]
                d = float(np.linalg.norm(p0 - pg))
                d_all.append(d)
                vac_all.append(d < args.pos_thresh)
                cross_all.append((p0[0] < args.wall_center) != (pg[0] < args.wall_center))
        d_all = np.array(d_all)
        vac = float(np.mean(vac_all))
        crs = float(np.mean(cross_all))
        need = d_all / AGENT_SPEED
        print(f"\n=== goal_offset t={off}  (eval budget {budget} steps, "
              f"n={len(d_all)} over seeds {args.seeds}) ===")
        print("  start->goal distance p10/p50/p90 = %.1f / %.1f / %.1f px"
              % tuple(np.percentile(d_all, [10, 50, 90])))
        print("  VACUOUS (start already within %.0f px of goal): %.1f%%"
              % (args.pos_thresh, 100 * vac))
        print("  start and goal in DIFFERENT rooms:                %.1f%%" % (100 * crs))
        print("  straight-line steps needed p50/p90 = %.0f / %.0f  vs budget %d"
              % (np.percentile(need, 50), np.percentile(need, 90), budget))
        starved = float(np.mean(need > budget))
        print("  fraction whose STRAIGHT-LINE need alone exceeds budget: %.1f%%"
              % (100 * starved))


if __name__ == "__main__":
    main()
