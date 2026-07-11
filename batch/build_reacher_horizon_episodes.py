"""Build the fixed-population episode file for the Reacher horizon sweep.

Samples n (episode, start) pairs from the HOLDOUT episodes (>= episode_min)
whose start is valid for the LARGEST sweep offset, so ONE file serves every
--goal-offset in {25, 50, 75, 100}: the start states are byte-identical across
the whole curve and horizon is the only variable. This is the t75-vs-t150
sampling-artifact lesson (PushT audit) applied up front -- per-offset sampling
gives each offset a different episode population and confounds horizon with
population. Reacher improves on the PushT reindexing scheme: because goals sit
at start+offset (not pinned to episode ends), the same START can simply reach
further, rather than the start moving backwards.

Deterministic given --seed, so concurrent array tasks can regenerate it
identically. Usage (inside the sweep sbatch, on the box):
  python batch/build_reacher_horizon_episodes.py --h5 <reacher.h5> \
      --out reacher_horizon.episodes.json
"""
import argparse
import json

import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass
import h5py


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--max-offset", type=int, default=100,
                   help="largest sweep goal_offset; starts must satisfy "
                        "start <= ep_len - 1 - max_offset")
    p.add_argument("--episode-min", type=int, default=8000,
                   help="holdout guard: only episodes >= this (drafters train on < 8000)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    with h5py.File(args.h5, "r") as f:
        # PushT names this column "episode_idx"; the reacher dump names it "ep_idx"
        episode_idx = f["episode_idx"][:] if "episode_idx" in f else f["ep_idx"][:]
        step_idx = f["step_idx"][:]
        ep_len = f["ep_len"][:]

    ep_len_per_row = ep_len[episode_idx]
    valid = (step_idx <= ep_len_per_row - args.max_offset - 1) & (episode_idx >= args.episode_min)
    valid_rows = np.nonzero(valid)[0]
    g = np.random.default_rng(args.seed)
    chosen = g.choice(len(valid_rows), size=args.n, replace=False)
    rows = np.sort(valid_rows[chosen])
    pairs = [[int(episode_idx[r]), int(step_idx[r])] for r in rows]

    out = {
        "max_offset": args.max_offset,
        "episode_min": args.episode_min,
        "seed": args.seed,
        "note": f"fixed population for the Reacher horizon sweep: same {args.n} "
                f"(episode,start) pairs at every goal_offset <= {args.max_offset}",
        "episodes": pairs,
    }
    with open(args.out, "w") as jf:
        json.dump(out, jf)
    print(f"wrote {args.out}: {len(pairs)} pairs "
          f"(episodes >= {args.episode_min}, start <= ep_len-1-{args.max_offset}), "
          f"sample {pairs[:3]}")


if __name__ == "__main__":
    main()
