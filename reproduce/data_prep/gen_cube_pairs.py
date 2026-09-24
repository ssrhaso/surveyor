"""Fixed Cube evaluation populations: cube_c2222.episodes{t}.json.

Why. Cube expert episodes are all 201 steps, and `--start final` (start = 201-1-t, goal = the final frame) is
degenerate: the block already sits within the 4 cm success radius of its goal-frame position in 100% of holdout
episodes at t=25 and t=100 (37% at t=50), so every arm scores 100 at step 0 there.

Protocol (mirrors pusht_c2222.episodes{t}.json, seed 2222): uniform draw without replacement over the valid
(episode, start) pairs of the holdout episodes [8000, 10000), start <= ep_len-1-t, goal = start+t, restricted to pairs
whose block moves at least MIN_DISP between the start frame and the goal frame, so no pair is solved before the first
action. 128 pairs per t, one file per t (the populations differ across t, as on PushT). Also writes a 16-pair smoke
file at t=100 (seed 2223).

    python reproduce/data_prep/gen_cube_pairs.py --h5 $DATA/cube_single_expert.h5 --out-dir . \
        --ts 25 50 75 100 150 175 195
"""
import argparse
import json
import os

import h5py
import numpy as np

H5 = os.path.join(os.environ.get("DATA", "."), "cube_single_expert.h5")
TS = (25, 50, 75, 100, 150)
SEED, EP_MIN, NUM = 2222, 8000, 128
MIN_DISP = 0.08          # metres; twice the 4 cm success radius
TRIVIAL = 0.04           # the success radius itself


def valid_pairs(pos, ep_len, ep_off, lo_row, t):
    eps, starts, disps = [], [], []
    for e in range(EP_MIN, len(ep_len)):
        L = int(ep_len[e]); o = int(ep_off[e]) - lo_row
        if L <= t + 1:
            continue
        p = pos[o:o + L]
        s = np.arange(0, L - t)                    # start <= L-1-t, goal = start+t
        d = np.linalg.norm(p[s + t] - p[s], axis=1)
        eps.append(np.full(len(s), e)); starts.append(s); disps.append(d)
    return np.concatenate(eps), np.concatenate(starts), np.concatenate(disps)


def draw(eps, starts, disps, num, seed):
    keep = np.nonzero(disps >= MIN_DISP)[0]
    g = np.random.default_rng(seed)
    chosen = np.sort(keep[g.choice(len(keep), size=num, replace=False)])
    return [[int(eps[i]), int(starts[i])] for i in chosen], disps[chosen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=H5)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--ts", type=int, nargs="+", default=list(TS),
                    help="goal distances to write (episodes are 201 steps)")
    a = ap.parse_args()
    with h5py.File(a.h5, "r") as f:
        ep_len = f["ep_len"][:]; ep_off = f["ep_offset"][:]
        assert np.all(np.diff(ep_off) == ep_len[:-1]), "episodes must be contiguous"
        lo = int(ep_off[EP_MIN]); hi = int(ep_off[-1] + ep_len[-1])
        pos = f["privileged_block_0_pos"][lo:hi]
    for t in a.ts:
        eps, starts, disps = valid_pairs(pos, ep_len, ep_off, lo, t)
        pairs, d = draw(eps, starts, disps, NUM, SEED)
        meta = {"goal_offset": t, "seed": SEED, "episode_min": EP_MIN, "num_eval": NUM,
                "min_block_disp_m": MIN_DISP, "n_valid_pairs": int(len(eps)),
                "n_pairs_after_filter": int((disps >= MIN_DISP).sum()),
                "frac_trivial_before_filter": float((disps < TRIVIAL).mean()),
                "chosen_disp_m": {"min": float(d.min()), "median": float(np.median(d)), "max": float(d.max())},
                "note": "Cube fixed population: start uniform over valid (episode,start) pairs of the holdout "
                        "episodes with block displacement start->start+t >= min_block_disp_m; goal = start+t",
                "episodes": pairs}
        fn = f"{a.out_dir}/cube_c2222.episodes{t}.json"
        with open(fn, "w") as f:
            json.dump(meta, f)
        print(f"t={t:3d} valid={len(eps):6d} kept={meta['n_pairs_after_filter']:6d} "
              f"trivial_before={meta['frac_trivial_before_filter']:.3f} chosen disp min/med/max="
              f"{d.min():.3f}/{np.median(d):.3f}/{d.max():.3f} -> {fn}")
        if t == 100:
            pairs16, d16 = draw(eps, starts, disps, 16, SEED + 1)
            with open(f"{a.out_dir}/cube_c2222.episodes100.smoke16.json", "w") as f:
                json.dump({**meta, "num_eval": 16, "seed": SEED + 1, "episodes": pairs16}, f)
            print(f"      smoke16 disp med={np.median(d16):.3f}")


if __name__ == "__main__":
    main()
