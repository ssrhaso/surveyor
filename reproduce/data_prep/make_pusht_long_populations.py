"""PushT populations beyond t = 150: pusht_c2222.episodes{175,200}.json.

The drafter and the GC-IDM checkpoints were trained with the episodes of pusht_c2222.h5 held out, so a population beyond
t = 150 must come from that file or it would evaluate on training episodes. The file holds 197 episodes longer than 176
steps and 50 longer than 201 (the longest demonstration is 246 steps), fewer than the 256 that the shorter distances
draw, so the population at each distance is EVERY held-out episode that is long enough, with no sampling. As at the
other distances the start is ep_len - 1 - t and the goal is the last frame; every c2222 episode already passes the
success5 filter.

    python reproduce/data_prep/make_pusht_long_populations.py --h5 pusht_c2222.h5 --goal-offsets 175 200
"""
import argparse
import json

import h5py

ap = argparse.ArgumentParser()
ap.add_argument("--h5", default="pusht_c2222.h5")
ap.add_argument("--goal-offsets", type=int, nargs="+", default=[175, 200])
args = ap.parse_args()
with h5py.File(args.h5, "r") as f:
    ep_len = f["ep_len"][:]
for t in args.goal_offsets:
    pairs = [[int(e), int(ep_len[e] - 1 - t)] for e in range(len(ep_len)) if ep_len[e] > t + 1]
    out = args.h5.replace(".h5", f".episodes{t}.json")
    with open(out, "w") as g:
        json.dump({"goal_offset": t, "seed": None, "eval_filter": "success5",
                   "selection": "every held-out episode with ep_len > t + 1", "episodes": pairs}, g)
    print(f"{out}: {len(pairs)} episodes, start steps {min(p[1] for p in pairs)} to {max(p[1] for p in pairs)}")
