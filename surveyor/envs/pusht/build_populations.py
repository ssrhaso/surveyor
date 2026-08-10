"""Regenerate the PushT fixed-population horizon files.

Builds pusht.episodes150.json (n=256, offset 150, seed 42, driver-default
eligibility) via the eval driver's own sample_long, then reindexes the SAME 256
episodes to goal_offset in {25,50,75,100} at start = ep_len-1-off. One
population, horizon as the only variable, applying the t75-vs-t150
sampling-artifact lesson up front.

The population is regenerated rather than byte-copied from the development
run, whose eval_filter field is not recoverable; cross-pipeline absolute
comparisons lean on the short-protocol replication instead, and the horizon
curve is internally consistent by construction.

Usage: python -m surveyor.envs.pusht.build_populations [dataset.h5]
"""
import json
import os
import sys

from surveyor.envs.pusht.eval import sample_long

H5 = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/data/pusht/pusht_expert_train.h5")
N, LAST, SEED = 256, 150, 42

episodes_idx, start_steps = sample_long(H5, N, LAST, SEED)
pairs = [[int(e), int(s)] for e, s in zip(episodes_idx, start_steps)]
base = {"goal_offset": LAST, "seed": SEED, "eval_filter": None,
        "note": "regenerated: sample_long(n=256, last_n=150, seed=42), "
                "driver-default eligibility", "episodes": pairs}
with open("pusht.episodes150.json", "w") as f:
    json.dump(base, f)
print(f"wrote pusht.episodes150.json: {len(pairs)} pairs, sample {pairs[:3]}")

import h5py  # noqa: E402
try:
    import hdf5plugin  # noqa: F401, E402
except ImportError:
    pass
with h5py.File(H5, "r") as f:
    ep_len = f["ep_len"][:]

for off in (25, 50, 75, 100):
    new_pairs = []
    for idx, _old in pairs:
        L = int(ep_len[idx])
        new_start = L - 1 - off
        assert new_start >= 0, (idx, L, off)
        new_pairs.append([idx, new_start])
    out = dict(base, goal_offset=off, episodes=new_pairs,
               note=f"same 256 episodes as pusht.episodes150.json reindexed "
                    f"to goal_offset={off}")
    name = f"pusht.episodes150as{off}.json"
    with open(name, "w") as f:
        json.dump(out, f)
    print(f"wrote {name}: {len(new_pairs)} pairs")
print("PUSHT-POPULATIONS-OK")
