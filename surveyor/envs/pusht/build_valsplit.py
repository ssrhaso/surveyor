"""Disjoint VALIDATION population for hyperparameter selection (audit #1).

pusht.episodes150.json (seed 42) is the TEST population every reported number
uses, and tau/k selection must not touch it. This builds
pusht.episodes150.val.json: 256 (episode, start) pairs from the same eligibility
pool at seed 777, with any episode index overlapping the test set removed BEFORE
truncation to 256. Selection sweeps run on val, then the chosen configuration is
reported on the untouched test set.

Usage: python -m surveyor.envs.pusht.build_valsplit [dataset.h5]
"""
import json
import os
import sys

from surveyor.envs.pusht.eval import sample_long

H5 = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/data/pusht/pusht_expert_train.h5")
N, LAST = 256, 150

with open("pusht.episodes150.json") as f:
    test_eps = {int(e) for e, _ in json.load(f)["episodes"]}

episodes_idx, start_steps = sample_long(H5, 4 * N, LAST, 777)
pairs = [[int(e), int(s)] for e, s in zip(episodes_idx, start_steps)
         if int(e) not in test_eps][:N]
assert len(pairs) == N, f"only {len(pairs)} disjoint pairs available"
assert not ({e for e, _ in pairs} & test_eps)

with open("pusht.episodes150.val.json", "w") as f:
    json.dump({"goal_offset": LAST, "seed": 777, "eval_filter": None,
               "note": "VALIDATION population, episode-disjoint from "
                       "pusht.episodes150.json (test); for tau/k selection only",
               "episodes": pairs}, f)
print(f"wrote pusht.episodes150.val.json: {len(pairs)} pairs, "
      f"disjoint from {len(test_eps)} test episodes")
print("PUSHT-VALSPLIT-OK")
