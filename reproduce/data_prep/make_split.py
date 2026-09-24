"""Episode split file in the format LeFlow's trainer reads
(external/LeFlow/train_latent_planner.py::load_or_create_episode_split):
train_episodes = every episode minus our evaluation holdout, eval_episodes =
the holdout. Holdout is either explicit files (PushT: pusht_c2222.source_
episodes*.json, full-dataset indices) or an index threshold (Reacher/Cube:
episodes >= 8000).

    python reproduce/data_prep/make_split.py --h5 X.h5 --name pusht_expert_train --out split.json --exclude-json a.json b.json
    python reproduce/data_prep/make_split.py --h5 X.h5 --name reacher --out split.json --episode-min 8000
"""
import argparse
import json

import h5py

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

p = argparse.ArgumentParser()
p.add_argument("--h5", required=True)
p.add_argument("--name", required=True)
p.add_argument("--out", required=True)
p.add_argument("--exclude-json", nargs="*", default=None)
p.add_argument("--episode-min", type=int, default=None)
a = p.parse_args()

with h5py.File(a.h5, "r") as f:
    n = int(len(f["ep_len"]))
hold = set()
if a.exclude_json:
    for path in a.exclude_json:
        with open(path) as fh:
            hold.update(int(pair[0]) for pair in json.load(fh)["episodes"])
if a.episode_min is not None:
    hold.update(range(a.episode_min, n))
train = sorted(set(range(n)) - hold)
payload = {"dataset": a.name, "seed": 0, "train_fraction": len(train) / n, "num_episodes": n,
           "train_episodes": train, "eval_episodes": sorted(hold),
           "note": "our evaluation holdout, not a random split"}
with open(a.out, "w") as fh:
    json.dump(payload, fh)
print(f"[split] {a.name}: {n} episodes, train {len(train)}, held out {len(hold)} -> {a.out}")
