"""Derive fixed-population horizon-sweep episode files.

Reindexes the SAME 256 episodes as the trusted t=150 GDM result
(subset_longeval.episodes150.json) to goal_offset in {25, 50, 100}; the 75
variant already exists as subset_longeval.episodes150as75.json from the
t75-vs-t150 resolution.

This applies the t75-vs-t150 sampling-artifact lesson up front: sample_long's
eligibility filter gives each offset a DIFFERENT episode population (90.3% vs
22.7% eligible at 75 vs 150), so an SR-vs-horizon curve drawn per-offset
confounds horizon with population. Reindexing one population isolates horizon
as the only variable across the whole curve.
"""
import json

import hdf5plugin  # noqa: F401
import h5py

OFFSETS = [25, 50, 100]

with h5py.File("subset_longeval.h5", "r") as f:
    ep_len = f["ep_len"][:]

with open("subset_longeval.episodes150.json") as jf:
    payload = json.load(jf)

pairs = payload["episodes"]
for off in OFFSETS:
    new_pairs = []
    for idx, _old_start in pairs:
        L = int(ep_len[idx])
        new_start = L - 1 - off
        assert new_start >= 0, (idx, L, off)
        new_pairs.append([idx, new_start])
    out = {
        "goal_offset": off,
        "seed": payload["seed"],
        "eval_filter": payload.get("eval_filter"),
        "note": f"same 256 episodes as episodes150.json, reindexed to "
                f"goal_offset={off} for the fixed-population horizon sweep",
        "episodes": new_pairs,
    }
    name = f"subset_longeval.episodes150as{off}.json"
    with open(name, "w") as jf:
        json.dump(out, jf)
    print(f"wrote {name}: {len(new_pairs)} pairs, sample {new_pairs[:3]}")
