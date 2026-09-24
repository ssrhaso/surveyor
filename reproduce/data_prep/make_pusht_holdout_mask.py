"""Per-episode training mask for the PushT drafters: the in_5deg population of the dense subgoal file MINUS every
episode any c2222 evaluation population draws from, so no drafter has seen its test starts. Same ordering as
blob['episode_idx'].

    python reproduce/data_prep/make_pusht_holdout_mask.py \
        --subgoals subgoals_dense_full.pt --out pusht_train_mask.npy \
        --exclude 'episodes/pusht_c2222.source_episodes*.json'
"""
import argparse
import glob
import json

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--subgoals", required=True)
p.add_argument("--out", required=True)
p.add_argument("--exclude", nargs="+", required=True)
a = p.parse_args()

blob = torch.load(a.subgoals, map_location="cpu", weights_only=False)
ep = blob["episode_idx"].numpy()
in5 = blob["in_5deg"].numpy().astype(bool)
banned = set()
files = [f for pat in a.exclude for f in sorted(glob.glob(pat))]
if not files:
    raise SystemExit(f"no exclusion files matched {a.exclude}")
for f in files:
    with open(f) as fh:
        banned.update(int(pair[0]) for pair in json.load(fh)["episodes"])
drop = np.isin(ep, np.fromiter(banned, dtype=np.int64, count=len(banned)))
mask = in5 & ~drop
np.save(a.out, mask)
print(f"[mask] episodes in file={len(ep)} in_5deg={int(in5.sum())} "
      f"banned(listed)={len(banned)} banned(in_5deg)={int((in5 & drop).sum())} "
      f"-> train={int(mask.sum())}  ({len(files)} exclusion files)")
