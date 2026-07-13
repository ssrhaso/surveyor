"""Build a held-out episodes file for the DSpark closed-loop eval: the val
episodes (episode_id % 5 == 0) of real_chains_n3.pt, disjoint from the DSpark
head's training episodes, at a given goal_offset (start = ep_len-1-offset,
matching sample_long). Output is an --episodes-file JSON for the PushT eval
driver.
"""
import argparse, json
import numpy as np, torch, h5py

p = argparse.ArgumentParser()
p.add_argument("--chains", default="real_chains_n3.pt")
p.add_argument("--h5", default="subset_longeval.h5")
p.add_argument("--goal-offset", type=int, default=150)
p.add_argument("--out", default="dspark_val.episodes.json")
args = p.parse_args()

d = torch.load(args.chains, weights_only=False)
ep = d["episode_id"].numpy()
val_eps = np.unique(ep[ep % 5 == 0])                 # held-out episodes
with h5py.File(args.h5, "r") as f:
    ep_len = f["ep_len"][:]
pairs = []
for e in val_eps:
    L = int(ep_len[e])
    if L > args.goal_offset + 1:                     # sample_long eligibility
        pairs.append([int(e), int(L - 1 - args.goal_offset)])
payload = {"episodes": pairs, "goal_offset": args.goal_offset, "seed": 42,
           "eval_filter": "success5-heldout",
           "note": "DSpark head val split (episode_id%5==0), disjoint from head training"}
with open(args.out, "w") as f:
    json.dump(payload, f)
print(f"[val-episodes] {len(pairs)} held-out episodes at goal_offset={args.goal_offset} -> {args.out}")
