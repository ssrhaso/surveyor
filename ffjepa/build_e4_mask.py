"""Compute the E4 paper-matched training population (pos<15px & ang<5deg) as a
per-episode boolean mask aligned to a subgoals file's episode order, without
re-encoding anything (the existing subgoals_dense_full.pt already covers this
population as a strict subset of its pos<20/ang<20 headline filter).

The paper's FF-JEPA III-A training filter keeps 8,318 episodes; no natural
(pos, angle) threshold hits that exactly. pos<15px & ang<5deg -> 8,441 episodes
is the closest natural cut (established in the forensic audit, see
FFJEPA_HANDOFF.md section 12.4 item 4 and the H7 diff-table row).

Run on ofs-v01 (needs the real h5 + the subgoals file):
    cd ~/le-wm
    python -m ffjepa.build_e4_mask --subgoals subgoals_dense_full.pt \
        --h5 ~/.stable-wm/datasets/pusht_expert_train.h5 --out e4_mask.npy \
        --pos-thresh 15 --angle-deg 5
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np
import torch

from ffjepa import lewm_io


def parse_args():
    p = argparse.ArgumentParser(description="Build the E4 paper-matched population mask")
    p.add_argument("--subgoals", required=True)
    p.add_argument("--h5", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pos-thresh", type=float, default=15.0)
    p.add_argument("--angle-deg", type=float, default=5.0)
    return p.parse_args()


def main():
    args = parse_args()
    blob = torch.load(args.subgoals, map_location="cpu", weights_only=False)
    episode_idx = blob["episode_idx"]
    if hasattr(episode_idx, "numpy"):
        episode_idx = episode_idx.numpy()
    episode_idx = np.asarray(episode_idx).astype(int)
    n_eps = len(blob["lengths"])
    print(f"[subgoals] {n_eps} episodes in file, episode_idx range "
          f"[{episode_idx.min()}, {episode_idx.max()}]")

    with h5py.File(args.h5, "r") as f:
        ep_off = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
        finals = f["state"][ep_off[episode_idx] + ep_len[episode_idx] - 1]

    mask = np.array([
        lewm_io.eval_state_tol(lewm_io.canonical_target_for(s), s, args.angle_deg, args.pos_thresh)
        for s in finals
    ], dtype=bool)

    print(f"[mask] pos<{args.pos_thresh}px & ang<{args.angle_deg}deg: "
          f"{int(mask.sum())}/{n_eps} episodes kept "
          f"(paper's own filter keeps 8,318; closest natural cut, not exact)")
    np.save(args.out, mask)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
