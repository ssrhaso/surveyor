"""Drafter-quality read on held-out episodes (the suffix-decay probe at
V-JEPA 2 scale). Runs directly off saved latents - no encoder, no world
model, CPU-friendly.

For every anchor m in each episode: draft a block from pooled tokens[m]
(+ pooled goal), then per position j=1..N compare against the true latent
at m+j*S in pooled space:
  rel_err  = ||p_pred - p_true|| / ||p_true||
  cos_move = cosine(p_pred - p_cond, p_true - p_cond)   (movement direction)
Baselines: NO-OP (predict the arm stays: p_pred = p_cond) and the LERP
drafter. The LeWM-stack reference shape (probe_suffix_decay): GDM rel_err
0.159/0.229/0.318 vs no-op 7.2x/6.2x/4.5x worse - a real drafter must beat
no-op at every position by a clear factor.

  python -m surveyor.vjepa2.probe_drafter --ckpt gdm_vj2_s10.pt \
      --episodes droid_lat/lat_droid_ep09*.npz --stride 10
"""

from __future__ import annotations

import argparse
import glob

import numpy as np
import torch

from .drafter import load_token_gdm
from .sources import GDMDraft, LerpBlockDrafter
from .wm import pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--k", type=int, default=8, help="DDIM steps per draft")
    ap.add_argument("--anchor-stride", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = sorted(sum([glob.glob(p) for p in args.episodes], []))
    assert files, f"no files match {args.episodes}"
    planner = load_token_gdm(args.ckpt, device=args.device)
    N, S = planner.cfg.n_future, args.stride
    gdm = GDMDraft(planner, seed=args.seed)
    lerp = LerpBlockDrafter(n_future=N)

    from .train_drafter import _load_tokens

    stats = {arm: {j: {"rel": [], "cos": []} for j in range(1, N + 1)}
             for arm in ("gdm", "lerp", "noop")}
    n_anchors = 0
    for f in files:
        tok = torch.from_numpy(np.asarray(_load_tokens(f), dtype=np.float32)).to(args.device)
        T = tok.shape[0]
        goal = tok[-1:]
        for m in range(0, T - N * S, args.anchor_stride):
            cond = tok[m:m + 1]
            p_cond = pool(cond)[0]
            blocks = {"gdm": gdm.draft(cond, goal, k=args.k)[0],
                      "lerp": lerp.draft(cond, goal)[0]}
            n_anchors += 1
            for j in range(1, N + 1):
                p_true = pool(tok[m + j * S])
                mv_true = p_true - p_cond
                for arm in ("gdm", "lerp", "noop"):
                    p_pred = p_cond if arm == "noop" else pool(blocks[arm][j - 1])
                    rel = float((p_pred - p_true).norm() / p_true.norm().clamp_min(1e-8))
                    stats[arm][j]["rel"].append(rel)
                    if arm != "noop":
                        mv_pred = p_pred - p_cond
                        cos = float(torch.nn.functional.cosine_similarity(
                            mv_pred.view(1, -1), mv_true.view(1, -1)))
                        stats[arm][j]["cos"].append(cos)

    print(f"==== DRAFTER QUALITY ({len(files)} held-out eps, {n_anchors} anchors, "
          f"S={S}, k={args.k}) ====")
    print(f"{'pos':>5} | {'gdm rel':>8} {'cos':>6} | {'lerp rel':>8} {'cos':>6} | "
          f"{'noop rel':>8} | {'noop/gdm':>8}")
    for j in range(1, N + 1):
        g = np.mean(stats["gdm"][j]["rel"])
        gc = np.mean(stats["gdm"][j]["cos"])
        lp = np.mean(stats["lerp"][j]["rel"])
        lc = np.mean(stats["lerp"][j]["cos"])
        o = np.mean(stats["noop"][j]["rel"])
        print(f"  m+{j}  | {g:8.4f} {gc:6.3f} | {lp:8.4f} {lc:6.3f} | {o:8.4f} | "
              f"{o / max(g, 1e-8):8.2f}x")
    print("(LeWM-stack reference: gdm rel 0.159/0.229/0.318, noop/gdm 7.2/6.2/4.5x; "
          "a drafter that loses to lerp or noop is not yet usable)")


if __name__ == "__main__":
    main()
