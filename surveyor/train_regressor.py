"""Train the deterministic z_cond -> N-block subgoal regressor (JointMLP),
the regression arm of the diffusion-vs-regression comparison. Reuses
train_drafter's data pipeline, so architecture and objective are the only
differences vs the GDM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from surveyor.train_drafter import build_pairs


def parse_args():
    p = argparse.ArgumentParser(description="deterministic z_cond->block regressor trainer")
    p.add_argument("--subgoals", required=True, help="dense subgoal .pt (file_stride=1)")
    p.add_argument("--out", required=True)
    p.add_argument("--subgoal-step", type=int, required=True,
                   help="index gap between positions (dense file => horizon in frames)")
    p.add_argument("--n-future", type=int, default=3)
    p.add_argument("--mask", choices=["5", "20"], default="5")
    p.add_argument("--window-rule", choices=["clamp", "full"], default="clamp")
    p.add_argument("--hid", type=int, default=512)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3,
                   help="the probe-verified non-overfit setting")
    p.add_argument("--val-frac-mod", type=int, default=10,
                   help="episodes with idx %% this == 0 form the val split")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class BlockRegressor(nn.Module):
    """JointMLP: standardized z_cond -> standardized (N, D) block, one shot."""

    def __init__(self, dim, n_future, hid):
        super().__init__()
        self.dim, self.n_future = dim, n_future
        self.net = nn.Sequential(nn.Linear(dim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, n_future * dim))

    def forward(self, zc):
        return self.net(zc).view(-1, self.n_future, self.dim)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dev = args.device

    blob = torch.load(args.subgoals, map_location="cpu", weights_only=False)
    latents = blob["latents"].float()
    lengths = blob["lengths"].numpy()
    offsets = blob["offsets"].numpy()
    in_5deg = blob["in_5deg"].numpy().astype(bool)
    mask = in_5deg if args.mask == "5" else np.ones(len(lengths), dtype=bool)
    file_stride = int(blob.get("stride", 25))
    D = latents.shape[1]

    # per-episode split BEFORE pairing (no window leakage across the split)
    ep_ids = np.arange(len(lengths))
    val_eps = (ep_ids % args.val_frac_mod == 0)
    tr_mask = mask & ~val_eps
    va_mask = mask & val_eps

    conds_tr, tgts_tr, _, n_tr = build_pairs(latents, lengths, offsets, tr_mask,
                                             args.n_future, args.window_rule,
                                             step=args.subgoal_step)
    conds_va, tgts_va, _, n_va = build_pairs(latents, lengths, offsets, va_mask,
                                             args.n_future, args.window_rule,
                                             step=args.subgoal_step)
    print(f"[data] horizon={file_stride * args.subgoal_step} frames; "
          f"train {conds_tr.shape[0]} pairs / {n_tr} eps; "
          f"val {conds_va.shape[0]} pairs / {n_va} eps")

    pool = torch.cat([conds_tr, tgts_tr.reshape(-1, D)], dim=0)
    mean, std = pool.mean(0), pool.std(0).clamp_min(1e-6)
    stdz = lambda z: (z - mean.to(z.device)) / std.to(z.device)
    unstd = lambda z: z * std.to(z.device) + mean.to(z.device)

    model = BlockRegressor(D, args.n_future, args.hid).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    zc_tr = stdz(conds_tr).to(dev)
    zt_tr = stdz(tgts_tr.reshape(-1, D)).reshape(tgts_tr.shape).to(dev)
    zc_va = stdz(conds_va).to(dev)
    M = zc_tr.shape[0]

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(M, device=dev)
        tot = 0.0
        for i in range(0, M, args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(zc_tr[idx]), zt_tr[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            pred = unstd(model(zc_va).reshape(-1, D)).reshape(-1, args.n_future, D).cpu()
        rel = [(pred[:, k] - tgts_va[:, k]).norm(dim=1)
               / tgts_va[:, k].norm(dim=1).clamp_min(1e-8) for k in range(args.n_future)]
        print(f"[epoch {ep + 1:02d}/{args.epochs}] train_mse={tot / M:.5f}  "
              f"val rel_err: " + " / ".join(f"{r.mean():.4f}" for r in rel))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "mean": mean, "std": std,
                "dim": D, "n_future": args.n_future, "hid": args.hid,
                "subgoal_step": args.subgoal_step, "file_stride": file_stride,
                "horizon_frames": file_stride * args.subgoal_step,
                "mask": args.mask, "wd": args.weight_decay, "seed": args.seed},
               args.out)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
