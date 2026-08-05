"""Gap-maximizing verification readout: derive the METRIC, not just tau.

The gap statistic (gap_stat.py) shows the verifier's operating interval
[equiv p90, hop p10] is closed on some substrates: TwoRoom saturated, DROID
pooled compressed, Reacher tail-overlapped. The information often survives
anyway (token-space c* separates on DROID), it is the fixed readout that loses
it. This script learns a small lens used ONLY by the verifier, leaving the world
model and planner frozen and the plug-in claim untouched:

  vector mode   z (D,)          -> L2-normalized W z            (linear, D->d)
  attentive     tokens (T_tok,D) -> L2-normalized W (sum_i a_i z_i),
                a = softmax(z q / sqrt(D))                      (query + linear)

Training is time-contrastive InfoNCE on cached latent sequences: positives are
frames <= equiv_stride apart, negatives are frames >= hop_stride apart in the
same episode plus every cross-episode frame in the batch. That objective IS the
gap, pulling the equiv distribution down and pushing the hop distribution up.
Afterward, re-run gap_stat with --readout <ckpt> and read tau off the new gap's
midpoint. Report before and after; no closed-loop claim until the lens is
validated there.

  python -m specaccept.probes.learn_readout --name reacher \
      --subgoals-pt subgoals_reacher_dense.pt --hop-stride 10 --out readout_reacher.pt
  python -m specaccept.probes.learn_readout --name droid --attentive \
      --lat-glob "droid_lat/lat_*.npz" --hop-stride 10 --out readout_droid.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import nn

from specaccept.probes.gap_stat import sequences_from_glob, sequences_from_pt


def token_sequences_from_glob(pattern, max_eps=None):
    """fp16 in RAM (explicit copy closes each file): 2k-ep token grids are
    ~115GB fp32; fp16 + --max-eps keeps the scale-up on one node."""
    import glob
    files = sorted(glob.glob(pattern))
    if max_eps:
        files = files[:max_eps]
    seqs = []
    for f in files:
        tok = np.load(f, mmap_mode="r") if f.endswith(".npy") else np.load(f)["tokens"]
        seqs.append(torch.from_numpy(np.array(tok, dtype=np.float16)))
    return seqs        # each (T, T_tok, D)


class Readout(nn.Module):
    def __init__(self, dim, out_dim=64, attentive=False):
        super().__init__()
        self.attentive = attentive
        self.proj = nn.Linear(dim, out_dim, bias=False)
        if attentive:
            self.query = nn.Parameter(torch.zeros(dim))
            nn.init.normal_(self.query, std=dim ** -0.5)

    def forward(self, z):
        if self.attentive:                       # (..., T_tok, D)
            a = torch.softmax(z @ self.query / z.shape[-1] ** 0.5, dim=-1)
            z = (a.unsqueeze(-1) * z).sum(dim=-2)
        out = self.proj(z)
        return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--subgoals-pt", default=None)
    ap.add_argument("--lat-glob", default=None)
    ap.add_argument("--attentive", action="store_true",
                    help="token-grid inputs: learned-query attention pooling "
                         "(the fixed mean-pool is what compresses DROID)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-dim", type=int, default=64)
    ap.add_argument("--equiv-stride", type=int, default=1)
    ap.add_argument("--hop-stride", type=int, default=10)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-eps", type=int, default=16)
    ap.add_argument("--max-eps", type=int, default=None,
                    help="cap episodes loaded from --lat-glob (RAM)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.subgoals_pt:
        seqs = sequences_from_pt(args.subgoals_pt)
    elif args.attentive:
        seqs = token_sequences_from_glob(args.lat_glob, max_eps=args.max_eps)
    else:
        seqs = sequences_from_glob(args.lat_glob)
    seqs = [s for s in seqs if len(s) > 2 * args.hop_stride + 2]
    n_val = max(2, int(len(seqs) * args.val_frac))
    tr, va = seqs[:-n_val], seqs[-n_val:]
    dim = seqs[0].shape[-1]
    print(f"[data] {len(tr)} train / {len(va)} val episodes, dim={dim}, "
          f"attentive={args.attentive}")

    model = Readout(dim, args.out_dim, attentive=args.attentive).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(args.seed)

    def sample_batch(pool):
        anchors, positives, hopnegs = [], [], []
        for _ in range(args.batch_eps):
            z = pool[int(torch.randint(0, len(pool), (1,), generator=g))]
            hi = len(z) - args.hop_stride - args.equiv_stride - 1
            m = int(torch.randint(0, hi, (1,), generator=g))
            anchors.append(z[m])
            positives.append(z[m + args.equiv_stride])
            hopnegs.append(z[m + args.hop_stride])
        st = torch.stack
        return st(anchors).to(args.device).float(), \
            st(positives).to(args.device).float(), \
            st(hopnegs).to(args.device).float()

    def infonce(a, p, hn):
        za, zp, zh = model(a), model(p), model(hn)
        # negatives: the explicit hop frame + every other episode's anchor/pos
        logits_pos = (za * zp).sum(-1, keepdim=True) / args.temp
        neg_bank = torch.cat([zh, za.roll(1, 0), zp.roll(1, 0)], dim=0)
        logits_neg = za @ neg_bank.T / args.temp
        logits = torch.cat([logits_pos, logits_neg], dim=1)
        return nn.functional.cross_entropy(
            logits, torch.zeros(len(za), dtype=torch.long, device=za.device))

    def gap_of(pool):
        eq, hop = [], []
        with torch.no_grad():
            for z in pool:
                r = model(z.to(args.device).float())
                eq += (r[args.equiv_stride:] - r[:-args.equiv_stride]).norm(dim=1).tolist()
                hop += (r[args.hop_stride:] - r[:-args.hop_stride]).norm(dim=1).tolist()
        eq, hop = torch.tensor(eq), torch.tensor(hop)
        return float(torch.quantile(eq, .9)), float(torch.quantile(hop, .1))

    lo0, hi0 = gap_of(va)
    print(f"[before] val equiv p90={lo0:.3f} hop p10={hi0:.3f} "
          f"gap={'OPEN' if hi0 > lo0 else 'closed'} ({hi0 - lo0:+.3f})")
    for step in range(args.steps):
        loss = infonce(*sample_batch(tr))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 400 == 0 or step == args.steps - 1:
            lo, hi = gap_of(va)
            print(f"step {step:5d} loss {float(loss):.4f}  val gap "
                  f"[{lo:.3f}, {hi:.3f}] width {hi - lo:+.3f}", flush=True)
    lo, hi = gap_of(va)
    torch.save({"state": model.state_dict(), "dim": dim, "out_dim": args.out_dim,
                "attentive": args.attentive, "name": args.name,
                "gap_before": [lo0, hi0], "gap_after": [lo, hi],
                "tau_derived": (lo + hi) / 2 if hi > lo else None,
                "args": vars(args)}, args.out)
    print(f"[after] val gap [{lo:.3f}, {hi:.3f}] width {hi - lo:+.3f} "
          f"(was {hi0 - lo0:+.3f}); derived tau = "
          f"{(lo + hi) / 2:.3f}" if hi > lo else "[after] gap still closed")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
