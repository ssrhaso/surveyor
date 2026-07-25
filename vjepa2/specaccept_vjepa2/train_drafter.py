"""Train TokenGDM on encoded episodes (lat_*.npz from encode_episodes.py).

Pairs mirror the LeWM-stack build: for every anchor frame m with m + N*S in
range, cond = tokens[m], target block = tokens at m+S..m+N*S, goal = final
frame. Pairs are indexed lazily (tokens held in RAM as fp16, ~0.7MB/frame at
ViT-g); materializing all blocks in fp32 would need tens of GB. Per-dim
standardization stats (shared across tokens) come from a frame subsample;
pooling commutes with the affine standardize, so pooled conditioning here
matches GDMDraft's pool-then-standardize at inference exactly.

The last --val-files files are held out; val loss is printed every log
interval as the overfit check.

  python -m specaccept_vjepa2.train_drafter --data droid_lat/lat_*.npz \
      --out gdm_vj2_s10.pt --stride 10 --epochs 150 --device cuda
"""

from __future__ import annotations

import argparse
import collections
import glob

import numpy as np
import torch

from .drafter import GaussianDiffusion, TokenGDM, TokenGDMConfig, save_token_gdm


def _load_tokens(f: str):
    """fp16 (T, T_tok, D). .npy is memory-mapped (large-scale path: 150k
    frames of ViT-g tokens is ~100GB - far beyond RAM); .npz is loaded."""
    if f.endswith(".npy"):
        return np.load(f, mmap_mode="r")
    return np.load(f)["tokens"]


def _fd_budget():
    """Per-process open-memmap budget: raise the soft fd limit to the hard
    limit, keep 512 fds of headroom for torch/cuda/sockets. Holding every
    episode's memmap open died on EMFILE at 2k files (job 2292554)."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY
                                               or hard > soft):
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
                soft = hard
            except (ValueError, OSError):
                pass
        if soft == resource.RLIM_INFINITY:
            return 65536
        return max(64, soft - 512)
    except ImportError:          # windows smoke runs
        return 512


class PairDataset(torch.utils.data.Dataset):
    """Lazy (cond, block, goal) pairs over per-episode token arrays.
    Memmaps open on demand through an LRU capped at the fd budget."""

    MAX_OPEN = _fd_budget()

    def __init__(self, files: list[str], stride: int, n_future: int):
        self.stride, self.N = int(stride), int(n_future)
        self.files = list(files)
        self._open = collections.OrderedDict()
        self.index = []
        for i, f in enumerate(self.files):
            t = _load_tokens(f)
            self.index += [(i, m)
                           for m in range(0, t.shape[0] - self.N * self.stride)]
            del t

    def tokens(self, i):
        t = self._open.get(i)
        if t is None:
            while len(self._open) >= self.MAX_OPEN:
                self._open.popitem(last=False)
            t = self._open[i] = _load_tokens(self.files[i])
        else:
            self._open.move_to_end(i)
        return t

    def __len__(self):
        return len(self.index)

    def __getitem__(self, j):
        i, m = self.index[j]
        t = self.tokens(i)
        cond = torch.from_numpy(np.asarray(t[m], dtype=np.float32))
        block = torch.from_numpy(np.stack(
            [np.asarray(t[m + (k + 1) * self.stride], dtype=np.float32)
             for k in range(self.N)]))
        goal = torch.from_numpy(np.asarray(t[-1], dtype=np.float32))
        return cond, block, goal


def stats_from(files: list[str], every: int = 10):
    """Per-dim (D,) mean/std over a frame subsample across files."""
    acc = []
    for f in files:
        tok = _load_tokens(f)[::every]
        acc.append(np.asarray(tok).reshape(-1, tok.shape[-1]).astype(np.float32))
    flat = torch.from_numpy(np.concatenate(acc))
    return flat.mean(0), flat.std(0).clamp_min(1e-6)


def residual_stats_from(files: list[str], stride: int, n_future: int, every: int = 7):
    """Per-dim (D,) mean/std of (tokens[m+j*S] - tokens[m]) over an anchor
    subsample; the quantity a residual drafter denoises."""
    acc = []
    for f in files:
        tok = _load_tokens(f)
        for m in range(0, tok.shape[0] - n_future * stride, every):
            base = np.asarray(tok[m], dtype=np.float32)
            for j in range(1, n_future + 1):
                acc.append(np.asarray(tok[m + j * stride], dtype=np.float32) - base)
    flat = torch.from_numpy(np.stack(acc).reshape(-1, acc[0].shape[-1]))
    return flat.mean(0), flat.std(0).clamp_min(1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="latent .npz files/globs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=10, help="S in (5fps) frames")
    ap.add_argument("--n-future", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--goal-cond", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--residual", action=argparse.BooleanOptionalAction, default=False,
                    help="denoise (block - cond_tokens); v2 fix for the no-op loss")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--schedule", default="cosine", choices=["linear", "cosine"])
    ap.add_argument("--parameterization", default="v", choices=["eps", "v"])
    ap.add_argument("--min-snr-gamma", type=float, default=5.0)
    ap.add_argument("--val-files", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    files = sorted(sum([glob.glob(p) for p in args.data], []))
    assert len(files) > args.val_files, f"only {len(files)} files match {args.data}"
    tr_files, va_files = files[:-args.val_files], files[-args.val_files:]

    tr = PairDataset(tr_files, args.stride, args.n_future)
    va = PairDataset(va_files, args.stride, args.n_future)
    T_tok, D = tr.tokens(0).shape[-2:]
    print(f"[pairs] train {len(tr)} / val {len(va)} from {len(files)} files "
          f"(N={args.n_future}, T_tok={T_tok}, D={D}, S={args.stride})")

    frame_a, frame_b = stats_from(tr_files)
    if args.residual:
        stat_a, stat_b = residual_stats_from(tr_files, args.stride, args.n_future)
        print(f"[stats] residual std p50 {float(stat_b.median()):.4f} "
              f"(frame std p50 {float(frame_b.median()):.4f})")
    else:
        stat_a, stat_b = frame_a, frame_b
    dev = args.device
    a = stat_a.to(dev).view(1, 1, 1, -1)
    b = stat_b.to(dev).view(1, 1, 1, -1)
    fa = frame_a.to(dev).view(1, 1, -1)
    fb = frame_b.to(dev).view(1, 1, -1)

    cfg = TokenGDMConfig(latent_dim=D, tokens_per_frame=T_tok, n_future=args.n_future,
                         hidden=args.hidden, depth=args.depth, heads=args.heads,
                         goal_cond=args.goal_cond, residual=args.residual)
    model = TokenGDM(cfg).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] TokenGDM {n_par / 1e6:.1f}M params, goal_cond={args.goal_cond}")
    diffusion = GaussianDiffusion(timesteps=args.timesteps, schedule=args.schedule,
                                  parameterization=args.parameterization,
                                  min_snr_gamma=args.min_snr_gamma, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def batch_loss(cond, block, goal):
        cond, block, goal = cond.to(dev), block.to(dev), goal.to(dev)
        target = block - cond.unsqueeze(1) if args.residual else block
        x0 = (target - a) / b                                     # (B, N, T_tok, D) std
        B = x0.shape[0]
        c = ((cond - fa) / fb).mean(dim=-2)                       # frame-std, pool -> (B, D)
        g = ((goal - fa) / fb).mean(dim=-2) if args.goal_cond else None
        return diffusion.p_losses(model, x0.reshape(B, -1, D), c, goal=g)

    tl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True,
                                     num_workers=args.workers, drop_last=True)
    vl = torch.utils.data.DataLoader(va, batch_size=args.batch, shuffle=False,
                                     num_workers=2)
    for ep in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for cond, block, goal in tl:
            loss = batch_loss(cond, block, goal)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                vtot, vnb = 0.0, 0
                for cond, block, goal in vl:
                    vtot += float(batch_loss(cond, block, goal))
                    vnb += 1
            print(f"epoch {ep:4d}  train {tot / nb:.5f}  val {vtot / max(vnb, 1):.5f}",
                  flush=True)
        if ep and ep % 25 == 0:
            save_token_gdm(args.out + f".ep{ep}", model, diffusion, stat_a, stat_b,
                           cond_stat_a=frame_a, cond_stat_b=frame_b,
                           extra={"stride": args.stride, "epoch": ep})

    save_token_gdm(args.out, model.cpu(), diffusion, stat_a, stat_b,
                   cond_stat_a=frame_a, cond_stat_b=frame_b,
                   extra={"stride": args.stride, "files": files,
                          "pairs": len(tr), "args": vars(args)})
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
