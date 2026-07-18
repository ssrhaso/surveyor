"""Offline gate probes for the DSpark/GDM direction (no CEM, one data load).

A1: few-step-draft acceptance curve (matched-seed DDIM, per-k error to the
50-step reference). A2: speculative-acceptance simulator, a demo-replay
proxy for the closed loop reporting per-tau call ratio, accepted depth, and
consumed-subgoal error (an optimistic bound). A3: posterior multimodality
via repeat draws with a per-condition two-means separation test.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from specaccept import encoder
from specaccept.drafter import load_gdm_planner

RUNWAY = 150  # episodes150 guarantee: start + 150 <= last frame


def parse_args():
    p = argparse.ArgumentParser(description="DSpark->GDM Round 2 Phase A gates")
    p.add_argument("--gdm-ckpt", required=True)
    p.add_argument("--stride", type=int, default=25, help="S: env-steps per subgoal hop")
    p.add_argument("--h5", required=True)
    p.add_argument("--episodes-file", required=True)
    p.add_argument("--source", choices=["pretrained", "local"], default="local")
    p.add_argument("--encoder-id", default="quentinll/lewm-pusht")
    p.add_argument("--local-dir", default=None)
    p.add_argument("--swm-src", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-probe", type=int, default=256)
    p.add_argument("--probes", nargs="+", default=["a1", "a2", "a3"],
                   choices=["a1", "a2", "a3"])
    p.add_argument("--ddim-ks", type=int, nargs="+", default=[2, 4, 8, 16, 50])
    p.add_argument("--n-seeds", type=int, default=4, help="seeds for A1 spread/matched dist")
    p.add_argument("--taus", type=float, nargs="+",
                   default=[0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 99.0],
                   help="A2 accept tolerances (99 = never reject: pure suffix-decay row)")
    p.add_argument("--k-samples", type=int, default=32, help="A3 draws per condition")
    p.add_argument("--gdm-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json-out", default=None)
    return p.parse_args()


def rel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """||a - b|| / ||b|| per row."""
    return (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-8)


def kmeans2(x: np.ndarray, iters: int = 12):
    """Tiny 2-means for one condition's (K, D) samples. Farthest-pair init.
    Returns (sep_ratio, minor_frac, inter_dist): centroid distance over mean
    rms cluster radius, minor-cluster mass, and the raw centroid distance."""
    d2 = ((x[:, None] - x[None]) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    c = np.stack([x[i], x[j]])
    for _ in range(iters):
        assign = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        if assign.min() == assign.max():  # collapsed to one cluster
            return 0.0, 0.0, 0.0
        c = np.stack([x[assign == 0].mean(0), x[assign == 1].mean(0)])
    r = [np.sqrt(((x[assign == m] - c[m]) ** 2).sum(-1).mean()) for m in (0, 1)]
    inter = float(np.linalg.norm(c[0] - c[1]))
    sep = inter / (0.5 * (r[0] + r[1]) + 1e-8)
    minor = float(min((assign == 0).mean(), (assign == 1).mean()))
    return float(sep), minor, inter


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dev = args.device
    planner = load_gdm_planner(args.gdm_ckpt, device=dev)
    N = planner.cfg.n_future
    H = RUNWAY // args.stride  # hops that fit the runway
    print(f"[gdm] ckpt={args.gdm_ckpt} N={N} sampler={planner.diffusion.sampler} "
          f"stride={args.stride} hops={H}")
    assert planner.diffusion.sampler != "ddpm" or "a1" not in args.probes, \
        "A1's matched-seed comparison assumes deterministic DDIM (eta=0)"

    with open(args.episodes_file) as f:
        payload = json.load(f)
    pairs = payload["episodes"][:args.n_probe]
    print(f"[episodes-file] {args.episodes_file}: {len(pairs)} windows")

    model = encoder.load_lewm(source=args.source, encoder_id=args.encoder_id,
                              local_dir=args.local_dir, swm_src=args.swm_src, device=dev)

    # encode Z: (B, H+1, D) latents at start + h*stride, h = 0..H
    import h5py
    with h5py.File(args.h5, "r") as f:
        ep_off = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
        pixels = f["pixels"]
        rows = []
        for ep, st in pairs:
            base = int(ep_off[ep]); last = base + int(ep_len[ep]) - 1
            for h in range(H + 1):
                r = base + int(st) + h * args.stride
                assert r <= last, f"episode {ep} shorter than its {RUNWAY} runway?"
                rows.append(r)
        rows = np.array(rows)
        order = np.argsort(rows, kind="stable")
        sorted_rows = rows[order]
        # chunked read+encode: encode_frames preprocesses its whole input to
        # (n,3,224,224) float32 up front; the stride-5 config has ~8k frames
        chunks = []
        for i in range(0, len(sorted_rows), 512):
            chunks.append(encoder.encode_frames(model, pixels[sorted_rows[i:i + 512]],
                                                device=dev))
        lat = torch.cat(chunks, 0)
        out = torch.empty_like(lat)
        out[order] = lat
    B = len(pairs)
    D = out.shape[1]
    Z = out.reshape(B, H + 1, D).to(dev)
    print(f"[encode] Z: {tuple(Z.shape)}")

    gen = lambda s: torch.Generator(device=planner.device).manual_seed(int(s))
    z_cond, z_true = Z[:, 0], Z[:, 1]
    results = {"config": {"ckpt": args.gdm_ckpt, "stride": args.stride, "N": N,
                          "hops": H, "n": B, "seed": args.seed}}

    # A1: few-step-draft acceptance curve
    if "a1" in args.probes:
        ks = sorted(set(args.ddim_ks) | {args.gdm_steps})
        drafts = {k: torch.stack([planner.sample_next(z_cond, n_steps=k,
                                                      generator=gen(args.seed + s))
                                  for s in range(args.n_seeds)], dim=1)  # (B, S, D)
                  for k in ks}
        ref = drafts[args.gdm_steps]
        print(f"\n[A1] few-step drafts vs {args.gdm_steps}-step reference "
              f"(matched x_T per seed; n={B} x {args.n_seeds} seeds)")
        print(f"{'k':>4} {'rel_err(true)':>14} {'cos_move':>9} {'spread':>8} "
              f"{'dist_to_full':>13} {'acc@0.05':>9} {'acc@0.10':>9} {'acc@0.20':>9}")
        a1 = {}
        for k in ks:
            d = drafts[k]
            re_true = rel(d[:, 0], z_true).mean().item()
            mv = torch.nn.functional.cosine_similarity(
                d[:, 0] - z_cond, z_true - z_cond, dim=-1).mean().item()
            spread = d.std(dim=1).mean().item()
            dist = rel(d.reshape(-1, D), ref.reshape(-1, D))
            accs = [(dist <= t).float().mean().item() for t in (0.05, 0.10, 0.20)]
            a1[k] = dict(rel_err_true=re_true, cos_move=mv, spread=spread,
                         dist_to_full=dist.mean().item(),
                         acc005=accs[0], acc010=accs[1], acc020=accs[2])
            print(f"{k:>4} {re_true:>14.4f} {mv:>9.3f} {spread:>8.4f} "
                  f"{dist.mean().item():>13.4f} {accs[0]:>9.3f} {accs[1]:>9.3f} {accs[2]:>9.3f}")
        print("[A1 gate] pick smallest k with rel_err(true)/spread ~= 50-step row "
              "(dist_to_full is the strictness reference; the loop consumes samples, "
              "not the exact 50-step draw).")
        results["a1"] = a1

    # A2: env-verifier speculative-acceptance simulator
    if "a2" in args.probes:
        blocks = torch.stack([planner.sample_sequence(Z[:, h], n_steps=args.gdm_steps,
                                                      generator=gen(args.seed + 1000 + h))
                              for h in range(H)])              # (H, B, N, D)
        fresh_err = torch.stack([rel(blocks[h][:, 0], Z[:, h + 1]) for h in range(H)])
        fresh_mean = fresh_err.mean().item()
        print(f"\n[A2] speculative-acceptance simulator: n={B}, hops={H}, N={N}, "
              f"fresh every-step baseline rel_err={fresh_mean:.4f}")
        print("  (demo-replay proxy: expert acts, so acceptance is an optimistic "
              "bound; Phase B closed-loop arm re-measures at the chosen tau)")
        print(f"{'tau':>6} {'call_ratio':>11} {'depth':>7} {'acc@p1':>7} {'acc@p2':>7} "
              f"{'acc@p3':>7} {'consumed_err':>13} {'stale_pen':>10}")
        a2 = {}
        Zn = Z  # alias
        for tau in args.taus:
            h0 = np.zeros(B, dtype=int)          # hop the active block was drafted at
            drafts_used = np.ones(B)
            pos_seen = np.zeros(N); pos_acc = np.zeros(N)
            consumed = []
            for h in range(H):
                j = h - h0                        # 0-based position consumed this hop
                tgt = blocks[torch.as_tensor(h0), torch.arange(B), torch.as_tensor(j)]
                err = rel(tgt, Zn[:, h + 1]).cpu().numpy()
                consumed.append(err)
                accept = err <= tau
                for p in range(N):
                    m = (j == p)
                    pos_seen[p] += m.sum(); pos_acc[p] += (accept & m).sum()
                exhausted = (j + 1) >= N
                redraft = (~accept) | exhausted
                if h + 1 < H:
                    h0 = np.where(redraft, h + 1, h0)
                    drafts_used += redraft
            consumed = np.stack(consumed)          # (H, B)
            call_ratio = drafts_used.mean() / H
            depth = H / drafts_used.mean()
            accs = [pos_acc[p] / max(pos_seen[p], 1) for p in range(N)]
            cons_mean = float(consumed.mean())
            a2[tau] = dict(call_ratio=float(call_ratio), depth=float(depth),
                           acc_by_pos=[float(a) for a in accs],
                           consumed_err=cons_mean, fresh_err=fresh_mean,
                           stale_penalty=cons_mean - fresh_mean,
                           pos_seen=[int(s) for s in pos_seen])
            print(f"{tau:>6.2f} {call_ratio:>11.3f} {depth:>7.2f} "
                  f"{accs[0]:>7.3f} {accs[1] if N > 1 else float('nan'):>7.3f} "
                  f"{accs[2] if N > 2 else float('nan'):>7.3f} "
                  f"{cons_mean:>13.4f} {cons_mean - fresh_mean:>10.4f}")
        print("[A2 gate] want a tau with depth >= ~1.5 and stale_pen ~= 0: that tau "
              "goes to the Phase B closed-loop arm. tau=99 row = unfiltered suffix "
              "decay (cross-check vs the drafter battery).")
        results["a2"] = {str(k): v for k, v in a2.items()}

    # A3: m+1 posterior multimodality
    if "a3" in args.probes:
        K = args.k_samples
        samples = torch.stack([planner.sample_next(z_cond, n_steps=args.gdm_steps,
                                                   generator=gen(args.seed + 2000 + s))
                               for s in range(K)], dim=1)      # (B, K, D)
        spread = samples.std(dim=1).mean().item()
        move = (z_true - z_cond).norm(dim=-1).cpu().numpy()    # actual move scale
        X = samples.cpu().numpy()
        seps, minors, gaps = np.zeros(B), np.zeros(B), np.zeros(B)
        for i in range(B):
            seps[i], minors[i], gaps[i] = kmeans2(X[i])
        gap_vs_move = gaps / np.maximum(move, 1e-8)
        cand = (seps > 2.0) & (minors >= 0.25) & (gap_vs_move > 0.5)
        print(f"\n[A3] m+1 posterior multimodality: n={B}, K={K}, "
              f"cond_spread={spread:.4f}")
        print(f"  2-means separation: p50={np.median(seps):.2f} p90={np.percentile(seps, 90):.2f}"
              f"  minor-mass p50={np.median(minors):.2f}"
              f"  modegap/move p50={np.median(gap_vs_move):.2f}")
        print(f"  candidate-multimodal conditions (sep>2 & minor>=0.25 & gap>0.5*move): "
              f"{int(cand.sum())}/{B} ({100 * cand.mean():.1f}%)")
        print("[A3 gate] <~5% candidates => posterior effectively unimodal on this "
              "population; K-wide CEM targets are dead on PushT (document and move on). "
              "A real multimodal fraction => spec the set-valued cost as a Phase B arm.")
        results["a3"] = dict(spread=spread, sep_p50=float(np.median(seps)),
                             sep_p90=float(np.percentile(seps, 90)),
                             minor_p50=float(np.median(minors)),
                             gap_vs_move_p50=float(np.median(gap_vs_move)),
                             frac_candidate=float(cand.mean()))

    if args.json_out:
        from pathlib import Path
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\n[save] {args.json_out}")


if __name__ == "__main__":
    main()
