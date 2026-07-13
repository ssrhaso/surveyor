"""Pre-gate for the stride-spacing direction: is a fine-stride subgoal target
well-posed in the frozen encoder's latent space?

Measures, over successful expert episodes and random anchors t, the realized
latent displacement disp(S) = ||E(t+S) - E(t)|| / ||E(t+S)|| for S in
{1, 5, 10, 15, 25} (S=1 is the encoder noise floor), plus two directional
checks: cos_long(S) (does the short move point where the stride-25 move
goes) and cos_succ(S) (do successive S-segments continue in a consistent
direction). If disp(5) collapses toward disp(1), the fine-stride target is
degenerate regardless of drafter.

Requires only the h5 and the frozen encoder; CPU is sufficient.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from specaccept import encoder

STRIDES = (1, 5, 10, 15, 25)
MAX_OFFSET = 50  # need t+2S for cos_succ at S=25


def parse_args():
    p = argparse.ArgumentParser(description="Stage 0: latent displacement vs stride")
    p.add_argument("--h5", required=True, help="pusht_expert_train.h5 (full or subset)")
    p.add_argument("--source", choices=["pretrained", "local"], default="local")
    p.add_argument("--encoder-id", default="quentinll/lewm-pusht")
    p.add_argument("--local-dir", default=None)
    p.add_argument("--swm-src", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-episodes", type=int, default=128)
    p.add_argument("--anchors-per-ep", type=int, default=4)
    p.add_argument("--angle-headline", type=float, default=20.0)
    p.add_argument("--pos-thresh", type=float, default=encoder.POS_THRESH)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json-out", default=None, help="optional path for raw stats dump")
    return p.parse_args()


def pct(x, q):
    return float(np.percentile(x, q))


def dist_stats(x):
    x = np.asarray(x, dtype=np.float64)
    return dict(mean=float(x.mean()), std=float(x.std()),
                p10=pct(x, 10), p50=pct(x, 50), p90=pct(x, 90), n=int(x.size))


@torch.no_grad()
def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    try:  # standalone h5 read path (see extract_subset.py note on the pixel filter)
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    import h5py

    model = encoder.load_lewm(source=args.source, encoder_id=args.encoder_id,
                              local_dir=args.local_dir, swm_src=args.swm_src,
                              device=args.device)

    offsets = sorted({0} | set(STRIDES) | {2 * s for s in STRIDES} | {MAX_OFFSET})
    with h5py.File(args.h5, "r") as f:
        ep_off = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
        state = f["state"]
        pixels = f["pixels"]
        n_total = len(ep_off)

        # success filter (same rule as build_subgoals pass 1), on a random scan
        # order so the kept sample is unbiased in episode index
        scan = rng.permutation(n_total)
        kept = []
        for e in scan:
            off, L = int(ep_off[e]), int(ep_len[e])
            if L <= MAX_OFFSET + 1:
                continue
            final = state[off + L - 1].astype(np.float64)
            target = encoder.canonical_target_for(final)
            if encoder.eval_state_tol(target, final, args.angle_headline, args.pos_thresh):
                kept.append((int(e), off, L))
            if len(kept) >= args.n_episodes:
                break
        print(f"[filter] scanned {np.where(scan == kept[-1][0])[0][0] + 1 if kept else 0} eps, "
              f"kept {len(kept)} successful@{args.angle_headline:g}deg with len>{MAX_OFFSET + 1}")

        # anchors + one contiguous span read per episode
        lat_by_key = {}  # (ep, t+off) -> latent row index
        all_rows_meta = []
        for e, off, L in kept:
            hi_anchor = L - 1 - MAX_OFFSET
            anchors = rng.choice(hi_anchor, size=min(args.anchors_per_ep, hi_anchor),
                                 replace=False)
            for t in anchors:
                for o in offsets:
                    all_rows_meta.append((e, off + int(t) + o, int(t), o))

        uniq = sorted({(e, row) for e, row, _, _ in all_rows_meta})
        rows_arr = np.array([row for _, row in uniq])
        order = np.argsort(rows_arr, kind="stable")
        # chunk the h5 read + encode: encode_frames preprocesses its whole input
        # to (n,3,224,224) float32 up front, which for thousands of frames is
        # multiple GB; 512-frame chunks keep peak RAM small at no fidelity cost
        # (latents are batch-independent, see encode_frames docstring)
        sorted_rows = rows_arr[order]
        chunks = []
        for i in range(0, len(sorted_rows), 512):
            chunks.append(encoder.encode_frames(model, pixels[sorted_rows[i:i + 512]],
                                                device=args.device,
                                                batch_size=args.batch_size))
            print(f"  encoded {min(i + 512, len(sorted_rows))}/{len(sorted_rows)}", flush=True)
        lat_sorted = torch.cat(chunks, 0)
        lat = torch.empty_like(lat_sorted)
        lat[order] = lat_sorted
        lat_by_key = {key: i for i, key in enumerate(uniq)}
        print(f"[encode] {len(uniq)} unique frames encoded "
              f"({len(kept)} eps x {args.anchors_per_ep} anchors x {len(offsets)} offsets)")

    # regroup per (episode, anchor)
    anchors_set = sorted({(e, t) for e, _, t, _ in all_rows_meta})
    Z = {}  # (e, t) -> {offset: latent}
    for e, row, t, o in all_rows_meta:
        Z.setdefault((e, t), {})[o] = lat[lat_by_key[(e, row)]]

    disp = {s: [] for s in STRIDES}
    cos_long = {s: [] for s in STRIDES if s < 25}
    cos_succ = {s: [] for s in STRIDES}
    for key in anchors_set:
        z = Z[key]
        z0 = z[0]
        d25 = z[25] - z0
        for s in STRIDES:
            ds = z[s] - z0
            disp[s].append(float(ds.norm() / z[s].norm()))
            dnext = z[2 * s] - z[s]
            cos_succ[s].append(float(torch.nn.functional.cosine_similarity(
                ds, dnext, dim=0)))
            if s < 25:
                cos_long[s].append(float(torch.nn.functional.cosine_similarity(
                    ds, d25, dim=0)))

    floor = np.median(disp[1])
    print(f"\n[STAGE 0] latent displacement vs stride  (n_anchors={len(anchors_set)}, "
          f"seed={args.seed}, h5={args.h5})")
    print(f"{'S':>4} {'disp mean':>10} {'p10':>7} {'p50':>7} {'p90':>7} "
          f"{'x floor(p50)':>12} {'cos_long':>9} {'cos_succ':>9}")
    results = {}
    for s in STRIDES:
        d = dist_stats(disp[s])
        cl = float(np.mean(cos_long[s])) if s < 25 else 1.0
        cs = float(np.mean(cos_succ[s]))
        results[s] = dict(disp=d, x_floor=d["p50"] / floor, cos_long=cl, cos_succ=cs)
        print(f"{s:>4} {d['mean']:>10.4f} {d['p10']:>7.4f} {d['p50']:>7.4f} {d['p90']:>7.4f} "
              f"{d['p50'] / floor:>12.2f} {cl:>9.3f} {cs:>9.3f}")

    print("\n[read-off] KILL if disp(5) ~ disp(1) (x_floor near 1): 5-step targets are "
          "latent-noise. PROCEED if disp(5) sits well above the floor (and cos_long/"
          "cos_succ show the short move is directionally structured, not jitter).")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"args": vars(args), "n_anchors": len(anchors_set),
                       "floor_p50_S1": floor, "per_stride": results}, fh, indent=2)
        print(f"[save] {args.json_out}")


if __name__ == "__main__":
    main()
