"""TwoRoom verification gap in the PAIRED verifier's space (frozen DINOv2).

Derives the accept threshold for surveyor.paired BEFORE any closed-loop cell,
in the exact space the verifier will use (encode_frames_dino: ImageNet
preprocessing, pooled patch tokens). A tau lifted from another encode pass or
another pooling would not be the same number, so this probe -- not the DINO-WM
lens checkpoint -- is what the TwoRoom paired arm is calibrated from.

Three distance populations, all within cross-room SUCCESSFUL episodes (the
drafter's own training population) and restricted to the TRAINING episode range
so the eval holdout is never touched:

  equiv_criterion  pairs whose agent positions are within the env success
                   radius (16 px). This is what "arrived" means, so it is the
                   population tau must sit above.
  equiv_temporal   consecutive frames (stride 1); the encoder's noise floor.
  hop              frames S apart (the serving subgoal stride).
  cross            frames from different episodes; the saturation reference.

gap = [equiv p90, hop p10]; OPEN iff equiv p90 < hop p10. Derived tau = the gap
midpoint when open. Reported for several S so the choice of stride is visible
rather than assumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # tworoom.h5 pixels are blosc-compressed; this registers the filter
    import hdf5plugin  # noqa: F401
except ImportError:
    pass
import h5py
import numpy as np
import torch

from surveyor import encoder, paired


def parse_args():
    p = argparse.ArgumentParser(description="TwoRoom DINOv2 verification-gap probe")
    p.add_argument("--h5", required=True)
    p.add_argument("--out", default="Results/gap_stat/gap_tworoom_paired.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--hops", type=int, nargs="+", default=[5, 10, 25])
    p.add_argument("--episodes", type=int, default=250,
                   help="episodes sampled from the TRAIN range")
    p.add_argument("--episode-max", type=int, default=4000,
                   help="exclusive upper episode index; the eval holdout starts here")
    p.add_argument("--pos-thresh", type=float, default=encoder.TWOROOM_POS_THRESH)
    p.add_argument("--wall-center", type=float, default=112.0)
    p.add_argument("--pairs-per-pop", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dump-latents", default=None,
                   help="directory to write per-episode lat_tworoom_paired_*.npz "
                        "(key 'tokens'), the input format learn_readout --lat-glob "
                        "expects. Lets the lens be trained in exactly this space.")
    return p.parse_args()


def pct(x):
    x = np.asarray(x, dtype=np.float64)
    return {"n": int(x.size),
            "p10": float(np.percentile(x, 10)),
            "p50": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90))}


def rel(a, b):
    """Relative L2 in the verifier's convention: ||a - b|| / ||b||, where b is
    the pursued waypoint (SurveyorPairedSource divides by the target norm)."""
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-8))


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    dev = args.device

    dino = paired.load_dinov2(device=dev)
    print(f"[probe] DINOv2 loaded on {dev}")

    with h5py.File(args.h5, "r") as f:
        ep_off = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
        state = f["state"]
        goal_state = f["goal_state"]
        dist = f["distance_to_target"]
        pixels = f["pixels"]

        n_total = min(len(ep_off), args.episode_max)
        eligible = []
        for e in range(n_total):
            off, L = int(ep_off[e]), int(ep_len[e])
            cross = (state[off][0] < args.wall_center) != \
                    (goal_state[off][0] < args.wall_center)
            if not cross:
                continue
            if not encoder.tworoom_success(float(dist[off + L - 1]), args.pos_thresh):
                continue
            eligible.append(e)
        print(f"[probe] {len(eligible)}/{n_total} train episodes cross-room AND successful")
        if len(eligible) < 20:
            raise SystemExit("too few eligible episodes")

        eps = rng.choice(eligible, size=min(args.episodes, len(eligible)),
                         replace=False)

        dump = Path(args.dump_latents) if args.dump_latents else None
        if dump:
            dump.mkdir(parents=True, exist_ok=True)

        lat, pos, epid = [], [], []
        for k, e in enumerate(eps):
            off, L = int(ep_off[e]), int(ep_len[e])
            z = paired.encode_frames_dino(dino, pixels[off:off + L], device=dev)
            zn = z.numpy()
            lat.append(zn)
            pos.append(np.asarray(state[off:off + L], dtype=np.float32))
            epid.append(np.full(L, k, dtype=np.int64))
            if dump:
                np.savez(dump / f"lat_tworoom_paired_{k:04d}.npz", tokens=zn,
                         state=np.asarray(state[off:off + L], dtype=np.float32))
            if k % 50 == 0:
                print(f"  encoded {k}/{len(eps)} episodes", flush=True)

    Z = np.concatenate(lat, 0)
    P = np.concatenate(pos, 0)
    E = np.concatenate(epid, 0)
    print(f"[probe] {Z.shape[0]} frames, dim={Z.shape[1]}")

    # ---- equiv_criterion: same-episode pairs within the success radius -------
    eq_c = []
    tries = 0
    while len(eq_c) < args.pairs_per_pop and tries < args.pairs_per_pop * 200:
        tries += 1
        i = rng.integers(0, Z.shape[0])
        j = rng.integers(0, Z.shape[0])
        if E[i] != E[j] or i == j:
            continue
        if np.linalg.norm(P[i] - P[j]) < args.pos_thresh:
            eq_c.append(rel(Z[i], Z[j]))

    # ---- equiv_temporal: consecutive frames ---------------------------------
    eq_t = []
    for _ in range(args.pairs_per_pop):
        i = rng.integers(0, Z.shape[0] - 1)
        if E[i] != E[i + 1]:
            continue
        eq_t.append(rel(Z[i], Z[i + 1]))

    # ---- cross: different episodes ------------------------------------------
    cr = []
    for _ in range(args.pairs_per_pop):
        i, j = rng.integers(0, Z.shape[0], size=2)
        if E[i] == E[j]:
            continue
        cr.append(rel(Z[i], Z[j]))

    out = {"name": "tworoom-paired-dino384", "h5": args.h5,
           "episodes": int(len(eps)), "frames": int(Z.shape[0]),
           "pos_thresh": args.pos_thresh,
           "equiv_criterion": pct(eq_c), "equiv_temporal": pct(eq_t),
           "cross": pct(cr), "hops": {}}

    print("\nequiv_criterion (within %.0f px)  p10/p50/p90 = %.4f / %.4f / %.4f"
          % (args.pos_thresh, out["equiv_criterion"]["p10"],
             out["equiv_criterion"]["p50"], out["equiv_criterion"]["p90"]))
    print("equiv_temporal  (stride 1)       p10/p50/p90 = %.4f / %.4f / %.4f"
          % (out["equiv_temporal"]["p10"], out["equiv_temporal"]["p50"],
             out["equiv_temporal"]["p90"]))
    print("cross           (random eps)     p10/p50/p90 = %.4f / %.4f / %.4f"
          % (out["cross"]["p10"], out["cross"]["p50"], out["cross"]["p90"]))

    for S in args.hops:
        hp = []
        for _ in range(args.pairs_per_pop):
            i = rng.integers(0, Z.shape[0] - S)
            if E[i] != E[i + S]:
                continue
            hp.append(rel(Z[i], Z[i + S]))
        h = pct(hp)
        lo = out["equiv_criterion"]["p90"]
        hi = h["p10"]
        open_ = hi > lo
        tau = 0.5 * (lo + hi) if open_ else None
        out["hops"][str(S)] = {"hop": h, "gap": [lo, hi], "gap_exists": bool(open_),
                               "tau_derived": tau}
        print("hop S=%-3d p10/p50/p90 = %.4f / %.4f / %.4f | gap [%.4f, %.4f] %s%s"
              % (S, h["p10"], h["p50"], h["p90"], lo, hi,
                 "OPEN" if open_ else "CLOSED",
                 ("  tau=%.4f" % tau) if open_ else ""))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
