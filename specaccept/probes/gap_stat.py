"""The verification-gap statistic: spec-accept's preflight applicability probe.

The verifier compares the ACHIEVED latent against the TARGET it pursued for one
subgoal interval. Arrival puts that distance at the criterion-equivalence noise
scale (adjacent states, task tolerance); failure to move leaves it at one full
hop. tau must separate the two, so the applicability gap is

    EQUIV p90  <  tau  <  HOP p10

  EQUIV = rel L2 between entries `equiv_stride` apart (adjacent frames:
          the "arrived, up to tolerance" noise scale)
  HOP   = rel L2 between entries `hop_stride` apart (one subgoal interval:
          the "did not move" scale the verifier must reject)
  CROSS = random entry vs episode end (context only)

An empty or inverted interval means no tau exists and verification cannot
operate. It fails in two directions: the EQUIV bulk above any usable tau
(TwoRoom, where even physically-arrived states read ~sqrt(2), corroborated by
voracle advances=14/113), or the HOP bulk below tau, the degenerate-accept case
where everything verifies (DROID at tau=0.20, hop p50 ~ 0.10).

Inputs: either a LeWM-stack subgoals .pt (latents/lengths/offsets) or a glob
of lat_*.npz|npy episode files (V-JEPA 2 tokens; pooled over the token axis).

  python -m specaccept.probes.gap_stat --name pusht --subgoals-pt subgoals_pusht.pt
  python -m specaccept.probes.gap_stat --name droid --lat-glob "droid_lat/lat_*.npz" --hop-stride 10
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import torch


def sequences_from_pt(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    lat, lens, offs = blob["latents"], blob["lengths"], blob["offsets"]
    return [lat[offs[k]:offs[k] + lens[k]].float() for k in range(len(lens))]


def sequences_from_glob(pattern):
    seqs = []
    for f in sorted(glob.glob(pattern)):
        if f.endswith(".npy"):
            tok = np.load(f, mmap_mode="r")
        else:
            tok = np.load(f)["tokens"]
        t = torch.from_numpy(np.asarray(tok, dtype=np.float32))
        seqs.append(t.mean(dim=-2) if t.ndim == 3 else t)   # pool token grids
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--subgoals-pt", default=None)
    ap.add_argument("--lat-glob", default=None)
    ap.add_argument("--equiv-stride", type=int, default=1,
                    help="entries per EQUIV pair (adjacent-frame tolerance scale)")
    ap.add_argument("--hop-stride", type=int, default=10,
                    help="entries per HOP (one subgoal interval, S in file units)")
    ap.add_argument("--tau", type=float, default=0.20)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert (args.subgoals_pt is None) != (args.lat_glob is None), \
        "exactly one of --subgoals-pt / --lat-glob"

    seqs = (sequences_from_pt(args.subgoals_pt) if args.subgoals_pt
            else sequences_from_glob(args.lat_glob))
    g = torch.Generator().manual_seed(args.seed)

    def rel_pairs(z, stride):
        a, b = z[:-stride], z[stride:]
        return ((b - a).norm(dim=1) / b.norm(dim=1).clamp_min(1e-8)).tolist()

    equivs, hops, crosses = [], [], []
    for z in seqs:
        if len(z) < 2 * args.hop_stride + 1:
            continue
        equivs += rel_pairs(z, args.equiv_stride)
        hops += rel_pairs(z, args.hop_stride)
        i = int(torch.randint(0, max(len(z) - 2 * args.hop_stride, 1), (1,), generator=g))
        crosses.append(float((z[i] - z[-1]).norm() / z[i].norm().clamp_min(1e-8)))

    eq_t, hop_t, cross_t = map(torch.tensor, (equivs, hops, crosses))

    def q(t, p):
        return float(torch.quantile(t, p))

    lo, hi = q(eq_t, .9), q(hop_t, .1)
    rec = {
        "name": args.name, "episodes": len(seqs),
        "equiv_stride": args.equiv_stride, "hop_stride": args.hop_stride,
        "equiv": {"n": len(equivs), "p10": q(eq_t, .1), "p50": q(eq_t, .5), "p90": lo},
        "hop": {"n": len(hops), "p10": hi, "p50": q(hop_t, .5), "p90": q(hop_t, .9)},
        "cross": {"p10": q(cross_t, .1), "p50": q(cross_t, .5), "p90": q(cross_t, .9)},
        "gap": [lo, hi], "gap_exists": hi > lo,
        "tau_in_gap": bool(hi > lo and lo < args.tau < hi),
    }
    print(f"[{args.name}] equiv p10/50/90 = {rec['equiv']['p10']:.3f}/{rec['equiv']['p50']:.3f}/"
          f"{lo:.3f} | hop p10/50/90 = {hi:.3f}/{rec['hop']['p50']:.3f}/{rec['hop']['p90']:.3f} | "
          f"cross p50 = {rec['cross']['p50']:.3f} | gap "
          + (f"[{lo:.3f}, {hi:.3f}] tau={args.tau} "
             f"{'IN GAP' if rec['tau_in_gap'] else 'OUTSIDE gap'}"
             if rec["gap_exists"] else "EMPTY/INVERTED -> verification cannot operate"))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rec, f, indent=1)


if __name__ == "__main__":
    main()
