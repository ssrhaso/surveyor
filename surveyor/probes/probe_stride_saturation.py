"""Where does an encoder saturate, as a function of subgoal stride?

Motivation. spec-accept serves a subgoal S frames ahead and the planner scores
CEM candidates by terminal L2 to it. Once the encoder's distance to a frame S
ahead reaches the level of two UNRELATED frames, the planner cannot tell that
subgoal from the far goal it was meant to replace, and no drafter can help. That
is a property of the encoder and the stride, measurable offline in seconds, and
distinct from the verification gap, which asks whether ARRIVAL is detectable.

Reports, per candidate stride S:
  hop(S)      distance between frames S apart, within an episode
  saturation  hop(S) p50 / cross p50, where cross is unrelated frames.
              At 1.0 the subgoal is indistinguishable from noise.
  gap         [equiv p90, hop p10]; open means a fixed threshold can certify
              arrival at that stride.

S should be the largest stride that stays inside the informative range, which
turns the method's one tuned knob into a derived one using the same cached
latents the gap statistic already reads.

Works on either a subgoals .pt (LeWM latents) or an npz glob (pooled DINOv2),
so the two verification spaces can be compared on one axis.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="stride-saturation curve")
    p.add_argument("--subgoals", default=None, help="subgoals .pt (stride-1 latents)")
    p.add_argument("--lat-glob", default=None, help="npz glob with a 'tokens' key")
    p.add_argument("--name", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--strides", type=int, nargs="+", default=[1, 2, 3, 5, 8, 10, 15, 25])
    p.add_argument("--pairs", type=int, default=8000)
    p.add_argument("--max-eps", type=int, default=400)
    p.add_argument("--pos-thresh", type=float, default=16.0,
                   help="criterion equivalence radius, used when states are available")
    p.add_argument("--sat-cutoff", type=float, default=0.80,
                   help="hop p50 / cross p50 above which the stride is called saturated")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_episodes(args):
    """-> list of (latents (T, D), states (T, 2) or None)."""
    if args.subgoals:
        import torch
        blob = torch.load(args.subgoals, map_location="cpu", weights_only=False)
        lat = blob["latents"].float().numpy()
        lengths = blob["lengths"].numpy()
        offsets = blob["offsets"].numpy()
        eps = []
        for i in range(min(len(lengths), args.max_eps)):
            o, L = int(offsets[i]), int(lengths[i])
            eps.append((lat[o:o + L], None))
        print(f"[data] {args.subgoals}: {len(eps)} episodes, dim={lat.shape[1]}, "
              f"file stride={blob.get('stride')}")
        return eps
    files = sorted(glob.glob(args.lat_glob))[:args.max_eps]
    if not files:
        raise SystemExit(f"no latents at {args.lat_glob}")
    eps = []
    for f in files:
        d = np.load(f)
        eps.append((d["tokens"].astype(np.float32),
                    d["state"].astype(np.float32) if "state" in d else None))
    print(f"[data] {len(eps)} episodes from {args.lat_glob}, dim={eps[0][0].shape[1]}")
    return eps


def pct(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"n": 0, "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {"n": int(x.size), "p10": float(np.percentile(x, 10)),
            "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90))}


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-8))


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    eps = load_episodes(args)
    has_state = eps[0][1] is not None

    # ---- cross: unrelated frames, the saturation reference --------------
    cross = []
    for _ in range(args.pairs):
        i, j = rng.integers(0, len(eps), size=2)
        if i == j:
            continue
        a = eps[i][0][rng.integers(0, len(eps[i][0]))]
        b = eps[j][0][rng.integers(0, len(eps[j][0]))]
        cross.append(rel(a, b))
    cross = pct(cross)

    # ---- equivalence: criterion when states exist, else temporal --------
    eq = []
    if has_state:
        tries = 0
        while len(eq) < args.pairs and tries < args.pairs * 200:
            tries += 1
            e = int(rng.integers(0, len(eps)))
            Z, P = eps[e]
            i, j = rng.integers(0, len(Z), size=2)
            if i == j:
                continue
            if np.linalg.norm(P[i] - P[j]) < args.pos_thresh:
                eq.append(rel(Z[i], Z[j]))
        eq_kind = f"criterion (within {args.pos_thresh:g} px)"
    else:
        for _ in range(args.pairs):
            e = int(rng.integers(0, len(eps)))
            Z = eps[e][0]
            if len(Z) < 2:
                continue
            i = int(rng.integers(0, len(Z) - 1))
            eq.append(rel(Z[i], Z[i + 1]))
        eq_kind = "temporal (stride 1)"
    eq = pct(eq)

    print(f"\n[{args.name}]  equivalence = {eq_kind}")
    print("  equiv  p10/p50/p90 = %.4f / %.4f / %.4f" % (eq["p10"], eq["p50"], eq["p90"]))
    print("  cross  p10/p50/p90 = %.4f / %.4f / %.4f  <- saturation reference"
          % (cross["p10"], cross["p50"], cross["p90"]))

    out = {"name": args.name, "equiv_kind": eq_kind, "equiv": eq, "cross": cross,
           "sat_cutoff": args.sat_cutoff, "strides": {}}

    print("\n   S   hop p10   hop p50   hop p90   sat=hop50/cross50   gap[eq90,hop10]")
    print("  ---------------------------------------------------------------------")
    usable = []
    for S in args.strides:
        hp = []
        for _ in range(args.pairs):
            e = int(rng.integers(0, len(eps)))
            Z = eps[e][0]
            if len(Z) <= S:
                continue
            i = int(rng.integers(0, len(Z) - S))
            hp.append(rel(Z[i], Z[i + S]))
        h = pct(hp)
        if h["n"] == 0:
            continue
        sat = h["p50"] / max(cross["p50"], 1e-8)
        gap_open = h["p10"] > eq["p90"]
        out["strides"][str(S)] = {"hop": h, "saturation": sat,
                                  "gap": [eq["p90"], h["p10"]],
                                  "gap_exists": bool(gap_open)}
        flag = ""
        if sat > args.sat_cutoff:
            flag = "  SATURATED"
        if gap_open:
            flag += "  gap OPEN"
        print("  %3d   %.4f    %.4f    %.4f      %.3f            [%.3f, %.3f]%s"
              % (S, h["p10"], h["p50"], h["p90"], sat, eq["p90"], h["p10"], flag))
        if sat <= args.sat_cutoff:
            usable.append(S)

    print()
    if usable:
        print("  => largest stride inside the informative range (sat <= %.2f): S = %d"
              % (args.sat_cutoff, max(usable)))
        out["S_recommended"] = int(max(usable))
    else:
        print("  => EVERY tested stride is saturated; this encoder cannot support "
              "subgoal serving at any of them")
        out["S_recommended"] = None

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
