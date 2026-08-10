"""Criterion floor of the TwoRoom lens space.

learn_readout only writes tau_derived when the gap OPENS; on TwoRoom it stayed
closed (-0.080 -> -0.027), so the lens checkpoint carries tau_derived=None and
the lens arm has no threshold. Rather than invent one, this derives it the same
way the raw-space tau was derived: tau = criterion-floor p50, measured in the
lens's own output space over the same cached latents (agent positions are
stored alongside them, so criterion equivalence is exact).

Reads the dumped npz latents, so it costs seconds and needs no GPU.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from surveyor import encoder


def parse_args():
    p = argparse.ArgumentParser(description="TwoRoom lens-space criterion floor")
    p.add_argument("--lat-glob", required=True)
    p.add_argument("--readout-ckpt", required=True)
    p.add_argument("--out", default="Results/gap_stat/lens_floor_tworoom.json")
    p.add_argument("--hop", type=int, default=10)
    p.add_argument("--pos-thresh", type=float, default=encoder.TWOROOM_POS_THRESH)
    p.add_argument("--pairs", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def pct(x):
    x = np.asarray(x, dtype=np.float64)
    return {"n": int(x.size), "p10": float(np.percentile(x, 10)),
            "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90))}


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    from surveyor.probes.learn_readout import Readout
    ck = torch.load(args.readout_ckpt, map_location="cpu", weights_only=False)
    ro = Readout(ck["dim"], ck["out_dim"], attentive=ck["attentive"])
    ro.load_state_dict(ck["state"])
    ro.eval()

    files = sorted(glob.glob(args.lat_glob))
    if not files:
        raise SystemExit(f"no latents at {args.lat_glob}")
    Z, P, E = [], [], []
    for k, f in enumerate(files):
        d = np.load(f)
        t = torch.from_numpy(d["tokens"]).float()
        with torch.no_grad():
            Z.append(ro(t).numpy())          # lens space, unit-norm output
        P.append(d["state"].astype(np.float32))
        E.append(np.full(len(t), k, dtype=np.int64))
    Z = np.concatenate(Z, 0)
    P = np.concatenate(P, 0)
    E = np.concatenate(E, 0)
    print(f"[lensfloor] {len(files)} episodes, {Z.shape[0]} frames, lens dim {Z.shape[1]}")

    # lens distances are absolute L2 in the readout space (learn_readout's gap_of)
    def d(i, j):
        return float(np.linalg.norm(Z[i] - Z[j]))

    eq, tries = [], 0
    while len(eq) < args.pairs and tries < args.pairs * 200:
        tries += 1
        i, j = rng.integers(0, Z.shape[0], size=2)
        if E[i] != E[j] or i == j:
            continue
        if np.linalg.norm(P[i] - P[j]) < args.pos_thresh:
            eq.append(d(i, j))

    hop = []
    for _ in range(args.pairs):
        i = rng.integers(0, Z.shape[0] - args.hop)
        if E[i] != E[i + args.hop]:
            continue
        hop.append(d(i, i + args.hop))

    cross = []
    for _ in range(args.pairs):
        i, j = rng.integers(0, Z.shape[0], size=2)
        if E[i] == E[j]:
            continue
        cross.append(d(i, j))

    out = {"readout": args.readout_ckpt, "hop": args.hop,
           "pos_thresh": args.pos_thresh,
           "equiv_criterion": pct(eq), "hop_dist": pct(hop), "cross": pct(cross)}
    out["gap"] = [out["equiv_criterion"]["p90"], out["hop_dist"]["p10"]]
    out["gap_exists"] = bool(out["gap"][1] > out["gap"][0])
    out["tau_criterion_floor_p50"] = out["equiv_criterion"]["p50"]

    print("equiv_criterion p10/p50/p90 = %.4f / %.4f / %.4f"
          % (out["equiv_criterion"]["p10"], out["equiv_criterion"]["p50"],
             out["equiv_criterion"]["p90"]))
    print("hop(%d)          p10/p50/p90 = %.4f / %.4f / %.4f"
          % (args.hop, out["hop_dist"]["p10"], out["hop_dist"]["p50"],
             out["hop_dist"]["p90"]))
    print("cross           p10/p50/p90 = %.4f / %.4f / %.4f"
          % (out["cross"]["p10"], out["cross"]["p50"], out["cross"]["p90"]))
    print("gap [%.4f, %.4f] %s" % (out["gap"][0], out["gap"][1],
                                   "OPEN" if out["gap_exists"] else "CLOSED"))
    print("=> tau (criterion floor p50, lens space) = %.4f"
          % out["tau_criterion_floor_p50"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
