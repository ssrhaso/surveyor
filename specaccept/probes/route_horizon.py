"""Ternary router (gate v4): offline routing pass for flat2 / flat5 / spec.

Per episode, from its true start state: route to flat2 if c*_2 <= tau, else
flat5 if c*_5 <= tau, else spec, where c* is the planner-predicted terminal rel
of one flat CEM plan at the corresponding window depth. Writes one
eval-compatible episodes file per branch, and the companion sbatch runs each
episode once under its routed arm, so the composite stays a single policy with
the planning-depth degree of freedom restored. tau is read from the floor
probe's recorded value, never swept here.
"""

from __future__ import annotations

import argparse
import copy
import json

import numpy as np

from specaccept.probes.horizon_gate import make_latent_lookup
from specaccept.probes.horizon_gate_cem import flat_plan_c_star
from specaccept.probes.probe_floor import dist_stats
from specaccept import encoder
from specaccept.sources import SubgoalCostModel


def parse_args():
    p = argparse.ArgumentParser(description="Gate v4 ternary router: per-episode "
                                            "c*_2/c*_5 -> flat2/flat5/spec subsets")
    p.add_argument("--h5", default=None, help="raw pixel dataset (encoded here)")
    p.add_argument("--dense-pt", default="subgoals_reacher_dense.pt",
                   help="stride-1 latent dump (pixel-free path, validated exact)")
    p.add_argument("--source", choices=["pretrained", "local"], default="local")
    p.add_argument("--encoder-id", default="quentinll/lewm-reacher")
    p.add_argument("--local-dir", default="encoder_reacher")
    p.add_argument("--swm-src", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256, help="encode batch (h5 path)")
    p.add_argument("--episodes-file", required=True,
                   help="population JSON from build_reacher_horizon_episodes.py")
    p.add_argument("--offsets", type=int, nargs="+", required=True)
    p.add_argument("--floor-json", default="Results/floor_reacher.json",
                   help="probe_floor output; tau is READ from its recorded args")
    # CEM = the eval driver's defaults, used unmodified by every reacher sbatch
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--var-scale", type=float, default=1.0)
    p.add_argument("--cem-seed", type=int, default=42,
                   help="routing CEM seed: ONE fixed decision per episode, "
                        "shared by every eval seed downstream")
    p.add_argument("--cem-batch", type=int, default=128)
    p.add_argument("--rh-short", type=int, default=2)
    p.add_argument("--rh-deep", type=int, default=5)
    p.add_argument("--adim", type=int, default=2,
                   help="env action dim for the probe CEM plans "
                        "(reacher/pusht 2, cube 5)")
    p.add_argument("--out-prefix", required=True,
                   help="subset files: {prefix}_t{t}_{flat2|flat5|spec}.json")
    p.add_argument("--json-out", default=None, help="routing summary JSON")
    return p.parse_args()


BRANCHES = ("flat2", "flat5", "spec")


def main():
    args = parse_args()
    if args.h5:
        args.dense_pt = None

    with open(args.floor_json) as fh:
        floor = json.load(fh)
    tau = float(floor["args"]["tau"])
    print(f"[tau] {tau:.2f} read from {args.floor_json} (frozen verifier "
          f"threshold; third use, not swept)")
    print(f"[rule] c*_{args.rh_short}<=tau -> flat{args.rh_short}; "
          f"elif c*_{args.rh_deep}<=tau -> flat{args.rh_deep}; else spec. "
          f"One decision per (episode, t), routing cem-seed={args.cem_seed}.")

    z_at, ep_len_of, h5_handle = make_latent_lookup(args)
    lewm = encoder.load_lewm(source=args.source, encoder_id=args.encoder_id,
                             local_dir=args.local_dir, swm_src=args.swm_src,
                             device=args.device)
    cost_model = SubgoalCostModel(lewm)
    print(f"[model] frozen LeWM predictor loaded "
          f"({sum(p.numel() for p in lewm.parameters())/1e6:.2f}M), "
          f"device={args.device}")

    with open(args.episodes_file) as fh:
        payload = json.load(fh)
    pairs = payload["episodes"]
    eps = np.array([p[0] for p in pairs], dtype=np.int64)
    starts = np.array([p[1] for p in pairs], dtype=np.int64)
    print(f"[pop] {args.episodes_file}: {len(pairs)} pairs "
          f"(max_offset={payload.get('max_offset')}, seed={payload.get('seed')})")

    z0_all = z_at(eps, starts)
    L = ep_len_of(eps)

    def probe_args(rh):
        a = copy.copy(args)
        a.horizon = rh
        a.receding_horizon = rh
        return a

    out = {"args": {k: v for k, v in vars(args).items()},
           "tau": tau, "rule": "c2<=tau -> flat2; elif c5<=tau -> flat5; else spec",
           "population": args.episodes_file, "cells": {}}

    for t in args.offsets:
        goal_steps = starts + t
        valid = goal_steps <= L - 1
        if (~valid).sum():
            print(f"[t={t}] WARNING {int((~valid).sum())} episodes dropped "
                  f"(goal past episode end); population-shifted cell")
        vidx = np.flatnonzero(valid)
        z_goal = z_at(eps[vidx], goal_steps[vidx])
        c2, _ = flat_plan_c_star(lewm, cost_model, z0_all[vidx], z_goal,
                                 probe_args(args.rh_short))
        c5, _ = flat_plan_c_star(lewm, cost_model, z0_all[vidx], z_goal,
                                 probe_args(args.rh_deep))
        branch = np.where(c2 <= tau, 0, np.where(c5 <= tau, 1, 2))

        cell = {"n": int(len(vidx)), "n_dropped": int((~valid).sum()),
                "tau": tau,
                "c2": dist_stats(c2), "c5": dist_stats(c5),
                "counts": {BRANCHES[b]: int((branch == b).sum()) for b in range(3)},
                "per_episode": [[int(eps[i]), int(starts[i]),
                                 float(c2[j]), float(c5[j]), BRANCHES[branch[j]]]
                                for j, i in enumerate(vidx)]}
        out["cells"][str(t)] = cell
        print(f"[t={t:>3d}] route counts: " +
              " ".join(f"{BRANCHES[b]}={cell['counts'][BRANCHES[b]]}"
                       for b in range(3)) +
              f"   c2 p50={cell['c2']['p50']:.3f} c5 p50={cell['c5']['p50']:.3f}")

        for b, name in enumerate(BRANCHES):
            sel = np.flatnonzero(branch == b)
            sub_pairs = [[int(eps[vidx[j]]), int(starts[vidx[j]])] for j in sel]
            sub = {"max_offset": payload.get("max_offset"),
                   "episode_min": payload.get("episode_min"),
                   "seed": payload.get("seed"),
                   "goal_offset": t, "branch": name, "tau": tau,
                   "routing_cem_seed": args.cem_seed,
                   "parent": args.episodes_file,
                   "note": f"gate v4 ternary route, t={t}, branch={name} "
                           f"({len(sub_pairs)}/{len(vidx)} episodes)",
                   "episodes": sub_pairs}
            fn = f"{args.out_prefix}_t{t}_{name}.json"
            with open(fn, "w") as fh:
                json.dump(sub, fh)
            print(f"        wrote {fn}: {len(sub_pairs)} episodes")

    if h5_handle is not None:
        h5_handle.close()
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"[out] wrote {args.json_out}")


if __name__ == "__main__":
    main()
