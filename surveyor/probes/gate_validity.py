"""Gate validity post-mortem: score routing signals against recorded
per-episode outcomes.

Diagnostic post-hoc analysis, not a pre-registered test. The dumped per-episode
success flags of the fixed arms determine any episode-level gate's closed-loop
result exactly (a fired episode gets flat's recorded outcome, otherwise spec's),
so candidate signals (h, c*, drafter arrival) are evaluated exactly and offline
with zero new rollouts. Every threshold is read from a prior derivation, nothing
is fitted here, and the whole probe runs from artifacts on disk: traces, dense
latents, floor json, drafter checkpoint.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings

import numpy as np
import torch

from surveyor.drafter import load_gdm_planner
from surveyor.probes.horizon_gate import make_latent_lookup
from surveyor.probes.horizon_gate_cem import flat_plan_c_star
from surveyor.probes.probe_floor import dist_stats, rel
from surveyor.sources import SubgoalCostModel
from surveyor import encoder


def parse_args():
    p = argparse.ArgumentParser(description="per-episode gate validity vs recorded outcomes")
    p.add_argument("--traces-glob", default="runs/reacher/traces_*.pt")
    p.add_argument("--h5", default=None)
    p.add_argument("--dense-pt", default="subgoals_reacher_dense.pt")
    p.add_argument("--source", choices=["pretrained", "local"], default="local")
    p.add_argument("--encoder-id", default="quentinll/lewm-reacher")
    p.add_argument("--local-dir", default="encoder_reacher")
    p.add_argument("--swm-src", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--floor-json", default="Results/floor_reacher.json")
    p.add_argument("--gdm-ckpt", default="gdm_reacher_s10.pt")
    p.add_argument("--gdm-steps", type=int, default=8, help="deployed spec draft k")
    # flat-plan config for c* (v2's instrument, unchanged)
    p.add_argument("--horizon", type=int, default=2)
    p.add_argument("--receding-horizon", type=int, default=2)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--var-scale", type=float, default=1.0)
    p.add_argument("--cem-seed", type=int, default=42)
    p.add_argument("--cem-batch", type=int, default=128)
    p.add_argument("--json-out", default=None)
    return p.parse_args()


def load_traces(pattern):
    """Group runs by (population, t): flat / spec (deployed config) / gdm / gated."""
    cells = {}
    for path in sorted(glob.glob(pattern)):
        r = torch.load(path, map_location="cpu", weights_only=False)
        a = r.get("args", {})
        t = a.get("goal_offset")
        if a.get("eval_budget") != 2 * t:      # 2t regime only (b50 = other paper axis)
            continue
        eps = tuple(r["episodes_idx"]); starts = tuple(r["start_steps"])
        key = (hash((eps, starts)), t)
        succ = np.asarray(r["successes"], dtype=float)
        arm = None
        if a.get("subgoal") == "baseline":
            arm = "flat"
        elif a.get("subgoal") == "gdm":
            arm = "gdm"
        elif a.get("subgoal") == "specaccept":
            if a.get("goal_gate"):
                arm = "gated"
            elif a.get("accept_tau") == 0.2 and a.get("gdm_steps") == 8:
                arm = "spec"
        if arm is None:
            continue
        c = cells.setdefault(key, {"eps": np.array(eps), "starts": np.array(starts),
                                   "t": t, "runs": {}})
        c["runs"].setdefault(arm, []).append(
            {"succ": succ, "seed": a.get("seed"), "rh": a.get("receding_horizon"),
             "file": os.path.basename(path),
             "trace": r.get("trace") if arm in ("spec", "gdm") else None})
    return cells


def pop_label(eps, starts, t):
    for f, name in [("reacher_horizon.ep.t0.json", "max100"),
                    ("reacher_horizon150.ep.t0.json", "max150")]:
        if os.path.exists(f):
            pl = json.load(open(f))["episodes"]
            if (len(pl) == len(eps)
                    and all(p[0] == e and p[1] == s
                            for p, e, s in zip(pl, eps, starts))):
                return f"{name}_t{t}"
    return f"native_t{t}"


def auc(score, label):
    """P(score_pos < score_neg): low score should predict label=1 (flat-preferred)."""
    pos, neg = score[label > 0.5], score[label <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] < neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    with open(args.floor_json) as fh:
        floor = json.load(fh)
    tau = float(floor["args"]["tau"])
    theta = float(floor["disp_S10"]["p50"])
    print(f"[thresholds] tau={tau:.2f} (verifier, frozen) theta={theta:.4f} "
          f"(disp_p50(S=10)); both read, nothing fitted in this script")

    z_at, ep_len_of, _ = make_latent_lookup(args)
    lewm = encoder.load_lewm(source=args.source, encoder_id=args.encoder_id,
                             local_dir=args.local_dir, swm_src=args.swm_src,
                             device=args.device)
    cost_model = SubgoalCostModel(lewm)
    planner = load_gdm_planner(args.gdm_ckpt, device=args.device)
    print(f"[drafter] {args.gdm_ckpt} goal_cond={planner.goal_cond} k={args.gdm_steps}")

    cells = load_traces(args.traces_glob)
    print(f"[traces] {len(cells)} (population, t) cells in the 2t regime")

    out = {"tau": tau, "theta": theta, "cells": {}}
    for (_, t), c in sorted(cells.items(), key=lambda kv: (kv[0][1],)):
        runs = c["runs"]
        if "flat" not in runs or "spec" not in runs:
            continue
        eps, starts = c["eps"], c["starts"]
        label = pop_label(eps, starts, t)
        if label.startswith("native"):      # per-seed populations: keep cells distinct
            label += f"_s{runs['flat'][0]['seed']}"
        n = len(eps)

        # per-episode success probability across seeds, per arm
        P = {arm: np.mean([r["succ"] for r in rr], axis=0) for arm, rr in runs.items()}
        sr = {arm: (100 * np.mean([r["succ"].mean() for r in rr]),
                    100 * np.std([r["succ"].mean() for r in rr]),
                    len(rr), runs[arm][0]["rh"]) for arm, rr in runs.items()}

        # ---- signals ----
        z0 = z_at(eps, starts)
        z_goal = z_at(eps, starts + t)
        h = rel(z0, z_goal)
        c_star, _ = flat_plan_c_star(lewm, cost_model, z0, z_goal, args)
        gen = torch.Generator(device=planner.device); gen.manual_seed(42)
        block = planner.sample_sequence(z0.to(planner.device), n_steps=args.gdm_steps,
                                        generator=gen,
                                        z_goal_native=z_goal.to(planner.device))
        zg = z_goal.to(planner.device)
        rel_j = torch.stack([(block[:, j] - zg).norm(dim=-1)
                             / block[:, j].norm(dim=-1).clamp_min(1e-8)
                             for j in range(block.shape[1])])       # (N, n)
        arrive = rel_j.min(dim=0).values.cpu().numpy()

        # ---- deployed first-draft detour stats (mechanism check) ----
        detour = None
        spec_trace = runs["spec"][0]["trace"]
        if spec_trace:
            first = spec_trace[0]
            idx = np.asarray(first["replan_idx"])
            zc, blk = first["z_cond"], first["block"]
            zgoal_rows = z_goal[idx]
            d_now = (zc - zgoal_rows).norm(dim=-1)                  # plain L2: CEM's metric
            d_w1 = (blk[:, 0] - zgoal_rows).norm(dim=-1)
            d_min = torch.stack([(blk[:, j] - zgoal_rows).norm(dim=-1)
                                 for j in range(blk.shape[1])]).min(0).values
            detour = {"n": int(len(idx)),
                      "frac_w1_away": float((d_w1 > d_now).float().mean()),
                      "frac_block_away": float((d_min > d_now).float().mean()),
                      "med_dnow": float(d_now.median()),
                      "med_dw1": float(d_w1.median())}

        # ---- gates (derived thresholds only) + exact gated SR ----
        gates = {"G_h": h <= theta, "G_c": c_star <= tau, "G_arr": arrive <= tau}
        gates["G_c&arr"] = gates["G_c"] & gates["G_arr"]
        gates["G_best"] = P["flat"] > P["spec"]                     # oracle bound
        pref = (P["flat"] > P["spec"]).astype(float)                # flat-preferred label

        mix = {name: float(100 * np.mean(np.where(g, P["flat"], P["spec"])))
               for name, g in gates.items()}
        fire = {name: float(np.mean(g)) for name, g in gates.items()}
        oracle_mix = float(100 * np.mean(np.maximum(P["flat"], P["spec"])))
        aucs = {"h": auc(h, pref), "c*": auc(c_star, pref), "arrive": auc(arrive, pref),
                "c*_vs_flatsucc": auc(c_star, (P["flat"] > 0.5).astype(float))}

        print(f"\n==== {label} (n={n}) ====")
        for arm in ("flat", "spec", "gdm", "gated"):
            if arm in sr:
                m, s, k, rh = sr[arm]
                print(f"  {arm:>5}: SR={m:5.1f} +/- {s:.1f} ({k} seeds, RH={rh})")
        print(f"  oracle-mix (perfect per-episode gate): {oracle_mix:.1f}")
        for name in ("G_h", "G_c", "G_arr", "G_c&arr", "G_best"):
            print(f"  {name:>7}: fire={fire[name]:.3f} gated-SR={mix[name]:5.1f}")
        print(f"  AUC(flat-preferred): h={aucs['h']:.3f} c*={aucs['c*']:.3f} "
              f"arrive={aucs['arrive']:.3f}   AUC(c* vs flat-success)="
              f"{aucs['c*_vs_flatsucc']:.3f}")
        print(f"  signals: h p50={np.median(h):.3f} c* p50={np.median(c_star):.3f} "
              f"arrive p50={np.median(arrive):.3f}")
        if detour:
            print(f"  deployed first drafts: w1 farther-from-goal on "
                  f"{detour['frac_w1_away']*100:.0f}% of envs, whole block farther on "
                  f"{detour['frac_block_away']*100:.0f}% "
                  f"(med L2 now={detour['med_dnow']:.2f} -> w1={detour['med_dw1']:.2f})")

        out["cells"][label] = {
            "n": n, "t": t, "sr": {k: v[:2] for k, v in sr.items()},
            "oracle_mix": oracle_mix, "fire": fire, "gated_sr": mix, "auc": aucs,
            "detour": detour,
            "per_episode": {"episodes": eps.tolist(), "starts": starts.tolist(),
                            "h": h.tolist(), "c_star": c_star.tolist(),
                            "arrive": arrive.tolist(),
                            "P_flat": P["flat"].tolist(), "P_spec": P["spec"].tolist()},
        }

    print("\n[disclaimer] diagnostic post-mortem on already-seen populations; any "
          "surviving gate must be pre-registered and re-run on fresh populations "
          "(new sampling seed) before claims.")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"[out] wrote {args.json_out}")


if __name__ == "__main__":
    main()
