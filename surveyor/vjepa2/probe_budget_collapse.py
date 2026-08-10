"""Flagship V-JEPA 2 probe: budget collapse + demo agreement.

Two claims, measured together on HELD-OUT DROID episodes. lat_*.npz needs no
re-encoding: tokens plus states suffice, and the AC predictor does the planning.

  CAPABILITY: their planner is myopic (rollout=2, capped step), so for far
  goals flat CEM's plan should stop agreeing with what the human demonstrator
  actually did, while planning toward a NEAR drafted subgoal should keep
  agreeing. Metric: cosine between the plan's total xyz displacement and the
  demonstrator's true displacement over the same window, plus the magnitude
  ratio. Grounded, with no model-space self-grading.

  SPEED: CEM toward a near subgoal is an easier optimization problem than
  toward a far goal. Sweep the sample budget; if the subgoal arm matches
  full-budget flat quality at a fraction of the budget, the wall-clock claim at
  1B scale is measured rather than argued. Solve seconds recorded per cell.

Arms: flat (goal tokens); spec (first TokenGDM subgoal at k DDIM steps, drafted
ONCE per anchor and reused across budgets, since the draft is
budget-independent); lerp (first straight-line subgoal, the decomposition
control separating "subgoals help" from "the learned drafter helps"). Anchors
are frames where the demo moves, xyz displacement over the plan window >
--move-thresh. Expected shape: parity at t<=S, divergence beyond.

  python -m surveyor.vjepa2.probe_budget_collapse --ckpt gdm_vj2_s10.pt \
      --episodes droid_lat/lat_droid_ep09*.npz --device cuda
"""

from __future__ import annotations

import argparse
import glob
import time
from collections import defaultdict

import numpy as np
import torch

from .drafter import load_token_gdm
from .planner import cstar, flat_plan
from .sources import GDMDraft, LerpBlockDrafter
from .wm import VJEPA2WM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", nargs="+", required=True, help="lat_*.npz (tokens+states)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="vjepa2_ac_vit_giant")
    ap.add_argument("--offsets", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--budgets", type=int, nargs="+", default=[25, 100, 400])
    ap.add_argument("--anchors-per-ep", type=int, default=3)
    ap.add_argument("--max-eps", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--cem-steps", type=int, default=5)
    ap.add_argument("--cem-rollout", type=int, default=2)
    ap.add_argument("--move-thresh", type=float, default=0.01,
                    help="min demo xyz displacement (m) over the plan window")
    ap.add_argument("--router-theta", type=float, default=None,
                    help="enable the ROUTER-COMPOSITE arm: per anchor compute the "
                         "calibrated token certificate tcal = c*_token(goal) - "
                         "c*_token(self); tcal <= theta routes to the flat plan, "
                         "else the spec plan (selection over already-computed "
                         "plans; the certified method at this scale)")
    ap.add_argument("--cstar-samples", type=int, default=100,
                    help="CEM samples for the router's certificate solves")
    ap.add_argument("--out", default="budget_collapse_raw.npz")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import os
    from pathlib import Path

    files = sorted(sum([glob.glob(p) for p in args.episodes], []))[:args.max_eps]
    assert files, f"no files match {args.episodes}"
    wm = VJEPA2WM(model_name=args.model, device=args.device)
    planner = load_token_gdm(args.ckpt, device=args.device)
    gdm = GDMDraft(planner, seed=args.seed)
    lerp = LerpBlockDrafter(n_future=planner.cfg.n_future)
    H = args.cem_rollout

    rows = []  # (ep, t, budget, arm, cos, mag_ratio, secs)
    done_eps = set()
    if os.path.exists(args.out):
        rows = [tuple(r) for r in
                np.load(args.out, allow_pickle=True)["rows"]]
        done_eps = {r[0] for r in rows}
        print(f"[resume] {len(rows)} rows for {len(done_eps)} episodes already in {args.out}")
    for f in files:
        ep_name = Path(f).stem
        if ep_name in done_eps:
            print(f"[skip] {ep_name} already complete")
            continue
        ep = np.load(f)
        tokens = torch.from_numpy(ep["tokens"]).float().to(wm.device)
        states = torch.from_numpy(ep["states"]).float().to(wm.device)
        T = tokens.shape[0]
        for t in args.offsets:
            if t + H >= T:
                continue
            # anchors where the demo moves over the plan window
            cands = []
            for m in range(0, T - t - H):
                d = float((states[m + H] - states[m])[:3].norm())
                if d >= args.move_thresh:
                    cands.append(m)
            if not cands:
                continue
            sel = cands[:: max(1, len(cands) // args.anchors_per_ep)][:args.anchors_per_ep]
            for m in sel:
                z_now = tokens[m:m + 1]
                pose = states[m].view(1, 1, 7)
                goal = tokens[m + t:m + t + 1]
                disp_true = (states[m + H] - states[m])[:3].cpu()
                targets = {"flat": goal,
                           "spec": gdm.draft(z_now, goal, k=args.k)[0][0:1],
                           "lerp": lerp.draft(z_now, goal)[0][0:1]}
                tcal = None
                if args.router_theta is not None:
                    rc = dict(samples=args.cstar_samples,
                              cem_steps=args.cem_steps, rollout=H)
                    c_goal, _ = cstar(wm, z_now, pose, goal, cem_args=rc, metric="token")
                    c_self, _ = cstar(wm, z_now, pose, z_now, cem_args=rc, metric="token")
                    tcal = c_goal - c_self
                for B in args.budgets:
                    cem_args = dict(samples=B, cem_steps=args.cem_steps, rollout=H)
                    cell = {}
                    for arm, tgt in targets.items():
                        t0 = time.perf_counter()
                        plan = flat_plan(wm, z_now, pose, tgt, cem_args=cem_args)
                        secs = time.perf_counter() - t0
                        dp = plan[0, :, :3].sum(0).float().cpu()
                        cos = float(torch.nn.functional.cosine_similarity(
                            dp.view(1, -1), disp_true.view(1, -1)))
                        mag = float(dp.norm() / disp_true.norm().clamp_min(1e-8))
                        rows.append((ep_name, t, B, arm, cos, mag, secs))
                        cell[arm] = (cos, mag, secs)
                    if tcal is not None:
                        pick = "flat" if tcal <= args.router_theta else "spec"
                        rows.append((ep_name, t, B, "route", *cell[pick]))
        np.savez(args.out, rows=np.array(rows, dtype=object), allow_pickle=True)
        print(f"[{ep_name}] done ({len(rows)} rows total, checkpointed)", flush=True)

    agg = defaultdict(list)
    for _ep, t, B, arm, cos, mag, secs in rows:
        agg[(t, B, arm)].append((cos, mag, secs))
    print(f"\n==== BUDGET COLLAPSE x DEMO AGREEMENT ({len(files)} held-out eps, "
          f"H={H}, k={args.k}, S={planner.cfg.n_future}x{ planner.cfg.tokens_per_frame} tokens) ====")
    arms = ["flat", "spec", "lerp"] + (["route"] if args.router_theta is not None else [])
    print(f"{'t':>4} {'budget':>7} | " + " ".join(f"{a + ' cos':>9}" for a in arms)
          + f" | {'flat s':>7} {'spec s':>7}")
    for t in args.offsets:
        for B in args.budgets:
            cell = {}
            for arm in arms:
                v = agg.get((t, B, arm))
                cell[arm] = (np.mean([x[0] for x in v]) if v else float("nan"),
                             np.mean([x[2] for x in v]) if v else float("nan"))
            print(f"{t:>4} {B:>7} | " + " ".join(f"{cell[a][0]:>9.3f}" for a in arms)
                  + f" | {cell['flat'][1]:>7.1f} {cell['spec'][1]:>7.1f}")
    print(f"\n[raw] {args.out}  "
          "(win shape: spec cos >= flat cos everywhere, gap grows with t; "
          "spec at budget 25-100 matching flat at 400 = the wall-clock claim)")


if __name__ == "__main__":
    main()
