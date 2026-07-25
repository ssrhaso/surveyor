"""Closed-loop-in-the-model trajectory agreement: the proper offline metric.

The 2-step displacement cosine died of anchor noise (flagship cut, 7/22).
This probe runs the full receding-horizon MPC loop INSIDE the AC world model
for K steps per anchor (plan toward the arm's target, execute the plan's H
actions through wm.step with a W-frame context window, replan) and scores
the whole resulting POSE trajectory against the demonstrator's:

  traj_rmse  = RMSE of xyz over the K-step window
  final_err  = |xyz_K - demo xyz_K|
  disp_cos   = cosine(net xyz displacement, demo net displacement)

Pose evolution is action arithmetic (compute_new_pose), so the score is
grounded in what the plan DOES, with ~K/H times the signal of one plan.

Arms: flat (goal tokens), spec (drafter via SpecAcceptTokenSource; note the
pooled verifier is degenerate-accept on DROID per gap-stat, so spec here ==
sequential block consumption; stated, not hidden), lerp (first straight-line
subgoal, recomputed each replan), route (token-cal certificate at the anchor
picks flat or spec for the whole rollout).

  python -m specaccept_vjepa2.probe_rollout_agreement --ckpt gdm_vj2_s10_v3.pt \
      --episodes "droid2k_lat/lat_droid_ep19*.npy" --device cuda
"""

from __future__ import annotations

import argparse
import glob
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .drafter import load_token_gdm
from .planner import cstar, flat_plan
from .sources import GDMDraft, LerpBlockDrafter, SpecAcceptTokenSource
from .train_drafter import _load_tokens
from .wm import VJEPA2WM

W_CTX = 8   # sliding context frames for the in-model simulator


def rollout_block(wm, reps, acts, poses, plan):
    """Execute all H actions of `plan` through the model; returns grown
    (reps, acts, poses), trimmed to the last W_CTX frames."""
    for h in range(plan.shape[1]):
        a = plan[:, h:h + 1].to(reps.device, dtype=reps.dtype)
        acts = a if acts is None else torch.cat([acts, a], dim=1)
        nxt, npse = wm.step(reps, acts, poses)
        reps = torch.cat([reps, nxt], dim=1)
        poses = torch.cat([poses, npse.to(poses.device, dtype=poses.dtype)], dim=1)
    if reps.shape[1] > W_CTX:
        reps, acts, poses = reps[:, -W_CTX:], acts[:, -(W_CTX - 1):], poses[:, -W_CTX:]
    return reps, acts, poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", nargs="+", required=True, help="lat_* files (tokens+states)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="vjepa2_ac_vit_giant")
    ap.add_argument("--rollout-steps", type=int, default=20, help="K executed steps per anchor")
    ap.add_argument("--goal-offset", type=int, default=32)
    ap.add_argument("--budget", type=int, default=50, help="CEM samples per replan")
    ap.add_argument("--cem-steps", type=int, default=5)
    ap.add_argument("--cem-rollout", type=int, default=2, help="H actions per replan")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--router-theta", type=float, default=0.07)
    ap.add_argument("--anchors-per-ep", type=int, default=3)
    ap.add_argument("--max-eps", type=int, default=8)
    ap.add_argument("--move-thresh", type=float, default=0.02,
                    help="min demo xyz displacement over the K window")
    ap.add_argument("--arms", default="flat,spec,lerp,route",
                    help="comma subset of flat,spec,lerp,route (v3 arrays re-run "
                         "spec only; flat/lerp rows are seed-identical to v1's)")
    ap.add_argument("--out", default="rollout_agreement_raw.npz")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())

    files = sorted(sum([glob.glob(p) for p in args.episodes], []))[:args.max_eps]
    assert files, f"no files match {args.episodes}"
    wm = VJEPA2WM(model_name=args.model, device=args.device)
    planner = load_token_gdm(args.ckpt, device=args.device)
    lerp = LerpBlockDrafter(n_future=planner.cfg.n_future)
    K, H = args.rollout_steps, args.cem_rollout
    cem_args = dict(samples=args.budget, cem_steps=args.cem_steps, rollout=H)

    rows, done_eps = [], set()
    import os
    if os.path.exists(args.out):
        rows = [tuple(r) for r in np.load(args.out, allow_pickle=True)["rows"]]
        done_eps = {r[0] for r in rows}
        print(f"[resume] {len(rows)} rows, {len(done_eps)} eps in {args.out}")

    def states_of(f):
        if f.endswith(".npy"):
            return np.load(f.replace(".npy", ".states.npy"))
        return np.load(f)["states"]

    for f in files:
        ep_name = Path(f).stem
        if ep_name in done_eps:
            print(f"[skip] {ep_name}")
            continue
        tokens = torch.from_numpy(np.asarray(_load_tokens(f), dtype=np.float32)).to(wm.device)
        states = torch.from_numpy(states_of(f).astype(np.float32)).to(wm.device)
        T = tokens.shape[0]
        cands = [m for m in range(0, T - max(args.goal_offset, K) - 1)
                 if float((states[m + K] - states[m])[:3].norm()) >= args.move_thresh]
        sel = cands[:: max(1, len(cands) // args.anchors_per_ep)][:args.anchors_per_ep]
        for m in sel:
            goal = tokens[min(m + args.goal_offset, T - 1)].unsqueeze(0)
            demo = states[m:m + K + 1, :3].cpu()
            tcal = None
            if "route" in run_arms and args.router_theta is not None:
                rc = dict(samples=100, cem_steps=args.cem_steps, rollout=H)
                z0, p0 = tokens[m:m + 1], states[m].view(1, 1, 7)
                cg, _ = cstar(wm, z0, p0, goal, cem_args=rc, metric="token")
                cs, _ = cstar(wm, z0, p0, z0, cem_args=rc, metric="token")
                tcal = cg - cs
            for arm in run_arms:
                if arm == "route" and tcal is None:
                    continue
                mode = arm if arm != "route" else ("flat" if tcal <= args.router_theta else "spec")
                source = (SpecAcceptTokenSource(GDMDraft(planner, seed=args.seed),
                                                n_envs=1, device=str(wm.device),
                                                tau=args.tau, k=args.k)
                          if mode == "spec" else None)
                reps = tokens[m:m + 1].unsqueeze(1)          # (1, 1, T_tok, D)
                poses = states[m].view(1, 1, 7)
                acts = None
                t0 = time.perf_counter()
                for _ in range(K // H):
                    z_cur = reps[:, -1]
                    p_cur = poses[:, -1:]
                    if mode == "flat":
                        tgt = goal
                    elif mode == "lerp":
                        tgt = lerp.draft(z_cur, goal)[0][0:1]
                    else:
                        tgt = source.current(z_cur, [0], goal)[0].unsqueeze(0).to(wm.device)
                    plan = flat_plan(wm, z_cur, p_cur, tgt, cem_args=cem_args)
                    reps, acts, poses = rollout_block(wm, reps, acts, poses, plan)
                secs = time.perf_counter() - t0
                sim = poses[0, -(K + 1):, :3].cpu() if poses.shape[1] >= K + 1 else None
                # poses window may be trimmed; rebuild full xyz path is overkill -
                # score net displacement + final pose (always available) + rmse
                # over the retained tail aligned from the end
                n = min(poses.shape[1], K + 1)
                sim_tail = poses[0, -n:, :3].cpu()
                demo_tail = demo[-n:]
                rmse = float(((sim_tail - demo_tail) ** 2).mean().sqrt())
                fin = float((sim_tail[-1] - demo_tail[-1]).norm())
                dp, dd = sim_tail[-1] - sim_tail[0], demo_tail[-1] - demo_tail[0]
                cos = float(torch.nn.functional.cosine_similarity(
                    dp.view(1, -1), dd.view(1, -1)))
                rows.append((ep_name, m, arm, cos, fin, rmse, secs))
        np.savez(args.out, rows=np.array(rows, dtype=object), allow_pickle=True)
        print(f"[{ep_name}] done ({len(rows)} rows, checkpointed)", flush=True)

    agg = defaultdict(list)
    for _e, _m, arm, cos, fin, rmse, secs in rows:
        agg[arm].append((cos, fin, rmse, secs))
    print(f"\n==== ROLLOUT AGREEMENT (K={K}, goal t+{args.goal_offset}, "
          f"budget {args.budget}, {len(files)} eps) ====")
    print(f"{'arm':>6} | {'disp cos':>9} | {'final err (m)':>13} | {'rmse (m)':>9} | {'s/anchor':>8} | n")
    for arm in ("flat", "spec", "lerp", "route"):
        v = agg.get(arm)
        if not v:
            continue
        print(f"{arm:>6} | {np.mean([x[0] for x in v]):>9.3f} | "
              f"{np.mean([x[1] for x in v]):>13.4f} | {np.mean([x[2] for x in v]):>9.4f} | "
              f"{np.mean([x[3] for x in v]):>8.1f} | {len(v)}")
    print(f"[raw] {args.out}")


if __name__ == "__main__":
    main()
