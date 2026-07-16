"""Horizon gate v2 (Phase 1, offline): planner-predicted reachability.

v1 (probes/horizon_gate.py) failed its pre-registered condition: rel(z0,
z_goal) saturates at ~sqrt(2) (decorrelated latents) by t=50, so encoder
distance carries no horizon information above one hop. That result stands;
this probe changes the INSTRUMENT, not the hypothesis. "Inside the lookahead"
is a property of the planner and its learned dynamics, not of latent
distance -- so ask the planner: run ONE flat CEM plan from z0 toward z_goal
and read its predicted terminal discrepancy

    c* = rel(z_hat_H, z_goal) = ||z_hat_H - z_goal|| / ||z_hat_H||

where z_hat_H is the predicted terminal latent of the plan CEM would hand to
the env (the final elite mean -- exactly what WorldModelPolicy executes at
replan 0). Predicted, not achieved: no environment stepping anywhere here.

Gate rule: plan flat iff c* <= tau, with tau the verifier's own frozen
accept threshold (0.20, criterion-floor-derived; every closed-loop sbatch
pins --accept-tau 0.20). tau is READ from the floor probe's recorded args --
same number, same unit, second use; no new constant, no sweep.

Flat branch config: RH=2, flat's strongest short-range configuration per the
existing sweep (batch/isca/run_reacher_strongbase.sbatch: "RH=2 flat hit
95.7 at t25 but 67.6 at t100"). CEM params are the eval driver's defaults
(300 samples / 30 steps / topk 30 / var_scale 1.0, seed 42). The plan is the
same call the policy makes: swm CEMSolver + SubgoalCostModel with
info_dict['subgoal_emb'] = z_goal (the degenerate flat case, terminal-L2^2
math identical to the baseline arm's LeWM.criterion). Nothing reimplemented:
CEMSolver is the shipped solver, the terminal rollout is
probe_cost_snr.rollout_terminal, latents come via horizon_gate's validated
make_latent_lookup (pixel-free dense path).

Pre-registered condition (module constants, cells with zero dropped
episodes): fire-rate >= 0.70 at t=25 and <= 0.15 at t=150. KNOWN RISK,
stated not corrected: c* is the planner's own optimistic estimate under its
learned dynamics and may under-report difficulty uniformly. If fire is high
at EVERY t including t=150, that is model exploitation, not a horizon
signal, and the gate fails for that reason -- the verdict flags it
explicitly rather than passing on the t=25 check alone.

Fails => the gate is dead a second time, tau/theta are not amended, and the
decomposition tax stands as the finding. No Phase 2 without approval.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from specaccept import encoder
from specaccept.probes.horizon_gate import make_latent_lookup
from specaccept.probes.probe_cost_snr import rollout_terminal
from specaccept.probes.probe_floor import dist_stats, rel
from specaccept.sources import SubgoalCostModel

# pre-registered separation condition -- constants, not flags (v1 convention)
T_SHORT, FIRE_MIN_SHORT = 25, 0.70
T_LONG, FIRE_MAX_LONG = 150, 0.15


def parse_args():
    p = argparse.ArgumentParser(description="Horizon gate v2, Phase 1: planner-"
                                            "predicted reachability c* per (pop, t)")
    p.add_argument("--h5", default=None, help="raw pixel dataset (encoded here)")
    p.add_argument("--dense-pt", default="subgoals_reacher_dense.pt",
                   help="stride-1 latent dump from build_subgoals (pixel-free path)")
    p.add_argument("--source", choices=["pretrained", "local"], default="local")
    p.add_argument("--encoder-id", default="quentinll/lewm-reacher")
    p.add_argument("--local-dir", default="encoder_reacher")
    p.add_argument("--swm-src", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256, help="encode batch (h5 path)")
    p.add_argument("--episodes-files", nargs="+", required=True)
    p.add_argument("--offsets", type=int, nargs="+",
                   default=[25, 50, 75, 100, 125, 150])
    p.add_argument("--floor-json", default="Results/floor_reacher.json",
                   help="probe_floor output; tau is READ from its recorded args "
                        "(the frozen verifier threshold), never set here")
    # flat branch: RH=2 = flat's strongest short-range config (strongbase sweep);
    # CEM params = the eval driver's defaults, used unmodified by every reacher sbatch
    p.add_argument("--horizon", type=int, default=2)
    p.add_argument("--receding-horizon", type=int, default=2)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--var-scale", type=float, default=1.0)
    p.add_argument("--cem-seed", type=int, default=42)
    p.add_argument("--cem-batch", type=int, default=128,
                   help="envs per CEM batch (algorithm-identical to the closed "
                        "loop's batch_size=1; only the RNG partitioning differs)")
    p.add_argument("--json-out", default=None)
    return p.parse_args()


def flat_plan_c_star(lewm, cost_model, z0, z_goal, args):
    """One flat CEM plan per episode toward z_goal; returns (c*, median L2^2).

    The exact replan-0 call: shipped CEMSolver, SubgoalCostModel cost
    (terminal L2^2 to subgoal_emb), pre-injected emb so LeWM.rollout skips
    the pixel encode (probe_cost_snr's validated path). z_hat_H is the
    terminal latent of the returned final elite mean.
    """
    import stable_worldmodel as swm
    from gymnasium.spaces import Box

    n = z0.shape[0]
    config = swm.PlanConfig(horizon=args.horizon,
                            receding_horizon=args.receding_horizon,
                            action_block=args.action_block)
    solver = swm.solver.CEMSolver(model=cost_model, batch_size=args.cem_batch,
                                  num_samples=args.num_samples,
                                  var_scale=args.var_scale, n_steps=args.n_steps,
                                  topk=args.topk, device=args.device,
                                  seed=args.cem_seed)
    solver.configure(action_space=Box(low=-1.0, high=1.0,
                                      shape=(n, getattr(args, "adim", 2)),
                                      dtype=np.float32),
                     n_envs=n, config=config)
    z0d = z0.to(args.device)
    info = {
        # only .size(2)==H(=1) is read once emb is pre-injected
        "pixels": torch.zeros(n, 1, 1, 1, 1, device=args.device),
        "emb": z0d[:, None, :],                       # (n, H=1, D)
        SubgoalCostModel.SUBGOAL_KEY: z_goal.to(args.device),
    }
    out = solver.solve(info)
    plan = out["actions"].to(args.device)             # (n, horizon, block*adim)
    z_hat = rollout_terminal(lewm, z0d, plan.unsqueeze(1), args.device)[:, 0]
    c_star = rel(z_hat, z_goal.to(args.device))       # ||z_hat - z_goal||/||z_hat||
    return c_star, float(np.median(out["costs"]))


def main():
    args = parse_args()
    if args.h5:
        args.dense_pt = None

    # ---- tau: the verifier's frozen threshold, read, not set ----
    with open(args.floor_json) as fh:
        floor = json.load(fh)
    tau = float(floor["args"]["tau"])
    print(f"[tau] {tau:.2f} = the frozen accept-test threshold recorded by "
          f"{args.floor_json} (closed-loop sbatches pin --accept-tau {tau:.2f}); "
          f"criterion-floor-derived, second use, not swept here")
    print(f"[pre-registered] fire-rate >= {FIRE_MIN_SHORT} at t={T_SHORT} and "
          f"<= {FIRE_MAX_LONG} at t={T_LONG}; high fire at EVERY t = model "
          f"exploitation and the gate fails for that reason")
    print(f"[flat-branch] RH={args.receding_horizon} (strongest short-range flat, "
          f"strongbase sweep), plan window = "
          f"{args.horizon * args.action_block} env steps, CEM "
          f"{args.num_samples}x{args.n_steps} topk={args.topk} "
          f"var={args.var_scale} seed={args.cem_seed}")

    z_at, ep_len_of, h5_handle = make_latent_lookup(args)
    lewm = encoder.load_lewm(source=args.source, encoder_id=args.encoder_id,
                             local_dir=args.local_dir, swm_src=args.swm_src,
                             device=args.device)
    cost_model = SubgoalCostModel(lewm)
    print(f"[model] frozen LeWM predictor loaded "
          f"({sum(p.numel() for p in lewm.parameters())/1e6:.2f}M), "
          f"device={args.device}")

    out = {"args": vars(args), "tau": tau,
           "tau_source": f"{args.floor_json}:args.tau (frozen verifier threshold)",
           "pre_registered": {"t_short": T_SHORT, "fire_min_short": FIRE_MIN_SHORT,
                              "t_long": T_LONG, "fire_max_long": FIRE_MAX_LONG},
           "populations": {}}
    for ep_file in args.episodes_files:
        with open(ep_file) as fh:
            payload = json.load(fh)
        pairs = payload["episodes"]
        label = f"max{payload['max_offset']}"
        eps = np.array([p[0] for p in pairs], dtype=np.int64)
        starts = np.array([p[1] for p in pairs], dtype=np.int64)
        print(f"\n[pop {label}] {ep_file}: {len(pairs)} pairs "
              f"(max_offset={payload['max_offset']}, seed={payload.get('seed')}, "
              f"episode_min={payload.get('episode_min')})")

        z0_all = z_at(eps, starts)
        L = ep_len_of(eps)

        cells = {}
        for t in args.offsets:
            goal_steps = starts + t
            valid = goal_steps <= L - 1
            n_drop = int((~valid).sum())
            if valid.sum() == 0:
                print(f"[pop {label}] t={t}: no valid episodes, skipped")
                continue
            vidx = np.flatnonzero(valid)
            z_goal = z_at(eps[vidx], goal_steps[vidx])
            c_star, med_l2sq = flat_plan_c_star(lewm, cost_model,
                                                z0_all[vidx], z_goal, args)
            cell = dist_stats(c_star)
            cell["fire_rate"] = float((c_star <= tau).mean())
            cell["n_dropped"] = n_drop
            cell["median_elite_l2sq"] = med_l2sq
            cells[str(t)] = cell
            drop_note = (f"  [{n_drop} dropped: goal past episode end -- "
                         f"population-shifted cell]" if n_drop else "")
            print(f"[pop {label}] t={t:>3d}: c* p10={cell['p10']:.3f} "
                  f"p50={cell['p50']:.3f} p90={cell['p90']:.3f} "
                  f"fire(c*<={tau:.2f})={cell['fire_rate']:.3f} "
                  f"n={cell['n']}{drop_note}")
        out["populations"][label] = {"file": ep_file, "cells": cells}
    if h5_handle is not None:
        h5_handle.close()

    # ---- verdict against the pre-registered condition ----
    print("\n==== verdict ====")
    checks, evaluated_long = [], False
    short_fires, long_fires, all_fires = [], [], []
    for label, pop in out["populations"].items():
        cells = pop["cells"]
        all_fires += [c["fire_rate"] for c in cells.values() if c["n_dropped"] == 0]
        short = cells.get(str(T_SHORT))
        if short is not None and short["n_dropped"] == 0:
            ok = short["fire_rate"] >= FIRE_MIN_SHORT
            checks.append(ok)
            short_fires.append(short["fire_rate"])
            print(f"{label} t={T_SHORT}: fire={short['fire_rate']:.3f} "
                  f"(need >= {FIRE_MIN_SHORT}) -> {'PASS' if ok else 'FAIL'}")
        long_cell = cells.get(str(T_LONG))
        if long_cell is not None:
            if long_cell["n_dropped"] == 0:
                ok = long_cell["fire_rate"] <= FIRE_MAX_LONG
                checks.append(ok)
                evaluated_long = True
                long_fires.append(long_cell["fire_rate"])
                print(f"{label} t={T_LONG}: fire={long_cell['fire_rate']:.3f} "
                      f"(need <= {FIRE_MAX_LONG}) -> {'PASS' if ok else 'FAIL'}")
            else:
                print(f"{label} t={T_LONG}: fire={long_cell['fire_rate']:.3f} but "
                      f"{long_cell['n_dropped']} episodes dropped -- population-"
                      f"shifted, excluded from the verdict")
        med = [(int(t), c["p50"])
               for t, c in sorted(cells.items(), key=lambda kv: int(kv[0]))]
        mono = all(a[1] <= b[1] for a, b in zip(med, med[1:]))
        print(f"{label} c* p50 across t: "
              + " ".join(f"{t}:{m:.3f}" for t, m in med)
              + f"  (monotone increasing: {mono})")

    passed = bool(checks) and all(checks) and evaluated_long
    # pre-registered exploitation kill: short passing on planner optimism that
    # also floods the long cells is not a horizon signal
    exploitation = (bool(short_fires) and bool(long_fires)
                    and min(short_fires) >= FIRE_MIN_SHORT
                    and min(long_fires) > FIRE_MAX_LONG)
    if exploitation:
        print("\nMODEL EXPLOITATION FLAG: fire-rate is high at t=150 as well as "
              "t=25 -- c* is the planner's own optimistic estimate under its "
              "learned dynamics, and a uniformly low c* means CEM believes it can "
              "reach ANY goal in one plan. That is optimism, not a horizon "
              "signal; the gate fails for this reason regardless of the t=25 "
              "check.")
        passed = False
    out["verdict"] = {"passed": passed, "n_checks": len(checks),
                      "model_exploitation": exploitation}
    verdict_msg = ("SEPARATES -- Phase 1 PASS" if passed else
                   "does NOT meet the pre-registered condition -- Phase 1 FAIL, "
                   "gate v2 is dead as specified")
    print(f"\ngate {verdict_msg}")
    print("(tau is the verifier's frozen threshold; not amended either way)")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"[out] wrote {args.json_out}")


if __name__ == "__main__":
    main()
