"""TwoRoom evaluation driver.

TwoRoom analog of the PushT driver; reuses the policy, cost model, subgoal
sources, and episode sampling unmodified. Simpler than PushT: the env's own
step() sets terminated when dist(agent, target) < 16 and world.evaluate reads
that flag directly, so no eval_state monkeypatch or angle sweep is needed.
State and goal callables map to TwoRoomEnv's _set_state/_set_goal_state and
the h5's state/goal_state columns.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from specaccept import encoder
from specaccept.envs.pusht.eval import sample_short, sample_long
from specaccept.sources import (SubgoalCostModel, OracleSubgoalSource,
                                    GDMSubgoalSource, SpecAcceptSubgoalSource,
                                    CstarRetireSource, LerpSubgoalSource,
                                    build_oracle_table, make_ffjepa_policy)


def parse_args():
    p = argparse.ArgumentParser(description="FF-JEPA eval driver (TwoRoom)")
    p.add_argument("--h5", required=True)
    p.add_argument("--source", choices=["pretrained", "local"], default="local")
    p.add_argument("--encoder-id", default="quentinll/lewm-tworooms")
    p.add_argument("--local-dir", default="encoder_tworoom")
    p.add_argument("--swm-src", default=None)
    p.add_argument("--device", default="cuda")
    # what to evaluate
    p.add_argument("--subgoal", choices=["oracle", "voracle", "gdm", "baseline",
                                         "specaccept", "unified", "lerp"],
                   default="oracle")
    p.add_argument("--lerp-frac", type=float, default=0.5,
                   help="lerp diagnostic source: fraction along current->goal "
                        "(1.0 = instrumented-flat tautology)")
    p.add_argument("--gdm-ckpt", default=None, help="(Task C) trained planner checkpoint")
    p.add_argument("--gdm-steps", type=int, default=50, help="DDIM sampling steps for GDM")
    p.add_argument("--accept-tau", type=float, default=0.2,
                   help="specaccept: relative-L2 tolerance for the reality verifier")
    p.add_argument("--mode", choices=["short", "long"], default="short")
    p.add_argument("--goal-offset", type=int, default=None,
                   help="override the mode-derived goal_offset (25 short / 75 long)")
    p.add_argument("--start", choices=["final", "random"], default="final",
                   help="final = paper FF-JEPA protocol (last goal_offset steps, goal = "
                        "last frame). long mode is always 'final'.")
    p.add_argument("--eval-filter", choices=["none", "success"], default="none",
                   help="restrict eval episodes to demos whose final frame reaches "
                        "within --pos-thresh of their OWN target (TwoRoom's own "
                        "success rule); the TwoRoom analog of PushT's success20/5.")
    p.add_argument("--pos-thresh", type=float, default=encoder.TWOROOM_POS_THRESH)
    p.add_argument("--require-cross-room", action="store_true",
                   help="restrict eval episodes to ones whose agent and target START in "
                        "different rooms (the intended door-crossing task; ~48%% of "
                        "collected episodes are trivially same-room; see "
                        "build_subgoals_tworoom's matching training-side filter)")
    p.add_argument("--wall-center", type=float, default=112.0)
    p.add_argument("--episode-min", type=int, default=None,
                   help="restrict eval to episode indices >= this (holdout guard "
                        "against training episodes; drafters train on < 4000)")
    p.add_argument("--episode-max", type=int, default=None,
                   help="restrict eval to episode indices < this (exclusive)")
    p.add_argument("--dump-traces", default=None,
                   help="path (.pt): per-episode success flags + per-replan latents")
    p.add_argument("--num-eval", type=int, default=32)
    p.add_argument("--eval-budget", type=int, default=None, help="override mode default")
    p.add_argument("--seed", type=int, default=42)
    # plan / CEM (same validated defaults as the PushT driver)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--receding-horizon", type=int, default=5)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--var-scale", type=float, default=1.0)
    p.add_argument("--cem-seed", type=int, default=None,
                   help="CEM seed; defaults to --seed")
    p.add_argument("--stride", type=int, default=25)
    return p.parse_args()


def img_transform():
    import stable_pretraining as spt
    from torchvision.transforms import v2 as transforms
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=224),
    ])


def img_transform_fallback():
    from torchvision.transforms import v2 as transforms
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=224),
    ])


def build_process(dataset, keys):
    from sklearn import preprocessing
    process = {}
    for col in keys:
        if col == "pixels":
            continue
        proc = preprocessing.StandardScaler()
        data = dataset.get_col_data(col)
        data = data[~np.isnan(data).any(axis=1)]
        proc.fit(data)
        process[col] = proc
        if col != "action":
            process[f"goal_{col}"] = proc
    return process


def tworoom_success_mask(h5, pos_thresh=encoder.TWOROOM_POS_THRESH):
    """Per-episode success flags: final-frame distance_to_target < pos_thresh."""
    import h5py
    with h5py.File(h5, "r") as f:
        ep_off = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
        rows = ep_off + ep_len - 1
        order = np.argsort(rows)
        d = np.empty(len(rows))
        d[order] = f["distance_to_target"][np.sort(rows)]
    return np.array([encoder.tworoom_success(v, pos_thresh) for v in d], dtype=bool)


def cross_room_mask(h5, wall_center=112.0):
    """Per-episode flags: agent and target START in different rooms (the
    intended door-crossing task; mirrors build_subgoals_tworoom's filter)."""
    import h5py
    with h5py.File(h5, "r") as f:
        rows = f["ep_offset"][:]
        order = np.argsort(rows)
        srt = np.sort(rows)
        agent_x = np.empty(len(rows))
        agent_x[order] = f["state"][srt][:, 0]
        target_x = np.empty(len(rows))
        target_x[order] = f["goal_state"][srt][:, 0]
    return (agent_x < wall_center) != (target_x < wall_center)


def main():
    args = parse_args()
    if args.subgoal in ("gdm", "specaccept", "unified") and not args.gdm_ckpt:
        raise ValueError(f"--subgoal {args.subgoal} requires --gdm-ckpt <trained planner checkpoint>")

    encoder._ensure_swm_importable(args.swm_src)
    import stable_worldmodel as swm

    goal_offset = args.goal_offset if args.goal_offset is not None else (25 if args.mode == "short" else 75)
    eval_budget = args.eval_budget or (50 if args.mode == "short" else 150)
    cem_seed = args.cem_seed if args.cem_seed is not None else args.seed

    model = encoder.load_lewm(source=args.source, encoder_id=args.encoder_id,
                              local_dir=args.local_dir, swm_src=args.swm_src,
                              device=args.device)
    is_baseline = args.subgoal == "baseline"
    cost_model = model if is_baseline else SubgoalCostModel(model)
    print(f"[model] frozen LeWM ({sum(p.numel() for p in model.parameters())/1e6:.2f}M), "
          f"device={args.device}, training={model.training}, subgoal={args.subgoal}")

    gdm_planner = None
    if args.subgoal in ("gdm", "specaccept", "unified"):
        from specaccept.drafter import load_gdm_planner, count_params
        gdm_planner = load_gdm_planner(args.gdm_ckpt, device=args.device)
        d = gdm_planner.diffusion
        samp = (f"ddpm_ancestral/all-{d.timesteps}-steps (--gdm-steps ignored)"
                if d.sampler == "ddpm" else f"ddim/{args.gdm_steps}-steps")
        print(f"[gdm] planner from {args.gdm_ckpt}: head={count_params(gdm_planner.model)/1e6:.2f}M, "
              f"N={gdm_planner.cfg.n_future} WG={gdm_planner.cfg.wg} "
              f"goal_cond={gdm_planner.goal_cond} "
              f"T={d.timesteps} sampler={samp} param={d.parameterization} "
              f"schedule={d.schedule} norm={gdm_planner.normalization}")

    from stable_worldmodel.data.formats.hdf5 import HDF5Dataset
    keys = ["action", "proprio", "state"]
    dataset = HDF5Dataset(path=args.h5, keys_to_cache=keys)

    try:
        tf = img_transform()
    except Exception:
        tf = img_transform_fallback()
    transform = {"pixels": tf, "goal": tf}
    process = build_process(dataset, keys)

    use_final = (args.mode == "long") or (args.start == "final")
    ep_mask = None
    if args.eval_filter == "success":
        ep_mask = tworoom_success_mask(args.h5, args.pos_thresh)
        print(f"[eval-filter] success@{args.pos_thresh:g}px: {int(ep_mask.sum())}/{len(ep_mask)} "
              f"episodes eligible (goal frame reaches within threshold of its own target)")
    if args.require_cross_room:
        cr = cross_room_mask(args.h5, args.wall_center)
        ep_mask = cr if ep_mask is None else (ep_mask & cr)
        print(f"[eval-filter] cross-room: {int(ep_mask.sum())} episodes remain eligible")
    if args.episode_min is not None or args.episode_max is not None:
        import h5py
        with h5py.File(args.h5, "r") as f:
            n_eps = len(f["ep_len"])
        lo = args.episode_min or 0
        hi = args.episode_max if args.episode_max is not None else n_eps
        rng_mask = np.zeros(n_eps, dtype=bool)
        rng_mask[lo:hi] = True
        ep_mask = rng_mask if ep_mask is None else (ep_mask & rng_mask)
        print(f"[holdout] eval restricted to episodes [{lo}, {hi}); "
              f"{int(ep_mask.sum())} episodes remain eligible")
    if use_final:
        episodes_idx, start_steps = sample_long(args.h5, args.num_eval, goal_offset, args.seed,
                                                ep_mask=ep_mask)
    else:
        episodes_idx, start_steps = sample_short(args.h5, args.num_eval, goal_offset, args.seed,
                                                 ep_mask=ep_mask)
    print(f"[eval] mode={args.mode} start={'final' if use_final else 'random'} "
          f"num_eval={args.num_eval} goal_offset={goal_offset} budget={eval_budget} "
          f"eval_filter={args.eval_filter}")
    print(f"[eval] episodes_idx[:5]={episodes_idx[:5]} start_steps[:5]={start_steps[:5]}")

    table = None
    if args.subgoal in ("oracle", "voracle"):
        table = build_oracle_table(args.h5, model, episodes_idx, start_steps, goal_offset,
                                   stride=args.stride, device=args.device)
        print(f"[{args.subgoal}] built {len(table)} per-env subgoal tables; "
              f"K (subgoals/ep) = {table[0].shape[0]}")

    config = swm.PlanConfig(horizon=args.horizon, receding_horizon=args.receding_horizon,
                            action_block=args.action_block)
    callables = [
        {"method": "_set_state", "args": {"state": {"value": "state", "in_dataset": True}}},
        {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state", "in_dataset": True}}},
    ]

    PolicyCls = make_ffjepa_policy(swm.policy.WorldModelPolicy)

    world = swm.World(env_name="swm/TwoRoom-v1", num_envs=args.num_eval,
                      max_episode_steps=2 * eval_budget, image_shape=(224, 224))
    solver = swm.solver.CEMSolver(model=cost_model, batch_size=1,
                                  num_samples=args.num_samples, var_scale=args.var_scale,
                                  n_steps=args.n_steps, topk=args.topk,
                                  device=args.device, seed=cem_seed)
    if is_baseline:
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform)
    else:
        if args.subgoal == "gdm":
            source = GDMSubgoalSource(gdm_planner, n_envs=args.num_eval,
                                      dim=gdm_planner.cfg.latent_dim,
                                      device=args.device, n_steps=args.gdm_steps,
                                      seed=cem_seed, record=bool(args.dump_traces))
        elif args.subgoal == "specaccept":
            source = SpecAcceptSubgoalSource(gdm_planner, n_envs=args.num_eval,
                                             device=args.device,
                                             n_steps=args.gdm_steps, seed=cem_seed,
                                             tau=args.accept_tau,
                                             record=bool(args.dump_traces))
        elif args.subgoal == "lerp":
            source = LerpSubgoalSource(n_envs=args.num_eval, device=args.device,
                                       frac=args.lerp_frac)
        elif args.subgoal == "voracle":
            from specaccept.sources import VerifiedOracleSource
            source = VerifiedOracleSource(table, device=args.device,
                                          tau=args.accept_tau)
            print(f"[voracle] achievement-verified oracle: tau={args.accept_tau}, "
                  f"advance only when the achieved latent reaches the waypoint")
        elif args.subgoal == "unified":
            source = CstarRetireSource(gdm_planner, model, n_envs=args.num_eval,
                                       device=args.device, n_steps=args.gdm_steps,
                                       seed=cem_seed, tau=args.accept_tau,
                                       horizon=args.horizon,
                                       action_block=args.action_block,
                                       num_samples=args.num_samples,
                                       cem_steps=args.n_steps, topk=args.topk,
                                       var_scale=args.var_scale, adim=2,
                                       record=bool(args.dump_traces))
            print(f"[unified] c*-retire spec-accept: tau={args.accept_tau}, "
                  f"retire window={args.horizon * args.action_block} steps; "
                  f"drafting only while the goal is out of certified reach")
        else:
            source = OracleSubgoalSource(table, device=args.device)
        policy = PolicyCls(cost_model=cost_model, subgoal_source=source,
                           solver=solver, config=config, process=process, transform=transform)
    world.set_policy(policy)
    metrics = world.evaluate(
        dataset=dataset, start_steps=list(start_steps), goal_offset=goal_offset,
        eval_budget=eval_budget, episodes_idx=list(episodes_idx), callables=callables,
    )
    sr = metrics["success_rate"]
    n_succ = int(metrics["episode_successes"].sum())
    world.close()

    if args.subgoal == "specaccept":
        total = source.n_redraft + source.n_advance
        print(f"[specaccept] tau={args.accept_tau} gdm_steps={args.gdm_steps} "
              f"re-drafts={source.n_redraft} advances={source.n_advance} "
              f"rejects={source.n_reject} "
              f"call_ratio={source.n_redraft / max(total, 1):.3f} "
              f"(every-step=1.000; lower = fewer diffusion calls)")
    if args.subgoal == "voracle":
        pt = source._ptr
        print(f"[voracle] tau={source.tau} advances={source.n_advance} holds={source.n_hold} "
              f"final ptr p10/p50/p90={np.percentile(pt, 10):.0f}/"
              f"{np.percentile(pt, 50):.0f}/{np.percentile(pt, 90):.0f} "
              f"(K={source.table[0].shape[0]})")
    if args.subgoal == "unified":
        seen = source._seen
        retired = int(source._retired.sum())
        fire0 = int((source.c_first[seen] <= source.tau).sum())
        rr = source.retire_replan[source.retire_replan >= 0]
        cf = source.c_first[seen]
        sp = source.spec
        sp_total = sp.n_redraft + sp.n_advance
        print(f"[unified] tau={source.tau} retired={retired}/{int(seen.sum())} "
              f"(fired-at-first-replan={fire0}, == the router fire test) "
              f"retire_replan p10/p50/p90="
              + (f"{np.percentile(rr, 10):.0f}/{np.percentile(rr, 50):.0f}/"
                 f"{np.percentile(rr, 90):.0f} " if len(rr) else "-/-/- ")
              + f"c*_first p50={np.percentile(cf, 50):.3f} | "
              f"spec: re-drafts={sp.n_redraft} advances={sp.n_advance} "
              f"rejects={sp.n_reject} "
              f"call_ratio={sp.n_redraft / max(sp_total, 1):.3f}")

    if args.dump_traces:
        from pathlib import Path
        rec = {"args": vars(args), "sr": float(sr),
               "successes": np.asarray(metrics["episode_successes"]).astype(bool).tolist(),
               "episodes_idx": list(episodes_idx), "start_steps": list(start_steps)}
        if not is_baseline and getattr(source, "record", False):
            rec["trace"] = source.trace
        Path(args.dump_traces).parent.mkdir(parents=True, exist_ok=True)
        torch.save(rec, args.dump_traces)
        print(f"[traces] saved {args.dump_traces}")

    print("\n==== FF-JEPA eval summary (TwoRoom) ====")
    print(f"mode={args.mode} subgoal={args.subgoal} num_eval={args.num_eval} "
          f"seed={args.seed} cem_seed={cem_seed} eval_filter={args.eval_filter}")
    print(f"  SR = {sr:.2f}%  ({n_succ}/{args.num_eval})")


if __name__ == "__main__":
    main()
