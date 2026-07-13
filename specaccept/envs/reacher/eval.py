"""Reacher (DMControl qpos_match) evaluation driver.

Reacher analog of the PushT driver; reuses the policy, cost model, subgoal
sources, and episode sampling unmodified. Differences:

* Success is the env's own termination rule (all |qpos - target_qpos| <
  qpos_threshold); no eval_state monkeypatch or angle sweep.
* State/goal callables mirror stable_worldmodel's reacher eval config:
  set_state from the start frame, set_target_qpos from the goal frame.
* The dataset is random-policy data, so sampling follows LeWM's own protocol
  (random valid start, goal = start + goal_offset; defaults 25/50).
* The GDM/spec-accept arms expect a goal-conditioned drafter (train_drafter
  --goal-cond --goal-rule window); the policy encodes the goal image at each
  replan and the sources pass it through to the sampler.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from specaccept import encoder
from specaccept.envs.pusht.eval import sample_short, sample_long
from specaccept.sources import (SubgoalCostModel, OracleSubgoalSource,
                                    GDMSubgoalSource, SpecAcceptSubgoalSource,
                                    LerpSubgoalSource,
                                    build_oracle_table, make_ffjepa_policy)


def parse_args():
    p = argparse.ArgumentParser(description="FF-JEPA eval driver (Reacher qpos_match)")
    p.add_argument("--h5", required=True)
    p.add_argument("--source", choices=["pretrained", "local"], default="local")
    p.add_argument("--encoder-id", default="quentinll/lewm-reacher")
    p.add_argument("--local-dir", default="encoder_reacher")
    p.add_argument("--swm-src", default=None)
    p.add_argument("--device", default="cuda")
    # what to evaluate
    p.add_argument("--lerp-frac", type=float, default=0.5,
                   help="lerp source: waypoint at this fraction along the latent "
                        "segment from the CURRENT latent to the goal (re-anchored "
                        "each replan); frac=1.0 degenerates to flat planning")
    p.add_argument("--subgoal", choices=["oracle", "gdm", "baseline", "specaccept", "lerp"],
                   default="oracle")
    p.add_argument("--gdm-ckpt", default=None, help="trained (goal-cond) planner checkpoint")
    p.add_argument("--gdm-steps", type=int, default=50, help="DDIM sampling steps for GDM")
    p.add_argument("--accept-tau", type=float, default=0.2,
                   help="specaccept: relative-L2 tolerance for the reality verifier")
    p.add_argument("--goal-gate", action="store_true",
                   help="specaccept: serve a drafted waypoint only if it reduces "
                        "latent distance to the goal; otherwise serve the goal "
                        "itself (native-regime parity, no extra diffusion)")
    p.add_argument("--goal-offset", type=int, default=25,
                   help="steps from start to the goal frame (LeWM reacher protocol: 25)")
    p.add_argument("--start", choices=["random", "final"], default="random",
                   help="random = LeWM protocol (random valid start, goal = start+offset). "
                        "final = last goal_offset steps (meaningless for random-policy "
                        "data; kept for symmetry).")
    p.add_argument("--num-eval", type=int, default=50)
    p.add_argument("--eval-budget", type=int, default=50)
    p.add_argument("--episode-min", type=int, default=None,
                   help="restrict eval to episode indices >= this (holdout guard: the "
                        "drafter trains on episodes < 8000, eval draws from >= 8000)")
    p.add_argument("--episode-max", type=int, default=None,
                   help="restrict eval to episode indices < this (exclusive)")
    p.add_argument("--episodes-file", default=None,
                   help="JSON from batch/build_reacher_horizon_episodes.py with "
                        "precomputed [episode, start] pairs. Bypasses sampling entirely "
                        "-- the fixed-population horizon sweep reuses ONE file at every "
                        "--goal-offset (starts are valid for the largest offset), so the "
                        "start states are byte-identical across the curve and horizon is "
                        "the only variable (the t75-vs-t150 sampling-artifact lesson).")
    p.add_argument("--qpos-threshold", type=float, default=None,
                   help="override the qpos_match success tolerance (rad/joint; env "
                        "default 0.05 = the official criterion)")
    p.add_argument("--seed", type=int, default=42)
    # plan / CEM (same validated defaults as the PushT driver; S = RH * action_block)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--receding-horizon", type=int, default=5)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--var-scale", type=float, default=1.0)
    p.add_argument("--cem-seed", type=int, default=None,
                   help="CEM seed; defaults to --seed")
    p.add_argument("--stride", type=int, default=25,
                   help="oracle subgoal table stride (oracle arm only)")
    p.add_argument("--dump-traces", default=None,
                   help="path (.pt): per-episode success flags + per-replan latents")
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


def main():
    args = parse_args()
    if args.subgoal in ("gdm", "specaccept") and not args.gdm_ckpt:
        raise ValueError(f"--subgoal {args.subgoal} requires --gdm-ckpt")

    encoder._ensure_swm_importable(args.swm_src)
    import stable_worldmodel as swm

    goal_offset = args.goal_offset
    eval_budget = args.eval_budget
    cem_seed = args.cem_seed if args.cem_seed is not None else args.seed

    model = encoder.load_lewm(source=args.source, encoder_id=args.encoder_id,
                              local_dir=args.local_dir, swm_src=args.swm_src,
                              device=args.device)
    is_baseline = args.subgoal == "baseline"
    cost_model = model if is_baseline else SubgoalCostModel(model)
    print(f"[model] frozen LeWM ({sum(p.numel() for p in model.parameters())/1e6:.2f}M), "
          f"device={args.device}, training={model.training}, subgoal={args.subgoal}")

    gdm_planner = None
    if args.subgoal in ("gdm", "specaccept"):
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
        if not gdm_planner.goal_cond:
            print("[warn] drafter is GOAL-FREE on random-policy Reacher data - "
                  "expected to be ill-posed (see module docstring); proceeding anyway")

    from stable_worldmodel.data.formats.hdf5 import HDF5Dataset
    keys = ["action"]
    dataset = HDF5Dataset(path=args.h5, keys_to_cache=keys)

    try:
        tf = img_transform()
    except Exception:
        tf = img_transform_fallback()
    transform = {"pixels": tf, "goal": tf}
    process = build_process(dataset, keys)

    if args.episodes_file:
        import json
        with open(args.episodes_file) as f:
            payload = json.load(f)
        pairs = payload["episodes"]
        episodes_idx = [p[0] for p in pairs]
        start_steps = [p[1] for p in pairs]
        if args.num_eval != len(pairs):
            raise ValueError(f"--num-eval {args.num_eval} != {len(pairs)} pairs in "
                             f"{args.episodes_file} (pass --num-eval {len(pairs)})")
        print(f"[episodes-file] {args.episodes_file}: {len(pairs)} precomputed pairs "
              f"(built for max_offset={payload.get('max_offset')}, seed={payload.get('seed')}, "
              f"episode_min={payload.get('episode_min')}) -- sampling bypassed; "
              f"running at goal_offset={goal_offset}")
    else:
        ep_mask = None
        if args.episode_min is not None or args.episode_max is not None:
            import h5py
            with h5py.File(args.h5, "r") as f:
                n_eps = len(f["ep_len"])
            lo = args.episode_min or 0
            hi = args.episode_max if args.episode_max is not None else n_eps
            ep_mask = np.zeros(n_eps, dtype=bool)
            ep_mask[lo:hi] = True
            print(f"[holdout] eval restricted to episodes [{lo}, {hi}) "
                  f"({int(ep_mask.sum())}/{n_eps} eligible)")
        if args.start == "final":
            episodes_idx, start_steps = sample_long(args.h5, args.num_eval, goal_offset,
                                                    args.seed, ep_mask=ep_mask)
        else:
            episodes_idx, start_steps = sample_short(args.h5, args.num_eval, goal_offset,
                                                     args.seed, ep_mask=ep_mask)
    print(f"[eval] start={args.start} num_eval={args.num_eval} goal_offset={goal_offset} "
          f"budget={eval_budget} qpos_threshold={args.qpos_threshold or 0.05:g} rad/joint")
    print(f"[eval] episodes_idx[:5]={episodes_idx[:5]} start_steps[:5]={start_steps[:5]}")

    table = None
    if args.subgoal == "oracle":
        table = build_oracle_table(args.h5, model, episodes_idx, start_steps, goal_offset,
                                   stride=args.stride, device=args.device)
        print(f"[oracle] built {len(table)} per-env subgoal tables; "
              f"K (subgoals/ep) = {table[0].shape[0]}")

    config = swm.PlanConfig(horizon=args.horizon, receding_horizon=args.receding_horizon,
                            action_block=args.action_block)
    # mirrors stable-worldmodel config/eval/reacher.yaml (in_dataset defaults True)
    callables = [
        {"method": "set_state", "args": {"qpos": {"value": "qpos"},
                                         "qvel": {"value": "qvel"}}},
        {"method": "set_target_qpos", "args": {"target_qpos": {"value": "goal_qpos"}}},
    ]

    PolicyCls = make_ffjepa_policy(swm.policy.WorldModelPolicy)

    world = swm.World(env_name="swm/ReacherDMControl-v0", num_envs=args.num_eval,
                      max_episode_steps=2 * eval_budget, image_shape=(224, 224),
                      task="qpos_match")
    if args.qpos_threshold is not None:
        for e in world.envs.envs:
            e.unwrapped.env.task.qpos_threshold = float(args.qpos_threshold)
        print(f"[pin] qpos_threshold overridden to {args.qpos_threshold:g} rad/joint")
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
                                             goal_gate=args.goal_gate,
                                             record=bool(args.dump_traces))
        elif args.subgoal == "lerp":
            source = LerpSubgoalSource(n_envs=args.num_eval, device=args.device,
                                       frac=args.lerp_frac)
            print(f"[lerp] straight-line oracle, frac={args.lerp_frac:g}")
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
        total_all = total + source.n_gate
        print(f"[specaccept] tau={args.accept_tau} gdm_steps={args.gdm_steps} "
              f"goal_gate={args.goal_gate} "
              f"re-drafts={source.n_redraft} advances={source.n_advance} "
              f"rejects={source.n_reject} gate_serves={source.n_gate} "
              f"call_ratio={source.n_redraft / max(total, 1):.3f} "
              f"call_ratio_all={source.n_redraft / max(total_all, 1):.3f} "
              f"(every-step=1.000; lower = fewer diffusion calls)")

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

    print("\n==== FF-JEPA eval summary (Reacher qpos_match) ====")
    print(f"subgoal={args.subgoal} num_eval={args.num_eval} goal_offset={goal_offset} "
          f"budget={eval_budget} seed={args.seed} cem_seed={cem_seed} "
          f"S={args.receding_horizon * args.action_block}")
    print(f"  SR = {sr:.2f}%  ({n_succ}/{args.num_eval})")


if __name__ == "__main__":
    main()
