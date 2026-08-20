"""OGBench-Cube (single) evaluation driver.

Cube analog of the Reacher/PushT drivers, reusing their policy, cost model, and
subgoal sources. The env is swm/OGBCube-v0 and success is its own criterion,
cube within 0.04 m of target. Callables set the start-frame MuJoCo state and the
goal-frame cube position, making evaluation hindsight goal-reaching. tau and k
follow the derive-first protocol: probe_floor on the frozen cube encoder,
registered before any closed-loop cell.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from surveyor import encoder
from surveyor.envs.pusht.eval import sample_short, sample_long
from surveyor.sources import (SubgoalCostModel, OracleSubgoalSource,
                                    GDMSubgoalSource, SurveyorSource,
                                    LerpSubgoalSource, CstarRetireSource,
                                    build_oracle_table, make_ffjepa_policy)

CUBE_SUCCESS_THRESH = 0.04  # metres; cube_env._compute_successes


def parse_args():
    p = argparse.ArgumentParser(description="FF-JEPA eval driver (OGBench-Cube single)")
    p.add_argument("--h5", required=True)
    p.add_argument("--source", choices=["pretrained", "local"], default="pretrained")
    p.add_argument("--encoder-id", default="quentinll/lewm-cube")
    p.add_argument("--local-dir", default=None)
    p.add_argument("--swm-src", default=None)
    p.add_argument("--device", default="cuda")
    # what to evaluate
    p.add_argument("--lerp-frac", type=float, default=0.5,
                   help="lerp source: waypoint this fraction along the latent segment "
                        "from the current latent to the goal (decomposition-tax control)")
    p.add_argument("--subgoal", choices=["oracle", "ffjepa", "flat", "surveyor",
                                         "lerp", "router+surveyor", "random", "gcidm",
                                         "gcidm+surveyor"],
                   default="flat")
    p.add_argument("--gcidm-ckpt", default=None,
                   help="GC-IDM checkpoint (arXiv 2605.08732 comparator); amortised "
                        "controller, one MLP forward per step and no solver")
    p.add_argument("--sg-steps", type=int, default=10,
                   help="gcidm+surveyor: subgoal spacing S the executor tracks")
    p.add_argument("--gdm-ckpt", default=None, help="trained planner checkpoint")
    p.add_argument("--gdm-steps", type=int, default=50, help="DDIM sampling steps for GDM")
    p.add_argument("--accept-tau", type=float, default=0.2,
                   help="surveyor: relative-L2 tolerance for the reality verifier "
                        "(derive from probe_floor on the cube encoder before use)")
    p.add_argument("--goal-gate", action="store_true")
    p.add_argument("--goal-offset", type=int, default=25,
                   help="steps from start to the goal frame (hindsight target)")
    p.add_argument("--start", choices=["random", "final"], default="final",
                   help="final = last goal_offset steps, goal = the frame where the "
                        "expert has moved the cube to its target (paper protocol). "
                        "random = random valid start, goal = start+offset.")
    p.add_argument("--num-eval", type=int, default=50)
    p.add_argument("--eval-budget", type=int, default=None, help="default 2*goal_offset")
    p.add_argument("--episode-min", type=int, default=None,
                   help="restrict eval to episode indices >= this (holdout guard)")
    p.add_argument("--episode-max", type=int, default=None)
    p.add_argument("--episodes-file", default=None,
                   help="JSON with precomputed [episode, start] pairs (fixed-population "
                        "horizon sweep); bypasses sampling.")
    p.add_argument("--success-thresh", type=float, default=None,
                   help="override cube success tolerance (m); env default 0.04")
    p.add_argument("--env-type", default="single",
                   help="OGBCube env_type (single/double/...); dataset is single")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug-keys", action="store_true",
                   help="print the resolved dataset/goal state keys for one env and exit "
                        "after setup (smoke-test the callable wiring)")
    # plan / CEM (validated defaults; S = receding_horizon * action_block)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--receding-horizon", type=int, default=5)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--var-scale", type=float, default=1.0)
    p.add_argument("--cem-seed", type=int, default=None)
    p.add_argument("--stride", type=int, default=25, help="oracle subgoal table stride")
    p.add_argument("--dump-traces", default=None)
    p.add_argument("--time-instrument", action="store_true",
                   help="CUDA-synced drafter/CEM wall-clock per episode (timing figure)")
    p.add_argument("--dump-strip", type=int, default=0,
                   help="capture the raw frame at EVERY replan boundary for the "
                        "first N envs, plus per-replan advance/redraft/gate events; "
                        "npz for filmstrip rendering (render_strips.py)")
    p.add_argument("--dump-strip-out", default="strips",
                   help="output directory for --dump-strip npz files")
    return p.parse_args()


def img_transform():
    """ImageNet-normalized 224px transform, from stable_pretraining's stats."""
    import stable_pretraining as spt
    from torchvision.transforms import v2 as transforms
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=224),
    ])


def img_transform_fallback():
    """`img_transform` without stable_pretraining, for CPU-only hosts.

    The ImageNet constants are inlined, so the two paths normalize alike.
    """
    from torchvision.transforms import v2 as transforms
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=224),
    ])


def build_process(dataset, keys):
    """Fit one StandardScaler per non-pixel column of `dataset`.

    Goal columns reuse their observation's scaler, so a state and the goal it
    is scored against land in the same units. `action` has no goal twin.
    """
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
    """Run one evaluation cell and print its summary.

    The final line is the cell's success rate as a pooled numerator over
    denominator, so runs that combine seeds sum the counts rather than
    averaging percentages.
    """
    args = parse_args()
    if args.subgoal in ("ffjepa", "surveyor", "router+surveyor") and not args.gdm_ckpt:
        raise ValueError(f"--subgoal {args.subgoal} requires --gdm-ckpt")

    encoder._ensure_swm_importable(args.swm_src)
    import stable_worldmodel as swm

    goal_offset = args.goal_offset
    eval_budget = args.eval_budget if args.eval_budget is not None else 2 * goal_offset
    cem_seed = args.cem_seed if args.cem_seed is not None else args.seed

    model = encoder.load_lewm(source=args.source, encoder_id=args.encoder_id,
                              local_dir=args.local_dir, swm_src=args.swm_src,
                              device=args.device)
    # random = swm's uniform action-space policy, LeWM Table I's chance floor.
    # Both solver-free arms take the plain cost model and never build a subgoal.
    is_random = args.subgoal == "random"
    is_gcidm = args.subgoal == "gcidm"
    is_gcidm_surveyor = args.subgoal == "gcidm+surveyor"
    is_baseline = args.subgoal in ("flat", "random", "gcidm")
    cost_model = model if is_baseline else SubgoalCostModel(model)
    print(f"[model] frozen LeWM ({sum(p.numel() for p in model.parameters())/1e6:.2f}M), "
          f"device={args.device}, training={model.training}, subgoal={args.subgoal}")

    gdm_planner = None
    if args.subgoal in ("ffjepa", "surveyor", "router+surveyor", "gcidm+surveyor"):
        from surveyor.drafter import load_gdm_planner
        gdm_planner = load_gdm_planner(args.gdm_ckpt, device=args.device)
        print(f"[gdm] planner from {args.gdm_ckpt}: "
              f"N={gdm_planner.cfg.n_future} goal_cond={gdm_planner.goal_cond} "
              f"sampler={gdm_planner.diffusion.sampler} norm={gdm_planner.normalization}")

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
            raise ValueError(f"--num-eval {args.num_eval} != {len(pairs)} pairs")
        print(f"[episodes-file] {args.episodes_file}: {len(pairs)} pairs, "
              f"goal_offset={goal_offset}")
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
                  f"({int(ep_mask.sum())}/{n_eps})")
        sampler = sample_long if args.start == "final" else sample_short
        episodes_idx, start_steps = sampler(args.h5, args.num_eval, goal_offset,
                                            args.seed, ep_mask=ep_mask)
    print(f"[eval] start={args.start} num_eval={args.num_eval} goal_offset={goal_offset} "
          f"budget={eval_budget} success_thresh={args.success_thresh or CUBE_SUCCESS_THRESH:g} m")

    table = None
    if args.subgoal == "oracle":
        table = build_oracle_table(args.h5, model, episodes_idx, start_steps, goal_offset,
                                   stride=args.stride, device=args.device)

    config = swm.PlanConfig(horizon=args.horizon, receding_horizon=args.receding_horizon,
                            action_block=args.action_block)
    # manip-space callables. World._extract_init_goal exposes start-frame columns
    # as init_state and the frame at start+goal_offset as goal_<col>, so we
    # initialise from the start frame's MuJoCo state (qpos/qvel) and set the goal
    # to the GOAL frame's cube position, making success goal-reaching.
    callables = [
        {"method": "set_state", "args": {"qpos": {"value": "qpos"},
                                         "qvel": {"value": "qvel"}}},
        {"method": "set_target_pos",
         "args": {"cube_id": {"value": 0, "in_dataset": False},
                  "target_pos": {"value": "goal_privileged_block_0_pos"}}},
    ]

    PolicyCls = make_ffjepa_policy(swm.policy.WorldModelPolicy)
    world = swm.World(env_name="swm/OGBCube-v0", num_envs=args.num_eval,
                      max_episode_steps=2 * eval_budget, image_shape=(224, 224),
                      env_type=args.env_type)
    if args.success_thresh is not None:
        for e in world.envs.envs:
            e.unwrapped.success_threshold = float(args.success_thresh)
        print(f"[pin] success threshold overridden to {args.success_thresh:g} m")

    if args.debug_keys:
        # smoke: surface the state keys the callables must resolve against, then stop.
        env0 = world.envs.envs[0].unwrapped
        print("[debug-keys] env unwrapped:", type(env0).__name__)
        print("[debug-keys] has set_state:", hasattr(env0, "set_state"),
              "| has set_target_pos:", hasattr(env0, "set_target_pos"))
        import h5py
        with h5py.File(args.h5, "r") as f:
            print("[debug-keys] h5 columns:", list(f.keys()))
            for col in ("prev_qpos", "prev_qvel", "privileged_block_0_pos",
                        "privileged_target_block_pos"):
                if col in f:
                    print(f"    {col}: shape {f[col].shape}")
        world.close()
        return

    solver = swm.solver.CEMSolver(model=cost_model, batch_size=1,
                                  num_samples=args.num_samples, var_scale=args.var_scale,
                                  n_steps=args.n_steps, topk=args.topk,
                                  device=args.device, seed=cem_seed)
    if is_random:
        # seeded via set_policy() from .seed, deterministic at cem_seed like every arm
        policy = swm.policy.RandomPolicy(seed=cem_seed)
    elif is_gcidm:
        from surveyor.gcidm import GCIDMPolicy, load_gcidm
        gci, ascaler, gmeta = load_gcidm(args.gcidm_ckpt, device=args.device)
        policy = GCIDMPolicy(gci, model, budget=eval_budget,
                             device=args.device, action_scaler=ascaler)
        print(f"[gcidm] {args.gcidm_ckpt}: H_max={gci.h_max} "
              f"params={sum(p.numel() for p in gci.parameters())/1e6:.2f}M "
              f"budget={eval_budget} (1 forward/step, no solver)")
    elif is_gcidm_surveyor:
        from surveyor.gcidm import load_gcidm
        from surveyor.gcidm_executor import SurveyorGCIDMPolicy
        gci, ascaler, gmeta = load_gcidm(args.gcidm_ckpt, device=args.device)
        policy = SurveyorGCIDMPolicy(gci, gdm_planner, model,
                                 sg_steps=args.sg_steps, tau=args.accept_tau,
                                 n_steps=args.gdm_steps, seed=cem_seed,
                                 device=args.device, action_scaler=ascaler,
                                 budget=eval_budget, cstar_route=False,
                                 goal_gate=args.goal_gate, adim=5)
        print(f"[gcidm+surveyor] executor={args.gcidm_ckpt} (H_max={gci.h_max}) "
              f"drafter={args.gdm_ckpt} S={args.sg_steps} tau={args.accept_tau} "
              f"k={args.gdm_steps} goal_gate={args.goal_gate} "
              f"(accept rule unchanged, no CEM in the arm)")
    elif is_baseline:
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform)
    else:
        if args.subgoal == "ffjepa":
            source = GDMSubgoalSource(gdm_planner, n_envs=args.num_eval,
                                      dim=gdm_planner.cfg.latent_dim,
                                      device=args.device, n_steps=args.gdm_steps,
                                      seed=cem_seed, record=bool(args.dump_traces))
        elif args.subgoal == "surveyor":
            source = SurveyorSource(gdm_planner, n_envs=args.num_eval,
                                             device=args.device, n_steps=args.gdm_steps,
                                             seed=cem_seed, tau=args.accept_tau,
                                             goal_gate=args.goal_gate,
                                             record=bool(args.dump_traces))
        elif args.subgoal == "router+surveyor":
            source = CstarRetireSource(gdm_planner, model, n_envs=args.num_eval,
                                       device=args.device, n_steps=args.gdm_steps,
                                       seed=cem_seed, tau=args.accept_tau,
                                       horizon=args.horizon,
                                       action_block=args.action_block,
                                       num_samples=args.num_samples,
                                       cem_steps=args.n_steps, topk=args.topk,
                                       var_scale=args.var_scale, adim=5,
                                       record=bool(args.dump_traces))
            print(f"[router+surveyor] c*-retire surveyor: tau={args.accept_tau}, "
                  f"retire window={args.horizon * args.action_block} steps; "
                  f"drafting only while the goal is out of certified reach")
        elif args.subgoal == "lerp":
            source = LerpSubgoalSource(n_envs=args.num_eval, device=args.device,
                                       frac=args.lerp_frac)
        else:
            source = OracleSubgoalSource(table, device=args.device)
        policy = PolicyCls(cost_model=cost_model, subgoal_source=source,
                           time_instrument=args.time_instrument,
                           dump_frames=args.dump_strip > 0, dump_strip=args.dump_strip,
                           solver=solver, config=config, process=process, transform=transform)
    world.set_policy(policy)
    metrics = world.evaluate(
        dataset=dataset, start_steps=list(start_steps), goal_offset=goal_offset,
        eval_budget=eval_budget, episodes_idx=list(episodes_idx), callables=callables,
    )
    if is_gcidm:
        # cost column: decisions per episode, one MLP forward each vs a CEM solve
        print(f"[gcidm-cost] decisions={policy.n_calls / max(args.num_eval, 1):.1f} "
              f"total={policy.n_calls} (1 MLP forward each, no solver)")
    sr = metrics["success_rate"]
    n_succ = int(metrics["episode_successes"].sum())
    world.close()

    if args.time_instrument and not is_baseline:
        t_d, t_c = policy.t_drafter, policy.t_cem
        t_tot = t_d + t_c
        print(f"[timing] drafter={t_d:.3f}s cem={t_c:.3f}s total={t_tot:.3f}s "
              f"drafter_frac={100.0 * t_d / max(t_tot, 1e-9):.2f}% "
              f"per_episode_ms={{'drafter_ms': {1000.0 * t_d / max(args.num_eval, 1)}, "
              f"'cem_ms': {1000.0 * t_c / max(args.num_eval, 1)}, "
              f"'total_ms': {1000.0 * t_tot / max(args.num_eval, 1)}}}")

    if args.subgoal == "surveyor":
        total = source.n_redraft + source.n_advance
        print(f"[surveyor] tau={args.accept_tau} gdm_steps={args.gdm_steps} "
              f"re-drafts={source.n_redraft} advances={source.n_advance} "
              f"rejects={source.n_reject} "
              f"call_ratio={source.n_redraft / max(total, 1):.3f} "
              f"(every-step=1.000; lower = fewer diffusion calls)")

    if args.subgoal == "router+surveyor":
        seen = source._seen
        retired = int(source._retired.sum())
        fire0 = int((source.c_first[seen] <= source.tau).sum())
        rr = source.retire_replan[source.retire_replan >= 0]
        cf = source.c_first[seen]
        sp = source.spec
        sp_total = sp.n_redraft + sp.n_advance
        print(f"[router+surveyor] tau={source.tau} retired={retired}/{int(seen.sum())} "
              f"(fired-at-first-replan={fire0}, == the router fire test) "
              f"retire_replan p10/p50/p90="
              + (f"{np.percentile(rr, 10):.0f}/{np.percentile(rr, 50):.0f}/"
                 f"{np.percentile(rr, 90):.0f} " if len(rr) else "-/-/- ")
              + f"c*_first p50={np.percentile(cf, 50):.3f} | "
              f"spec: re-drafts={sp.n_redraft} advances={sp.n_advance} "
              f"rejects={sp.n_reject} "
              f"call_ratio={sp.n_redraft / max(sp_total, 1):.3f}")

    if args.dump_strip and not is_baseline and getattr(policy, "_strip", None):
        import json as _json
        from pathlib import Path
        outdir = Path(args.dump_strip_out)
        outdir.mkdir(parents=True, exist_ok=True)
        fn = outdir / (f"strip_cube_{args.subgoal}_t{goal_offset}"
                       f"_seed{args.seed}.npz")
        succ = np.asarray(metrics["episode_successes"]).astype(bool)
        payload = {"success": succ,
                   "meta": np.array(_json.dumps({
                       "env": "cube", "subgoal": args.subgoal,
                       "goal_offset": int(goal_offset), "seed": int(args.seed),
                       "tau": float(args.accept_tau), "sr": float(sr)}))}
        for i, fr_list in enumerate(policy._strip):
            if not fr_list:
                continue
            payload[f"ep{i}_tags"] = np.array([t for t, _ in fr_list], np.int64)
            payload[f"ep{i}_frames"] = np.stack([f for _, f in fr_list]).astype(np.uint8)
            for tag, fr in (("start", policy._frame_start[i]),
                            ("goal", policy._frame_goal[i]),
                            ("last", policy._frame_last[i])):
                if fr is not None:
                    payload[f"ep{i}_{tag}"] = np.asarray(fr).astype(np.uint8)
        ev = getattr(source, "events", None)
        if ev:
            payload["ev_env"] = np.array([e for e, _, _ in ev], np.int64)
            payload["ev_kind"] = np.array([k for _, k, _ in ev])
            payload["ev_rel"] = np.array([r for _, _, r in ev], np.float32)
        np.savez_compressed(fn, **payload)
        print(f"[strip] saved {sum(1 for s in policy._strip if s)} episode strips -> {fn}")

    if args.dump_traces:
        from pathlib import Path
        rec = {"args": vars(args), "sr": float(sr),
               "successes": np.asarray(metrics["episode_successes"]).astype(bool).tolist(),
               "episodes_idx": list(episodes_idx), "start_steps": list(start_steps)}
        if not is_baseline and getattr(source, "record", False):
            rec["trace"] = source.trace
            rec["cal"] = getattr(source, "cal", [])
        Path(args.dump_traces).parent.mkdir(parents=True, exist_ok=True)
        torch.save(rec, args.dump_traces)
        print(f"[traces] saved {args.dump_traces}")

    print("\n==== FF-JEPA eval summary (OGBench-Cube single) ====")
    print(f"subgoal={args.subgoal} num_eval={args.num_eval} goal_offset={goal_offset} "
          f"budget={eval_budget} seed={args.seed} cem_seed={cem_seed} "
          f"S={args.receding_horizon * args.action_block}")
    print(f"  SR = {sr:.2f}%  ({n_succ}/{args.num_eval})")
    if is_gcidm_surveyor:
        bnd = policy.n_advance + policy.n_redraft
        cr = policy.n_redraft / max(bnd, 1)
        print(f"[gcidm+surveyor] redrafts={policy.n_redraft} advances={policy.n_advance} "
              f"rejects={policy.n_reject} arrived={policy.n_arrive} "
              f"call_ratio={cr:.3f} NFE/replan={cr * args.gdm_steps:.2f}")


if __name__ == "__main__":
    main()
