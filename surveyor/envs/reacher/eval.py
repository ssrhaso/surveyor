"""Reacher (DMControl qpos_match) evaluation driver.

Reacher analog of the PushT driver, reusing the policy, cost model, and subgoal
sources. Success is the env's own qpos_match rule, and the callables set the
start-frame state and the goal-frame target qpos. The dataset is random-policy
data, so drafting arms expect a goal-conditioned drafter: the policy encodes the
goal image at each replan and sources pass it through.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from surveyor import encoder
from surveyor.envs.pusht.eval import sample_short, sample_long
from surveyor.sources import (SubgoalCostModel, OracleSubgoalSource,
                                    GDMSubgoalSource, SpecAcceptSubgoalSource,
                                    LerpSubgoalSource, HorizonGatedSource,
                                    CstarRetireSource,
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
    p.add_argument("--subgoal", choices=["oracle", "gdm", "baseline", "specaccept",
                                         "lerp", "horizon_gated", "unified", "random", "gcidm",
                                         "specgcidm"],
                   default="oracle")
    p.add_argument("--gcidm-ckpt", default=None,
                   help="GC-IDM checkpoint (arXiv 2605.08732 comparator); amortised "
                        "controller, one MLP forward per step and no solver")
    p.add_argument("--gdm-ckpt", default=None, help="trained (goal-cond) planner checkpoint")
    p.add_argument("--gdm-steps", type=int, default=50, help="DDIM sampling steps for GDM")
    p.add_argument("--verify-readout", default=None,
                   help="learn_readout.py checkpoint: verify drafts in the "
                        "gap-maximized lens space instead of native rel-L2 "
                        "(verification-side only; drafting/cost stay native)")
    p.add_argument("--readout-tau", type=float, default=None,
                   help="override the lens's derived tau (default: ckpt value)")
    p.add_argument("--accept-tau", type=float, default=0.2,
                   help="specaccept: relative-L2 tolerance for the reality verifier")
    p.add_argument("--sg-steps", type=int, default=10,
                   help="specgcidm: subgoal spacing S in env steps; accept test at "
                        "every S-step boundary, executor horizon clock counts to it")
    p.add_argument("--cstar-route", action="store_true",
                   help="specgcidm: certified scope (P-EXEC-6). One flat-CEM c* "
                        "probe at each env's first boundary routes the episode to "
                        "plain GC-IDM when c* <= tau (arbiter window 2x5); drafting "
                        "envs carry the tau arrival gate. One CEM solve/episode.")
    p.add_argument("--random-reject", type=float, default=None,
                   help="decision-content control (docs/randreject_prereg.md): "
                        "replace the accept test with an i.i.d. coin rejecting "
                        "at this matched probability; all other mechanics "
                        "identical to the banked spec arm")
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
    if args.subgoal in ("gdm", "specaccept", "horizon_gated", "unified", "specgcidm") and not args.gdm_ckpt:
        raise ValueError(f"--subgoal {args.subgoal} requires --gdm-ckpt")
    if args.subgoal == "specgcidm" and not args.gcidm_ckpt:
        raise ValueError("--subgoal specgcidm requires --gcidm-ckpt (amortised executor)")

    encoder._ensure_swm_importable(args.swm_src)
    import stable_worldmodel as swm

    goal_offset = args.goal_offset
    eval_budget = args.eval_budget
    cem_seed = args.cem_seed if args.cem_seed is not None else args.seed

    model = encoder.load_lewm(source=args.source, encoder_id=args.encoder_id,
                              local_dir=args.local_dir, swm_src=args.swm_src,
                              device=args.device)
    # random = swm's uniform action-space policy, LeWM Table I's chance floor.
    # Both solver-free arms take the plain cost model and never build a subgoal.
    is_random = args.subgoal == "random"
    is_gcidm = args.subgoal == "gcidm"
    is_specgcidm = args.subgoal == "specgcidm"
    is_baseline = args.subgoal in ("baseline", "random", "gcidm", "specgcidm")
    cost_model = model if is_baseline else SubgoalCostModel(model)
    print(f"[model] frozen LeWM ({sum(p.numel() for p in model.parameters())/1e6:.2f}M), "
          f"device={args.device}, training={model.training}, subgoal={args.subgoal}")

    gdm_planner = None
    if args.subgoal in ("gdm", "specaccept", "horizon_gated", "unified", "specgcidm"):
        from surveyor.drafter import load_gdm_planner, count_params
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
              f"episode_min={payload.get('episode_min')}); sampling bypassed, "
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
    elif is_specgcidm:
        from surveyor.gcidm import load_gcidm
        from surveyor.spec_gcidm import SpecGCIDMPolicy
        gci, ascaler, gmeta = load_gcidm(args.gcidm_ckpt, device=args.device)
        policy = SpecGCIDMPolicy(gci, gdm_planner, model,
                                 sg_steps=args.sg_steps, tau=args.accept_tau,
                                 n_steps=args.gdm_steps, seed=cem_seed,
                                 device=args.device, action_scaler=ascaler,
                                 budget=eval_budget,
                                 cstar_route=args.cstar_route,
                                 cem=dict(horizon=2, action_block=5,
                                          num_samples=args.num_samples,
                                          n_steps=args.n_steps, topk=args.topk,
                                          var_scale=args.var_scale,
                                          seed=cem_seed),
                                 adim=2)
        print(f"[specgcidm] executor={args.gcidm_ckpt} (H_max={gci.h_max}) "
              f"drafter={args.gdm_ckpt} S={args.sg_steps} tau={args.accept_tau} "
              f"k={args.gdm_steps} cstar_route={args.cstar_route} "
              f"(accept rule unchanged)")
    elif is_baseline:
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform)
    else:
        if args.subgoal == "gdm":
            source = GDMSubgoalSource(gdm_planner, n_envs=args.num_eval,
                                      dim=gdm_planner.cfg.latent_dim,
                                      device=args.device, n_steps=args.gdm_steps,
                                      seed=cem_seed, record=bool(args.dump_traces))
        elif args.subgoal == "specaccept":
            readout, rtau = None, None
            if args.verify_readout:
                from surveyor.probes.learn_readout import Readout
                ck = torch.load(args.verify_readout, map_location="cpu",
                                weights_only=False)
                readout = Readout(ck["dim"], ck["out_dim"],
                                  attentive=ck["attentive"])
                readout.load_state_dict(ck["state"])
                readout.to(args.device).eval()
                rtau = (args.readout_tau if args.readout_tau is not None
                        else ck["tau_derived"])
                print(f"[readout-verify] lens={args.verify_readout} "
                      f"({ck['dim']}->{ck['out_dim']}), tau={rtau:.3f} "
                      f"(gap after: {ck['gap_after']})")
            source = SpecAcceptSubgoalSource(gdm_planner, n_envs=args.num_eval,
                                             device=args.device,
                                             n_steps=args.gdm_steps, seed=cem_seed,
                                             tau=args.accept_tau,
                                             goal_gate=args.goal_gate,
                                             readout=readout, readout_tau=rtau,
                                             random_reject=args.random_reject,
                                             record=bool(args.dump_traces))
            if args.random_reject is not None:
                print(f"[randreject] coin control p={args.random_reject} "
                      f"(matched-rate; latent test overridden)")
        elif args.subgoal == "horizon_gated":
            source = HorizonGatedSource(gdm_planner, model, n_envs=args.num_eval,
                                        device=args.device, n_steps=args.gdm_steps,
                                        seed=cem_seed, tau=args.accept_tau,
                                        horizon=args.horizon,
                                        action_block=args.action_block,
                                        num_samples=args.num_samples,
                                        cem_steps=args.n_steps, topk=args.topk,
                                        var_scale=args.var_scale,
                                        record=bool(args.dump_traces))
            print(f"[horizon-gate] episode-level c* gate: tau={args.accept_tau} "
                  f"plan window={args.horizon * args.action_block} steps; "
                  f"fired episodes run flat on the goal latent, others spec-accept")
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
        elif args.subgoal == "lerp":
            source = LerpSubgoalSource(n_envs=args.num_eval, device=args.device,
                                       frac=args.lerp_frac)
            print(f"[lerp] straight-line oracle, frac={args.lerp_frac:g}")
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

    if is_specgcidm:
        b = policy.n_redraft + policy.n_advance
        print(f"[specgcidm-cost] exec_forwards={policy.n_calls} "
              f"redrafts={policy.n_redraft} advances={policy.n_advance} "
              f"rejects={policy.n_reject} call_ratio={policy.n_redraft / max(b, 1):.3f} "
              f"routed={policy.n_routed} arrived={policy.n_arrive} "
              f"draft_s={policy.t_draft:.2f} exec_s={policy.t_exec:.2f} "
              f"probe_s={policy.t_probe:.2f}")
    if args.time_instrument and not is_baseline:
        t_d, t_c = policy.t_drafter, policy.t_cem
        t_tot = t_d + t_c
        print(f"[timing] drafter={t_d:.3f}s cem={t_c:.3f}s total={t_tot:.3f}s "
              f"drafter_frac={100.0 * t_d / max(t_tot, 1e-9):.2f}% "
              f"per_episode_ms={{'drafter_ms': {1000.0 * t_d / max(args.num_eval, 1)}, "
              f"'cem_ms': {1000.0 * t_c / max(args.num_eval, 1)}, "
              f"'total_ms': {1000.0 * t_tot / max(args.num_eval, 1)}}}")

    if args.subgoal == "horizon_gated":
        dec = int(source._decided.sum())
        fired = int(source._fired.sum())
        cs = source.c_star[source._decided]
        sp = source.spec
        sp_total = sp.n_redraft + sp.n_advance
        print(f"[horizon-gate] tau={args.accept_tau} fired={fired}/{dec} "
              f"(fire-rate={fired / max(dec, 1):.3f}) "
              f"c* p10/p50/p90={np.percentile(cs, 10):.3f}/"
              f"{np.percentile(cs, 50):.3f}/{np.percentile(cs, 90):.3f} | "
              f"spec branch: re-drafts={sp.n_redraft} advances={sp.n_advance} "
              f"rejects={sp.n_reject} "
              f"call_ratio={sp.n_redraft / max(sp_total, 1):.3f} "
              f"(fired episodes make zero drafter calls by construction)")

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

    if args.dump_strip and not is_baseline and getattr(policy, "_strip", None):
        import json as _json
        from pathlib import Path
        outdir = Path(args.dump_strip_out)
        outdir.mkdir(parents=True, exist_ok=True)
        fn = outdir / (f"strip_reacher_{args.subgoal}_t{goal_offset}"
                       f"_seed{args.seed}.npz")
        succ = np.asarray(metrics["episode_successes"]).astype(bool)
        payload = {"success": succ,
                   "meta": np.array(_json.dumps({
                       "env": "reacher", "subgoal": args.subgoal,
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

    print("\n==== FF-JEPA eval summary (Reacher qpos_match) ====")
    print(f"subgoal={args.subgoal} num_eval={args.num_eval} goal_offset={goal_offset} "
          f"budget={eval_budget} seed={args.seed} cem_seed={cem_seed} "
          f"S={args.receding_horizon * args.action_block}")
    print(f"  SR = {sr:.2f}%  ({n_succ}/{args.num_eval})")


if __name__ == "__main__":
    main()
