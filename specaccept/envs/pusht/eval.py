"""PushT evaluation driver.

Runs FFJEPAPolicy + SubgoalCostModel through the stable_worldmodel harness and
reports success at 20 deg (headline) and 5 deg (strict) under the same
eval_state pin as the baseline. --subgoal selects the source; horizon modes
follow the paper protocol (final-N window, goal = last frame) in short, long,
and random_init variants.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from specaccept import encoder
from specaccept.sources import (SubgoalCostModel, OracleSubgoalSource,
                                    GDMSubgoalSource, DSparkSubgoalSource,
                                    SpecAcceptSubgoalSource, RegressorSubgoalSource,
                                    CstarRetireSource, LerpSubgoalSource,
                                    build_oracle_table, make_ffjepa_policy)


# success-criterion pin (mirrors run_eval.py): pos<20 AND angle<ANGLE_DEG
def patch_eval_state(PushT, angle_deg, score_mode="env"):
    """score_mode: 'env' takes pos_diff over 4-D agent+block (env-native,
    baseline-comparable), 'block' over block xy only (the paper's criterion)."""
    def eval_state(self, goal_state, cur_state):
        if score_mode == "block":
            pos_diff = np.linalg.norm(goal_state[2:4] - cur_state[2:4])
        else:
            pos_diff = np.linalg.norm(goal_state[:4] - cur_state[:4])
        angle_diff = np.abs(goal_state[4] - cur_state[4])
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        success = pos_diff < 20 and angle_diff < np.radians(angle_deg)
        return success, np.linalg.norm(goal_state - cur_state)
    PushT.eval_state = eval_state
    tag = "block xy only (paper)" if score_mode == "block" else "4-D agent+block (env-native)"
    print(f"[PIN] success = pos_diff<20 [{tag}] AND angle<{angle_deg:g}deg")


def parse_args():
    p = argparse.ArgumentParser(description="FF-JEPA eval driver")
    p.add_argument("--h5", required=True)
    p.add_argument("--source", choices=["pretrained", "local"], default="pretrained")
    p.add_argument("--encoder-id", default="quentinll/lewm-pusht")
    p.add_argument("--local-dir", default=None)
    p.add_argument("--swm-src", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dataset-name", default="pusht_expert_train",
                   help="swm dataset name (box: resolved under STABLEWM_HOME); --h5 used for oracle/state")
    # what to evaluate
    p.add_argument("--subgoal", choices=["oracle", "gdm", "baseline", "dspark", "specaccept",
                                         "regressor", "unified", "lerp", "specpaired",
                                         "random", "gcidm", "specgcidm"],
                   default="oracle")
    p.add_argument("--gcidm-ckpt", default=None,
                   help="GC-IDM checkpoint (arXiv 2605.08732 comparator); amortised "
                        "controller, one MLP forward per step and no solver")
    p.add_argument("--verify-half", choices=["dino", "lewm"], default="dino",
                   help="specpaired: which half of the paired draft the accept "
                        "test reads. The C1/C2 verify-space control runs BOTH "
                        "on one drafter; tau must be derived in the chosen "
                        "space (lewm: the env's criterion floor; dino: "
                        "probe_floor --verify-space dino).")
    p.add_argument("--lerp-frac", type=float, default=0.5,
                   help="lerp source: waypoint at this fraction along the latent segment "
                        "from the CURRENT latent to the goal (re-anchored each replan); "
                        "frac=1.0 degenerates to flat planning (instrumented-flat tautology)")
    p.add_argument("--regressor-ckpt", default=None,
                   help="train_regressor.py checkpoint (--subgoal regressor)")
    p.add_argument("--accept-tau", type=float, default=0.2,
                   help="specaccept: relative-L2 tolerance for the reality verifier "
                        "(accept the next drafted position iff the achieved latent is "
                        "within tau of the subgoal just driven toward)")
    p.add_argument("--sg-steps", type=int, default=10,
                   help="specgcidm: subgoal spacing S in env steps; the accept test "
                        "runs at every S-step boundary and the executor's horizon "
                        "clock counts steps to the next boundary")
    p.add_argument("--cstar-route", action="store_true",
                   help="specgcidm: certified scope (P-EXEC-6). One flat-CEM c* "
                        "probe at each env's first boundary routes the episode to "
                        "plain GC-IDM when c* <= tau (arbiter window 2x5); drafting "
                        "envs carry the tau arrival gate. One CEM solve/episode.")
    p.add_argument("--draft-noise", type=float, default=0.0,
                   help="corruption sweep (certification prereg): displace every "
                        "drafted waypoint by this relative-norm sigma along a "
                        "random unit direction, every draft incl. re-drafts. "
                        "Blind consumption = --accept-tau 999 (accepts all).")
    p.add_argument("--random-reject", type=float, default=None,
                   help="decision-content control (docs/randreject_prereg.md): "
                        "replace the accept test with an i.i.d. coin rejecting "
                        "at this matched probability; all other mechanics "
                        "identical to the banked spec arm")
    p.add_argument("--gdm-ckpt", default=None, help="(Task C) trained planner checkpoint")
    p.add_argument("--gdm-steps", type=int, default=50, help="DDIM sampling steps for GDM")
    p.add_argument("--gdm-noise-scale", type=float, default=1.0,
                   help="gamma: scales every injected noise draw in the drafter sampler "
                        "(0=deterministic mode-seeking, 1=native, >1=inflated spread). "
                        "Inference-only dose-response knob for the spread thesis.")
    # DSpark mechanism (--subgoal dspark): frozen GDM drafter + trained refiner/confidence
    p.add_argument("--dspark-ckpt", default=None, help="train_dspark_head.py checkpoint")
    p.add_argument("--commit", choices=["adaptive", "fixed"], default="adaptive",
                   help="commit-depth policy: adaptive (confidence k*=max{k:Pi c_i>theta}) or fixed")
    p.add_argument("--commit-theta", type=float, default=0.5, help="adaptive commit threshold")
    p.add_argument("--commit-k", type=int, default=None, help="fixed commit depth (<=N)")
    p.add_argument("--no-sts", action="store_true", help="disable sequential temperature scaling")
    p.add_argument("--chain-n", type=int, default=None,
                   help="AR-chain the native-N drafter to this block length (e.g. 6) for a "
                        "longer subgoal chain; default = drafter's native N")
    p.add_argument("--no-refine", action="store_true",
                   help="skip the DSparkHead: use the RAW (AR-chained) draft block. Requires "
                        "--commit fixed. Isolates the longer-chain claim from refinement.")
    p.add_argument("--mode", choices=["short", "long", "random_init"], default="short",
                   help="random_init: paper's most extreme test (III-A col 3). Fully "
                        "synthetic random start (PushT's own reset() default variation "
                        "sampling, not tied to any recorded episode) scored against the "
                        "fixed canonical block target. --subgoal gdm only for now (oracle "
                        "has no natural subgoal chain for a synthetic start; baseline needs "
                        "a goal image from a cross-episode source not yet implemented).")
    p.add_argument("--goal-offset", type=int, default=None,
                   help="override the mode-derived goal_offset (25 short / 75 long), e.g. "
                        "--mode long --goal-offset 150 for a beyond-paper long-long horizon. "
                        "eval-budget still defaults off --mode; pass --eval-budget explicitly too.")
    p.add_argument("--start", choices=["final", "random"], default="final",
                   help="episode sampling. final = paper FF-JEPA protocol: the LAST "
                        "goal_offset steps, goal = last frame (the task target). "
                        "random = eval.py/[13,5] common setting: random valid start, "
                        "goal = start+goal_offset. long mode is always 'final'.")
    p.add_argument("--episodes-file", default=None,
                   help="JSON file from specaccept.envs.pusht.extract_subset (produces a small subset --h5 "
                        "plus this file). Bypasses sample_long/short entirely and uses the "
                        "exact precomputed (episode, start_step) pairs; for running against "
                        "a transferred subset h5 on a box without the full dataset.")
    p.add_argument("--eval-filter", choices=["none", "success20", "success5"], default="none",
                   help="restrict eval episodes to demos whose FINAL frame reaches the "
                        "canonical target (paper III-A: the last frame, where the "
                        "T-piece reaches its target position, serves as the goal "
                        "state). none = current behavior: ALL episodes, of which ~34%% "
                        "are FAILED demos whose goal a goal-free planner cannot target.")
    p.add_argument("--num-eval", type=int, default=32)
    p.add_argument("--eval-budget", type=int, default=None, help="override mode default")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--angles", type=float, nargs="+", default=[20.0, 5.0])
    p.add_argument("--score", choices=["env", "block", "both"], default="both",
                   help="env=4-D agent+block (baseline-comparable); block=paper's block-only")
    # plan / CEM (validated defaults)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--receding-horizon", type=int, default=5)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--var-scale", type=float, default=1.0)
    p.add_argument("--cem-seed", type=int, default=None,
                   help="CEM seed; defaults to --seed (matches eval.py cem.yaml seed=${seed})")
    p.add_argument("--stride", type=int, default=25)
    p.add_argument("--dump-frames-dir", default=None,
                   help="save per-env start/goal/last PNG frames (visual sanity check; "
                        "e.g. for --mode random_init). Additive/opt-in: forces "
                        "reset_mode='wait' for the random_init eval call so each env "
                        "slot runs exactly one episode (frame<->success stays aligned).")
    p.add_argument("--time-instrument", action="store_true",
                   help="wall-clock CEM-vs-drafter split (additive; zero overhead when "
                        "unset). Prints a [timing] line per (score_mode, angle) run. "
                        "Non-baseline subgoal sources only.")
    p.add_argument("--dump-traces", default=None,
                   help="path (.pt): save per-episode success flags plus, for the gdm "
                        "source, every replan's (z_cond, drafted subgoal) pair; the "
                        "raw material for failure anatomy (unreachable-subgoal vs "
                        "drift vs budget) without re-running the eval")
    p.add_argument("--dump-strip", type=int, default=0,
                   help="capture the raw frame at EVERY replan boundary for the "
                        "first N envs (the achieved state the verifier sees), "
                        "plus per-replan advance/redraft/gate events for sources "
                        "that log them; saved as npz for filmstrip rendering "
                        "(render_strips.py). Poster/paper qualitative figures.")
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
    # used when stable_pretraining isn't installed (CPU box): identical ImageNet stats
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


def success_mask(h5, angle_deg):
    """Per-episode success flags under the Task-A canonical block-only filter:
    block within 20px AND angle < angle_deg of the canonical target, agent term
    neutralized. Same math as build_subgoals."""
    import h5py
    with h5py.File(h5, "r") as f:
        ep_off = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
        rows = ep_off + ep_len - 1
        order = np.argsort(rows)
        finals = np.empty((len(rows), f["state"].shape[1]), dtype=np.float64)
        finals[order] = f["state"][np.sort(rows)]
    return np.array([encoder.eval_state_tol(encoder.canonical_target_for(s), s, angle_deg)
                     for s in finals], dtype=bool)


def sample_short(h5, num_eval, goal_offset, seed, ep_mask=None):
    """eval.py's random-valid-start sampling, so runs stay baseline-comparable.
    ep_mask (per-episode bool) optionally restricts to successful demos."""
    import h5py
    with h5py.File(h5, "r") as f:
        # PushT names this column "episode_idx"; the reacher dump names it "ep_idx"
        episode_idx = f["episode_idx"][:] if "episode_idx" in f else f["ep_idx"][:]
        step_idx = f["step_idx"][:]
        ep_len = f["ep_len"][:]
    ep_len_per_row = ep_len[episode_idx]
    max_start_per_row = ep_len_per_row - goal_offset - 1
    valid = step_idx <= max_start_per_row
    if ep_mask is not None:
        valid &= ep_mask[episode_idx]
    valid_indices = np.nonzero(valid)[0]
    g = np.random.default_rng(seed)
    chosen = g.choice(len(valid_indices) - 1, size=num_eval, replace=False)
    rows = np.sort(valid_indices[chosen])
    episodes_idx = episode_idx[rows].tolist()
    start_steps = step_idx[rows].tolist()
    return episodes_idx, start_steps


def sample_long(h5, num_eval, last_n, seed, ep_mask=None):
    """Per-episode start = ep_len-1-last_n (the 'last last_n steps'); goal=last frame.
    ep_mask (per-episode bool) optionally restricts to successful demos."""
    import h5py
    with h5py.File(h5, "r") as f:
        ep_len = f["ep_len"][:]
    valid = ep_len > last_n + 1
    if ep_mask is not None:
        valid &= ep_mask
    valid_eps = np.nonzero(valid)[0]
    g = np.random.default_rng(seed)
    eps = np.sort(g.choice(valid_eps, size=num_eval, replace=False))
    episodes_idx = eps.tolist()
    start_steps = [int(ep_len[e] - 1 - last_n) for e in eps]
    return episodes_idx, start_steps


def sample_random_init_goals(h5, num_eval, seed, angle_deg=20.0):
    """Per-episode-random goal states for --mode random_init.

    Paper III-A puts the goal at the final position of a random successful
    episode, not at one fixed canonical target. Draws num_eval goals with
    replacement from episodes whose final frame passes the success filter, each
    goal being that episode's final block pose with the agent zeroed (score=block
    ignores the agent term)."""
    import h5py
    mask = success_mask(h5, angle_deg)
    eligible = np.nonzero(mask)[0]
    with h5py.File(h5, "r") as f:
        ep_off = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
        rows = (ep_off + ep_len - 1)[eligible]
        order = np.argsort(rows)
        finals = np.empty((len(rows), f["state"].shape[1]), dtype=np.float64)
        finals[order] = f["state"][np.sort(rows)]
    g = np.random.default_rng(seed)
    chosen = g.integers(0, len(eligible), size=num_eval)
    return [np.array([0.0, 0.0, finals[i, 2], finals[i, 3], finals[i, 4], 0.0, 0.0])
            for i in chosen]


def main():
    args = parse_args()
    if args.subgoal in ("gdm", "specaccept", "unified", "specpaired", "specgcidm") and not args.gdm_ckpt:
        raise ValueError(f"--subgoal {args.subgoal} requires --gdm-ckpt <trained planner checkpoint>")
    if args.subgoal == "specgcidm" and not args.gcidm_ckpt:
        raise ValueError("--subgoal specgcidm requires --gcidm-ckpt (amortised executor)")
    if args.subgoal == "regressor" and not args.regressor_ckpt:
        raise ValueError("--subgoal regressor requires --regressor-ckpt (train_regressor.py)")
    if args.subgoal == "dspark":
        if not args.gdm_ckpt:
            raise ValueError("--subgoal dspark requires --gdm-ckpt (frozen drafter)")
        if not args.no_refine and not args.dspark_ckpt:
            raise ValueError("--subgoal dspark requires --dspark-ckpt (refiner/confidence); "
                             "or pass --no-refine to use the raw draft block")
    if args.mode == "random_init" and args.subgoal not in ("gdm", "specaccept"):
        raise ValueError("--mode random_init supports --subgoal gdm/specaccept (closed-loop "
                         "sources, same interface); oracle/baseline have no natural subgoal "
                         "chain for a synthetic start (see --mode help)")

    encoder._ensure_swm_importable(args.swm_src)
    import stable_worldmodel as swm
    from stable_worldmodel.envs.pusht.env import PushT

    goal_offset = args.goal_offset if args.goal_offset is not None else (25 if args.mode == "short" else 75)
    eval_budget = args.eval_budget or (50 if args.mode == "short" else (150 if args.mode == "long" else 300))
    cem_seed = args.cem_seed if args.cem_seed is not None else args.seed  # == eval.py ${seed}

    # ---- frozen model + cost model
    model = encoder.load_lewm(source=args.source, encoder_id=args.encoder_id,
                              local_dir=args.local_dir, swm_src=args.swm_src,
                              device=args.device)
    # random = swm's uniform action-space policy, LeWM Table I's chance floor.
    # All solver-free arms take the plain cost model and never build a subgoal.
    is_random = args.subgoal == "random"
    is_gcidm = args.subgoal == "gcidm"
    is_specgcidm = args.subgoal == "specgcidm"
    is_baseline = args.subgoal in ("baseline", "random", "gcidm", "specgcidm")
    # baseline mode = raw LeWM + goal image (== eval.py), to isolate harness vs injection
    cost_model = model if is_baseline else SubgoalCostModel(model)
    print(f"[model] frozen LeWM ({sum(p.numel() for p in model.parameters())/1e6:.2f}M), "
          f"device={args.device}, training={model.training}, subgoal={args.subgoal}")

    # GDM planner (goal-free) loaded once; the per-env source is rebuilt per run
    gdm_planner = None
    if args.subgoal in ("gdm", "dspark", "specaccept", "unified", "specpaired", "specgcidm"):
        from specaccept.drafter import load_gdm_planner, count_params
        gdm_planner = load_gdm_planner(args.gdm_ckpt, device=args.device)
        gdm_planner.noise_scale = args.gdm_noise_scale
        d = gdm_planner.diffusion
        # report the sampler that ACTUALLY runs (GDMPlanner branches on the ckpt's
        # sampler field): ddpm ancestral visits all T steps and ignores --gdm-steps
        samp = (f"ddpm_ancestral/all-{d.timesteps}-steps (--gdm-steps ignored)"
                if d.sampler == "ddpm" else f"ddim/{args.gdm_steps}-steps")
        print(f"[gdm] planner from {args.gdm_ckpt}: head={count_params(gdm_planner.model)/1e6:.2f}M, "
              f"N={gdm_planner.cfg.n_future} WG={gdm_planner.cfg.wg} gamma={args.gdm_noise_scale} "
              f"T={d.timesteps} sampler={samp} param={d.parameterization} "
              f"schedule={d.schedule} norm={gdm_planner.normalization}")

    # ---- dataset (use local path so it works on CPU and box)
    from stable_worldmodel.data.formats.hdf5 import HDF5Dataset
    keys = ["action", "proprio", "state"]
    dataset = HDF5Dataset(path=args.h5, keys_to_cache=keys)

    # ---- transform + process (mirror eval.py)
    try:
        tf = img_transform()
    except Exception:
        tf = img_transform_fallback()
    transform = {"pixels": tf, "goal": tf}
    process = build_process(dataset, keys)

    # ---- episode sampling
    #   paper FF-JEPA protocol (default): the final `goal_offset` steps of each
    #   episode with the LAST frame as the goal, for short (25) and long (75)
    #   alike. `--start random` falls back to eval.py's common 25-step setting
    #   (random valid start, goal = start+goal_offset). random_init samples no
    #   episode at all, see the world.evaluate(episodes=...) branch below.
    use_final = (args.mode == "long") or (args.start == "final")
    episodes_idx = start_steps = None
    random_init_goals = None
    if args.mode == "random_init":
        # sampled once so every score_mode x angle iteration below scores the
        # same (random start, random goal) pairs.
        random_init_goals = sample_random_init_goals(args.h5, args.num_eval, args.seed,
                                                      angle_deg=20.0)
        print(f"[eval] mode=random_init subgoal={args.subgoal} num_eval={args.num_eval} "
              f"budget={eval_budget}, per-episode random goal (paper III-A protocol)")
    elif args.episodes_file:
        import json
        with open(args.episodes_file) as f:
            payload = json.load(f)
        pairs = payload["episodes"]
        episodes_idx = [p[0] for p in pairs]
        start_steps = [p[1] for p in pairs]
        print(f"[episodes-file] {args.episodes_file}: {len(pairs)} precomputed pairs "
              f"(goal_offset={payload.get('goal_offset')}, seed={payload.get('seed')}, "
              f"eval_filter={payload.get('eval_filter')}); sample_long/short bypassed")
    else:
        ep_mask = None
        if args.eval_filter != "none":
            ang = 20.0 if args.eval_filter == "success20" else 5.0
            ep_mask = success_mask(args.h5, ang)
            print(f"[eval-filter] success@{ang:g}deg: {int(ep_mask.sum())}/{len(ep_mask)} "
                  f"episodes eligible (goal frame reaches the canonical target)")
        if use_final:
            episodes_idx, start_steps = sample_long(args.h5, args.num_eval, goal_offset, args.seed,
                                                    ep_mask=ep_mask)
        else:
            episodes_idx, start_steps = sample_short(args.h5, args.num_eval, goal_offset, args.seed,
                                                     ep_mask=ep_mask)
    if args.mode != "random_init":
        print(f"[eval] mode={args.mode} start={'final' if use_final else 'random'} "
              f"num_eval={args.num_eval} goal_offset={goal_offset} budget={eval_budget} "
              f"eval_filter={args.eval_filter}")
        print(f"[eval] episodes_idx[:5]={episodes_idx[:5]} start_steps[:5]={start_steps[:5]}")

    # ---- oracle subgoal table (oracle mode only; gdm samples closed-loop, baseline uses goal image)
    table = None
    if args.subgoal == "oracle":
        table = build_oracle_table(args.h5, model, episodes_idx, start_steps, goal_offset,
                                   stride=args.stride, device=args.device)
        print(f"[oracle] built {len(table)} per-env subgoal tables; "
              f"K (subgoals/ep) = {table[0].shape[0]}")

    config = swm.PlanConfig(horizon=args.horizon, receding_horizon=args.receding_horizon,
                            action_block=args.action_block)
    callables = [
        {"method": "_set_state", "args": {"state": {"value": "state", "in_dataset": True}}},
        {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state", "in_dataset": True}}},
    ]

    PolicyCls = make_ffjepa_policy(swm.policy.WorldModelPolicy)

    score_modes = ["env", "block"] if args.score == "both" else [args.score]
    results = {}
    trace_dump = {}
    for score_mode in score_modes:
        for angle in args.angles:
            patch_eval_state(PushT, angle, score_mode)
            # random_init has no separate eval_budget in world.evaluate(); its
            # max_episode_steps is the real cap, so don't double it like the
            # dataset-driven modes (whose 2x is just a loose outer safety net).
            ep_max_steps = eval_budget if args.mode == "random_init" else 2 * eval_budget
            world = swm.World(env_name="swm/PushT-v1", num_envs=args.num_eval,
                              max_episode_steps=ep_max_steps, image_shape=(224, 224))
            solver = swm.solver.CEMSolver(model=cost_model, batch_size=1,
                                          num_samples=args.num_samples, var_scale=args.var_scale,
                                          n_steps=args.n_steps, topk=args.topk,
                                          device=args.device, seed=cem_seed)
            if is_random:
                # seeded via set_policy() from .seed, deterministic at cem_seed like every arm
                policy = swm.policy.RandomPolicy(seed=cem_seed)
            elif is_gcidm:
                from specaccept.gcidm import GCIDMPolicy, load_gcidm
                gci, ascaler, gmeta = load_gcidm(args.gcidm_ckpt, device=args.device)
                policy = GCIDMPolicy(gci, model, budget=eval_budget,
                                     device=args.device, action_scaler=ascaler)
                print(f"[gcidm] {args.gcidm_ckpt}: H_max={gci.h_max} "
                      f"params={sum(p.numel() for p in gci.parameters())/1e6:.2f}M "
                      f"budget={eval_budget} (1 forward/step, no solver)")
            elif is_specgcidm:
                from specaccept.gcidm import load_gcidm
                from specaccept.spec_gcidm import SpecGCIDMPolicy
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
                    # fresh source per run: resets per-env cache + reseeds the
                    # sampling generator so each run is deterministic at cem_seed
                    source = GDMSubgoalSource(gdm_planner, n_envs=args.num_eval,
                                              dim=gdm_planner.cfg.latent_dim,
                                              device=args.device, n_steps=args.gdm_steps,
                                              seed=cem_seed,
                                              record=bool(args.dump_traces))
                elif args.subgoal == "regressor":
                    source = RegressorSubgoalSource(args.regressor_ckpt,
                                                    n_envs=args.num_eval,
                                                    device=args.device,
                                                    record=bool(args.dump_traces))
                elif args.subgoal == "specaccept":
                    source = SpecAcceptSubgoalSource(gdm_planner, n_envs=args.num_eval,
                                                     device=args.device,
                                                     n_steps=args.gdm_steps, seed=cem_seed,
                                                     tau=args.accept_tau,
                                                     draft_noise=args.draft_noise,
                                                     random_reject=args.random_reject,
                                                     record=bool(args.dump_traces))
                    if args.random_reject is not None:
                        print(f"[randreject] coin control p={args.random_reject} "
                              f"(matched-rate; latent test overridden)")
                elif args.subgoal == "specpaired":
                    # verify-space control (C1/C2): one paired drafter, the
                    # accept test reads --verify-half; everything else fixed.
                    from specaccept.paired import (PairedEncoder,
                                                   SpecAcceptPairedSource,
                                                   load_dinov2)
                    penc = PairedEncoder(model, load_dinov2(device=args.device),
                                         device=args.device)
                    source = SpecAcceptPairedSource(
                        gdm_planner, penc, n_envs=args.num_eval,
                        device=args.device, n_steps=args.gdm_steps,
                        seed=cem_seed, tau=args.accept_tau,
                        record=bool(args.dump_traces),
                        verify_half=args.verify_half)
                    print(f"[specpaired] verify-half={args.verify_half} "
                          f"tau={args.accept_tau} N={source.N}")
                elif args.subgoal == "lerp":
                    source = LerpSubgoalSource(n_envs=args.num_eval, device=args.device,
                                               frac=args.lerp_frac)
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
                elif args.subgoal == "dspark":
                    source = DSparkSubgoalSource(gdm_planner, args.dspark_ckpt,
                                                 n_envs=args.num_eval, device=args.device,
                                                 n_steps=args.gdm_steps, seed=cem_seed,
                                                 commit=args.commit, theta=args.commit_theta,
                                                 fixed_k=args.commit_k, use_sts=not args.no_sts,
                                                 chain_n=args.chain_n, refine=not args.no_refine)
                else:
                    source = OracleSubgoalSource(table, device=args.device)
                policy = PolicyCls(cost_model=cost_model, subgoal_source=source,
                                   time_instrument=args.time_instrument,
                                   dump_frames=bool(args.dump_frames_dir) or args.dump_strip > 0,
                                   dump_strip=args.dump_strip,
                                   solver=solver, config=config, process=process, transform=transform)
            world.set_policy(policy)
            if args.mode == "random_init":
                # synthetic random start (PushT's own reset() sampling); goal_state
                # is per-env (random_init_goals), not one fixed canonical target.
                options = [{"goal_state": gv} for gv in random_init_goals]
                eval_kwargs = {}
                if args.dump_frames_dir:
                    # episodic eval defaults to reset_mode='auto', which restarts
                    # finished envs until `episodes` complete and so can reuse an
                    # env slot for a second, DIFFERENT episode. That would desync
                    # the frame capture (start frame from episode 1, last frame
                    # from episode 2). 'wait' freezes each env after its own first
                    # episode, guaranteeing one episode per env slot.
                    eval_kwargs["reset_mode"] = "wait"
                metrics = world.evaluate(episodes=args.num_eval, seed=cem_seed,
                                         options=options, **eval_kwargs)
            else:
                metrics = world.evaluate(
                    dataset=dataset, start_steps=list(start_steps), goal_offset=goal_offset,
                    eval_budget=eval_budget, episodes_idx=list(episodes_idx), callables=callables,
                )
            sr = metrics["success_rate"]
            n_succ = int(metrics["episode_successes"].sum())
            results[(score_mode, angle)] = (sr, n_succ)
            print(f"[RESULT] score={score_mode} angle={angle:g}deg  SR={sr:.2f}%  ({n_succ}/{args.num_eval})")
            if is_gcidm:
                # one MLP forward per step against CEM's full solve; per episode
                print(f"[gcidm-cost] decisions={policy.n_calls / max(args.num_eval, 1):.1f} "
                      f"total={policy.n_calls} (1 MLP forward each, no solver)")
            if is_specgcidm:
                b = policy.n_redraft + policy.n_advance
                print(f"[specgcidm-cost] exec_forwards={policy.n_calls} "
                      f"redrafts={policy.n_redraft} advances={policy.n_advance} "
                      f"rejects={policy.n_reject} call_ratio={policy.n_redraft / max(b, 1):.3f} "
                      f"routed={policy.n_routed} arrived={policy.n_arrive} "
                      f"draft_s={policy.t_draft:.2f} exec_s={policy.t_exec:.2f} "
                      f"probe_s={policy.t_probe:.2f}")
            if args.dump_frames_dir and not is_baseline and getattr(policy, "dump_frames", False):
                import json
                from pathlib import Path
                from PIL import Image
                outdir = Path(args.dump_frames_dir) / f"{score_mode}_{angle:g}"
                outdir.mkdir(parents=True, exist_ok=True)
                # world.terminateds is in env_idx order and, under
                # reset_mode='wait', every env terminates exactly once before
                # evaluate() returns, so it is the per-episode success flag
                # aligned with the frames captured at the SAME env index.
                # metrics['episode_successes'] follows COMPLETION order instead,
                # so it cannot be used here.
                term = np.asarray(world.terminateds).astype(bool)
                manifest = []
                for i in range(args.num_eval):
                    for tag, frame in (("start", policy._frame_start[i]),
                                       ("goal", policy._frame_goal[i]),
                                       ("last", policy._frame_last[i])):
                        if frame is None:
                            continue
                        Image.fromarray(np.asarray(frame).astype(np.uint8)).save(
                            outdir / f"ep{i:02d}_{tag}.png")
                    manifest.append({"env": i, "success": bool(term[i])})
                with open(outdir / "manifest.json", "w") as f:
                    json.dump(manifest, f, indent=2)
                print(f"[frames] saved {sum(m['success'] for m in manifest)}/{len(manifest)} "
                      f"episode frame-sets (start/goal/last PNG) to {outdir}")
            if args.dump_strip and not is_baseline and getattr(policy, "_strip", None):
                import json as _json
                from pathlib import Path
                outdir = Path(args.dump_strip_out)
                outdir.mkdir(parents=True, exist_ok=True)
                fn = outdir / (f"strip_pusht_{args.subgoal}_{score_mode}_{angle:g}"
                               f"_t{goal_offset}_seed{args.seed}.npz")
                # dataset eval assigns one episode per env slot; episode_successes
                # follows the episodes_idx slot order (same convention as the
                # cross-room split diagnostic in the tworoom driver)
                succ = np.asarray(metrics["episode_successes"]).astype(bool)
                payload = {"success": succ,
                           "meta": np.array(_json.dumps({
                               "env": "pusht", "subgoal": args.subgoal,
                               "score": score_mode, "angle": float(angle),
                               "goal_offset": int(goal_offset), "seed": int(args.seed),
                               "tau": float(getattr(args, "accept_tau", float("nan"))),
                               "sr": float(sr)}))}
                for i, fr_list in enumerate(policy._strip):
                    if not fr_list:
                        continue
                    payload[f"ep{i}_tags"] = np.array([t for t, _ in fr_list], np.int64)
                    payload[f"ep{i}_frames"] = np.stack(
                        [f for _, f in fr_list]).astype(np.uint8)
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
                print(f"[strip] saved {sum(1 for s in policy._strip if s)} episode "
                      f"strips -> {fn}")
            if args.dump_traces:
                rec = {"score": score_mode, "angle": float(angle), "sr": float(sr),
                       "successes": np.asarray(metrics["episode_successes"]).astype(bool).tolist()}
                if args.mode != "random_init":
                    rec["episodes_idx"] = list(episodes_idx)
                    rec["start_steps"] = list(start_steps)
                if not is_baseline and getattr(source, "record", False):
                    rec["trace"] = source.trace  # list of {replan_idx, z_cond, z_next}
                    rec["cal"] = getattr(source, "cal", [])
                trace_dump[f"{score_mode}_{angle:g}"] = rec
            if args.subgoal == "specaccept":
                total = source.n_redraft + source.n_advance
                print(f"[specaccept] tau={args.accept_tau} gdm_steps={args.gdm_steps} "
                      f"re-drafts={source.n_redraft} advances={source.n_advance} "
                      f"rejects={source.n_reject} "
                      f"call_ratio={source.n_redraft / max(total, 1):.3f} "
                      f"(every-step=1.000; lower = fewer diffusion calls)")
            if args.subgoal == "specpaired":
                print(source.stats())
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
            if args.subgoal == "dspark":
                cd = source.commit_depths
                ratio = source.n_redraft / max(source.n_advance, 1)
                print(f"[dspark] commit={args.commit} theta={args.commit_theta} "
                      f"re-drafts={source.n_redraft} advances={source.n_advance} "
                      f"mean_commit_depth={np.mean(cd) if cd else 0:.2f} "
                      f"redraft/advance={ratio:.3f} (GDM=1.000; lower=fewer diffusion calls)")
            if args.time_instrument and not is_baseline:
                t_d, t_c = policy.t_drafter, policy.t_cem
                t_tot = t_d + t_c
                frac = 100.0 * t_d / t_tot if t_tot > 0 else 0.0
                per_ep = {
                    "drafter_ms": 1000.0 * t_d / max(args.num_eval, 1),
                    "cem_ms": 1000.0 * t_c / max(args.num_eval, 1),
                    "total_ms": 1000.0 * t_tot / max(args.num_eval, 1),
                }
                print(f"[timing] score={score_mode} angle={angle:g}deg "
                      f"drafter={t_d:.3f}s cem={t_c:.3f}s total={t_tot:.3f}s "
                      f"drafter_frac={frac:.2f}% per_episode_ms={per_ep} "
                      f"timed_steps={policy._timed_steps}")
            world.close()

    if args.dump_traces:
        from pathlib import Path
        Path(args.dump_traces).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"args": vars(args), "records": trace_dump}, args.dump_traces)
        print(f"[traces] saved {args.dump_traces}")

    print("\n==== FF-JEPA eval summary ====")
    print(f"mode={args.mode} subgoal={args.subgoal} num_eval={args.num_eval} "
          f"seed={args.seed} cem_seed={cem_seed} eval_filter={args.eval_filter}")
    for (sm, angle), (sr, n) in results.items():
        print(f"  score={sm:5s} {angle:>4g}deg : {sr:6.2f}%  ({n}/{args.num_eval})")


if __name__ == "__main__":
    main()
