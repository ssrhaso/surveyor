"""FF-JEPA evaluation driver (Task B oracle test; Task C GDM later).

Runs FFJEPAPolicy + SubgoalCostModel through the validated swm harness and
reports success rate at BOTH 20 deg (headline) and 5 deg (supplementary), using
the same eval_state pin as the baseline run_eval.py.

Subgoal source:
  --subgoal oracle : the TRUE demo latents at start, start+25, ..., goal frame
                     (subgoal[1] == the baseline's goal latent in short-horizon).
  --subgoal gdm    : (Task C) a trained latent planner G. Not yet wired.

Horizon modes (short reproduces the baseline's episode sampling so the oracle is
directly comparable):
  short : goal_offset 25, budget 50, eval.py-style random valid starts.
  long  : goal_offset 75, budget 150, per-episode start = ep_len-1-75 (last 75).

CPU smoke (tiny CEM, local model, 2 envs):
  SDL_VIDEODRIVER=dummy python -m ffjepa.eval_ffjepa --source local \
    --local-dir ../drift_probe/model --swm-src ../lewm-investigation/stable-worldmodel \
    --h5 ../drift_probe/expert/pusht_expert_train.h5 --device cpu \
    --num-eval 2 --num-samples 8 --n-steps 2 --mode short --angles 20

Box (A100), short, n=32, matches baseline episode set (seed 42):
  cd ~/le-wm && STABLEWM_HOME=$HOME/.stable-wm SDL_VIDEODRIVER=dummy \
    python -m ffjepa.eval_ffjepa --source pretrained --encoder-id quentinll/lewm-pusht \
    --h5 $HOME/.stable-wm/datasets/pusht_expert_train.h5 --device cuda \
    --num-eval 32 --mode short --angles 20 5
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from ffjepa import lewm_io
from ffjepa.subgoal_planner import (SubgoalCostModel, OracleSubgoalSource,
                                    build_oracle_table, make_ffjepa_policy)


# success-criterion pin (mirrors run_eval.py): pos<20 AND angle<ANGLE_DEG
def patch_eval_state(PushT, angle_deg, score_mode="env"):
    """score_mode: 'env' = pos_diff over 4-D agent+block (env-native, baseline-
    comparable); 'block' = block xy only (the paper's criterion, agent ignored)."""
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
    p.add_argument("--subgoal", choices=["oracle", "gdm", "baseline"], default="oracle")
    p.add_argument("--gdm-ckpt", default=None, help="(Task C) trained planner checkpoint")
    p.add_argument("--mode", choices=["short", "long"], default="short")
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


def sample_short(h5, num_eval, goal_offset, seed):
    """Replicate eval.py's random-valid-start sampling (baseline-comparable)."""
    import h5py
    with h5py.File(h5, "r") as f:
        episode_idx = f["episode_idx"][:]
        step_idx = f["step_idx"][:]
        ep_len = f["ep_len"][:]
    ep_len_per_row = ep_len[episode_idx]
    max_start_per_row = ep_len_per_row - goal_offset - 1
    valid_indices = np.nonzero(step_idx <= max_start_per_row)[0]
    g = np.random.default_rng(seed)
    chosen = g.choice(len(valid_indices) - 1, size=num_eval, replace=False)
    rows = np.sort(valid_indices[chosen])
    episodes_idx = episode_idx[rows].tolist()
    start_steps = step_idx[rows].tolist()
    return episodes_idx, start_steps


def sample_long(h5, num_eval, last_n, seed):
    """Per-episode start = ep_len-1-last_n (the 'last last_n steps'); goal=last frame."""
    import h5py
    with h5py.File(h5, "r") as f:
        ep_len = f["ep_len"][:]
    valid_eps = np.nonzero(ep_len > last_n + 1)[0]
    g = np.random.default_rng(seed)
    eps = np.sort(g.choice(valid_eps, size=num_eval, replace=False))
    episodes_idx = eps.tolist()
    start_steps = [int(ep_len[e] - 1 - last_n) for e in eps]
    return episodes_idx, start_steps


def main():
    args = parse_args()
    if args.subgoal == "gdm":
        raise NotImplementedError("gdm subgoal source is Task C; not wired yet")

    lewm_io._ensure_swm_importable(args.swm_src)
    import stable_worldmodel as swm
    from stable_worldmodel.envs.pusht.env import PushT

    goal_offset = 25 if args.mode == "short" else 75
    eval_budget = args.eval_budget or (50 if args.mode == "short" else 150)
    cem_seed = args.cem_seed if args.cem_seed is not None else args.seed  # == eval.py ${seed}

    # -- frozen model + cost model
    model = lewm_io.load_lewm(source=args.source, encoder_id=args.encoder_id,
                              local_dir=args.local_dir, swm_src=args.swm_src,
                              device=args.device)
    is_baseline = args.subgoal == "baseline"
    # baseline mode = raw LeWM + goal image (== eval.py), to isolate harness vs injection
    cost_model = model if is_baseline else SubgoalCostModel(model)
    print(f"[model] frozen LeWM ({sum(p.numel() for p in model.parameters())/1e6:.2f}M), "
          f"device={args.device}, training={model.training}, subgoal={args.subgoal}")

    # -- dataset (use local path so it works on CPU and box)
    from stable_worldmodel.data.formats.hdf5 import HDF5Dataset
    keys = ["action", "proprio", "state"]
    dataset = HDF5Dataset(path=args.h5, keys_to_cache=keys)

    # -- transform + process (mirror eval.py)
    try:
        tf = img_transform()
    except Exception:
        tf = img_transform_fallback()
    transform = {"pixels": tf, "goal": tf}
    process = build_process(dataset, keys)

    # -- episode sampling
    if args.mode == "short":
        episodes_idx, start_steps = sample_short(args.h5, args.num_eval, goal_offset, args.seed)
    else:
        episodes_idx, start_steps = sample_long(args.h5, args.num_eval, goal_offset, args.seed)
    print(f"[eval] mode={args.mode} num_eval={args.num_eval} goal_offset={goal_offset} "
          f"budget={eval_budget}")
    print(f"[eval] episodes_idx[:5]={episodes_idx[:5]} start_steps[:5]={start_steps[:5]}")

    # -- subgoal source (oracle); skipped for baseline mode
    table = None
    if not is_baseline:
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
    for score_mode in score_modes:
        for angle in args.angles:
            patch_eval_state(PushT, angle, score_mode)
            world = swm.World(env_name="swm/PushT-v1", num_envs=args.num_eval,
                              max_episode_steps=2 * eval_budget, image_shape=(224, 224))
            solver = swm.solver.CEMSolver(model=cost_model, batch_size=1,
                                          num_samples=args.num_samples, var_scale=args.var_scale,
                                          n_steps=args.n_steps, topk=args.topk,
                                          device=args.device, seed=cem_seed)
            if is_baseline:
                policy = swm.policy.WorldModelPolicy(
                    solver=solver, config=config, process=process, transform=transform)
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
            results[(score_mode, angle)] = (sr, n_succ)
            print(f"[RESULT] score={score_mode} angle={angle:g}deg  SR={sr:.2f}%  ({n_succ}/{args.num_eval})")
            world.close()

    print("\n==== FF-JEPA eval summary ====")
    print(f"mode={args.mode} subgoal={args.subgoal} num_eval={args.num_eval} "
          f"seed={args.seed} cem_seed={cem_seed}")
    for (sm, angle), (sr, n) in results.items():
        print(f"  score={sm:5s} {angle:>4g}deg : {sr:6.2f}%  ({n}/{args.num_eval})")


if __name__ == "__main__":
    main()
