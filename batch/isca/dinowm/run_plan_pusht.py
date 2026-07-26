"""Inline DINO-WM PushT planning on ISCA: bypasses plan.py's submitit launch
(hardcoded h100/qos for their cluster) and calls planning_main directly with
the plan_pusht.yaml protocol. Overrides via env: N_EVALS, PLAN_SEED, GOAL_H."""
import os
import sys

REPO = "/lustre/home/ha676/dino_wm"
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch  # noqa: E402

try:  # make the hub 'dinov2' package importable before ckpt unpickling
    torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
except Exception as e:
    print(f"[runner] dinov2 hub preload: {e}", flush=True)

from plan import build_plan_cfg_dicts, planning_main  # noqa: E402

outdir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "plan_outputs/isca_run")
cfg = build_plan_cfg_dicts(
    plan_cfg_path=f"{REPO}/conf/plan_pusht.yaml",
    ckpt_base_path="/lustre/home/ha676/data/dinowm/checkpoints",
    model_name="pusht",
    model_epoch="latest",
    planner=["mpc_cem"],
    goal_source=["dset"],
    goal_H=[int(os.environ.get("GOAL_H", 5))],
    alpha=[1],
)[0]
cfg["saved_folder"] = outdir
cfg["wandb_logging"] = False
if os.environ.get("N_EVALS"):
    cfg["n_evals"] = int(os.environ["N_EVALS"])
if os.environ.get("PLAN_SEED"):
    cfg["seed"] = int(os.environ["PLAN_SEED"])
# Their released config leaves planner.max_iter null = plan FOREVER until every
# episode succeeds (impossible episodes spin to the slurm timeout). Cap at the
# env's episode budget: 12 iters x 5 taken actions x frameskip 5 = 300 env
# steps = max_episode_steps. Same cap for every arm; protocol interpretation.
cfg["planner"]["max_iter"] = int(os.environ.get("MAX_ITER", 12))
os.makedirs(outdir, exist_ok=True)
os.chdir(outdir)   # logs.json / plan_targets.pkl land here (planning_main_in_dir mimic)
print(f"[runner] n_evals={cfg['n_evals']} seed={cfg['seed']} goal_H={cfg['goal_H']} "
      f"planner={cfg['planner']['name']} out={outdir}", flush=True)
planning_main(cfg_dict=cfg)
