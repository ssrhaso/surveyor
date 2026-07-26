"""Arm-aware DINO-WM PushT planning runner: ARM=flat|spec, TAU, N_EVALS,
PLAN_SEED, MAX_ITER via env. spec swaps the planner _target_ to
specaccept_dinowm.SpecMPCPlanner (drafted-subgoal serving + verification);
flat is byte-identical to run_plan_pusht.py."""
import os
import sys

REPO = "/lustre/home/ha676/dino_wm"
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch  # noqa: E402

try:
    torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
except Exception as e:
    print(f"[runner] dinov2 hub preload: {e}", flush=True)

from plan import build_plan_cfg_dicts, planning_main  # noqa: E402

ARM = os.environ.get("ARM", "flat")
outdir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else f"plan_outputs/isca_{ARM}")
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
cfg["planner"]["max_iter"] = int(os.environ.get("MAX_ITER", 12))

if ARM == "spec":
    import specaccept_dinowm  # noqa: F401  (register module for hydra)
    cfg["planner"]["_target_"] = "specaccept_dinowm.SpecMPCPlanner"
    cfg["planner"]["gdm_ckpt"] = os.environ.get(
        "GDM_CKPT", "/lustre/home/ha676/le-wm/vjepa2/gdm_dinowm_pusht_s25.pt")
    cfg["planner"]["accept_tau"] = float(os.environ.get("TAU", 0.121))
    cfg["planner"]["gdm_steps"] = int(os.environ.get("GDM_STEPS", 8))
    cfg["planner"]["spec_seed"] = cfg["seed"]

os.makedirs(outdir, exist_ok=True)
os.chdir(outdir)
print(f"[runner] arm={ARM} n_evals={cfg['n_evals']} seed={cfg['seed']} "
      f"goal_H={cfg['goal_H']} max_iter={cfg['planner']['max_iter']} "
      f"tau={cfg['planner'].get('accept_tau')} out={outdir}", flush=True)
planning_main(cfg_dict=cfg)
