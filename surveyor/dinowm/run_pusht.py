"""Arm-aware DINO-WM PushT planning runner: ARM=flat|spec, TAU, N_EVALS,
PLAN_SEED, MAX_ITER via env. spec swaps the planner _target_ to the drafted
subgoal server in planner.py; flat calls their planning_main unmodified, so it
reproduces their released protocol exactly.

Bypasses plan.py's submitit launch (hardcoded to their cluster's partition and
QoS) and calls planning_main directly with the plan_pusht.yaml protocol."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = os.environ.get("DINOWM_REPO", os.path.expanduser("~/dino_wm"))
DATA = os.environ.get("DINOWM_DATA", os.path.expanduser("~/data/dinowm"))
sys.path.insert(0, REPO)
sys.path.insert(0, str(ROOT))
os.chdir(REPO)

import torch  # noqa: E402

try:
    torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
except Exception as e:
    print(f"[runner] dinov2 hub preload: {e}", flush=True)

from plan import build_plan_cfg_dicts, planning_main  # noqa: E402

ARM = os.environ.get("ARM", "flat")
outdir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else f"plan_outputs/{ARM}")
cfg = build_plan_cfg_dicts(
    plan_cfg_path=f"{REPO}/conf/plan_pusht.yaml",
    ckpt_base_path=f"{DATA}/checkpoints",
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
# Their released config leaves planner.max_iter null, which plans forever
# until every episode succeeds (impossible episodes spin to the job timeout).
# Cap at the env's episode budget: 12 iters x 5 taken actions x frameskip 5 =
# 300 env steps = max_episode_steps. Same cap for every arm; a protocol
# interpretation, applied identically to flat and drafted.
cfg["planner"]["max_iter"] = int(os.environ.get("MAX_ITER", 12))

if ARM == "spec":
    from surveyor.dinowm import planner  # noqa: F401  (register for hydra)
    cfg["planner"]["_target_"] = "surveyor.dinowm.planner.SurveyorMPCPlanner"
    cfg["planner"]["gdm_ckpt"] = os.environ.get(
        "GDM_CKPT", f"{DATA}/gdm_dinowm_pusht_s25.pt")
    cfg["planner"]["accept_tau"] = float(os.environ.get("TAU", 0.121))
    cfg["planner"]["gdm_steps"] = int(os.environ.get("GDM_STEPS", 8))
    cfg["planner"]["spec_seed"] = cfg["seed"]
    # autopsy serve modes (prereg/2026-07-26_dinowm_instrument.md AUTOPSY REGISTRATION):
    # draft (production) | goal (D1 tautology) | snap (D2 draft-and-snap)
    cfg["planner"]["spec_serve"] = os.environ.get("SPEC_SERVE", "draft")
    if os.environ.get("SNAP_DIR"):
        cfg["planner"]["snap_dir"] = os.environ["SNAP_DIR"]
        cfg["planner"]["snap_max"] = int(os.environ.get("SNAP_MAX", 6000))
    if os.environ.get("GOAL_GATE") == "1":
        cfg["planner"]["goal_gate"] = True

os.makedirs(outdir, exist_ok=True)
os.chdir(outdir)
print(f"[runner] arm={ARM} n_evals={cfg['n_evals']} seed={cfg['seed']} "
      f"goal_H={cfg['goal_H']} max_iter={cfg['planner']['max_iter']} "
      f"tau={cfg['planner'].get('accept_tau')} out={outdir}", flush=True)
planning_main(cfg_dict=cfg)
