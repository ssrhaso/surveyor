#!/usr/bin/env bash
#SBATCH --job-name=dwm-horizon
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --array=0-15%4
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_horizon_%A_%a.log
# ============ DINO-WM AS A SECOND-ARCHITECTURE HORIZON BASELINE =============
# Supervisor request (2026-08-07): put DINO-WM in the paper as a baseline.
# WHAT THIS CAN AND CANNOT BE. DINO-WM ships its own PushT (env, pusht_noise
# dataset, criterion, goal_source=dset), so its numbers CANNOT occupy a cell
# in the headline grid -- episodes, goals and criterion all differ. On its own
# protocol it CAN answer the paper's question: does a second frozen-encoder
# latent world-model planner also collapse as the goal outruns its planning
# window? SEPARATE PANEL in DINO-WM's own units, never a spliced row.
# WHY THE BANKED RUNS COULD NOT BE REUSED (checked 2026-08-07): the banked
# goal_H=5 vs 15 flat pair reads 0.88 -> 0.06 at face value but is NOT
# comparable: goal_H=5 converged (SR plateau over 7 MPC iterations) while
# goal_H=15 hit the 12h walltime after TWO iterations -- a truncation
# artifact, the same class this project has retracted before. Discarded.
# THIS ARRAY: matched budget at every horizon. MAX_ITER=4 for ALL cells
# (4 iters x 5 taken actions x frameskip 5 = 100 env steps), so no cell can be
# scheduler-truncated and each horizon gets an identical replan budget >= our
# 2t convention; a collapse here is a lookahead limit, not a starved budget.
#   goal_H in {5, 10, 15, 25} x seeds {99,100,101,102}
# n_evals=10 (their plan.yaml default) is a COST decision (~6h/iteration once
# episodes stop terminating early), a deviation from their released 50 that
# must be stated wherever these numbers appear. goal_H=5 re-runs inside the
# array as an internal reference; the full-protocol anchor (ours 0.86 vs their
# 0.90) certifies the stack separately. No bar frozen: characterisation only.
# ==============================================================================
set -euo pipefail
export DATASET_DIR=/lustre/home/ha676/data/dinowm/datasets
export TORCH_HOME=/lustre/home/ha676/torch_cache
export WANDB_MODE=disabled SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/dino_wm
source .venv/bin/activate

HS=(5 10 15 25)
ID=$SLURM_ARRAY_TASK_ID
export GOAL_H=${HS[$((ID / 4))]}
export PLAN_SEED=$((99 + ID % 4))
export N_EVALS=10
export MAX_ITER=4

OUT=plan_outputs/horizon_h${GOAL_H}_s${PLAN_SEED}
mkdir -p "$OUT"

# The n_evals deviation is recorded IN THE RUN CONFIG so a cell can never be
# quoted without its provenance; the harvester refuses to pool cells whose
# protocols disagree.
cat > "$OUT/run_config.json" <<JSON
{
  "arm": "flat",
  "env": "dinowm_pusht",
  "goal_H": ${GOAL_H},
  "seed": ${PLAN_SEED},
  "n_evals": ${N_EVALS},
  "released_n_evals": 50,
  "max_iter": ${MAX_ITER},
  "env_steps_budget": $((MAX_ITER * 5 * 5)),
  "deviations": [
    "n_evals=${N_EVALS} against the released protocol's 50; a cost decision forced by ~6h/iteration once episodes stop terminating early. Cells are DIRECTIONAL ONLY (per-cell SE approx 5-6pp): they may support 'flat collapses on a second architecture too' and may never carry a paired margin.",
    "max_iter=${MAX_ITER} fixed at every goal_H so each horizon gets an identical replan budget; the released config leaves max_iter null (plan forever)."
  ],
  "anchor_reference": {
    "note": "goal_H=5 is re-run inside this array at the reduced n so the curve has an internal reference; that cell is compared against the banked FULL-n run below.",
    "banked_run": "plan_outputs/isca_battery_flat_t0_s99",
    "banked_n_evals": 50,
    "banked_sr_plateau": 0.86,
    "banked_converged": true
  },
  "prereg": "prereg/2026-08-07_baselines.md"
}
JSON

echo "=== dwm-horizon goal_H=$GOAL_H seed=$PLAN_SEED n_evals=$N_EVALS max_iter=$MAX_ITER ==="
python run_plan_pusht.py "$OUT"
RC=$?
echo "=== DWM HORIZON DONE h=$GOAL_H seed=$PLAN_SEED rc=$RC ==="

# Walltime-truncation check, the failure mode that voided the banked goal_H=15
# pair: a cell is only complete if its SR trace reached max_iter. Recorded in the
# log AND in the config so the harvester can flag it without re-deriving.
python - "$OUT" "$MAX_ITER" <<'PY'
import json, sys, pathlib
out, max_iter = pathlib.Path(sys.argv[1]), int(sys.argv[2])
lp = out / "logs.json"
rows = []
if lp.exists():
    rows = [json.loads(l) for l in lp.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "mpc/success_rate" in r]
steps = len(rows)
trunc = steps < max_iter
print(f"[truncation-check] logged_steps={steps} max_iter={max_iter} "
      f"truncated={trunc} sr_trace={[r['mpc/success_rate'] for r in rows]}")
cp = out / "run_config.json"
cfg = json.loads(cp.read_text())
cfg["logged_steps"] = steps
cfg["truncated_by_walltime"] = trunc
cfg["sr_trace"] = [r["mpc/success_rate"] for r in rows]
cp.write_text(json.dumps(cfg, indent=2))
PY
