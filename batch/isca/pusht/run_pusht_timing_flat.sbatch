#!/usr/bin/env bash
#SBATCH --job-name=pusht-timing-f
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_timingF_%A_%a.log
# Instrumented FLAT wall-clock for the cost figure. The bare baseline bypasses
# the instrumented policy, so we use the validated oracle-stride tautology
# (stride > offset => the oracle's single waypoint IS the goal frame => flat-
# equivalent execution through the instrumented path; SR must reproduce the
# flat reference ~13% at t100 as the equivalence check).
#   0: flat-equiv RH=5   1: flat-equiv RH=2
# PushT t=100, s5as100 population, seed 42, n=256, budget 200.
set -euo pipefail
VENV=/lustre/home/ha676/le-wm/.venv
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
cd /lustre/home/ha676/le-wm
mkdir -p logs
export SDL_VIDEODRIVER=dummy
source "$VENV/bin/activate"
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then NAME=flatT_rh5; RH=5; else NAME=flatT_rh2; RH=2; fi
echo "=== pusht timing-flat arm=$NAME t=100 seed=42 (oracle-stride tautology, time-instrument) ==="
python -m specaccept.envs.pusht.eval --subgoal oracle --stride 200 \
  --source pretrained --h5 "$H5" --device cuda \
  --mode long --episodes-file pusht.episodes150s5as100.json \
  --goal-offset 100 --eval-budget 200 \
  --num-eval 256 --score block --angles 20 --seed 42 --cem-seed 42 \
  --horizon ${RH} --receding-horizon ${RH} \
  --time-instrument \
  2>&1 | tee logs/pusht_timingF_${NAME}.log
