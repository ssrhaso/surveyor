#!/usr/bin/env bash
#SBATCH --job-name=gcidm-hsweep
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/lustre/home/ha676/le-wm/logs/gcidm_hsweep_%A_%a.log
# ========= P-HMAX-1: DOES ANY H_max RESCUE GC-IDM AT LONG HORIZON? ==========
# Pre-registered in prereg/2026-08-07_hmax_tradeoff.md BEFORE submission.
# Two banked t=150 (budget 300) points fail for different reasons:
#   H_max=50  served at T=300 (clamped, breaks their rule)   -> 63.38
#   H_max=300 = T             (their rule, correct port)     -> 57.81
# Small H_max incurs clamping shift; large H_max dilutes 1.5M params across
# goal pairs up to 300 steps apart. All five trained checkpoints are served at
# the SAME cell, so the baseline is compared at its best configuration.
# PREDICTION (frozen): max over the five < 75.0% against our arm's 98.1.
# FALSIFIER: any cell >= 90.0% means the collapse is a config artifact and our
# PushT margin is not real -- reported as the headline correction.
# One seed: deterministic on this fixed population (v1's 8 seeds were
# byte-identical), so n_eff = 256 and extra seeds would be replicas.
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
mkdir -p logs
source /lustre/home/ha676/le-wm/.venv/bin/activate

HS=(50 100 150 200 300)
HMAX=${HS[$SLURM_ARRAY_TASK_ID]}
CKPT=gcidm_pusht_h${HMAX}.pt
[ -f "$CKPT" ] || { echo "MISSING $CKPT"; exit 1; }

EPFILE=pusht.episodes150s5.json
N=$(python -c "import json;print(len(json.load(open('$EPFILE'))['episodes']))")
echo "=== P-HMAX-1 pusht t=150 budget=300 H_max=$HMAX n=$N ==="
python -m specaccept.envs.pusht.eval --subgoal gcidm --gcidm-ckpt "$CKPT" \
  --source pretrained --h5 /lustre/home/ha676/data/pusht/pusht_expert_train.h5 \
  --device cuda --mode long --episodes-file $EPFILE \
  --goal-offset 150 --eval-budget 300 --num-eval $N \
  --score block --angles 20 5 --seed 42 --cem-seed 42 \
  2>&1 | tee logs/gcidm_hsweep_pusht_h${HMAX}.log
echo "=== DONE H_max=$HMAX ==="
