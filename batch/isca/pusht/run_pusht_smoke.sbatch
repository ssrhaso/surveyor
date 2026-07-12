#!/usr/bin/env bash
#SBATCH --job-name=pusht-smoke
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_smoke_%j.log
# PushT eval-stack smoke on ISCA: flat LeWM baseline, short mode, n=32,
# 20deg/5deg scoring. No drafter needed. Expect ~90s% SR (box reference:
# baseline short ~93-96 filter-dependent). Non-gating: informational.
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
cd /lustre/home/ha676/le-wm
mkdir -p runs/pusht
export SDL_VIDEODRIVER=dummy
source "$VENV/bin/activate"
echo '=== BASELINE n=32 short (validates env + encoder + harness + scoring) ==='
python -m specaccept.envs.pusht.eval --subgoal baseline \
  --source pretrained --h5 "$H5" --device cuda \
  --mode short --num-eval 32 --seed 42 \
  --score block --angles 20 5
