#!/usr/bin/env bash
#SBATCH --job-name=gcidm-anch2
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/lustre/home/ha676/le-wm/logs/gcidm_anch2_%A_%a.log
# ========== GC-IDM ANCHOR v2: PUSHT AT THEIR SAMPLING SEMANTICS =============
# The v1 pusht anchor (100.00 x3) ran on the success-filtered fixed population
# with start=final-25 -- not their protocol, gap in the FLATTERING direction,
# so it anchors nothing. Their Table 1 samples random valid starts over the
# dataset (goal = start+25, budget T=50, H_max=50); this runs exactly that:
# --mode short --start random, --eval-filter none, n=200 (targets 84.2 n=200 /
# 84.7 n=50), 3 population seeds. Sampling includes trained-on episodes, as
# theirs does -- fidelity anchor, never a comparison cell. Reacher v1 anchored
# exactly (100.0 x3); Cube v1 read 92.7 vs 99.3 with the declared held-out
# handicap pointing that way. PushT was the one wrong cell; this closes it.
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
mkdir -p logs
source /lustre/home/ha676/le-wm/.venv/bin/activate

SEED=$((42 + SLURM_ARRAY_TASK_ID))
echo "=== gcidm-anchor-v2 pusht t=25 budget=50 n=200 start=random filter=none seed=$SEED (targets 84.7 n=50 / 84.2 n=200) ==="
python -m specaccept.envs.pusht.eval --subgoal gcidm --gcidm-ckpt gcidm_pusht.pt \
  --source pretrained --h5 /lustre/home/ha676/data/pusht/pusht_expert_train.h5 \
  --device cuda --mode short --start random --eval-filter none \
  --goal-offset 25 --eval-budget 50 --num-eval 200 \
  --score block --angles 20 5 --seed $SEED --cem-seed $SEED \
  2>&1 | tee logs/gcidm_anch2_pusht_seed${SEED}.log
echo "=== DONE seed=$SEED ==="
