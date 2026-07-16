#!/usr/bin/env bash
#SBATCH --job-name=cube-k3c
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-15%8
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cubeK3C_%A_%a.log
# ================ CUBE CHAMPION CONFIRM (PRE-REGISTERED) ======================
# Written BEFORE any cell runs. The S=10 sweep (job 2274073) found the best
# cube config is EXACTLY the derived one: gc10 + goal-gate, k=3 (derive-k's
# sampler-convergence value), tau=0.20 (transferred; criterion floor OFF-FLOOR
# on cube), RH=2, S=10 -> 81.25% over seeds 42-45 vs flat 70.9 / oracle 82.4.
# Those seeds also picked the champion, so this array re-stamps it on EIGHT
# FRESH seeds (50-57, never used on cube) with flat re-run on the same seeds.
#
# SUCCESS CONDITION (frozen NOW; means over the 8 fresh seeds):
#   spec_k3 >= flat + 5.0pp   (exploratory margin was +10.3; half-margin bar)
# Reported, not gating: distance to the historic oracle mean (82.4, seeds
# 42-45). Config frozen; failure reported as-is, no post-hoc amendment.
#   arms: 0-7 spec_k3 seeds 50-57, 8-15 flat(RH5) seeds 50-57
# ==============================================================================
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
mkdir -p logs
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
SEED=$((50 + SLURM_ARRAY_TASK_ID % 8))
if [ $SLURM_ARRAY_TASK_ID -le 7 ]; then
  NAME=speck3
  ARM="--subgoal specaccept --gdm-ckpt gdm_cube_s10_gc.pt --goal-gate --accept-tau 0.20 --gdm-steps 3 --horizon 2 --receding-horizon 2"
else
  NAME=flat
  ARM="--subgoal baseline --horizon 5 --receding-horizon 5"
fi
echo "=== cube k3-confirm arm=$NAME seed=$SEED (certified: 600 samples, offset150 budget300) ==="
python -m specaccept.envs.cube.eval $ARM \
  --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
  --start final --goal-offset 150 --eval-budget 300 \
  --num-eval 128 --seed "$SEED" --cem-seed "$SEED" --episode-min 8000 \
  --num-samples 600 --n-steps 30 \
  2>&1 | tee logs/cubeK3C_${NAME}_seed${SEED}.log
