#!/usr/bin/env bash
#SBATCH --job-name=unif-cube
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/lustre/home/ha676/le-wm/logs/unif_cube_%A_%a.log
# ============ UNIFIED METHOD (c*-retire spec): CUBE, FULL ARM ===============
# PRE-REGISTERED. The unified source REPLACES the latent goal-gate on cube:
# same gc drafter (s10_gc), same derived k=3, same tau=0.20, RH=2, S=10; but
# arrival handling is now c*-retire (the planner's certificate) instead of the
# latent-distance goal-gate. Cube runs the WHOLE unified method in one process:
# the first-replan c* check IS the router's fire test (fired episodes serve the
# goal from replan 0 = the flat branch), and transport ~90 steps >> the 10-step
# window means the predicted fire-rate is ~0; the [unified] stats line
# reports it (fired-at-first-replan) so the prediction is checked, not assumed.
# Seeds 50-57 = the SAME eight fresh seeds as the k3c confirm (job 2276035), so
# flat and the goal-gate variant are directly comparable rows.
# BAR (frozen; 8-seed mean): unified >= flat + 5.0pp. Reported: vs speck3+goal-
# gate (k3c); does the principled retire match or beat the latent gate?
# ==============================================================================
set -euo pipefail
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
mkdir -p logs
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
SEED=$((50 + SLURM_ARRAY_TASK_ID))
echo "=== unif-cube seed=$SEED (unified c*-retire, certified protocol) ==="
python -m specaccept.envs.cube.eval --subgoal unified \
  --gdm-ckpt gdm_cube_s10_gc.pt --gdm-steps 3 --accept-tau 0.20 \
  --horizon 2 --receding-horizon 2 \
  --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
  --start final --goal-offset 150 --eval-budget 300 \
  --num-eval 128 --seed "$SEED" --cem-seed "$SEED" --episode-min 8000 \
  --num-samples 600 --n-steps 30 \
  2>&1 | tee logs/unif_cube_seed${SEED}.log
