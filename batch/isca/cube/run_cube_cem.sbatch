#!/usr/bin/env bash
#SBATCH --job-name=cube-cem
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cube_cemjob_%A_%a.log
# Close the 6.4pp cube baseline gap (ours 67.6 vs LeWM 74). Hypothesis: we are
# under-planning. Sweep CEM capacity at the certified protocol (offset150/bud300/RH5).
#   0-2: samples {300,600,1000} x n_steps 30
#   3-5: samples {300,600,1000} x n_steps 50
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
ID=$SLURM_ARRAY_TASK_ID
SAMP=(300 600 1000); NS=(30 50)
S=${SAMP[$((ID % 3))]}; N=${NS[$((ID / 3))]}
echo "=== cube CEM capacity: samples=$S n_steps=$N (baseline 67.6, LeWM 74) ==="
python -m specaccept.envs.cube.eval --subgoal baseline \
  --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
  --start final --goal-offset 150 --eval-budget 300 \
  --num-eval 128 --seed 42 --cem-seed 42 --episode-min 8000 \
  --horizon 5 --receding-horizon 5 \
  --num-samples $S --n-steps $N \
  2>&1 | tee logs/cube_cem_s${S}_n${N}.log
