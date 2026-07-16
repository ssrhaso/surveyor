#!/usr/bin/env bash
#SBATCH --job-name=cube-final
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cube_finaljob_%A_%a.log
# THE CUBE COMPARISON, at the FULLY CERTIFIED protocol.
# Baseline anchor SOLVED: the 6.4pp gap to LeWM's 74% was CEM under-planning --
# 300 samples (the inherited PushT default) -> 70.3; 600 samples -> 75.8; 1000 -> 75.0.
# So: offset=150, budget=300, num_samples=600, n_steps=30. Flat RH=5 (its strongest);
# drafters at the matched loop RH=2 (S=10). EVERY arm now gets the same CEM budget,
# so the comparison is apples-to-apples.
#   0-3  flat (RH5)   4-7  s10 goal-free every-step   8-11  s10 spec-accept (t.20,k4)
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
ID=$SLURM_ARRAY_TASK_ID
SEED=$((42 + ID % 4))
if   [ $ID -le 3 ]; then NAME=flat;    ARM="--subgoal baseline --horizon 5 --receding-horizon 5"
elif [ $ID -le 7 ]; then NAME=s10gdm;  ARM="--subgoal gdm --gdm-ckpt gdm_cube_s10.pt --gdm-steps 50 --horizon 2 --receding-horizon 2"
else                     NAME=s10spec; ARM="--subgoal specaccept --gdm-ckpt gdm_cube_s10.pt --accept-tau 0.20 --gdm-steps 4 --horizon 2 --receding-horizon 2"
fi
echo "=== cube FINAL (certified: 600 samples): $NAME seed=$SEED ==="
python -m specaccept.envs.cube.eval $ARM \
  --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
  --start final --goal-offset 150 --eval-budget 300 \
  --num-eval 128 --seed $SEED --cem-seed $SEED --episode-min 8000 \
  --num-samples 600 --n-steps 30 \
  2>&1 | tee logs/cubeF_${NAME}_seed${SEED}.log
