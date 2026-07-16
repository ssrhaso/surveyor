#!/usr/bin/env bash
#SBATCH --job-name=cube-arms
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cube_armsjob_%A_%a.log
# THE CUBE COMPARISON, at the CERTIFIED protocol (offset150 / budget300 / RH5 for
# flat; matched loop RH2 for S=10 drafters). Flat anchor = 67.6% (4 seeds, n=512).
# Arms: goal-free every-step drafter, and spec-accept (tau=0.20 transferred default
# since the criterion-floor rule is out of scope on cube; k=4 derived).
# 4 seeds each -> n=512/arm, SE ~2pp, directly comparable to the flat anchor.
#   0-3: s10 goal-free every-step (k=50)    4-7: s10 spec-accept (tau .20, k=4)
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
ID=$SLURM_ARRAY_TASK_ID
SEED=$((42 + ID % 4))
if [ $ID -le 3 ]; then
  NAME=s10gdm; ARM="--subgoal gdm --gdm-ckpt gdm_cube_s10.pt --gdm-steps 50"
else
  NAME=s10spec; ARM="--subgoal specaccept --gdm-ckpt gdm_cube_s10.pt --accept-tau 0.20 --gdm-steps 4"
fi
echo "=== cube arms (certified protocol): $NAME seed=$SEED  [flat anchor 67.6] ==="
python -m specaccept.envs.cube.eval $ARM \
  --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
  --start final --goal-offset 150 --eval-budget 300 \
  --num-eval 128 --seed $SEED --cem-seed $SEED --episode-min 8000 \
  --horizon 2 --receding-horizon 2 \
  2>&1 | tee logs/cube_arm_${NAME}_seed${SEED}.log
