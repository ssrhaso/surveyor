#!/usr/bin/env bash
#SBATCH --job-name=cube-gate2
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cubeG2_%A_%a.log
# Gate battery for the S={12,20} gc checkpoints (submitted afterok the
# cube-gc-S2 training array). Champion config verbatim (goal-gate, k=8,
# tau=0.20, RH=2, certified protocol) so the row is directly comparable to
# gc15+gate=79.30 / flat=70.9 / oracle=82.4.
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
mkdir -p runs/cube logs
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
SS=(12 12 12 12 20 20 20 20)
S=${SS[$SLURM_ARRAY_TASK_ID]}
SEED=$((42 + SLURM_ARRAY_TASK_ID % 4))
CK=gdm_cube_s${S}_gc.pt
echo "=== cube-gate2 S=$S ckpt=$CK seed=$SEED (champion config) ==="
python -m specaccept.envs.cube.eval \
  --subgoal specaccept --gdm-ckpt "$CK" --goal-gate \
  --accept-tau 0.20 --gdm-steps 8 --horizon 2 --receding-horizon 2 \
  --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
  --start final --goal-offset 150 --eval-budget 300 \
  --num-eval 128 --seed "$SEED" --cem-seed "$SEED" --episode-min 8000 \
  --num-samples 600 --n-steps 30 \
  2>&1 | tee logs/cubeG2_s${S}_seed${SEED}.log
