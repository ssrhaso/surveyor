#!/usr/bin/env bash
#SBATCH --job-name=cube-battery
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-29
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cube_batteryjob_%A_%a.log
# Cube horizon spine at the DERIVED config (sources cube_derived.env written by
# the extract job; NO closed-loop number chose tau/k). 3 arms x 5 offsets x
# 2 seeds, n=128. Arms: baseline (flat goal-image, RH5, strongest at range);
# s10 every-step GOAL-FREE (k=50 reference); s10 spec-accept at derived (tau,k).
# Drafters use matched loop RH2 (S=10). Predictions under test: goal-free works
# (expert data), drafting beats flat at long horizon, spec SR-neutral.
#   ARM = idx//10 (0 base,1 gdm,2 spec); OFF = (idx//2)%5; SEED = 42 + idx%2
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
source /lustre/home/ha676/le-wm/cube_derived.env   # TAU, K
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
ID=$SLURM_ARRAY_TASK_ID
ARMS=(baseline s10gdm s10spec)
OFFSETS=(25 50 75 100 150)
A=$((ID / 10)); OFF=${OFFSETS[$(((ID / 2) % 5))]}; SEED=$((42 + ID % 2))
NAME=${ARMS[$A]}
case $NAME in
  baseline) ARM="--subgoal baseline --horizon 5 --receding-horizon 5" ;;
  s10gdm)   ARM="--subgoal gdm --gdm-ckpt gdm_cube_s10.pt --gdm-steps 50 --horizon 2 --receding-horizon 2" ;;
  s10spec)  ARM="--subgoal specaccept --gdm-ckpt gdm_cube_s10.pt --accept-tau ${TAU} --gdm-steps ${K} --horizon 2 --receding-horizon 2" ;;
esac
echo "=== cube battery: arm=$NAME t=$OFF seed=$SEED (derived TAU=${TAU} K=${K}) ==="
python -m specaccept.envs.cube.eval $ARM \
  --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
  --start final --goal-offset $OFF --eval-budget $((2*OFF)) \
  --num-eval 128 --seed $SEED --cem-seed $SEED \
  2>&1 | tee logs/cube_${NAME}_t${OFF}_seed${SEED}.log
