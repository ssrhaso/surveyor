#!/usr/bin/env bash
#SBATCH --job-name=timing-grid
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:30:00
#SBATCH --array=0-59%8
#SBATCH --output=/lustre/home/ha676/le-wm/logs/timing_grid_%A_%a.log
# ================== CROSS-ENV TIMING GRID (cost figure) =======================
# CUDA-synced planning wall-clock per episode, one representative cell per env,
# 4 arms x 5 seeds (42-46). Flat is measured through the instrumented policy
# via the validated tautologies (PushT: oracle stride > offset => the single
# waypoint IS the goal; Reacher/Cube: lerp frac=1.0 == flat, harness-checked
# within 0.2pp) -- SR must reproduce the flat reference as the equivalence
# check on every cell. Arms: flat-equiv (strongest RH), FF-JEPA every-step
# k=50 (its recipe per env), spec-accept (certified config), certified
# spec-accept. Cells: PushT t=100 (s5as100, budget 200, n=256); Reacher t=100
# (pop-999 ep100, budget 200, n=128); Cube certified (offset 150, budget 300,
# 600-sample CEM, n=128).
# ==============================================================================
set -euo pipefail
VENV=/lustre/home/ha676/le-wm/.venv
cd /lustre/home/ha676/le-wm
mkdir -p logs
source "$VENV/bin/activate"
ID=$SLURM_ARRAY_TASK_ID
ENV=$((ID / 20)); REST=$((ID % 20)); A=$((REST / 5)); SEED=$((42 + REST % 5))
ARMS=(flat ffjepa spec cert)
NAME=${ARMS[$A]}
if [ "$ENV" -eq 0 ]; then
  H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
  export SDL_VIDEODRIVER=dummy
  case $NAME in
    flat)   ARM="--subgoal lerp --lerp-frac 1.0 --horizon 5 --receding-horizon 5";;
    ffjepa) ARM="--subgoal gdm --gdm-steps 50 --gdm-ckpt gdm_stride25.pt --horizon 5 --receding-horizon 5";;
    spec)   ARM="--subgoal specaccept --gdm-ckpt gdm_stride10.pt --accept-tau 0.20 --gdm-steps 3 --horizon 2 --receding-horizon 2";;
    cert)   ARM="--subgoal unified --gdm-ckpt gdm_stride10.pt --accept-tau 0.20 --gdm-steps 3 --horizon 2 --receding-horizon 2";;
  esac
  echo "=== timing-grid pusht arm=$NAME seed=$SEED ==="
  python -m specaccept.envs.pusht.eval $ARM \
    --source pretrained --h5 "$H5" --device cuda \
    --mode long --episodes-file pusht.episodes150s5as100.json \
    --goal-offset 100 --eval-budget 200 \
    --num-eval 256 --score block --angles 20 --seed ${SEED} --cem-seed ${SEED} \
    --time-instrument \
    2>&1 | tee logs/tgrid_pusht_${NAME}_seed${SEED}.log
elif [ "$ENV" -eq 1 ]; then
  H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
  export MUJOCO_GL=egl
  case $NAME in
    flat)   ARM="--subgoal lerp --lerp-frac 1.0 --horizon 5 --receding-horizon 5";;
    ffjepa) ARM="--subgoal gdm --gdm-steps 50 --gdm-ckpt gdm_reacher_s25.pt --horizon 5 --receding-horizon 5";;
    spec)   ARM="--subgoal specaccept --gdm-ckpt gdm_reacher_s10.pt --accept-tau 0.20 --gdm-steps 8 --horizon 2 --receding-horizon 2";;
    cert)   ARM="--subgoal unified --gdm-ckpt gdm_reacher_s10.pt --accept-tau 0.20 --gdm-steps 8 --horizon 2 --receding-horizon 2";;
  esac
  echo "=== timing-grid reacher arm=$NAME seed=$SEED ==="
  python -m specaccept.envs.reacher.eval $ARM \
    --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
    --num-eval 128 --seed ${SEED} --episodes-file reacher_gatev4c.ep100.s999.json \
    --goal-offset 100 --eval-budget 200 \
    --time-instrument \
    2>&1 | tee logs/tgrid_reacher_${NAME}_seed${SEED}.log
else
  H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
  export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
  case $NAME in
    flat)   ARM="--subgoal lerp --lerp-frac 1.0 --horizon 5 --receding-horizon 5";;
    ffjepa) ARM="--subgoal gdm --gdm-steps 50 --gdm-ckpt gdm_cube_s10.pt --horizon 2 --receding-horizon 2";;
    spec)   ARM="--subgoal specaccept --gdm-ckpt gdm_cube_s10_gc.pt --goal-gate --accept-tau 0.20 --gdm-steps 3 --horizon 2 --receding-horizon 2";;
    cert)   ARM="--subgoal unified --gdm-ckpt gdm_cube_s10_gc.pt --accept-tau 0.20 --gdm-steps 3 --horizon 2 --receding-horizon 2";;
  esac
  echo "=== timing-grid cube arm=$NAME seed=$SEED ==="
  python -m specaccept.envs.cube.eval $ARM \
    --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
    --start final --goal-offset 150 --eval-budget 300 \
    --num-eval 128 --seed ${SEED} --cem-seed ${SEED} --episode-min 8000 \
    --num-samples 600 --n-steps 30 \
    --time-instrument \
    2>&1 | tee logs/tgrid_cube_${NAME}_seed${SEED}.log
fi
