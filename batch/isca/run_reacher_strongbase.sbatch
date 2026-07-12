#!/usr/bin/env bash
#SBATCH --job-name=reacher-strong
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-15
#SBATCH --output=/lustre/home/ha676/le-wm/logs/reacher_strongjob_%A_%a.log
# Strong-baseline completion + frontier top-up (ICLR comparison hygiene):
#   0-3   flat baseline RH=1, t25 native, seeds 42-45
#   4-5   flat baseline RH=1, t100 (max-100 pop), seeds 42-43
#   6-7   flat baseline RH=1, t150 (max-150 pop), seeds 42-43
#   8-9   flat baseline RH=2, t100, seeds 44-45   (-> n=4 with 2270701)
#   10-11 flat baseline RH=2, t150, seeds 44-45   (-> n=4 with 2270701)
#   12-13 frontier s10spec tau=0.2 k=4, t100, seeds 44-45  (-> n=4)
#   14-15 frontier s10spec tau=0.1 k=4, t100, seeds 44-45  (-> n=4)
# Purpose: every paper table anchors to the STRONGEST flat baseline per cell
# (RH=2 flat hit 95.7 at t25 but 67.6 at t100 — replan-rate is a brittle
# trade-off; the crossover must be shown against best-per-cell).
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
ID=$SLURM_ARRAY_TASK_ID
cd /lustre/home/ha676/le-wm
mkdir -p runs/reacher
export MUJOCO_GL=egl
source "$VENV/bin/activate"

BASE="--subgoal baseline --accept-tau 0.20 --gdm-steps 50 \
  --source local --local-dir encoder_reacher --h5 $H5 --device cuda --num-eval 128"
SPEC="--subgoal specaccept --gdm-ckpt gdm_reacher_s10.pt --gdm-steps 4 \
  --source local --local-dir encoder_reacher --h5 $H5 --device cuda \
  --num-eval 128 --horizon 2 --receding-horizon 2"

epfile () {  # $1 = max-offset
  EPFILE=reacher_strong.ep$1.t${ID}.json
  python batch/build_reacher_horizon_episodes.py --h5 "$H5" --out $EPFILE \
    --n 128 --max-offset $1 --episode-min 8000 --seed 42
}

case $ID in
  0|1|2|3)
    SEED=$((42 + ID))
    python -m specaccept.envs.reacher.eval $BASE --horizon 1 --receding-horizon 1 \
      --seed $SEED --episode-min 8000 --goal-offset 25 --eval-budget 50 \
      2>&1 | tee logs/reacher_rh1_t25_seed${SEED}.log ;;
  4|5)
    SEED=$((42 + ID - 4)); epfile 100
    python -m specaccept.envs.reacher.eval $BASE --horizon 1 --receding-horizon 1 \
      --seed $SEED --episodes-file $EPFILE --goal-offset 100 --eval-budget 200 \
      2>&1 | tee logs/reacher_rh1_t100_seed${SEED}.log ;;
  6|7)
    SEED=$((42 + ID - 6)); epfile 150
    python -m specaccept.envs.reacher.eval $BASE --horizon 1 --receding-horizon 1 \
      --seed $SEED --episodes-file $EPFILE --goal-offset 150 --eval-budget 300 \
      2>&1 | tee logs/reacher_rh1_t150_seed${SEED}.log ;;
  8|9)
    SEED=$((44 + ID - 8)); epfile 100
    python -m specaccept.envs.reacher.eval $BASE --horizon 2 --receding-horizon 2 \
      --seed $SEED --episodes-file $EPFILE --goal-offset 100 --eval-budget 200 \
      2>&1 | tee logs/reacher_rh2_t100_seed${SEED}.log ;;
  10|11)
    SEED=$((44 + ID - 10)); epfile 150
    python -m specaccept.envs.reacher.eval $BASE --horizon 2 --receding-horizon 2 \
      --seed $SEED --episodes-file $EPFILE --goal-offset 150 --eval-budget 300 \
      2>&1 | tee logs/reacher_rh2_t150_seed${SEED}.log ;;
  12|13)
    SEED=$((44 + ID - 12)); epfile 100
    python -m specaccept.envs.reacher.eval $SPEC --accept-tau 0.20 \
      --seed $SEED --episodes-file $EPFILE --goal-offset 100 --eval-budget 200 \
      2>&1 | tee logs/reacher_frontier_tau0.2_k4_seed${SEED}.log ;;
  14|15)
    SEED=$((44 + ID - 14)); epfile 100
    python -m specaccept.envs.reacher.eval $SPEC --accept-tau 0.10 \
      --seed $SEED --episodes-file $EPFILE --goal-offset 100 --eval-budget 200 \
      2>&1 | tee logs/reacher_frontier_tau0.1_k4_seed${SEED}.log ;;
esac
