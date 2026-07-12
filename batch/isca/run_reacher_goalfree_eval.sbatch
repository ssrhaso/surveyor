#!/usr/bin/env bash
#SBATCH --job-name=reacher-gf-eval
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/lustre/home/ha676/le-wm/logs/reacher_gfeval_%A_%a.log
# Goal-free ablation evals (every-step GDM, S=10 goal-free drafter):
#   tasks 0-1: t25 native, seeds 42-43     (goal-cond ref: s10gdm 60.55)
#   tasks 2-3: t100 max-offset-100 pop     (goal-cond ref: 90.82)
#   tasks 4-5: t150 max-offset-150 pop     (goal-cond ref: 95.12)
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
ID=$SLURM_ARRAY_TASK_ID
cd /lustre/home/ha676/le-wm
mkdir -p runs/reacher
export MUJOCO_GL=egl
source "$VENV/bin/activate"

COMMON="--subgoal gdm --gdm-ckpt gdm_reacher_s10_goalfree.pt \
  --accept-tau 0.20 --gdm-steps 50 \
  --source local --local-dir encoder_reacher --h5 $H5 --device cuda \
  --num-eval 128 --horizon 2 --receding-horizon 2"

SEED=$((42 + ID % 2))
if [ $ID -le 1 ]; then
  echo "=== goal-free gdm t25 native seed=$SEED ==="
  python -m specaccept.envs.reacher.eval $COMMON \
    --seed $SEED --episode-min 8000 --goal-offset 25 --eval-budget 50 \
    2>&1 | tee logs/reacher_gf_t25_seed${SEED}.log
elif [ $ID -le 3 ]; then
  EPFILE=reacher_gf.ep100.t${ID}.json
  echo "=== goal-free gdm t100 seed=$SEED ==="
  python batch/build_reacher_horizon_episodes.py --h5 "$H5" --out $EPFILE \
    --n 128 --max-offset 100 --episode-min 8000 --seed 42
  python -m specaccept.envs.reacher.eval $COMMON \
    --seed $SEED --episodes-file $EPFILE --goal-offset 100 --eval-budget 200 \
    2>&1 | tee logs/reacher_gf_t100_seed${SEED}.log
else
  EPFILE=reacher_gf.ep150.t${ID}.json
  echo "=== goal-free gdm t150 seed=$SEED ==="
  python batch/build_reacher_horizon_episodes.py --h5 "$H5" --out $EPFILE \
    --n 128 --max-offset 150 --episode-min 8000 --seed 42
  python -m specaccept.envs.reacher.eval $COMMON \
    --seed $SEED --episodes-file $EPFILE --goal-offset 150 --eval-budget 300 \
    2>&1 | tee logs/reacher_gf_t150_seed${SEED}.log
fi
