#!/usr/bin/env bash
#SBATCH --job-name=reacher-audit
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-19
#SBATCH --output=/lustre/home/ha676/le-wm/logs/reacher_auditjob_%A_%a.log
# Reviewer-audit batteries (Reacher side):
#   0-11  STRAIGHT-LINE ORACLE (lerp): waypoint at frac along z_now->z_goal,
#         re-anchored per replan, RH2 (matched to the drafted-subgoal loop).
#         frac in {0.25, 0.5, 1.0} x seeds 42-45 at the native protocol.
#         Separates decomposition's intrinsic cost from the dataset-oracle's
#         meander: if lerp also fails to beat flat, decomposition overhead
#         alone explains the native-parity bound. frac=1.0 is the sanity arm
#         (must ~= the RH2 flat baseline, 95.7).
#   12-15 EVERY-STEP k=8 MIRROR at t100 (2t, max-100 pop): isolates the
#         consumption policy against an equally-cheap baseline -- the missing
#         twin of the k-isolation arm. Reference: every-step k=50 = 90.8,
#         spec tau=.2 k=8 = 91.0 (cr .61).
#   16-19 EVERY-STEP k=8 at the native protocol (completes the pair).
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
ID=$SLURM_ARRAY_TASK_ID
cd /lustre/home/ha676/le-wm
mkdir -p runs/reacher
export MUJOCO_GL=egl
source "$VENV/bin/activate"
COMMON="--source local --local-dir encoder_reacher --h5 $H5 --device cuda --num-eval 128"

if [ $ID -le 11 ]; then
  FRACS=(0.25 0.5 1.0)
  FRAC=${FRACS[$((ID / 4))]}
  SEED=$((42 + ID % 4))
  echo "=== audit lerp frac=$FRAC seed=$SEED (native t25) ==="
  python -m specaccept.envs.reacher.eval $COMMON --subgoal lerp --lerp-frac $FRAC \
    --seed $SEED --episode-min 8000 --goal-offset 25 --eval-budget 50 \
    --horizon 2 --receding-horizon 2 \
    2>&1 | tee logs/reacher_lerp${FRAC}_t25_seed${SEED}.log
elif [ $ID -le 15 ]; then
  SEED=$((42 + ID - 12))
  EPFILE=reacher_audit.ep100.t${ID}.json
  python batch/build_reacher_horizon_episodes.py --h5 "$H5" --out $EPFILE \
    --n 128 --max-offset 100 --episode-min 8000 --seed 42
  echo "=== audit every-step k=8 t100 seed=$SEED ==="
  python -m specaccept.envs.reacher.eval $COMMON --subgoal gdm \
    --gdm-ckpt gdm_reacher_s10.pt --gdm-steps 8 \
    --seed $SEED --episodes-file $EPFILE --goal-offset 100 --eval-budget 200 \
    --horizon 2 --receding-horizon 2 \
    2>&1 | tee logs/reacher_gdmk8_t100_seed${SEED}.log
else
  SEED=$((42 + ID - 16))
  echo "=== audit every-step k=8 native t25 seed=$SEED ==="
  python -m specaccept.envs.reacher.eval $COMMON --subgoal gdm \
    --gdm-ckpt gdm_reacher_s10.pt --gdm-steps 8 \
    --seed $SEED --episode-min 8000 --goal-offset 25 --eval-budget 50 \
    --horizon 2 --receding-horizon 2 \
    2>&1 | tee logs/reacher_gdmk8_t25_seed${SEED}.log
fi
