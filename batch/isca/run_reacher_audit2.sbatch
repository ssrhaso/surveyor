#!/usr/bin/env bash
#SBATCH --job-name=reacher-audit2
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-15
#SBATCH --output=/lustre/home/ha676/le-wm/logs/reacher_audit2job_%A_%a.log
# Audit round 2, long-horizon controls (follow-ups forced by round 1):
#   0-11  STRAIGHT-LINE ORACLE AT t100 (2t, max-100 pop): frac {0.25,0.5,0.75}
#         x seeds 42-45, RH2. Round 1 showed lerp loses badly at t25 (21.9/
#         70.7 vs flat 95.7). The long-horizon question is the dangerous one:
#         if a geometric lerp waypoint MATCHES the drafter at t100 (94.5), the
#         53M diffusion drafter adds nothing a one-line interpolation cannot;
#         if it does not, the learned draft distribution is proven necessary.
#   12-15 EVERY-STEP k=4 MIRROR at t100: the fair twin of the frontier's best
#         cell (spec tau=.2 k=4: 95.5 at 2.6 NFE/replan). k=8 every-step
#         already hit 94.9 at NFE 8; if k=4 every-step also lands ~95 at NFE 4,
#         spec-accept's marginal factor at the headline cell is ~1.5x and gets
#         reported as such.
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
ID=$SLURM_ARRAY_TASK_ID
cd /lustre/home/ha676/le-wm
mkdir -p runs/reacher
export MUJOCO_GL=egl
source "$VENV/bin/activate"
COMMON="--source local --local-dir encoder_reacher --h5 $H5 --device cuda --num-eval 128 \
  --horizon 2 --receding-horizon 2"
EPFILE=reacher_audit2.ep100.t${ID}.json
python batch/build_reacher_horizon_episodes.py --h5 "$H5" --out $EPFILE \
  --n 128 --max-offset 100 --episode-min 8000 --seed 42

if [ $ID -le 11 ]; then
  FRACS=(0.25 0.5 0.75)
  FRAC=${FRACS[$((ID / 4))]}
  SEED=$((42 + ID % 4))
  echo "=== audit2 lerp frac=$FRAC t100 seed=$SEED ==="
  python -m specaccept.envs.reacher.eval $COMMON --subgoal lerp --lerp-frac $FRAC \
    --seed $SEED --episodes-file $EPFILE --goal-offset 100 --eval-budget 200 \
    2>&1 | tee logs/reacher_lerp${FRAC}_t100_seed${SEED}.log
else
  SEED=$((42 + ID - 12))
  echo "=== audit2 every-step k=4 t100 seed=$SEED ==="
  python -m specaccept.envs.reacher.eval $COMMON --subgoal gdm \
    --gdm-ckpt gdm_reacher_s10.pt --gdm-steps 4 \
    --seed $SEED --episodes-file $EPFILE --goal-offset 100 --eval-budget 200 \
    2>&1 | tee logs/reacher_gdmk4_t100_seed${SEED}.log
fi
