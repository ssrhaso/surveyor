#!/usr/bin/env bash
#SBATCH --job-name=2room-rawtgt
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-11%6
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_rawtgt_%A_%a.log
# ===== TWO-ROOM FIXED-TARGET, ACTUALLY SERVED FOR THE FIRST TIME ============
# prereg/2026-08-07_composite_and_executor.md section D. The goalkey diff
# (job 2329041 counterparts) came back byte-identical because swm's DERIVED
# goal_{col} keys (column at the goal frame) SHADOW the h5's raw goal_state
# column, and state==proprio in this h5 -- both "keys" served the same
# hindsight value. The fixed-target task has therefore never been run.
# --goal-raw-target injects the raw column literally (in_dataset: False).
#
# P-2ROOM-KEY-1 (frozen): random on fixed-target < 5% (LeWM Fig. 6 Random=0
#   band; probe: 114.39px median, 0.00% pre-solved).
# P-2ROOM-KEY-2 (frozen): flat on fixed-target in [80,90] iff their published
#   87 is the fixed-target task; < 70 -> their figure is better explained as
#   hindsight-task and the discrepancy moves elsewhere. Reported either way.
# Protocol otherwise identical to the goalkey arms (their t=25: mode short,
# start random, budget 50, eval-filter none, episode-min 4000, n=64).
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
mkdir -p logs
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)

ID=$SLURM_ARRAY_TASK_ID
SEED=$((42 + ID % 6))
if [ "$((ID / 6))" -eq 0 ]; then
  NAME=random; ARM=(--subgoal random)
else
  NAME=flat;   ARM=(--subgoal baseline --receding-horizon 5)
fi

echo "=== 2room-rawtgt arm=$NAME goal=RAW-fixed-target t=25 seed=$SEED ==="
python -m specaccept.envs.tworoom.eval "${ARM[@]}" \
  --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
  --device cuda --seed "$SEED" --cem-seed "$SEED" \
  --mode short --start random --goal-offset 25 --eval-budget 50 \
  --eval-filter none --episode-min 4000 --num-eval 64 \
  --goal-raw-target \
  2>&1 | tee logs/2room_rawtgt_${NAME}_t25_seed${SEED}.log
echo "=== DONE task $ID ==="
