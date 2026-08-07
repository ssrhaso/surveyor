#!/usr/bin/env bash
#SBATCH --job-name=2room-goalkey
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-23%12
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_goalkey_%A_%a.log
# ======== TWO-ROOM ANCHOR: WHICH GOAL DEFINITION IS LeWM's Fig. 6? ==========
# THE DISCREPANCY. LeWM Fig. 6 reports Random = 0 on Two-Room; our measured
# random floor at their t=25 protocol is 22.53% (job 2329041, 12 seeds). Both
# cannot be right, and we separately reproduce their published number under
# their config (85.33 vs 87), so this questions a load-bearing anchor.
# RULED OUT offline (probe_tworoom_protocol_diff.py): _set_state proprio vs
# state (columns byte-identical); --episode-min 4000 (vacuity 8.75% vs 8.82%);
# vacuity alone (8.82% pre-solved explains at most ~9 of the 22.5 points).
# WHAT IS LEFT, decisive: the column feeding _set_goal_state IS the task.
#   goal_proprio (their released config): agent position at start+25, median
#       41.25 px away, 8.82% pre-solved  -> our 22.53% random
#   goal_state (= pos_target, the episode's FIXED target): median 114.39 px,
#       0.00% pre-solved -> a random walk should never arrive, i.e. Random = 0
# Their config says goal_proprio; their Fig. 6 Random says fixed target. This
# array asks the environment which one reproduces their figure.
# ARMS, at their protocol otherwise (mode short, start random, t=25, budget
# 50, eval-filter none, episode-min 4000, n=64), 12 seeds each:
#   random + goal_state -> if ~0, Fig. 6 is the FIXED-TARGET task and our
#                          Two-Room arc is on a different, easier one
#   flat   + goal_state -> does the strongest flat arm still reach ~87 there?
# goal_proprio counterparts are banked (random 22.53, flat 83.72), not re-run.
# No bar frozen: protocol diagnosis, both outcomes reportable. An anchor we
# cannot defend is worse than no anchor.
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
mkdir -p logs
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)

ID=$SLURM_ARRAY_TASK_ID
SEED=$((42 + ID % 12))
if [ "$((ID / 12))" -eq 0 ]; then
  NAME=random; ARM=(--subgoal random)
else
  NAME=flat;   ARM=(--subgoal baseline --receding-horizon 5)
fi

# NOTE: --goal-from-proprio is DELIBERATELY OMITTED, so the driver's default
# goal_state (the episode's fixed target) feeds _set_goal_state.
echo "=== 2room-goalkey arm=$NAME goal=goal_state t=25 seed=$SEED ==="
python -m specaccept.envs.tworoom.eval "${ARM[@]}" \
  --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
  --device cuda --seed "$SEED" --cem-seed "$SEED" \
  --mode short --start random --goal-offset 25 --eval-budget 50 \
  --eval-filter none --episode-min 4000 --num-eval 64 \
  2>&1 | tee logs/2room_goalkey_${NAME}_t25_seed${SEED}.log
echo "=== DONE task $ID ==="
