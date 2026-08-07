#!/usr/bin/env bash
#SBATCH --job-name=random-floor
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-83%10
#SBATCH --output=/lustre/home/ha676/le-wm/logs/randfloor_%A_%a.log
# =============== RANDOM-ACTION FLOOR: ALL FOUR ENVS x THE GRID ===============
# Supervisor request (2026-08-07): the headline table needs the chance floor
# LeWM's own Table I carries, so a reader can see what the criterion gives
# away for free. A BASELINE, not a claim: no bar frozen, cells reported as-is.
# Arm = swm.policy.RandomPolicy (the library's uniform action-space sampler,
# seeded through World.set_policy), reached by --subgoal random; no CEM, no
# drafter, no verifier. Everything else is the banked cell's protocol verbatim
# (episode files, budget 2t, criterion, seeds), comparable cell by cell:
#   PushT    t{25,50,75,100,150} x seeds 42-45, full pops, block, angles 20/5
#   Reacher  t{25,50,100,150}    x seeds 42-49, gatev4c s999 pops (max-offset
#            100 for t<=100, 150 for t=150)
#   Cube     t150                x seeds 42-49, episode-min 8000
#   Two-Room t25 (their protocol) and t75 (anchored) x seeds 42-53
# 84 tasks; cost is env stepping and rendering only.
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
mkdir -p logs
source /lustre/home/ha676/le-wm/.venv/bin/activate

ID=$SLURM_ARRAY_TASK_ID

# ---------------------------------------------------------------- PushT 0-19
if [ "$ID" -lt 20 ]; then
  H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
  OFFSETS=(25 50 75 100 150)
  OFF=${OFFSETS[$((ID / 4))]}
  SEED=$((42 + ID % 4))
  EPFILE=pusht.episodes150s5as${OFF}.json
  if [ "$OFF" -eq 150 ]; then EPFILE=pusht.episodes150s5.json; fi
  N=$(python -c "import json;print(len(json.load(open('$EPFILE'))['episodes']))")
  echo "=== random-floor pusht t=$OFF seed=$SEED n=$N epfile=$EPFILE (budget=$((2*OFF))) ==="
  python -m specaccept.envs.pusht.eval --subgoal random \
    --source pretrained --h5 "$H5" --device cuda \
    --mode long --episodes-file $EPFILE --goal-offset ${OFF} --eval-budget $((2*OFF)) \
    --num-eval $N --score block --angles 20 5 --seed ${SEED} --cem-seed ${SEED} \
    2>&1 | tee logs/randfloor_pusht_t${OFF}_seed${SEED}.log

# -------------------------------------------------------------- Reacher 20-51
elif [ "$ID" -lt 52 ]; then
  R=$((ID - 20))
  H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
  OFFSETS=(25 50 100 150)
  OFF=${OFFSETS[$((R / 8))]}
  SEED=$((42 + R % 8))
  MAXOFF=100; if [ "$OFF" -eq 150 ]; then MAXOFF=150; fi
  EPFILE=reacher_gatev4c.ep${MAXOFF}.s999.json
  N=$(python -c "import json;print(len(json.load(open('$EPFILE'))['episodes']))")
  echo "=== random-floor reacher t=$OFF seed=$SEED n=$N epfile=$EPFILE (budget=$((2*OFF))) ==="
  python -m specaccept.envs.reacher.eval --subgoal random \
    --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
    --num-eval $N --seed ${SEED} --cem-seed ${SEED} --episodes-file $EPFILE \
    --goal-offset ${OFF} --eval-budget $((2*OFF)) \
    2>&1 | tee logs/randfloor_reacher_t${OFF}_seed${SEED}.log

# ------------------------------------------------------------------ Cube 52-59
elif [ "$ID" -lt 60 ]; then
  H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
  SEED=$((42 + ID - 52))
  echo "=== random-floor cube t=150 seed=$SEED (budget=300) ==="
  python -m specaccept.envs.cube.eval --subgoal random \
    --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
    --start final --goal-offset 150 --eval-budget 300 \
    --num-eval 128 --seed $SEED --cem-seed $SEED --episode-min 8000 \
    2>&1 | tee logs/randfloor_cube_seed${SEED}.log

# -------------------------------------------------------------- Two-Room 60-83
else
  W=$((ID - 60))
  H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
  SEED=$((42 + W % 12))
  COMMON=(--h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms
          --device cuda --subgoal random --seed "$SEED" --cem-seed "$SEED"
          --eval-filter none --episode-min 4000 --goal-from-proprio --num-eval 64)
  if [ "$((W / 12))" -eq 0 ]; then
    # LeWM's own published protocol (the t=25 anchor)
    echo "=== random-floor tworoom t=25 (their protocol) seed=$SEED ==="
    python -m specaccept.envs.tworoom.eval "${COMMON[@]}" \
      --mode short --start random --goal-offset 25 --eval-budget 50 \
      2>&1 | tee logs/randfloor_tworoom_t25_seed${SEED}.log
  else
    # the anchored long-horizon cell where spec-accept wins +15.4pp
    echo "=== random-floor tworoom t=75 (anchored) seed=$SEED ==="
    python -m specaccept.envs.tworoom.eval "${COMMON[@]}" \
      --mode long --goal-offset 75 --eval-budget 150 \
      2>&1 | tee logs/randfloor_tworoom_t75_seed${SEED}.log
  fi
fi
echo "=== DONE task $ID ==="
