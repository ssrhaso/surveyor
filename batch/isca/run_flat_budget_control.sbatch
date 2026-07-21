#!/usr/bin/env bash
#SBATCH --job-name=flat-budget
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-19%8
#SBATCH --output=/lustre/home/ha676/le-wm/logs/flatbudget_%A_%a.log
# ============ COMPUTE-MATCHED FLAT CONTROL (PRE-REGISTERED) ===================
# The arbitrated policy spends up to 1.58x planning wall-clock on drafting
# episodes (measured). The fairness control this licenses: give FLAT planning
# the same extra compute; CEM samples 300 -> 600 (2x, a strict upper bound on
# the arbiter's overhead); at the long horizons where our margins are largest,
# on the SAME confirm populations. PREDICTION (frozen now): no material rescue.
# The long-horizon collapse is a LOOKAHEAD limit, not an optimization-power
# limit (cube showed sample-plateau 600->1000; no RH rescues it); doubling
# samples optimizes the same too-short plan better. If flat closes materially,
# our compute-fairness framing must be weakened and we report that.
#   reacher: {flat2, flat5} x {t100, t150} x seeds 42-45, pop 999   (16 cells)
#   pusht:   flat5 x {t100, t150} x seeds 42-43, s5 spine           (4 cells)
# ==============================================================================
set -euo pipefail
VENV=/lustre/home/ha676/le-wm/.venv
cd /lustre/home/ha676/le-wm
mkdir -p logs
source "$VENV/bin/activate"
ID=$SLURM_ARRAY_TASK_ID
if [ $ID -lt 16 ]; then
  H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
  ARMS=(flat2 flat5); RHS=(2 5)
  A=$((ID / 8)); REST=$((ID % 8)); O=$((REST / 4)); SEED=$((42 + REST % 4))
  NAME=${ARMS[$A]}; RH=${RHS[$A]}
  OFF=$((100 + 50 * O))
  MAXOFF=100; if [ "$OFF" -eq 150 ]; then MAXOFF=150; fi
  export MUJOCO_GL=egl
  echo "=== flat-budget reacher $NAME t=$OFF seed=$SEED samples=600 ==="
  python -m specaccept.envs.reacher.eval --subgoal baseline \
    --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
    --num-eval 128 --seed ${SEED} --episodes-file reacher_gatev4c.ep${MAXOFF}.s999.json \
    --goal-offset ${OFF} --eval-budget $((2*OFF)) \
    --horizon ${RH} --receding-horizon ${RH} --num-samples 600 \
    2>&1 | tee logs/flatbudget_rea_${NAME}600_t${OFF}_seed${SEED}.log
else
  H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
  REST=$((ID - 16)); O=$((REST / 2)); SEED=$((42 + REST % 2))
  OFF=$((100 + 50 * O))
  EPFILE=pusht.episodes150s5as${OFF}.json
  if [ "$OFF" -eq 150 ]; then EPFILE=pusht.episodes150s5.json; fi
  export SDL_VIDEODRIVER=dummy
  echo "=== flat-budget pusht flat5 t=$OFF seed=$SEED samples=600 ==="
  python -m specaccept.envs.pusht.eval --subgoal baseline \
    --source pretrained --h5 "$H5" --device cuda \
    --mode long --episodes-file $EPFILE --goal-offset ${OFF} --eval-budget $((2*OFF)) \
    --num-eval 256 --score block --angles 20 5 --seed ${SEED} --cem-seed ${SEED} \
    --horizon 5 --receding-horizon 5 --num-samples 600 \
    2>&1 | tee logs/flatbudget_pusht_flat5600_t${OFF}_seed${SEED}.log
fi
