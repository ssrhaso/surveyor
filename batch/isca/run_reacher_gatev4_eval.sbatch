#!/usr/bin/env bash
#SBATCH --job-name=gatev4-eval
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:30:00
#SBATCH --array=0-95%8
#SBATCH --output=/lustre/home/ha676/le-wm/logs/gatev4_eval_%A_%a.log
# ==================== GATE V4 TERNARY ROUTER: CLOSED-LOOP EVAL ==============
# Stage 2 of 2 (afterok the routing pass). EXPLORATORY (the ternary rule is
# new; rule/tau/populations were fixed BEFORE any cell ran (this header)
# but v4 has never been run, so a pass here graduates to a frozen-bar confirm
# on ANOTHER fresh builder seed before anything is cited).
#
# Arms x offsets{25,50,100,150} x seeds{42-45}, n=128 pop (builder seed 888),
# budget=2t, pop max100 for t<=100 / max150 for t=150 (the v3 conventions):
#   full-pop references: flat2, flat5, spec        (the bars)
#   routed subsets:      r_flat2, r_flat5, r_spec  (the composite policy)
# Composite SR per (t, seed) = (succ_r2 + succ_r5 + succ_rs) / 128; each
# episode executed exactly ONCE, under the arm the router chose from its
# start state. Empty subsets are skipped (contribute 0 successes, 0 episodes).
#
# DECLARED BARS (means over the 4 seeds, per t):
#   PRIMARY: composite >= max(flat2, flat5, spec) - 1.0pp at EVERY t
#            (one policy that never loses to any fixed arm)
#   STRETCH: composite >= max(flat2, flat5) at EVERY t (LeWM's planner beaten
#            outright at every horizon by a single policy)
# Diagnosed v3 miss this must fix: t50 gated 91.99 vs flat5 95.51 (RH=2 flat
# branch). Prediction: t50 routes mostly flat5 and recovers to ~95. KNOWN
# RISK (stated in route_horizon.py): CEM optimism at RH=5 may over-fire
# c*_5 at t150; if flat5 eats spec's territory there the composite drops
# below spec; reported as-is, no post-hoc amendment.
# ==============================================================================
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
ARMS=(flat2 flat5 spec r_flat2 r_flat5 r_spec)
SUBGOALS=(baseline baseline specaccept baseline baseline specaccept)
RHS=(2 5 2 2 5 2)
BRANCHES=(- - - flat2 flat5 spec)
OFFSETS=(25 50 100 150)
A=$((SLURM_ARRAY_TASK_ID / 16))
O=$(((SLURM_ARRAY_TASK_ID / 4) % 4))
SEED=$((42 + SLURM_ARRAY_TASK_ID % 4))
NAME=${ARMS[$A]}; SG=${SUBGOALS[$A]}; RH=${RHS[$A]}; BR=${BRANCHES[$A]}
OFF=${OFFSETS[$O]}
MAXOFF=100; if [ "$OFF" -eq 150 ]; then MAXOFF=150; fi
cd /lustre/home/ha676/le-wm
mkdir -p runs/reacher logs
export MUJOCO_GL=egl
source "$VENV/bin/activate"
if [ "$BR" = "-" ]; then
  EPFILE=reacher_gatev4.ep${MAXOFF}.s888.json
else
  EPFILE=reacher_gatev4_route_t${OFF}_${BR}.json
fi
N=$(python -c "import json;print(len(json.load(open('$EPFILE'))['episodes']))")
if [ "$N" -eq 0 ]; then
  echo "=== gatev4 arm=$NAME t=$OFF seed=$SEED: EMPTY SUBSET; SR = (0/0), skipped ==="
  exit 0
fi
CKPT_ARG=""
if [ "$SG" != "baseline" ]; then CKPT_ARG="--gdm-ckpt gdm_reacher_s10.pt --gdm-steps 8"; fi
echo "=== gatev4 arm=$NAME t=$OFF seed=$SEED n=$N epfile=$EPFILE (budget=$((2*OFF))) ==="
python -m specaccept.envs.reacher.eval --subgoal ${SG} ${CKPT_ARG} \
  --accept-tau 0.20 \
  --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
  --num-eval $N --seed ${SEED} --episodes-file $EPFILE \
  --goal-offset ${OFF} --eval-budget $((2*OFF)) \
  --horizon ${RH} --receding-horizon ${RH} \
  2>&1 | tee logs/gatev4_${NAME}_t${OFF}_seed${SEED}.log
