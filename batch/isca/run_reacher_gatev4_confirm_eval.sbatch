#!/usr/bin/env bash
#SBATCH --job-name=gatev4c-eval
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:30:00
#SBATCH --array=0-95%8
#SBATCH --output=/lustre/home/ha676/le-wm/logs/gatev4c_eval_%A_%a.log
# ============== GATE V4 CONFIRM -- CLOSED-LOOP EVAL (PRE-REGISTERED) ==========
# Frozen-bar confirmation of the v4 ternary router on FRESH populations
# (builder seed 999; 42=analysis, 777=v3 confirm, 888=v4 exploratory). Same
# grid as the exploratory run: arms x t{25,50,100,150} x seeds{42-45}, n=128,
# budget=2t, pop max100 for t<=100 / max150 for t=150.
#
# SUCCESS CONDITION (frozen NOW, before any cell runs; means over 4 seeds):
#   PRIMARY: composite >= max(flat2, flat5) at EVERY t  -- a single policy
#            beats LeWM's planner (either config) at every horizon.
#   SECONDARY (reported, not gating): composite vs spec at t100/t150; the
#            exploratory run showed -2.15pp vs our own spec at t150 (c*
#            optimism) -- if it repeats it is REPORTED AS A KNOWN LIMITATION.
# Composite SR per (t, seed) = (succ_r_flat2 + succ_r_flat5 + succ_r_spec)/128.
# tau=0.20, the routing rule, k=8, S=10 all frozen; no amendment on failure.
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
mkdir -p logs
export MUJOCO_GL=egl
source "$VENV/bin/activate"
if [ "$BR" = "-" ]; then
  EPFILE=reacher_gatev4c.ep${MAXOFF}.s999.json
else
  EPFILE=reacher_gatev4c_route_t${OFF}_${BR}.json
fi
N=$(python -c "import json;print(len(json.load(open('$EPFILE'))['episodes']))")
if [ "$N" -eq 0 ]; then
  echo "=== gatev4c arm=$NAME t=$OFF seed=$SEED: EMPTY SUBSET -- SR = (0/0), skipped ==="
  exit 0
fi
CKPT_ARG=""
if [ "$SG" != "baseline" ]; then CKPT_ARG="--gdm-ckpt gdm_reacher_s10.pt --gdm-steps 8"; fi
echo "=== gatev4c arm=$NAME t=$OFF seed=$SEED n=$N epfile=$EPFILE (budget=$((2*OFF))) ==="
python -m specaccept.envs.reacher.eval --subgoal ${SG} ${CKPT_ARG} \
  --accept-tau 0.20 \
  --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
  --num-eval $N --seed ${SEED} --episodes-file $EPFILE \
  --goal-offset ${OFF} --eval-budget $((2*OFF)) \
  --horizon ${RH} --receding-horizon ${RH} \
  2>&1 | tee logs/gatev4c_${NAME}_t${OFF}_seed${SEED}.log
