#!/usr/bin/env bash
#SBATCH --job-name=pusht-v4-eval
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-39%8
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_v4eval_%A_%a.log
# ================ GATE V4 ON PUSHT -- CLOSED-LOOP EVAL (PRE-REGISTERED) =======
# afterok the routing pass. Arms x t{25,50,75,100,150} x seeds{42,43} (the s5
# spine convention), n=256, budget=2t, score block, angles 20/5:
#   flat2   : baseline RH=2 on the FULL population (the one missing reference;
#             flat5=the existing baseline spine, spec=the existing speck3 spine)
#   r_flat2 / r_flat5 / r_spec : the routed subsets (composite policy)
# Composite SR per (t, seed) = (succ_r_flat2 + succ_r_flat5 + succ_r_spec)/256.
#
# SUCCESS CONDITION (frozen NOW; means over seeds 42-43, 20deg):
#   PRIMARY: composite >= speck3 spine - 1.0pp at EVERY t -- the router must
#            not break pure spec's dominance on the env where flat collapses
#            (the c*-optimism worst case: any episode mis-routed to a flat arm
#            at t>=100 is likely lost, flat=2.7-12.7 there).
#   Also reported: composite vs flat arms (expected >> by construction).
# Rule/tau/k frozen (tau=0.20 from floor_pusht.json, spec k=3 derived, S=10).
# Failure reported as-is, no post-hoc amendment.
# ==============================================================================
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
ARMS=(flat2 r_flat2 r_flat5 r_spec)
SUBGOALS=(baseline baseline baseline specaccept)
RHS=(2 2 5 2)
BRANCHES=(- flat2 flat5 spec)
OFFSETS=(25 50 75 100 150)
A=$((SLURM_ARRAY_TASK_ID / 10))
REST=$((SLURM_ARRAY_TASK_ID % 10))
O=$((REST / 2))
SEED=$((42 + REST % 2))
NAME=${ARMS[$A]}; SG=${SUBGOALS[$A]}; RH=${RHS[$A]}; BR=${BRANCHES[$A]}
OFF=${OFFSETS[$O]}
cd /lustre/home/ha676/le-wm
mkdir -p logs
export SDL_VIDEODRIVER=dummy
source "$VENV/bin/activate"
if [ "$BR" = "-" ]; then
  EPFILE=pusht.episodes150s5as${OFF}.json
  if [ "$OFF" -eq 150 ]; then EPFILE=pusht.episodes150s5.json; fi
else
  EPFILE=pusht_gatev4_route_t${OFF}_${BR}.json
fi
N=$(python -c "import json;print(len(json.load(open('$EPFILE'))['episodes']))")
if [ "$N" -eq 0 ]; then
  echo "=== pusht-v4 arm=$NAME t=$OFF seed=$SEED: EMPTY SUBSET -- SR = (0/0), skipped ==="
  exit 0
fi
CKPT_ARG=""
if [ "$SG" != "baseline" ]; then CKPT_ARG="--gdm-ckpt gdm_stride10.pt --accept-tau 0.20 --gdm-steps 3"; fi
echo "=== pusht-v4 arm=$NAME t=$OFF seed=$SEED n=$N epfile=$EPFILE (budget=$((2*OFF))) ==="
python -m specaccept.envs.pusht.eval --subgoal ${SG} ${CKPT_ARG} \
  --source pretrained --h5 "$H5" --device cuda \
  --mode long --episodes-file $EPFILE --goal-offset ${OFF} --eval-budget $((2*OFF)) \
  --num-eval $N --score block --angles 20 5 --seed ${SEED} --cem-seed ${SEED} \
  --horizon ${RH} --receding-horizon ${RH} \
  2>&1 | tee logs/pusht_v4_${NAME}_t${OFF}_seed${SEED}.log
