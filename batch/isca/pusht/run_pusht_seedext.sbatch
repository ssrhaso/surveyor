#!/usr/bin/env bash
#SBATCH --job-name=pusht-seedext
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-69%8
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_ext_%A_%a.log
# ================== SEED EXTENSION (declared before results) ==================
# Grid standardization: 1024 episodes per cell (cube 8x128, reacher 8x128 via
# rea-seedext). PushT goes 2 -> 4 seeds: adds seeds 44-45 for every headline
# arm on the SAME s5 spine populations and route files. HYGIENE: the 5/5
# pre-registered safety verdict remains that of seeds 42-43; the extension is
# reported IN FULL, all seeds pooled, no selection. Routing is per-episode and
# seed-independent (subsets byte-identical across seeds). GDM every-step stays
# at 2 seeds (context row only, not a claimed comparison).
#   arms: flat2, flat5, spec(k3), r_flat2, r_flat5, r_spec, unif(spec-leg)
#   x t{25,50,75,100,150} x seeds{44,45}
# ==============================================================================
set -euo pipefail
VENV=/lustre/home/ha676/le-wm/.venv
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
ARMS=(flat2 flat5 spec r_flat2 r_flat5 r_spec unif)
SUBGOALS=(baseline baseline specaccept baseline baseline specaccept unified)
RHS=(2 5 2 2 5 2 2)
BRANCHES=(- - - flat2 flat5 spec spec)
OFFSETS=(25 50 75 100 150)
A=$((SLURM_ARRAY_TASK_ID / 10))
REST=$((SLURM_ARRAY_TASK_ID % 10))
O=$((REST / 2))
SEED=$((44 + REST % 2))
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
  echo "=== pusht-ext arm=$NAME t=$OFF seed=$SEED: EMPTY SUBSET -- SR = (0/0), skipped ==="
  exit 0
fi
CKPT_ARG=""
if [ "$SG" != "baseline" ]; then CKPT_ARG="--gdm-ckpt gdm_stride10.pt --accept-tau 0.20 --gdm-steps 3"; fi
echo "=== pusht-ext arm=$NAME t=$OFF seed=$SEED n=$N (budget=$((2*OFF))) ==="
python -m specaccept.envs.pusht.eval --subgoal ${SG} ${CKPT_ARG} \
  --source pretrained --h5 "$H5" --device cuda \
  --mode long --episodes-file $EPFILE --goal-offset ${OFF} --eval-budget $((2*OFF)) \
  --num-eval $N --score block --angles 20 5 --seed ${SEED} --cem-seed ${SEED} \
  --horizon ${RH} --receding-horizon ${RH} \
  2>&1 | tee logs/pusht_ext_${NAME}_t${OFF}_seed${SEED}.log
