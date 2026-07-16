#!/usr/bin/env bash
#SBATCH --job-name=rea-seedext
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:30:00
#SBATCH --array=0-111%8
#SBATCH --output=/lustre/home/ha676/le-wm/logs/rea_ext_%A_%a.log
# ================== SEED EXTENSION (declared before results) ==================
# Grid standardization: 1024 episodes per cell across all envs (cube = 8x128
# already). Reacher goes 4 -> 8 seeds: this array adds seeds 46-49 for every
# headline arm on the SAME pop-999 populations and route files as the v4c
# confirm. HYGIENE: the pre-registered verdicts remain those of the original
# seeds 42-45; this extension is reported IN FULL (all seeds pooled, no
# selection) regardless of outcome. Routing is per-episode and seed-
# independent, so the routed subsets are byte-identical across seeds.
#   arms: flat2, flat5, spec, r_flat2, r_flat5, r_spec, unif(spec-leg)
#   x t{25,50,100,150} x seeds{46-49}
# ==============================================================================
set -euo pipefail
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
ARMS=(flat2 flat5 spec r_flat2 r_flat5 r_spec unif)
SUBGOALS=(baseline baseline specaccept baseline baseline specaccept unified)
RHS=(2 5 2 2 5 2 2)
BRANCHES=(- - - flat2 flat5 spec spec)
OFFSETS=(25 50 100 150)
A=$((SLURM_ARRAY_TASK_ID / 16))
O=$(((SLURM_ARRAY_TASK_ID / 4) % 4))
SEED=$((46 + SLURM_ARRAY_TASK_ID % 4))
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
  echo "=== rea-ext arm=$NAME t=$OFF seed=$SEED: EMPTY SUBSET -- SR = (0/0), skipped ==="
  exit 0
fi
CKPT_ARG=""
if [ "$SG" != "baseline" ]; then CKPT_ARG="--gdm-ckpt gdm_reacher_s10.pt --gdm-steps 8"; fi
echo "=== rea-ext arm=$NAME t=$OFF seed=$SEED n=$N (budget=$((2*OFF))) ==="
python -m specaccept.envs.reacher.eval --subgoal ${SG} ${CKPT_ARG} \
  --accept-tau 0.20 \
  --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
  --num-eval $N --seed ${SEED} --episodes-file $EPFILE \
  --goal-offset ${OFF} --eval-budget $((2*OFF)) \
  --horizon ${RH} --receding-horizon ${RH} \
  2>&1 | tee logs/rea_ext_${NAME}_t${OFF}_seed${SEED}.log
