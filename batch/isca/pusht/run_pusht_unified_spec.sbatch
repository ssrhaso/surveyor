#!/usr/bin/env bash
#SBATCH --job-name=unif-pusht
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-9%8
#SBATCH --output=/lustre/home/ha676/le-wm/logs/unif_pusht_%A_%a.log
# ============ UNIFIED METHOD (c*-retire spec): PUSHT SPEC LEG ===============
# PRE-REGISTERED. Same unified source as reacher/cube (CstarRetireSource: one
# rule, one tau, drafting only while c* > tau). Runs the composite's spec leg
# on the routed spec subsets (pusht_gatev4_route_t*_spec.json, route job
# afterok); flat legs shared with the pusht v4 eval array. PushT is where pure
# spec already wins everywhere, so the unified arm's job is DO-NO-HARM:
# BAR (frozen; means over seeds 42-43, 20deg): composite_unified >= speck3
# spine - 1.0pp at every t. The retire check should almost never fire early
# here (drafts arrive natively); c_first/retire stats in the log verify.
# k=3 (derived), tau=0.20 (derived on this env), S=10. Reported as-is.
# ==============================================================================
set -euo pipefail
VENV=/lustre/home/ha676/le-wm/.venv
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
OFFSETS=(25 50 75 100 150)
O=$((SLURM_ARRAY_TASK_ID / 2))
SEED=$((42 + SLURM_ARRAY_TASK_ID % 2))
OFF=${OFFSETS[$O]}
cd /lustre/home/ha676/le-wm
mkdir -p logs
export SDL_VIDEODRIVER=dummy
source "$VENV/bin/activate"
EPFILE=pusht_gatev4_route_t${OFF}_spec.json
N=$(python -c "import json;print(len(json.load(open('$EPFILE'))['episodes']))")
if [ "$N" -eq 0 ]; then
  echo "=== unif-pusht t=$OFF seed=$SEED: EMPTY SUBSET; SR = (0/0), skipped ==="
  exit 0
fi
echo "=== unif-pusht t=$OFF seed=$SEED n=$N (unified c*-retire, budget=$((2*OFF))) ==="
python -m specaccept.envs.pusht.eval --subgoal unified \
  --gdm-ckpt gdm_stride10.pt --gdm-steps 3 --accept-tau 0.20 \
  --source pretrained --h5 "$H5" --device cuda \
  --mode long --episodes-file $EPFILE --goal-offset ${OFF} --eval-budget $((2*OFF)) \
  --num-eval $N --score block --angles 20 5 --seed ${SEED} --cem-seed ${SEED} \
  --horizon 2 --receding-horizon 2 \
  2>&1 | tee logs/unif_pusht_t${OFF}_seed${SEED}.log
