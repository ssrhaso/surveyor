#!/usr/bin/env bash
#SBATCH --job-name=pusht-contain
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-11%8
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_contain_%A_%a.log
# ============== BAD-DRAFTER CONTAINMENT (PRE-REGISTERED) ======================
# Does the certificate bound the damage of a BROKEN drafter? Certified
# spec-accept with the drafter forced BELOW the convergence cliff (k=1: drafts
# are partially denoised noise, bias ~1.0; the measured worst case, no
# retraining needed). In-process unified (routing at first replan + retire),
# full s5as populations. PREDICTIONS (frozen now): (i) at t=25/50 the routing
# floor holds (certificate keeps flat-solvable episodes away from the garbage
# drafter -> ~flat2-level or better); (ii) at t=100 damage is confined to the
# drafting regime (degrades toward every-step k=1 ~56, but stays far above
# flat's 12.7). If (i) fails, containment is NOT a property and we report that.
#   t{25,50,100} x seeds{42-45}
# ==============================================================================
set -euo pipefail
VENV=/lustre/home/ha676/le-wm/.venv
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
OFFSETS=(25 50 100)
O=$((SLURM_ARRAY_TASK_ID / 4)); SEED=$((42 + SLURM_ARRAY_TASK_ID % 4))
OFF=${OFFSETS[$O]}
cd /lustre/home/ha676/le-wm
mkdir -p logs
export SDL_VIDEODRIVER=dummy
source "$VENV/bin/activate"
EPFILE=pusht.episodes150s5as${OFF}.json
echo "=== pusht containment: certified with k=1 drafter, t=$OFF seed=$SEED ==="
python -m specaccept.envs.pusht.eval --subgoal unified \
  --gdm-ckpt gdm_stride10.pt --gdm-steps 1 --accept-tau 0.20 \
  --source pretrained --h5 "$H5" --device cuda \
  --mode long --episodes-file $EPFILE --goal-offset ${OFF} --eval-budget $((2*OFF)) \
  --num-eval 256 --score block --angles 20 --seed ${SEED} --cem-seed ${SEED} \
  --horizon 2 --receding-horizon 2 \
  2>&1 | tee logs/pusht_contain_k1_t${OFF}_seed${SEED}.log
