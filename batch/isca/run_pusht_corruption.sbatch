#!/usr/bin/env bash
#SBATCH --job-name=pusht-corrupt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-23
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_corrupt_%A_%a.log
# ========== DRAFTER CORRUPTION SWEEP (PRE-REGISTERED, certification leg) ==========
# Bars + predictions frozen in docs/certification_prereg.md ("R1") BEFORE
# submission. What does reality-verification BUY when the drafter degrades?
# Every drafted waypoint displaced by sigma*||w|| (random unit direction, every
# draft incl. re-drafts). Verified (tau=0.20 derived) vs BLIND (tau=999 accepts
# all = blind commit-N). P-CERT-1: the offline-derived tau localizes the
# closed-loop rejection-ignition point (call_ratio -> 1 as sigma crosses
# [tau/2, 2tau]). P-CERT-2: verified degrades gracefully, blind compounds.
#   sigma in {0, 0.1, 0.2, 0.4} x {verified, blind} x seeds 42-44, n=64,
#   pusht t=150 mode long, budget 300, block-20, k=3, S=10, RH2.
# ==================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
mkdir -p logs
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5

ID=$SLURM_ARRAY_TASK_ID
SIGMAS=(0 0.1 0.2 0.4)
SIG=${SIGMAS[$((ID / 6))]}
REST=$((ID % 6))
SEED=$((42 + REST % 3))
if [ $((REST / 3)) -eq 0 ]; then NAME=ver;   TAU=0.20; else NAME=blind; TAU=999; fi

LOG=logs/pusht_corrupt_${NAME}_sig${SIG}_seed${SEED}.log
echo "=== corruption arm=$NAME tau=$TAU sigma=$SIG seed=$SEED ==="
python -m specaccept.envs.pusht.eval --subgoal specaccept \
  --gdm-ckpt gdm_stride10.pt --accept-tau "$TAU" --gdm-steps 3 \
  --draft-noise "$SIG" \
  --source pretrained --h5 "$H5" --device cuda --mode long \
  --goal-offset 150 --eval-budget 300 --num-eval 64 --score block --angles 20 \
  --seed "$SEED" --cem-seed "$SEED" --horizon 2 --receding-horizon 2 \
  2>&1 | tee "$LOG"
echo "=== DONE $NAME sig=$SIG seed=$SEED ==="
