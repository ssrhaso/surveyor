#!/usr/bin/env bash
#SBATCH --job-name=pusht-audit
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-11
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_auditjob_%A_%a.log
# Reviewer-audit batteries (PushT side), gated on tonight's drafters:
#   0-3   EVERY-STEP k=8 MIRROR (audit #2): s10 gdm, --gdm-steps 8, t150
#         protocol, seeds 42-45. Isolates spec-accept's call reduction against
#         an equally-cheap-per-call baseline. If this arm matches spec-accept's
#         SR at NFE = 8/replan-every-replan, the honest headline shrinks to the
#         call-ratio factor alone; we need to know either way.
#   4-11  TAU RE-SELECTION ON A DISJOINT VAL SPLIT (audit #1): tau in
#         {0.15,0.2,0.25,0.3} x k=8 x seeds 42-43 on pusht.episodes150.VAL
#         (episode-disjoint from the test population; task 0 builds it).
#         The selected tau is then reported on test (already measured for
#         tau=0.2; other winners get a follow-up test run).
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
ID=$SLURM_ARRAY_TASK_ID
cd /lustre/home/ha676/le-wm
mkdir -p runs/pusht
export SDL_VIDEODRIVER=dummy
export PYTHONPATH=/lustre/home/ha676/le-wm
source "$VENV/bin/activate"

# val split is idempotent + deterministic; every task ensures it exists
[ -f pusht.episodes150.val.json ] || \
  python batch/isca/pusht/build_pusht_valsplit.py "$H5"

if [ $ID -le 3 ]; then
  SEED=$((42 + ID))
  echo "=== audit every-step k=8 (t150 test pop) seed=$SEED ==="
  python -m specaccept.envs.pusht.eval --subgoal gdm --gdm-ckpt gdm_stride10.pt \
    --gdm-steps 8 \
    --source pretrained --h5 "$H5" --device cuda \
    --mode long --episodes-file pusht.episodes150.json --goal-offset 150 --eval-budget 300 \
    --num-eval 256 --score block --angles 20 5 --seed $SEED --cem-seed $SEED \
    --horizon 2 --receding-horizon 2 \
    2>&1 | tee logs/pusht_gdmk8_t150_seed${SEED}.log
else
  REL=$((ID - 4))
  TAUS=(0.15 0.20 0.25 0.30)
  TAU=${TAUS[$((REL / 2))]}
  SEED=$((42 + REL % 2))
  echo "=== audit tau-selection on VAL: tau=$TAU seed=$SEED ==="
  python -m specaccept.envs.pusht.eval --subgoal specaccept --gdm-ckpt gdm_stride10.pt \
    --accept-tau $TAU --gdm-steps 8 \
    --source pretrained --h5 "$H5" --device cuda \
    --mode long --episodes-file pusht.episodes150.val.json --goal-offset 150 --eval-budget 300 \
    --num-eval 256 --score block --angles 20 5 --seed $SEED --cem-seed $SEED \
    --horizon 2 --receding-horizon 2 \
    2>&1 | tee logs/pusht_valtau${TAU}_seed${SEED}.log
fi
