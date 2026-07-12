#!/usr/bin/env bash
#SBATCH --job-name=pusht-train
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_train_%A_%a.log
# PushT drafters on ISCA -- the EXACT faithful recipe (batch 128, lr 5e-5,
# wd 1e-3, clip 1.0, warmup_cosine, 20 ep, EMA off, bf16 amp, mask 5 =
# success5 training population):
#   task 0: gdm_stride10.pt      goal-FREE  (headline replication: PushT winner)
#   task 1: gdm_stride25.pt      goal-FREE  (faithful/native reference; its diag
#            curve must reproduce gdm_faithful.pt rel_err ~0.16 -- pipeline gate)
#   task 2: gdm_stride10_gc.pt   GOAL-COND  (--goal-rule final: expert demos end
#            at the goal, the TwoRoom rule) -- fills the last 2x2 ablation cell
#            {env} x {drafter conditioning}
# Stage-2 offline gate (diag_gdm rel_err vs noop degeneracy) runs in-job.
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
cd /lustre/home/ha676/le-wm
export SDL_VIDEODRIVER=dummy
source "$VENV/bin/activate"

case $SLURM_ARRAY_TASK_ID in
  0) OUT=gdm_stride10.pt;    S=10; EXTRA="" ;;
  1) OUT=gdm_stride25.pt;    S=25; EXTRA="" ;;
  2) OUT=gdm_stride10_gc.pt; S=10; EXTRA="--goal-cond --goal-rule final" ;;
esac
echo "=== train $OUT (faithful recipe, subgoal-step $S $EXTRA) ==="
python -m specaccept.train_drafter --subgoals subgoals_dense_full.pt --out $OUT \
  --device cuda --mask 5 --subgoal-step $S $EXTRA \
  --batch-size 128 --lr 5e-5 --weight-decay 1e-3 --grad-clip 1.0 \
  --lr-schedule warmup_cosine --epochs 20 --ema-decay 0 --amp
echo "=== Stage 2 offline gate: diag at subgoal-step $S ==="
python -m specaccept.diag_gdm --gdm-ckpt $OUT --subgoals subgoals_dense_full.pt \
  --subgoal-step $S --stride $S --h5 "$H5" --device cuda --source pretrained
