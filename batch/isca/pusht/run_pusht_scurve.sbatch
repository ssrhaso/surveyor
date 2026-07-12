#!/usr/bin/env bash
#SBATCH --job-name=pusht-scurve
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_scurve_train_%A_%a.log
# Interior-optimum symmetry (audit follow-up): Reacher's long-horizon S grid
# has 4 points {5,10,15,25}; PushT's has 2 {10,25} -- a curve vs a line
# segment. Train the missing PushT drafters (faithful recipe, mask 5, goal-
# free); evals at t150 + native follow gated (run_pusht_scurve_eval.sbatch).
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
S=$([ $SLURM_ARRAY_TASK_ID -eq 0 ] && echo 15 || echo 5)
cd /lustre/home/ha676/le-wm
export SDL_VIDEODRIVER=dummy
echo "=== train gdm_stride${S}.pt (faithful recipe, subgoal-step $S) ==="
python -m specaccept.train_drafter --subgoals subgoals_dense_full.pt --out gdm_stride${S}.pt \
  --device cuda --mask 5 --subgoal-step $S \
  --batch-size 128 --lr 5e-5 --weight-decay 1e-3 --grad-clip 1.0 \
  --lr-schedule warmup_cosine --epochs 20 --ema-decay 0 --amp
