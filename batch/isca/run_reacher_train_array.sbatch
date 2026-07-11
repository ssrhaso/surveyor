#!/usr/bin/env bash
#SBATCH --job-name=reacher-train
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/home/ha676/le-wm/logs/reacher_train_%A_%a.log
# ISCA port of batch/run_reacher_train_array.sbatch (Isambard). Reacher
# drafters at S in {5,10,15,25}, the EXACT faithful recipe (batch 128, lr 5e-5,
# wd 1e-3, clip 1.0, warmup_cosine, 20 ep, EMA off, bf16 amp), GOAL-CONDITIONED
# with SAMPLED hindsight gaps G~U[1,100], HOLDOUT episodes 0..7999 (H8 guard).
# See the Isambard original for the full rationale.
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
STRIDES=(5 10 15 25)
S=${STRIDES[$SLURM_ARRAY_TASK_ID]}
cd /lustre/home/ha676/le-wm

source "$VENV/bin/activate"
python - <<'PY'
import torch, numpy as np
blob = torch.load('subgoals_reacher_dense.pt', map_location='cpu', weights_only=False)
ep = blob['episode_idx'].numpy()
mask = ep < 8000
np.save('reacher_train_mask.npy', mask)
print(f'[mask] train episodes: {mask.sum()}/{len(mask)} (holdout = {int((~mask).sum())})')
PY
echo "=== train gdm_reacher_s${S}.pt (faithful recipe, goal-cond sampled gap 1..100, eps<8000) ==="
python -m specaccept.train_drafter --subgoals subgoals_reacher_dense.pt --out gdm_reacher_s${S}.pt \
  --device cuda --mask custom --custom-mask-file reacher_train_mask.npy \
  --subgoal-step ${S} --goal-cond --goal-rule window --goal-gap 1 --goal-gap-max 100 \
  --batch-size 128 --lr 5e-5 --weight-decay 1e-3 --grad-clip 1.0 \
  --lr-schedule warmup_cosine --epochs 20 --ema-decay 0 --amp
