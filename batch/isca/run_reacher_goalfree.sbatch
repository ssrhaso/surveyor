#!/usr/bin/env bash
#SBATCH --job-name=reacher-gf-train
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/reacher_gf_train_%j.log
# GOAL-FREE ablation drafter at S=10: identical faithful recipe to the
# goal-cond battery drafters MINUS the goal conditioning. Measures the
# "goal-free drafting is ill-posed on wander data" claim (the TwoRoom
# lesson) instead of asserting it. Prediction on record: well below the
# goal-cond drafter everywhere. Eval via run_reacher_goalfree_eval.sbatch
# (submitted afterok by submit_goalfree_chain).
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
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
echo "=== train gdm_reacher_s10_goalfree.pt (faithful recipe, NO goal-cond, eps<8000) ==="
python -m specaccept.train_drafter --subgoals subgoals_reacher_dense.pt \
  --out gdm_reacher_s10_goalfree.pt \
  --device cuda --mask custom --custom-mask-file reacher_train_mask.npy \
  --subgoal-step 10 \
  --batch-size 128 --lr 5e-5 --weight-decay 1e-3 --grad-clip 1.0 \
  --lr-schedule warmup_cosine --epochs 20 --ema-decay 0 --amp
