#!/usr/bin/env bash
#SBATCH --job-name=dwm-drafter
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_drafter_%j.log
# ========== DINO-WM PUSHT DRAFTER: TokenGDM on (257,384) grids, S=25 ==========
# specaccept_vjepa2.train_drafter runs VERBATIM on the grid npy files (same
# (T, T_tok, D) contract as V-JEPA 2 tokens; dims inferred from data).
# S=25 raw frames = goal_H (the hop where the pre-registered gap [0.108,
# 0.135] is OPEN), N=3, goal-cond on the final-frame grid. Then the held-out
# drafter-quality probe (rel-err vs no-op/lerp, 16 valid eps) runs in-job.
# ==============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm/vjepa2
source ../.venv/bin/activate
GRIDS=/lustre/home/ha676/data/dinowm/pusht_lat/grids
# 18,685 train eps -> ~1.6M pairs: 2 epochs = ~100k steps, already 2-3x the
# optimizer budget of every prior drafter (v1 33k, v3 48k); 12 would blow the
# walltime with no checkpoint saved.
python -m specaccept_vjepa2.train_drafter \
  --data "$GRIDS/grid_pusht_train_*.npy" \
  --out gdm_dinowm_pusht_s25.pt \
  --stride 25 --n-future 3 --epochs 2 --batch 32 --val-files 60 \
  --device cuda 2>&1 | tee ../logs/dwm_drafter_train.log

echo "=== drafter quality probe (held-out valid eps) ==="
VALEPS=$(ls $GRIDS/grid_pusht_valid_*.npy | head -16 | tr '\n' ' ')
python -m specaccept_vjepa2.probe_drafter \
  --ckpt gdm_dinowm_pusht_s25.pt \
  --episodes $VALEPS \
  --stride 25 --k 8 --device cuda 2>&1 | tee ../logs/dwm_drafter_probe.log
echo "=== DWM DRAFTER DONE ==="
