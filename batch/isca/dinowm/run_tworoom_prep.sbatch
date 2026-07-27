#!/usr/bin/env bash
#SBATCH --job-name=dwm-2roomprep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_2roomprep_%j.log
# ===== TWOROOM RECOVERY PREP: grids -> lens@serving-res -> drafter -> QC =====
# Runs in PARALLEL with WM training (2299632): the encoder is frozen DINOv2,
# so drafter food does not depend on the WM. Grids at img 224 (the WM's
# training resolution) via the training dataset class (stats consistency).
# S=25 raw frames = goal_H 5 x frameskip 5; lens retrained on the 224-res
# pooled latents at hop 25 (the 7/26 lens was on a different encode pass;
# re-derive tau in the exact serving space).
# =============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache PYTHONUNBUFFERED=1
cd /lustre/home/ha676/dino_wm
.venv/bin/python encode_tworoom_grids.py

echo "=== lens at serving resolution + hop (le-wm venv) ==="
cd /lustre/home/ha676/le-wm
.venv/bin/python -m specaccept.probes.learn_readout --name tworoom224_h25 \
  --lat-glob "/lustre/home/ha676/data/dinowm/tworoom_lat224/lat_tworoom224_*.npz" \
  --hop-stride 25 --out readout_tworoom224_h25.pt --device cuda \
  2>&1 | tee logs/lens_tworoom224_h25.log

echo "=== drafter S=25 (le-wm venv) ==="
cd /lustre/home/ha676/le-wm/vjepa2
GRIDS=/lustre/home/ha676/data/dinowm/tworoom_lat224/grids
../.venv/bin/python -m specaccept_vjepa2.train_drafter \
  --data "$GRIDS/grid_tworoom_train_*.npy" \
  --out gdm_dinowm_tworoom_s25.pt \
  --stride 25 --n-future 3 --epochs 20 --batch 32 --val-files 60 \
  --device cuda 2>&1 | tee ../logs/dwm_2room_drafter_train.log

echo "=== drafter QC ==="
VALEPS=$(ls $GRIDS/grid_tworoom_valid_*.npy | head -16 | tr '\n' ' ')
../.venv/bin/python -m specaccept_vjepa2.probe_drafter \
  --ckpt gdm_dinowm_tworoom_s25.pt \
  --episodes $VALEPS \
  --stride 25 --k 8 --device cuda 2>&1 | tee ../logs/dwm_2room_drafter_probe.log
echo "=== TWOROOM PREP DONE ==="
