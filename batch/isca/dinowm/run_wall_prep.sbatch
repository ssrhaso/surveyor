#!/usr/bin/env bash
#SBATCH --job-name=dwm-wallprep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_wallprep_%j.log
# ============ WALL: GRIDS -> LENS(h10) -> DRAFTER(S=10) -> QC ============
# Wall episodes are ~50 frames, so S=10 (N=3 block spans 30 frames). The
# lens is retrained at hop 10 to match the serving hop (the 7/26 lens used
# hop 20); its derived tau feeds the battery's lens arm.
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export DATASET_DIR=/lustre/home/ha676/data/dinowm/datasets
export WANDB_MODE=disabled SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/dino_wm
MODEL_NAME=wall_single MAX_PER_SPLIT=1500 WRITE_GRIDS=1 \
  .venv/bin/python encode_dinowm_env.py

echo "=== lens at the serving hop (le-wm venv) ==="
cd /lustre/home/ha676/le-wm
.venv/bin/python -m specaccept.probes.learn_readout --name dinowm_wall_h10 \
  --lat-glob "/lustre/home/ha676/data/dinowm/wall_single_lat/lat_wall_single_*.npz" \
  --hop-stride 10 --out readout_dinowm_wall_h10.pt --device cuda \
  2>&1 | tee logs/lens_dinowm_wall_h10.log

echo "=== drafter (le-wm venv) ==="
cd /lustre/home/ha676/le-wm/vjepa2
GRIDS=/lustre/home/ha676/data/dinowm/wall_single_lat/grids
../.venv/bin/python -m specaccept_vjepa2.train_drafter \
  --data "$GRIDS/grid_wall_single_train_*.npy" \
  --out gdm_dinowm_wall_s10.pt \
  --stride 10 --n-future 3 --epochs 15 --batch 32 --val-files 60 \
  --device cuda 2>&1 | tee ../logs/dwm_wall_drafter_train.log

echo "=== drafter QC ==="
VALEPS=$(ls $GRIDS/grid_wall_single_valid_*.npy | head -16 | tr '\n' ' ')
../.venv/bin/python -m specaccept_vjepa2.probe_drafter \
  --ckpt gdm_dinowm_wall_s10.pt \
  --episodes $VALEPS \
  --stride 10 --k 8 --device cuda 2>&1 | tee ../logs/dwm_wall_drafter_probe.log
echo "=== WALL PREP DONE ==="
