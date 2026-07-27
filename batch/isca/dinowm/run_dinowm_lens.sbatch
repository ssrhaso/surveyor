#!/usr/bin/env bash
#SBATCH --job-name=dwm-lens
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/dwm_lens_%j.log
# ===== LENS BATTERY: reacher-shaped DINO substrates (point_maze, wall, 2room) =====
# All three showed bulk-separated, tail-overlapped gaps (docs/dinowm_prereg.md
# P3). The time-contrastive lens opened exactly this shape on reacher
# (-0.039 -> +0.031). If it opens these, the P3 rows gain derived taus;
# still closed = the shape resists a learned metric on generic features too.
# Offline instrument rows only; reacher's lens lesson (efficiency, not SR)
# stands.
# ==================================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
python -m specaccept.probes.learn_readout --name dinowm_point_maze \
  --lat-glob "/lustre/home/ha676/data/dinowm/point_maze_lat/lat_point_maze_*.npz" \
  --hop-stride 25 --out readout_dinowm_point_maze.pt --device cuda \
  2>&1 | tee logs/lens_dinowm_point_maze.log
python -m specaccept.probes.learn_readout --name dinowm_wall \
  --lat-glob "/lustre/home/ha676/data/dinowm/wall_single_lat/lat_wall_single_*.npz" \
  --hop-stride 20 --out readout_dinowm_wall.pt --device cuda \
  2>&1 | tee logs/lens_dinowm_wall.log
python -m specaccept.probes.learn_readout --name tworoom_dino \
  --lat-glob "/lustre/home/ha676/data/dinowm/tworoom_dino_lat/lat_tworoom_*.npz" \
  --hop-stride 25 --out readout_tworoom_dino.pt --device cuda \
  2>&1 | tee logs/lens_tworoom_dino.log
echo "=== DWM LENS BATTERY DONE ==="
