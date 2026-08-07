#!/usr/bin/env bash
#SBATCH --job-name=comp-build
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/comp1111_build_%A.log
# ========= P-COMP-1: FRESH CONFIRMATION POPULATIONS, BUILDER SEED 1111 =======
# prereg/2026-08-07_composite_and_executor.md section A. Never-analyzed
# populations for the instrument-routed composite: pusht per-offset files
# (success5 filter, n=256, same convention as the banked episodes150s5*
# populations), reacher max-offset files (n=128, episode-min 8000, gatev4c
# convention), cube samples per-seed at eval time (no files).
# ==============================================================================
set -euo pipefail
export SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
mkdir -p logs
source /lustre/home/ha676/le-wm/.venv/bin/activate

PUSHT_H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
REACHER_H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)

echo "=== pusht populations (seed 1111, success5, n=256/offset) ==="
python -m specaccept.envs.pusht.extract_subset --h5 "$PUSHT_H5" --out pusht_c1111.h5 \
  --seed 1111 --eval-filter success5 --goal-offsets 25 50 75 100 150 \
  --num-eval 256 --compression lzf

echo "=== reacher populations (seed 1111, gatev4c convention) ==="
python batch/build_reacher_horizon_episodes.py --h5 "$REACHER_H5" \
  --out reacher_c1111.ep100.s1111.json --n 128 --max-offset 100 --episode-min 8000 --seed 1111
python batch/build_reacher_horizon_episodes.py --h5 "$REACHER_H5" \
  --out reacher_c1111.ep150.s1111.json --n 128 --max-offset 150 --episode-min 8000 --seed 1111

ls -la pusht_c1111.episodes*.json reacher_c1111.*.json
echo "=== BUILD DONE ==="
