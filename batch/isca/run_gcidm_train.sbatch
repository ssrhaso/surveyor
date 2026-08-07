#!/usr/bin/env bash
#SBATCH --job-name=gcidm-train
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=/lustre/home/ha676/le-wm/logs/gcidm_train_%A_%a.log
# ================== GC-IDM TRAINING (arXiv 2605.08732 comparator) ============
# Trained to the paper's spec on the stride-1 dense LeWM latent caches; one job
# per env (paper reports ~20 min/env on one GPU).
# HOLDOUT: reacher/cube eval draws episodes >= 8000, so training caps at < 8000.
# PushT eval populations span the whole index range, so every episode named in
# any eval population file is excluded explicitly; without this the comparator
# would train on its own test episodes and any cell it won would be worthless.
# H_max = 50 = the paper's eval budget T (also LeWM's), so the anchor runs at
# the conditioning the model saw. The grid serves the SAME checkpoint at budget
# 2t, where h saturates at 1.0 -- the paper's own rule, not a handicap we
# introduced, and the mechanism P-GCIDM-2 predicts will bite.
# ==============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
mkdir -p logs
source /lustre/home/ha676/le-wm/.venv/bin/activate

case $SLURM_ARRAY_TASK_ID in
  0)
    ENV=pusht
    LAT=subgoals_dense_full.pt
    H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
    EXTRA="--exclude-episodes pusht.episodes150s5as25.json pusht.episodes150s5as50.json pusht.episodes150s5as75.json pusht.episodes150s5as100.json pusht.episodes150s5.json"
    ;;
  1)
    ENV=reacher
    LAT=subgoals_reacher_dense.pt
    H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
    EXTRA="--episode-max 8000"
    ;;
  2)
    ENV=cube
    LAT=subgoals_cube_dense.pt
    H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
    EXTRA="--episode-max 8000"
    ;;
esac

echo "=== gcidm-train $ENV latents=$LAT h5=$H5 $EXTRA ==="
python -m specaccept.train_gcidm \
  --latents "$LAT" --h5 "$H5" --out gcidm_${ENV}.pt \
  --h-max 50 --hidden 512 --depth 3 --dropout 0.1 \
  --steps 40000 --batch-size 1024 --lr 3e-4 --weight-decay 1e-4 \
  --device cuda --seed 0 $EXTRA \
  2>&1 | tee logs/gcidm_train_${ENV}.log
echo "=== GCIDM TRAIN DONE $ENV ==="
