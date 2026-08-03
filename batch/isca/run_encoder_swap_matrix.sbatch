#!/usr/bin/env bash
#SBATCH --job-name=swapmatrix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/lustre/home/ha676/le-wm/logs/swapmatrix_%A_%a.log
# ===== VERIFICATION-SPACE MATRIX COMPLETION, pre-registered 2026-08-03 =====
# docs/encoder_swap_matrix_prereg.md frozen BEFORE this ran.
# Fills the two missing cells of the space x env matrix: Reacher and Cube
# measured in pooled frozen DINOv2, the same probe and flag that produced the
# existing PushT row (Results/gap_stat/floor_pusht_dino.json).
# Offline only: no CEM, no closed loop. No --gdm-ckpt, so part C is skipped.
# Gap read as [criterion_floor p90, disp_S10 p10]; OPEN iff hop p10 > crit p90.
# P-SWAP-1 (reacher): predict OPEN.  P-SWAP-2 (cube): descriptive control.
# Nothing here changes a deployed tau.
# ==========================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
export PYTHONPATH=/lustre/home/ha676/le-wm
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
mkdir -p Results/gap_stat logs

ID=$SLURM_ARRAY_TASK_ID
if [ "$ID" -eq 0 ]; then
  ENV=reacher
  H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
  ENCID=quentinll/lewm-reacher
  EXTRA="--episode-min 8000 --qpos-tol 0.05"
else
  ENV=cube
  H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
  ENCID=quentinll/lewm-cube
  EXTRA="--cube-tol 0.04"
fi

OUT=Results/gap_stat/floor_${ENV}_dino.json
echo "=== encoder-swap matrix: env=${ENV} space=dino h5=${H5} ==="

python -m specaccept.probes.probe_floor --env "$ENV" --h5 "$H5" \
  --source pretrained --encoder-id "$ENCID" \
  --verify-space dino --device cuda \
  --stride 10 --tau 0.20 --seed 42 \
  --n-anchors 512 --pair-pool 120000 \
  $EXTRA \
  --json-out "$OUT" \
  2>&1 | tee "logs/swapmatrix_${ENV}_dino.log"

echo "=== ${ENV} done -> ${OUT} ==="
python - <<EOF
import json
d = json.load(open("$OUT"))
crit = next(v for k, v in d.items() if k.startswith("criterion_floor"))
hop = d["disp_S10"]
lo, hi = crit["p90"], hop["p10"]
print(f"[gap] ${ENV} in DINOv2: criterion p90={lo:.4f}  hop p10={hi:.4f}  "
      f"-> {'OPEN' if hi > lo else 'CLOSED'}"
      + (f"  midpoint tau={0.5*(lo+hi):.4f}" if hi > lo else ""))
EOF
