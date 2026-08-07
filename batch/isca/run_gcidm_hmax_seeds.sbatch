#!/usr/bin/env bash
#SBATCH --job-name=gcidm-hseed
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/lustre/home/ha676/le-wm/logs/gcidm_hseed_%A_%a.log
# ===== H_max SEED REPLICATES: IS "BEST SWEPT 86.33" REAL OR TRAINING NOISE? ==
# prereg/2026-08-07_composite_and_executor.md section C. The t=150 H_max sweep
# behind P-HMAX-1's refutation has ONE training seed per point and is
# non-monotone (59.4/85.9/79.3/86.3/60.6), so its max may be selection over
# training noise. Two extra training seeds at each contending H_max, optimizer
# spec and holdout unchanged, evaluated on the identical t=150 cell.
# REPORTING RULE (no falsifier; measurement): "GC-IDM (best swept)" = max over
# all (H_max, seed) runs, labeled oracle selection; "GC-IDM (their spec)"
# stays H_max=T at seed 0.
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
mkdir -p logs
source /lustre/home/ha676/le-wm/.venv/bin/activate

HS=(100 100 150 150 200 200)
SD=(1   2   1   2   1   2)
H=${HS[$SLURM_ARRAY_TASK_ID]}
S=${SD[$SLURM_ARRAY_TASK_ID]}

LAT=subgoals_dense_full.pt
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
EXTRA="--exclude-episodes pusht.episodes150s5as25.json pusht.episodes150s5as50.json pusht.episodes150s5as75.json pusht.episodes150s5as100.json pusht.episodes150s5.json"
OUT=gcidm_pusht_h${H}_s${S}.pt

echo "=== gcidm-hseed train pusht H_max=$H seed=$S ==="
python -m specaccept.train_gcidm \
  --latents "$LAT" --h5 "$H5" --out "$OUT" \
  --h-max ${H} --hidden 512 --depth 3 --dropout 0.1 \
  --epochs 50 --batch-size 1024 --lr 1e-3 --weight-decay 1e-4 \
  --device cuda --seed ${S} $EXTRA \
  2>&1 | tee logs/gcidm_hseed_train_h${H}_s${S}.log

EPFILE=pusht.episodes150s5.json
N=$(python -c "import json;print(len(json.load(open('$EPFILE'))['episodes']))")
echo "=== gcidm-hseed eval pusht t=150 budget=300 H_max=$H seed=$S n=$N ==="
python -m specaccept.envs.pusht.eval --subgoal gcidm --gcidm-ckpt "$OUT" \
  --source pretrained --h5 "$H5" \
  --device cuda --mode long --episodes-file $EPFILE \
  --goal-offset 150 --eval-budget 300 --num-eval $N \
  --score block --angles 20 5 --seed 42 --cem-seed 42 \
  2>&1 | tee logs/gcidm_hseed_pusht_h${H}_s${S}.log
echo "=== DONE H_max=$H seed=$S ==="
