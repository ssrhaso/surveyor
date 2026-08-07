#!/usr/bin/env bash
#SBATCH --job-name=gcidm-hmaxfix
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --array=0-9%5
#SBATCH --output=/lustre/home/ha676/le-wm/logs/gcidm_hmaxfix_%A_%a.log
# ============ GC-IDM PROTOCOL CORRECTION: H_max = T, PER HORIZON ============
# Supersedes job 2329277. The paper states its rule twice (Sec 4.2 and Sec 5):
# "in all experiments H_max = T = 50", the horizon cap EQUALS the eval budget.
# Our first grid trained one H_max=50 model and served it at budget 2t (up to
# 300), clamping the normalised horizon at 1.0 -- an input never seen in
# training. The paper documents exactly this failure (Appendix E.7 Table J,
# budget sweep at offset 25: Push-T peaks 93.0 at budget 25, drops to 75.0 at
# 100, "consistent with the h_frac clamping effect"). So our measured PushT
# collapse (100.0 -> 63.38) is CONFOUNDED: part real goal-distance
# degradation, part an artifact of breaking their protocol; their genuine
# effect is milder (Table 2, budget 50: Push-T 94.0 -> 70.0, Reacher
# 99.5 -> 99.0, Cube 98.5 -> 93.0).
# THE FIX: one model per (env, horizon) with H_max = T = 2t, plus their
# Appendix F optimiser spec we had also deviated from: AdamW lr=1e-3 (was
# 3e-4), 50 epochs (was 40k steps), cosine to lr/100, batch 1024, grad clip
# 1.0, Kaiming init with a sigma=0.01 output head. Every correction runs
# AGAINST us: it can only strengthen the baseline and shrink our PushT margin.
# Holdout unchanged (reacher/cube < 8000; pusht eval populations excluded).
# ==============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
mkdir -p logs
source /lustre/home/ha676/le-wm/.venv/bin/activate

# (env, goal_offset) -> H_max = 2 * goal_offset
ENVS=(pusht  pusht  pusht  pusht  pusht  reacher reacher reacher reacher cube)
OFFS=(25     50     75     100    150    25      50      100     150     150)
ID=$SLURM_ARRAY_TASK_ID
E=${ENVS[$ID]}; OFF=${OFFS[$ID]}; HMAX=$((2 * OFF))

case $E in
  pusht)
    LAT=subgoals_dense_full.pt
    H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
    EXTRA="--exclude-episodes pusht.episodes150s5as25.json pusht.episodes150s5as50.json pusht.episodes150s5as75.json pusht.episodes150s5as100.json pusht.episodes150s5.json"
    ;;
  reacher)
    LAT=subgoals_reacher_dense.pt
    H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
    EXTRA="--episode-max 8000"
    ;;
  cube)
    LAT=subgoals_cube_dense.pt
    H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
    EXTRA="--episode-max 8000"
    ;;
esac

OUT=gcidm_${E}_h${HMAX}.pt
echo "=== gcidm-hmaxfix $E t=$OFF -> H_max=T=$HMAX (paper rule) $EXTRA ==="
python -m specaccept.train_gcidm \
  --latents "$LAT" --h5 "$H5" --out "$OUT" \
  --h-max ${HMAX} --hidden 512 --depth 3 --dropout 0.1 \
  --epochs 50 --batch-size 1024 --lr 1e-3 --weight-decay 1e-4 \
  --device cuda --seed 0 $EXTRA \
  2>&1 | tee logs/gcidm_hmaxfix_${E}_h${HMAX}.log
echo "=== DONE $E H_max=$HMAX ==="
