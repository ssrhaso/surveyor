#!/usr/bin/env bash
#SBATCH --job-name=gcidm-sdrep
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/lustre/home/ha676/le-wm/logs/gcidm_sdrep_%A_%a.log
# ====== THEIR-SPEC SEED REPLICATES: pusht H_max in {50,300} x seeds {1,2} =====
# prereg/2026-08-08_overnight_hardening.md section A, frozen before submission.
# The their-spec cells at t=25 (h50) and t=150 (h300, the "collapse to 60.6")
# are single-training-seed reads; the H200 replicates spread 19pp, so both ends
# of the curve need their seed range measured. Reporting rule in the prereg:
# their-spec stays seed 0 in every table; App H gains the 3-seed range; no
# selection in either direction.
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
mkdir -p logs
source /lustre/home/ha676/le-wm/.venv/bin/activate

HS=(50 50 300 300)
SD=(1  2  1   2)
H=${HS[$SLURM_ARRAY_TASK_ID]}
S=${SD[$SLURM_ARRAY_TASK_ID]}

LAT=subgoals_dense_full.pt
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
EXTRA="--exclude-episodes pusht.episodes150s5as25.json pusht.episodes150s5as50.json pusht.episodes150s5as75.json pusht.episodes150s5as100.json pusht.episodes150s5.json"
OUT=gcidm_pusht_h${H}_s${S}.pt

echo "=== gcidm-sdrep train pusht H_max=$H seed=$S ==="
python -m specaccept.train_gcidm \
  --latents "$LAT" --h5 "$H5" --out "$OUT" \
  --h-max ${H} --hidden 512 --depth 3 --dropout 0.1 \
  --epochs 50 --batch-size 1024 --lr 1e-3 --weight-decay 1e-4 \
  --device cuda --seed ${S} $EXTRA \
  2>&1 | tee logs/gcidm_sdrep_train_h${H}_s${S}.log

if [ "$H" -eq 50 ]; then
  OFF=25; EPFILE=pusht.episodes150s5as25.json
else
  OFF=150; EPFILE=pusht.episodes150s5.json
fi
BUD=$((2 * OFF))
N=$(python -c "import json;print(len(json.load(open('$EPFILE'))['episodes']))")
echo "=== gcidm-sdrep eval pusht t=$OFF budget=$BUD H_max=$H seed=$S n=$N ==="
python -m specaccept.envs.pusht.eval --subgoal gcidm --gcidm-ckpt "$OUT" \
  --source pretrained --h5 "$H5" \
  --device cuda --mode long --episodes-file $EPFILE \
  --goal-offset ${OFF} --eval-budget ${BUD} --num-eval $N \
  --score block --angles 20 5 --seed 42 --cem-seed 42 \
  2>&1 | tee logs/gcidm_sdrep_pusht_h${H}_s${S}.log
echo "=== DONE H_max=$H seed=$S ==="
