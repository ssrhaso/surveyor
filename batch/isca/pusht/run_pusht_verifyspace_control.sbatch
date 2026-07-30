#!/usr/bin/env bash
#SBATCH --job-name=pusht-c1c2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-11
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_c1c2_%A_%a.log
# ========= PUSHT VERIFY-SPACE CONTROL (C1/C2), pre-registered 2026-07-30 =========
# Question: does external-space (DINOv2) verification also work on an env whose
# OWN latent space passes the gap probe? Answers "why not DINOv2 everywhere"
# with a measurement. ONE paired drafter (gdm_pusht_s10_paired_e400v.pt, norm
# gate 1.001/1.002); the arms differ ONLY in which half the accept test reads:
#   C1 --verify-half lewm  tau=0.20 (the env's derived LeWM-space tau)
#   C2 --verify-half dino  tau = criterion floor p50 from probe_floor
#        --verify-space dino (read from the JSON at runtime; derived, no sweep)
# Declared readings frozen in docs/tworoom_paired_prereg.md before any cell.
# t=150 long protocol, internally controlled (C1 vs C2), n=64, seeds 42-47.
# ================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
export PYTHONPATH=/lustre/home/ha676/le-wm
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5

ID=$SLURM_ARRAY_TASK_ID
SEED=$((42 + ID % 6))
if [ "$ID" -lt 6 ]; then
  NAME=c1_lewm; HALF=lewm; TAU=0.20
else
  NAME=c2_dino; HALF=dino
  TAU=$(python - <<'EOF'
import json
d = json.load(open("Results/gap_stat/floor_pusht_dino.json"))
print(round(d["criterion_floor_20deg"]["p50"], 4))
EOF
)
fi

LOG=logs/pusht_c1c2_${NAME}_seed${SEED}.log
echo "=== verify-space control arm=$NAME half=$HALF tau=$TAU seed=$SEED ==="
python -m specaccept.envs.pusht.eval --subgoal specpaired \
  --gdm-ckpt gdm_pusht_s10_paired_e400v.pt --gdm-steps 50 \
  --verify-half "$HALF" --accept-tau "$TAU" \
  --source pretrained --h5 "$H5" --device cuda --mode long \
  --goal-offset 150 --eval-budget 300 --num-eval 64 --score block --angles 20 \
  --seed "$SEED" --cem-seed "$SEED" --horizon 2 --receding-horizon 2 \
  2>&1 | tee "$LOG"
echo "=== DONE $NAME seed=$SEED ==="
