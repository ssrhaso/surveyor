#!/usr/bin/env bash
#SBATCH --job-name=faith-prep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=/lustre/home/ha676/le-wm/logs/faith_prep_%A_%a.log
# ============= FAITHFULNESS PROGRAM PREP (all offline; no closed loop) =============
# Two independent legs, agreed 2026-07-29 evening:
#   0  TwoRoom k derivation: the one constant in the TwoRoom row not backed by a
#      derivation (50-step default). probe_k.py applies probe_floor section C's
#      rule (smallest k with bias-to-k50 p50 below the criterion floor 0.098,
#      measured in the verification space). A confirm run at k* follows
#      separately if k* < 50.
#   1  PushT DINOv2-verify CONTROL prep: paired stride-10 subgoals (--dino-pair)
#      -> 576-d goal-free drafter (v recipe, mask 5, matching the pusht headline
#      population) -> offline norm gate. Purpose: answer "why not DINOv2
#      everywhere" with a measurement -- both closed-loop arms will share THIS
#      drafter and differ only in which half the verifier reads. tau for the
#      DINO arm is derived tomorrow from a pusht-DINO criterion-floor probe;
#      closed-loop cells fire only after bars are pre-registered.
# ===================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  echo "=== TwoRoom k derivation (paired drafter, verification space) ==="
  python -m specaccept.envs.tworoom.probe_k \
    --subgoals subgoals_tworoom_s10_paired.pt \
    --ckpt gdm_tworoom_s10_gc_paired_e400v.pt \
    --device cuda --n 256 --n-draws 8 --floor 0.098 --seed 0
else
  H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
  SG=subgoals_pusht_s10_paired.pt
  CK=gdm_pusht_s10_paired_e400v.pt
  if [ ! -f "$SG" ]; then
    echo "=== build pusht paired stride-10 subgoals ==="
    python -m specaccept.envs.pusht.build_subgoals \
      --h5 "$H5" --out "$SG" --source pretrained \
      --device cuda --stride 10 --dino-pair
  else
    echo "[skip] $SG exists"
  fi
  if [ ! -f "$CK" ]; then
    echo "=== train 576-d goal-free paired drafter (v recipe, mask 5) ==="
    python -m specaccept.train_drafter \
      --subgoals "$SG" --out "$CK" --latent-dim 576 --n-future 3 \
      --mask 5 --device cuda --seed 42 \
      --epochs 400 --parameterization v --schedule cosine --min-snr-gamma 5.0 \
      2>&1 | tail -8
  else
    echo "[skip] $CK exists"
  fi
  echo "############ OFFLINE GATE: norm ratio + fidelity ############"
  python -m specaccept.envs.tworoom.probe_paired_drafter \
    --subgoals "$SG" --ckpt "$CK" --device cuda --n 256 --steps 50
fi
echo "=== FAITH-PREP TASK $SLURM_ARRAY_TASK_ID DONE ==="
