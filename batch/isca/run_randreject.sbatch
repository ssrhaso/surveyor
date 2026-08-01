#!/usr/bin/env bash
#SBATCH --job-name=randreject
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --array=0-11
#SBATCH --output=/lustre/home/ha676/le-wm/logs/randreject_%A_%a.log
# ========= MATCHED-RATE RANDOM-REJECTION CONTROL (PRE-REGISTERED) =========
# docs/randreject_prereg.md frozen 2026-08-01 BEFORE this submission. The
# decision-content control: the verifier's per-event accept/reject decision
# is replaced by an i.i.d. coin at the banked cells' own measured per-event
# rejection probability (derived from printed log counters: p = rejects /
# [advances + rejects + (redrafts - rejects - first-draws)]). Everything
# else byte-identical to the banked fixed-spec arms. Banked cells NOT re-run;
# comparison columns recorded in the prereg. Bar: banked spec >= coin + 3pp,
# paired t >= 2, per env. Validity gate: coin call_ratio within +/-0.05 of
# banked (pusht 0.598, reacher 0.612). Single shot, no p adjustment.
#   0-3  pusht  t150 seeds 42-45, p=0.4937, pop pusht.episodes150s5.json n=256
#   4-11 reacher t150 seeds 42-49, p=0.4902, pop reacher_gatev4c.ep150.s999 n=128
# ==========================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
mkdir -p logs

ID=$SLURM_ARRAY_TASK_ID
if [ "$ID" -lt 4 ]; then
  SEED=$((42 + ID))
  LOG=logs/randreject_pusht_t150_seed${SEED}.log
  echo "=== randreject pusht t=150 seed=$SEED p=0.4937 ==="
  python -m specaccept.envs.pusht.eval --subgoal specaccept \
    --gdm-ckpt gdm_stride10.pt --accept-tau 0.20 --gdm-steps 3 \
    --random-reject 0.4937 \
    --source pretrained --h5 /lustre/home/ha676/data/pusht/pusht_expert_train.h5 \
    --device cuda --mode long --episodes-file pusht.episodes150s5.json \
    --goal-offset 150 --eval-budget 300 --num-eval 256 \
    --score block --angles 20 5 --seed "$SEED" --cem-seed "$SEED" \
    --horizon 2 --receding-horizon 2 \
    2>&1 | tee "$LOG"
else
  SEED=$((42 + ID - 4))
  H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
  LOG=logs/randreject_reacher_t150_seed${SEED}.log
  echo "=== randreject reacher t=150 seed=$SEED p=0.4902 ==="
  python -m specaccept.envs.reacher.eval --subgoal specaccept \
    --gdm-ckpt gdm_reacher_s10.pt --gdm-steps 8 --accept-tau 0.20 \
    --random-reject 0.4902 \
    --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
    --num-eval 128 --seed "$SEED" --cem-seed "$SEED" \
    --episodes-file reacher_gatev4c.ep150.s999.json \
    --goal-offset 150 --eval-budget 300 \
    --horizon 2 --receding-horizon 2 \
    2>&1 | tee "$LOG"
fi
echo "=== DONE randreject task $ID ==="
