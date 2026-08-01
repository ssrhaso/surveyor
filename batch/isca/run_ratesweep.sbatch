#!/usr/bin/env bash
#SBATCH --job-name=ratesweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --array=0-19
#SBATCH --output=/lustre/home/ha676/le-wm/logs/ratesweep_%A_%a.log
# ======= REJECTION-RATE DOSE-RESPONSE (RATE-SWEEP AMENDMENT, PRE-REG) =======
# docs/randreject_prereg.md RATE-SWEEP AMENDMENT registered BEFORE this
# submission (openly post-B1-verdict). B1 showed the coin at the matched rate
# TIES certified spec on both t150 cells -> the verifier's closed-loop value
# at nominal = setting/adapting the re-anchoring RATE. This sweep measures
# SR(p) below the matched rate. Endpoints banked: pusht p=0 = blind commit-3
# (-18pp, Isambard logs, NOT re-run); matched-p cells = B1.
#   0-11  reacher t150 s999 pop, p in {0, 0.125, 0.25} x seeds 42-45
#   12-19 pusht  t150 s5 spine pop, p in {0.125, 0.25} x seeds 42-45
# P-RATE-1/2/3 frozen; single battery, no extension.
# ============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
mkdir -p logs

ID=$SLURM_ARRAY_TASK_ID
if [ "$ID" -lt 12 ]; then
  PS=(0 0.125 0.25)
  P=${PS[$((ID / 4))]}
  SEED=$((42 + ID % 4))
  H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
  LOG=logs/ratesweep_reacher_p${P}_seed${SEED}.log
  echo "=== ratesweep reacher t=150 p=$P seed=$SEED ==="
  python -m specaccept.envs.reacher.eval --subgoal specaccept \
    --gdm-ckpt gdm_reacher_s10.pt --gdm-steps 8 --accept-tau 0.20 \
    --random-reject "$P" \
    --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
    --num-eval 128 --seed "$SEED" --cem-seed "$SEED" \
    --episodes-file reacher_gatev4c.ep150.s999.json \
    --goal-offset 150 --eval-budget 300 \
    --horizon 2 --receding-horizon 2 \
    2>&1 | tee "$LOG"
else
  J=$((ID - 12))
  PS=(0.125 0.25)
  P=${PS[$((J / 4))]}
  SEED=$((42 + J % 4))
  LOG=logs/ratesweep_pusht_p${P}_seed${SEED}.log
  echo "=== ratesweep pusht t=150 p=$P seed=$SEED ==="
  python -m specaccept.envs.pusht.eval --subgoal specaccept \
    --gdm-ckpt gdm_stride10.pt --accept-tau 0.20 --gdm-steps 3 \
    --random-reject "$P" \
    --source pretrained --h5 /lustre/home/ha676/data/pusht/pusht_expert_train.h5 \
    --device cuda --mode long --episodes-file pusht.episodes150s5.json \
    --goal-offset 150 --eval-budget 300 --num-eval 256 \
    --score block --angles 20 5 --seed "$SEED" --cem-seed "$SEED" \
    --horizon 2 --receding-horizon 2 \
    2>&1 | tee "$LOG"
fi
echo "=== DONE ratesweep task $ID ==="
