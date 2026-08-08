#!/usr/bin/env bash
#SBATCH --job-name=sgc-blindR
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/lustre/home/ha676/le-wm/logs/sgc_blindR_%A_%a.log
# ========== P-EXEC-7: BLIND-COMMIT CONTROL FOR THE AMORTIZED EXECUTOR =========
# prereg/2026-08-08_overnight_hardening.md section B, frozen before submission.
# Byte-identical to the banked P-EXEC-5 executor arm except --accept-tau 999
# (blind consumption). Banked tau=0.20 references: t=100 91.4, t=150 95.3.
# Frozen: blind loses >= 2pp at BOTH t. Falsifier: ties or wins at both ->
# verification's reacher SR value is executor-dependent, reported as such.
# Serving path proven by the sgc_curve jobs; no smoke needed (tau is the only
# change).
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
mkdir -p logs
source /lustre/home/ha676/le-wm/.venv/bin/activate

ID=$SLURM_ARRAY_TASK_ID
REACHER_H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
OFFS=(100 100 150 150)
OFF=${OFFS[$ID]}; SEED=$((42 + ID % 2)); BUD=$((2 * OFF))
MAXOFF=100; if [ "$OFF" -eq 150 ]; then MAXOFF=150; fi
EPFILE=reacher_gatev4c.ep${MAXOFF}.s999.json
for f in gdm_reacher_s10.pt gcidm_reacher_h50.pt "$EPFILE"; do
  [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done

echo "=== P-EXEC-7 reacher BLIND tau=999 t=$OFF seed=$SEED ==="
python -m specaccept.envs.reacher.eval --subgoal specgcidm \
  --gcidm-ckpt gcidm_reacher_h50.pt --gdm-ckpt gdm_reacher_s10.pt --gdm-steps 4 \
  --accept-tau 999 --sg-steps 10 \
  --source local --local-dir encoder_reacher --h5 "$REACHER_H5" --device cuda \
  --num-eval 128 --seed $SEED --cem-seed $SEED --episodes-file $EPFILE \
  --goal-offset ${OFF} --eval-budget ${BUD} \
  2>&1 | tee logs/sgc_blindR_t${OFF}_seed${SEED}.log
echo "=== sgc-blindR task $ID DONE ==="
