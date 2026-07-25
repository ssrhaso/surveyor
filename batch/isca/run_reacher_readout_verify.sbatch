#!/usr/bin/env bash
#SBATCH --job-name=rea-roverify
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-15%8
#SBATCH --output=/lustre/home/ha676/le-wm/logs/rea_roverify_%A_%a.log
# ============ REACHER: CLOSED-LOOP VALIDATION OF THE LEARNED LENS =============
# Offline result (job 2292631): the time-contrastive readout OPENS reacher's
# verification gap (-0.039 -> +0.031, derived tau=0.309) where pooled rel-L2
# has tail overlap. This battery is the closed-loop test: the FULL-POP spec
# arm of the v4c protocol (pop seed-999 epfiles, n=128, budget 2t), verified
# in lens space at the DERIVED tau, seeds 42-45 x t{25,50,100,150}.
# BASELINE (banked, v4c job 2276034 spec arm, 4-seed means):
#   SR 60.55 / 76.76 / 87.30 / 90.82 at t=25/50/100/150.
# BARS (informal, frozen now): SR >= baseline - 1pp at every t; reject
# behavior non-degenerate (not ~0% and not ~100%); call_ratio <= baseline's.
# Prediction: SR moves little; false-reject churn (call_ratio) drops.
# Report as-is either way - a null here bounds the lens's closed-loop value.
# ==============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
OFFSETS=(25 50 100 150)
O=$((SLURM_ARRAY_TASK_ID / 4))
SEED=$((42 + SLURM_ARRAY_TASK_ID % 4))
OFF=${OFFSETS[$O]}
EPFILE=reacher_gatev4c.ep100.s999.json
[ "$OFF" -eq 150 ] && EPFILE=reacher_gatev4c.ep150.s999.json
cd /lustre/home/ha676/le-wm
mkdir -p logs
export MUJOCO_GL=egl
source "$VENV/bin/activate"
test -f "$EPFILE" || { echo "missing $EPFILE"; exit 1; }
test -f readout_reacher.pt || { echo "missing readout_reacher.pt"; exit 1; }
python -m specaccept.envs.reacher.eval --subgoal specaccept \
  --gdm-ckpt gdm_reacher_s10.pt --gdm-steps 8 --accept-tau 0.20 \
  --verify-readout readout_reacher.pt \
  --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
  --num-eval 128 --seed ${SEED} --cem-seed ${SEED} --episodes-file $EPFILE \
  --goal-offset ${OFF} --eval-budget $((2 * OFF)) \
  --horizon 2 --receding-horizon 2 \
  2>&1 | tee logs/rea_roverify_t${OFF}_seed${SEED}.log
