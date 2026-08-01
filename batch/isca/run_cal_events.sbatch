#!/usr/bin/env bash
#SBATCH --job-name=cal-events
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cal_events_%A_%a.log
# ===== M1 PER-EVENT DUMP (08-01 extraction addendum, registered first) =====
# Registered in docs/certification_prereg.md "08-01 EXTRACTION ADDENDUM"
# BEFORE computing: per-event (rel, accepted, state_dist) CSVs from the SAME
# frozen traces and probe definitions, for the descriptive FA(r)/FR(r)
# radius-sensitivity readout and the rel-vs-state operating curve with the
# derived tau marked. QUOTABLE probe class per env (conservative-row rule):
#   0 pusht (mlp) | 1 tworoom (ridge) | 2 cube (ridge) | 3 reacher (mlp circ)
# tau and every decision unchanged; banked JSONs untouched (rerun outputs go
# to *_events_rerun.json).
# ===========================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
mkdir -p logs Results/calibration

case $SLURM_ARRAY_TASK_ID in
  0)
    python -m specaccept.probes.calibrate_verifier --env pusht --probe mlp \
      --h5 /lustre/home/ha676/data/pusht/pusht_expert_train.h5 \
      --traces 'runs/pusht/traces_cal_spec_t150_seed*.pt' --tau 0.20 \
      --out Results/calibration/cal_pusht_events_rerun.json --device cuda \
      --dump-events Results/calibration/events_pusht.csv \
      2>&1 | tee logs/cal_events_pusht.log
    ;;
  1)
    H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
    python -m specaccept.probes.calibrate_verifier --env tworoom \
      --h5 "$H5" \
      --traces 'runs/tworoom/traces_cal_spec_t75_seed*.pt' --tau 0.098 \
      --out Results/calibration/cal_tworoom_events_rerun.json --device cuda \
      --dump-events Results/calibration/events_tworoom.csv \
      2>&1 | tee logs/cal_events_tworoom.log
    ;;
  2)
    python -m specaccept.probes.calibrate_verifier --env cube \
      --h5 /lustre/home/ha676/data/cube/cube_single_expert.h5 \
      --traces 'runs/cube/traces_cal_champion_seed*.pt' --tau 0.20 \
      --out Results/calibration/cal_cube_events_rerun.json --device cuda \
      --dump-events Results/calibration/events_cube.csv \
      2>&1 | tee logs/cal_events_cube.log
    ;;
  3)
    H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
    python -m specaccept.probes.calibrate_verifier --env reacher --probe mlp --circular \
      --h5 "$H5" --source local --local-dir encoder_reacher \
      --traces 'runs/reacher/traces_cal_spec_t150_seed*.pt' --tau 0.20 \
      --out Results/calibration/cal_reacher_events_rerun.json --device cuda \
      --dump-events Results/calibration/events_reacher.csv \
      2>&1 | tee logs/cal_events_reacher.log
    ;;
esac
echo "=== DONE cal-events task $SLURM_ARRAY_TASK_ID ==="
