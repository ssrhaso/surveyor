#!/usr/bin/env bash
#SBATCH --job-name=verif-cal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/home/ha676/le-wm/logs/verif_cal_%A_%a.log
# ============ VERIFIER CALIBRATION READOUT (M1, PRE-REGISTERED) ============
# Definitions frozen in docs/certification_prereg.md BEFORE any number was
# computed: probe R^2 gate 0.90 (below = row unavailable), accept-vs-reject
# state-distance separation, Spearman rho(rel, state dist), false-accept /
# false-reject rates at each env's own success-criterion radius. Runs after
# the trace-collection battery (submit with --dependency=afterok).
#   0 pusht | 1 tworoom | 2 cube | 3 reacher
# ===========================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
mkdir -p logs Results/calibration

case $SLURM_ARRAY_TASK_ID in
  0)
    python -m specaccept.probes.calibrate_verifier --env pusht \
      --h5 /lustre/home/ha676/data/pusht/pusht_expert_train.h5 \
      --traces 'runs/pusht/traces_cal_spec_t150_seed*.pt' --tau 0.20 \
      --out Results/calibration/cal_pusht.json --device cuda \
      2>&1 | tee logs/verif_cal_pusht.log
    ;;
  1)
    H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
    python -m specaccept.probes.calibrate_verifier --env tworoom \
      --h5 "$H5" \
      --traces 'runs/tworoom/traces_cal_spec_t75_seed*.pt' --tau 0.098 \
      --out Results/calibration/cal_tworoom.json --device cuda \
      2>&1 | tee logs/verif_cal_tworoom.log
    ;;
  2)
    python -m specaccept.probes.calibrate_verifier --env cube \
      --h5 /lustre/home/ha676/data/cube/cube_single_expert.h5 \
      --traces 'runs/cube/traces_cal_champion_seed*.pt' --tau 0.20 \
      --out Results/calibration/cal_cube.json --device cuda \
      2>&1 | tee logs/verif_cal_cube.log
    ;;
  3)
    H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
    python -m specaccept.probes.calibrate_verifier --env reacher \
      --h5 "$H5" --source local --local-dir encoder_reacher \
      --traces 'runs/reacher/traces_cal_spec_t150_seed*.pt' --tau 0.20 \
      --out Results/calibration/cal_reacher.json --device cuda \
      2>&1 | tee logs/verif_cal_reacher.log
    ;;
esac
echo "=== DONE verif-cal task $SLURM_ARRAY_TASK_ID ==="
