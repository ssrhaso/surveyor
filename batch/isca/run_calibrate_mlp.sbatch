#!/usr/bin/env bash
#SBATCH --job-name=verif-cal-mlp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/home/ha676/le-wm/logs/verif_calmlp_%A_%a.log
# ==== M1 PROBE-CLASS SENSITIVITY ROW (registered amendment, 2026-08-01) ====
# Same frozen definitions and traces, MLP probe instead of ridge. Declared
# reading: probe-independent if FA agrees within 2pp; else the more
# conservative row is the one the paper quotes.
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
      --out Results/calibration/cal_pusht_mlp.json --device cuda \
      2>&1 | tee logs/verif_calmlp_pusht.log
    ;;
  1)
    H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
    python -m specaccept.probes.calibrate_verifier --env tworoom --probe mlp \
      --h5 "$H5" \
      --traces 'runs/tworoom/traces_cal_spec_t75_seed*.pt' --tau 0.098 \
      --out Results/calibration/cal_tworoom_mlp.json --device cuda \
      2>&1 | tee logs/verif_calmlp_tworoom.log
    ;;
  2)
    python -m specaccept.probes.calibrate_verifier --env cube --probe mlp \
      --h5 /lustre/home/ha676/data/cube/cube_single_expert.h5 \
      --traces 'runs/cube/traces_cal_champion_seed*.pt' --tau 0.20 \
      --out Results/calibration/cal_cube_mlp.json --device cuda \
      2>&1 | tee logs/verif_calmlp_cube.log
    ;;
  3)
    H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
    python -m specaccept.probes.calibrate_verifier --env reacher --probe mlp \
      --h5 "$H5" --source local --local-dir encoder_reacher \
      --traces 'runs/reacher/traces_cal_spec_t150_seed*.pt' --tau 0.20 \
      --out Results/calibration/cal_reacher_mlp.json --device cuda \
      2>&1 | tee logs/verif_calmlp_reacher.log
    ;;
esac
echo "=== DONE verif-cal-mlp task $SLURM_ARRAY_TASK_ID ==="
