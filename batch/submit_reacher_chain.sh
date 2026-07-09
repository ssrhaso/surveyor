#!/usr/bin/env bash
# Submit the full Reacher pipeline as a slurm dependency chain. Run ON the
# Isambard login node from ~/le-wm:  bash batch/submit_reacher_chain.sh
#
# Chain (each stage gated on the previous with afterok):
#   download+inspect (skipped if reacher.h5 + encoder already present)
#     -> smoke  (oracle + baseline: validates env/encoder/harness, no drafter)
#     -> dense  (stride-1 latent build, 10k episodes)          [parallel w/ smoke]
#        -> train (S in {5,10,25} goal-cond drafters, array)
#           -> arms (6 configs x seeds 42-45, array)
#
# If a stage fails, its dependents sit in DependencyNeverSatisfied - scancel
# them, fix, resubmit. Check the inspect log BEFORE trusting arm results:
#   tail -40 logs/reacher_download_*.log   (schema: needs qpos/qvel/pixels/
#   action + ep_offset/ep_len/episode_idx/step_idx columns)
set -eu
cd ~/le-wm
mkdir -p logs runs/reacher

H5=$(find /scratch/u6ko/hasoshu.u6ko/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) 2>/dev/null | head -1 || true)
DL_DEP=""
QUEUED_DL=$(squeue --me --noheader --format='%i %j' 2>/dev/null | awk '$2=="reacher-download"{print $1; exit}' || true)
if [ -n "$H5" ] && [ -f encoder_reacher/weights.pt ]; then
  echo "[chain] dataset ($H5) + encoder already present; skipping download stage"
elif [ -n "$QUEUED_DL" ]; then
  echo "[chain] reusing already-queued download job: $QUEUED_DL"
  DL_DEP="--dependency=afterok:$QUEUED_DL"
else
  DL=$(sbatch --parsable batch/run_reacher_download_inspect.sbatch)
  echo "[chain] download+inspect: $DL"
  DL_DEP="--dependency=afterok:$DL"
fi

SMOKE=$(sbatch --parsable $DL_DEP batch/run_reacher_smoke.sbatch)
echo "[chain] smoke (oracle+baseline): $SMOKE"

DENSE=$(sbatch --parsable $DL_DEP batch/run_reacher_dense_build.sbatch)
echo "[chain] dense build: $DENSE"

TRAIN=$(sbatch --parsable --dependency=afterok:$DENSE batch/run_reacher_train_array.sbatch)
echo "[chain] train array (S=5,10,15,25): $TRAIN"

ARMS=$(sbatch --parsable --dependency=afterok:$TRAIN batch/run_reacher_eval_arms.sbatch)
echo "[chain] eval arms (9 configs x 4 seeds): $ARMS"

SWEEP=$(sbatch --parsable --dependency=afterok:$TRAIN batch/run_reacher_horizon_sweep.sbatch)
echo "[chain] horizon sweep (3 arms x 4 offsets x 2 seeds, 2t + vlwm-50 regimes): $SWEEP"

echo "[chain] submitted. squeue --me to watch; results land in logs/reacher_arms_*.log"
echo "[chain] and logs/reacher_horizon_*.log"
