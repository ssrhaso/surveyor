#!/usr/bin/env bash
# ISCA port of batch/submit_reacher_chain.sh. Run ON the ISCA login node from
# ~/le-wm:  bash batch/isca/submit_reacher_chain.sh
#
# Unlike Isambard there is NO download stage in the chain: the dataset +
# encoder land via the login-node downloader (~/download_reacher.sh); this
# script refuses to submit until they are present. Run the schema inspection
# on the login node before trusting arm results:
#   source .venv/bin/activate && python batch/inspect_reacher.py
#
# Chain (afterok-gated):
#   smoke  (oracle + baseline: validates env/encoder/harness, no drafter)
#   dense  (stride-1 latent build, 10k episodes)      [parallel w/ smoke]
#     -> train (S in {5,10,15,25} goal-cond drafters, array)
#        -> arms  (9 configs x seeds 42-45, array 0-35)
#        -> sweep (3 arms x 4 offsets x 2 seeds, array 0-23)
set -eu
cd ~/le-wm
mkdir -p logs runs/reacher

H5=$(find ~/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) 2>/dev/null | head -1 || true)
[ -n "$H5" ] || { echo "[chain] MISSING dataset: no .h5 under ~/data/reacher (wait for ~/download.log: DOWNLOAD-DONE)" >&2; exit 1; }
[ -f encoder_reacher/weights.pt ] || { echo "[chain] MISSING encoder_reacher/weights.pt" >&2; exit 1; }
[ -f .venv/bin/activate ] || { echo "[chain] MISSING .venv (wait for ~/bootstrap.log: BOOTSTRAP-DONE)" >&2; exit 1; }

SMOKE=$(sbatch --parsable batch/isca/run_reacher_smoke.sbatch)
echo "[chain] smoke (oracle+baseline): $SMOKE"

DENSE=$(sbatch --parsable batch/isca/run_reacher_dense_build.sbatch)
echo "[chain] dense build: $DENSE"

TRAIN=$(sbatch --parsable --dependency=afterok:$DENSE batch/isca/run_reacher_train_array.sbatch)
echo "[chain] train array (S=5,10,15,25): $TRAIN"

ARMS=$(sbatch --parsable --dependency=afterok:$TRAIN batch/isca/run_reacher_eval_arms.sbatch)
echo "[chain] eval arms (9 configs x 4 seeds): $ARMS"

SWEEP=$(sbatch --parsable --dependency=afterok:$TRAIN batch/isca/run_reacher_horizon_sweep.sbatch)
echo "[chain] horizon sweep (3 arms x 4 offsets x 2 seeds, 2t + vlwm-50 regimes): $SWEEP"

echo "[chain] submitted. squeue --me to watch; results land in logs/reacher_arms_*.log"
echo "[chain] and logs/reacher_horizon_*.log"
