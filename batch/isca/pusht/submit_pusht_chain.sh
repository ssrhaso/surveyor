#!/usr/bin/env bash
# PushT Stage A on ISCA: sanity -> dense -> train(3); smoke rides alongside.
# Run ON the ISCA login node from ~/le-wm AFTER deps + dataset are staged:
#   bash batch/isca/pusht/submit_pusht_chain.sh
# Stage B (replication arms, RH sweep, frontier, timing, goal-cond evals) is
# submitted separately once the drafters + eval populations exist.
set -eu
cd ~/le-wm
mkdir -p logs runs/pusht

H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
[ -f "$H5" ] || { echo "[chain] MISSING $H5 (wait for PUSHT-DOWNLOAD-DONE in ~/pusht_download.log)" >&2; exit 1; }
python -c "import Box2D" 2>/dev/null || { echo "[chain] MISSING box2d in venv (wait for PUSHT-DEPS-DONE in ~/pusht_deps.log)" >&2; exit 1; }

SAN=$(sbatch --parsable batch/isca/pusht/run_pusht_sanity.sbatch)
echo "[chain] sanity (encoder download + env + h5): $SAN"

SMOKE=$(sbatch --parsable --dependency=afterok:$SAN batch/isca/pusht/run_pusht_smoke.sbatch)
echo "[chain] smoke (baseline n=32, informational): $SMOKE"

DENSE=$(sbatch --parsable --dependency=afterok:$SAN batch/isca/pusht/run_pusht_dense_build.sbatch)
echo "[chain] dense build: $DENSE"

TRAIN=$(sbatch --parsable --dependency=afterok:$DENSE batch/isca/pusht/run_pusht_train_array.sbatch)
echo "[chain] train array (s10, s25, s10-goalcond): $TRAIN"

echo "[chain] submitted. Stage B follows once drafters exist."
squeue --me
