#!/usr/bin/env bash
# Submit the TwoRoom pipeline as a slurm dependency chain. Run ON the Isambard
# login node from ~/le-wm:  bash batch/submit_tworoom_chain.sh
#
# Unlike Reacher there is no download stage: the crossroom dense latents
# (subgoals_tworoom_dense_crossroom.pt), the h5 collections, and
# encoder_tworoom all exist from the parked TwoRoom phase.
#
#   train (S in {5,10,15,25} goal-cond drafters, array)
#     -> arms (9 configs x seeds 42-45, array)
set -eu
cd ~/le-wm
mkdir -p logs runs/tworoom

for req in subgoals_tworoom_dense_crossroom.pt encoder_tworoom/weights.pt; do
  [ -e "$req" ] || { echo "[chain] MISSING prerequisite: $req" >&2; exit 1; }
done

TRAIN=$(sbatch --parsable batch/run_tworoom_train_array.sbatch)
echo "[chain] train array (S=5,10,15,25): $TRAIN"

ARMS=$(sbatch --parsable --dependency=afterok:$TRAIN batch/run_tworoom_eval_arms.sbatch)
echo "[chain] eval arms (9 configs x 4 seeds): $ARMS"

echo "[chain] submitted. squeue --me to watch; results land in logs/tworoom_arms_*.log"
