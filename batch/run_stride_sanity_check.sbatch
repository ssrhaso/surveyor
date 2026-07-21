#!/usr/bin/env bash
#SBATCH --job-name=stride-sanity
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/stride_sanity_%j.log
# Resolve the Stage 2 gate's S=25 "sanity fail": the gate band [0.10,0.25] was
# calibrated to probe_suffix_decay's anchor (0.1593 @ m+1; end-anchored runway
# windows, success5 episodes), but diag_gdm Probe B samples arbitrary-phase
# windows incl. the fast early-push regime; a different, harder distribution.
# Two direct apples-to-apples reads:
#   1. diag_gdm on gdm_faithful.pt with the IDENTICAL settings the stride-25 arm
#      used -> if its B-success rel_err ~ 0.40 too, the rebuilt pipeline IS
#      faithful and the gate band was miscalibrated.
#   2. probe_suffix_decay on gdm_stride25.pt on the EXACT anchor protocol ->
#      must reproduce 0.1593/0.2288/0.3179 (m+1/m+2/m+3) if faithful.
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
ENC=/home/u6ko/hasoshu.u6ko/le-wm/encoder_local
H5=/scratch/u6ko/hasoshu.u6ko/data/pusht/pusht_expert_train.h5
cd /home/u6ko/hasoshu.u6ko/le-wm

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  echo '=== [1/2] diag_gdm on gdm_faithful.pt (identical settings to stride-25 arm) ==='
  python -m specaccept.diag_gdm --gdm-ckpt gdm_faithful.pt --subgoals subgoals_dense_full.pt \
    --subgoal-step 25 --stride 25 --h5 $H5 --device cuda --source local --local-dir $ENC
  echo '=== [2/2] probe_suffix_decay on gdm_stride25.pt (anchor protocol; expect ~0.159/0.229/0.318) ==='
  python -m specaccept.probes.probe_suffix_decay --gdm-ckpt gdm_stride25.pt \
    --source local --local-dir $ENC \
    --h5 subset_longeval.h5 --episodes-file subset_longeval.episodes150.json \
    --device cuda --n-probe 256 --stride 25 --seed 42
"
