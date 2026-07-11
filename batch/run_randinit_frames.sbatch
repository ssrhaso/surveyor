#!/usr/bin/env bash
#SBATCH --job-name=randinit-frames
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/randinit_frames_%j.log
# Reviewer-proofing (random-init visual sanity check): batch/run_randinit_winner.sbatch
# (arm s10) produced the paper's hardest-cell winner (100.00/100.00, n=256) but never
# saved a single frame. Small n=8 repeat of the EXACT same arm/config with the new
# --dump-frames-dir flag (additive; subgoal_planner.py/_FFJEPAPolicy captures the raw
# reset frame, the goal-render frame, and the terminal frame per env, forces
# reset_mode='wait' so env-slot <-> episode stays 1:1). Confirms the instrumentation
# doesn't change behavior (still ~perfect success) and gives real PNGs to eyeball.
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
ENC=/home/u6ko/hasoshu.u6ko/le-wm/encoder_local
cd /home/u6ko/hasoshu.u6ko/le-wm
mkdir -p runs/randinit_frames
echo "=== random_init frame capture arm=s10 seed=42 n=8 ==="

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python -m specaccept.envs.pusht.eval --subgoal gdm --gdm-ckpt gdm_stride10.pt \
    --source local --local-dir $ENC \
    --h5 subset_longeval.h5 --device cuda \
    --mode random_init --score block --angles 20 5 --num-eval 8 \
    --seed 42 --cem-seed 42 \
    --horizon 2 --receding-horizon 2 \
    --dump-frames-dir runs/randinit_frames
" 2>&1 | tee logs/randinit_frames.log
