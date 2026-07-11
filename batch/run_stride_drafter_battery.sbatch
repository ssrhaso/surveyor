#!/usr/bin/env bash
#SBATCH --job-name=stride-battery
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --array=0-3
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/stride_battery_%A_%a.log
# Offline block-quality battery for the four stride-S drafters: per-position
# suffix decay (anchor-comparable windows) + across-seed conditional spread
# (the offline health signal validated against closed-loop SR). Independent of
# the Stage 2 gate resolution; informs Round 2 (fine-stride block drafting for
# speculative acceptance) and doubles as another S=25 faithfulness read
# (must reproduce 0.1593/0.2288/0.3179).
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
ENC=/home/u6ko/hasoshu.u6ko/le-wm/encoder_local
STRIDES=(5 10 15 25)
S=${STRIDES[$SLURM_ARRAY_TASK_ID]}
cd /home/u6ko/hasoshu.u6ko/le-wm
mkdir -p runs/stride

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python -m ffjepa.probes.probe_stride_drafters --gdm-ckpt gdm_stride${S}.pt --stride ${S} \
    --source local --local-dir $ENC \
    --h5 subset_longeval.h5 --episodes-file subset_longeval.episodes150.json \
    --device cuda --n-probe 256 --n-seeds 8 --seed 42 \
    --json-out runs/stride/battery_S${S}.json
"
