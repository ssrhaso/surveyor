#!/usr/bin/env bash
#SBATCH --job-name=gates-s10
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/dspark_gates_s10_%j.log
# Phase A gate battery for the WINNING stride (S=10, found after the original
# gates ran on s25/s5): A1 few-step curve, A2 acceptance simulator (calibrates
# tau for the s10 spec-accept reading), A3 multimodality at s10.
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
ENC=/home/u6ko/hasoshu.u6ko/le-wm/encoder_local
cd /home/u6ko/hasoshu.u6ko/le-wm
mkdir -p runs/dspark2

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python -m specaccept.probes.probe_dspark_gates --gdm-ckpt gdm_stride10.pt --stride 10 \
    --source local --local-dir $ENC \
    --h5 subset_longeval.h5 --episodes-file subset_longeval.episodes150.json \
    --device cuda --n-probe 256 --seed 42 \
    --json-out runs/dspark2/gates_s10.json
"
