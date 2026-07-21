#!/usr/bin/env bash
#SBATCH --job-name=specaccept-densebuild
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/densebuild_%j.log
# Stage 1 prerequisite (stride-spacing round): rebuild subgoals_dense_full.pt on
# Isambard; the original lives only on unreachable ofs-v01. Same invocation as
# the original (FFJEPA_HANDOFF.md 10.6): build_subgoals --stride 1 (encode every
# frame of every successful@20deg episode; 5deg subset flagged) on the full
# 18,685-episode pusht_expert_train.h5, frozen local encoder. The stride-S
# drafters then train from this ONE file via train_gdm --subgoal-step S.
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
H5=/scratch/u6ko/hasoshu.u6ko/data/pusht/pusht_expert_train.h5
cd /home/u6ko/hasoshu.u6ko/le-wm

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python -m specaccept.envs.pusht.build_subgoals --source local --local-dir encoder_local \
    --h5 $H5 --out subgoals_dense_full.pt --stride 1 --device cuda --batch-size 256
"
