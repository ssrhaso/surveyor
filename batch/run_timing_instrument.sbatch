#!/usr/bin/env bash
#SBATCH --job-name=timing-inst
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --array=0-2
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/timing_inst_%A_%a.log
# Reviewer-proofing (wall-clock CEM-vs-drafter split): the writeup claims
# "wall-clock is CEM-dominated and near-identical across all three (within
# +-10% at matched workload; fine stride is never slower)"; an NFE/call-count
# claim, never actually measured in wall-clock. Uses the new --time-instrument
# flag (additive; subgoal_planner.py/_FFJEPAPolicy accumulates t_drafter/t_cem
# with torch.cuda.synchronize() at each boundary). Same controlled protocol as
# the stride-spacing round's tab:spec arms, single seed (42); wall-clock/step
# doesn't vary by seed the way SR does.
#   task 0: s25   every-step, stride-25 drafter, horizon=receding=5
#   task 1: s10   every-step, stride-10 drafter, horizon=receding=2
#   task 2: s10 spec-accept, tau=0.20, gdm-steps=8, horizon=receding=2
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
ENC=/home/u6ko/hasoshu.u6ko/le-wm/encoder_local
cd /home/u6ko/hasoshu.u6ko/le-wm
NEV=$(python3 -c "import json;print(len(json.load(open('dspark_val.episodes.json'))['episodes']))")
COMMON="--source local --local-dir $ENC --h5 subset_longeval.h5 \
  --episodes-file dspark_val.episodes.json --device cuda \
  --mode long --goal-offset 150 --eval-budget 300 \
  --score block --angles 20 5 --num-eval $NEV --seed 42 --time-instrument"

case $SLURM_ARRAY_TASK_ID in
  0)
    NAME=s25
    ARGS="--subgoal gdm --gdm-ckpt gdm_stride25.pt --horizon 5 --receding-horizon 5"
    ;;
  1)
    NAME=s10
    ARGS="--subgoal gdm --gdm-ckpt gdm_stride10.pt --horizon 2 --receding-horizon 2"
    ;;
  2)
    NAME=s10specaccept
    ARGS="--subgoal specaccept --gdm-ckpt gdm_stride10.pt --accept-tau 0.20 --gdm-steps 8 \
      --horizon 2 --receding-horizon 2"
    ;;
esac
echo "=== timing arm=$NAME n=$NEV ==="

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python -m specaccept.envs.pusht.eval $ARGS $COMMON
" 2>&1 | tee logs/timing_inst_${NAME}.log
