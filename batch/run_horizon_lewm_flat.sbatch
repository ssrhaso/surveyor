#!/usr/bin/env bash
#SBATCH --job-name=horizon-flat
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/horizon_flat_%A_%a.log
# Flat-LeWM baseline row for the fixed-population horizon sweep: raw LeWM +
# goal image (--subgoal baseline, == eval.py) on the SAME 256 episodes and
# budgets (2x offset) as the s25/s10/spec arms of job 5579798. This is the
# FF-JEPA Table-I discipline: the flat baseline re-run under OUR protocol,
# never imported. Expected: collapse at t>=50 (their t=75/budget-150 = 3.52%).
#   task = OFFSET(4) x SEED(2): idx = o*2 + s
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
ENC=/home/u6ko/hasoshu.u6ko/le-wm/encoder_local
OFFSETS=(25 50 75 100)
SEEDS=(42 43)
O=$((SLURM_ARRAY_TASK_ID / 2))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 2))]}
OFF=${OFFSETS[$O]}
BUDGET=$((2 * OFF))
EPFILE=subset_longeval.episodes150as${OFF}.json
cd /home/u6ko/hasoshu.u6ko/le-wm
echo "=== horizon-flat (LeWM baseline) offset=$OFF budget=$BUDGET seed=$SEED ==="

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python -m specaccept.envs.pusht.eval --subgoal baseline \
    --source local --local-dir $ENC \
    --h5 subset_longeval.h5 --episodes-file $EPFILE --device cuda \
    --mode long --goal-offset $OFF --eval-budget $BUDGET \
    --score block --angles 20 5 --num-eval 256 --seed ${SEED} --cem-seed ${SEED}
" 2>&1 | tee logs/horizon_sweep_flat_t${OFF}_seed${SEED}.log
