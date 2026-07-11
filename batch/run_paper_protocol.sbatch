#!/usr/bin/env bash
#SBATCH --job-name=paper-proto
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/paper_proto_%A_%a.log
# Fill the two remaining headline-table cells: the PAPER's short (t=25, budget
# 50) and long (t=75, budget 150) protocols, n=256, success5 population --
# the same protocol as our gdm_faithful repro (95.31 short / 90.0 long) --
# run with the winning configuration. Two configs per cell for attribution:
#   gdm  = S=10 drafter, every-step drafting (the SR treatment alone)
#   spec = + reality-verified consumption + 8-step drafts (the full system)
# Long uses the exact repro episode set (subset_longeval.episodes75.json);
# short samples the success5 pool as in the repro protocol.
#   task = CONFIG(2) x PROTO(2) x SEED(2): idx = c*4 + p*2 + s
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
ENC=/home/u6ko/hasoshu.u6ko/le-wm/encoder_local
CONFIGS=(gdm spec)
PROTOS=(short long75)
SEEDS=(42 43)
C=$((SLURM_ARRAY_TASK_ID / 4))
P=$(((SLURM_ARRAY_TASK_ID / 2) % 2))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 2))]}
CFG=${CONFIGS[$C]}; PROTO=${PROTOS[$P]}
cd /home/u6ko/hasoshu.u6ko/le-wm

if [ "$CFG" = "spec" ]; then
  SUBGOAL_ARGS="--subgoal specaccept --accept-tau 0.20 --gdm-steps 8"
else
  SUBGOAL_ARGS="--subgoal gdm"
fi
if [ "$PROTO" = "short" ]; then
  PROTO_ARGS="--mode short --eval-filter success5 --num-eval 256"
else
  PROTO_ARGS="--mode long --episodes-file subset_longeval.episodes75.json --goal-offset 75 --eval-budget 150 --num-eval 256"
fi
echo "=== paper-protocol cfg=$CFG proto=$PROTO seed=$SEED ==="

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python -m specaccept.envs.pusht.eval $SUBGOAL_ARGS --gdm-ckpt gdm_stride10.pt \
    --source local --local-dir $ENC --h5 subset_longeval.h5 --device cuda \
    $PROTO_ARGS --score block --angles 20 5 --seed ${SEED} --cem-seed ${SEED} \
    --horizon 2 --receding-horizon 2
" 2>&1 | tee logs/paper_proto_${CFG}_${PROTO}_seed${SEED}.log
