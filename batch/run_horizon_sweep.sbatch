#!/usr/bin/env bash
#SBATCH --job-name=horizon-sweep
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --array=0-23
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/horizon_sweep_%A_%a.log
# Fixed-population SR-vs-horizon curve (the BeyondNext-style question): goal
# offsets t in {25, 50, 75, 100} on the IDENTICAL 256 episodes
# (episodes150asXX reindexes of the trusted t=150 set), budgets 2x offset per
# the FF-JEPA convention. Three arms so the curve attributes cleanly:
#   s25  = gdm_faithful, stride-25, horizon=receding=5 (the FF-JEPA repro arm)
#   s10  = gdm_stride10 every-step, horizon=receding=2 (best-SR config)
#   spec = s10 + reality-verified consumption, tau=0.20 k=8 (best-cost config)
#   task = CONFIG(3) x OFFSET(4) x SEED(2): idx = c*8 + o*2 + s
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
ENC=/home/u6ko/hasoshu.u6ko/le-wm/encoder_local
CONFIGS=(s25 s10 spec)
OFFSETS=(25 50 75 100)
SEEDS=(42 43)
C=$((SLURM_ARRAY_TASK_ID / 8))
O=$(((SLURM_ARRAY_TASK_ID / 2) % 4))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 2))]}
CFG=${CONFIGS[$C]}; OFF=${OFFSETS[$O]}
BUDGET=$((2 * OFF))
EPFILE=subset_longeval.episodes150as${OFF}.json
cd /home/u6ko/hasoshu.u6ko/le-wm

case "$CFG" in
  s25)  ARM_ARGS="--subgoal gdm --gdm-ckpt gdm_faithful.pt --horizon 5 --receding-horizon 5" ;;
  s10)  ARM_ARGS="--subgoal gdm --gdm-ckpt gdm_stride10.pt --horizon 2 --receding-horizon 2" ;;
  spec) ARM_ARGS="--subgoal specaccept --accept-tau 0.20 --gdm-steps 8 --gdm-ckpt gdm_stride10.pt --horizon 2 --receding-horizon 2" ;;
esac
echo "=== horizon-sweep cfg=$CFG offset=$OFF budget=$BUDGET seed=$SEED epfile=$EPFILE ==="

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python -m ffjepa.eval_ffjepa $ARM_ARGS \
    --source local --local-dir $ENC \
    --h5 subset_longeval.h5 --episodes-file $EPFILE --device cuda \
    --mode long --goal-offset $OFF --eval-budget $BUDGET \
    --score block --angles 20 5 --num-eval 256 --seed ${SEED} --cem-seed ${SEED}
" 2>&1 | tee logs/horizon_sweep_${CFG}_t${OFF}_seed${SEED}.log
