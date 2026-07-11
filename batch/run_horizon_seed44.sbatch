#!/usr/bin/env bash
#SBATCH --job-name=horizon-s44
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --array=0-11
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/horizon_s44_%A_%a.log
# Seed extension for the fixed-population horizon sweep (job 5579798, seeds
# 42/43): adds seed 44 to every (config x offset) cell -> 3 seeds per cell.
# Invocations byte-identical to run_horizon_sweep.sbatch except --seed/--cem-seed;
# log naming follows the existing horizon_sweep_<cfg>_t<off>_seed<NN>.log
# convention so the grep patterns keep working.
#   task = CONFIG(3) x OFFSET(4): idx = c*4 + o
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
ENC=/home/u6ko/hasoshu.u6ko/le-wm/encoder_local
CONFIGS=(s25 s10 spec)
OFFSETS=(25 50 75 100)
SEED=44
C=$((SLURM_ARRAY_TASK_ID / 4))
O=$((SLURM_ARRAY_TASK_ID % 4))
CFG=${CONFIGS[$C]}; OFF=${OFFSETS[$O]}
BUDGET=$((2 * OFF))
EPFILE=subset_longeval.episodes150as${OFF}.json
cd /home/u6ko/hasoshu.u6ko/le-wm

case "$CFG" in
  s25)  ARM_ARGS="--subgoal gdm --gdm-ckpt gdm_faithful.pt --horizon 5 --receding-horizon 5" ;;
  s10)  ARM_ARGS="--subgoal gdm --gdm-ckpt gdm_stride10.pt --horizon 2 --receding-horizon 2" ;;
  spec) ARM_ARGS="--subgoal specaccept --accept-tau 0.20 --gdm-steps 8 --gdm-ckpt gdm_stride10.pt --horizon 2 --receding-horizon 2" ;;
esac
echo "=== horizon-seed44 cfg=$CFG offset=$OFF budget=$BUDGET seed=$SEED epfile=$EPFILE ==="

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  python -m ffjepa.eval_ffjepa $ARM_ARGS \
    --source local --local-dir $ENC \
    --h5 subset_longeval.h5 --episodes-file $EPFILE --device cuda \
    --mode long --goal-offset $OFF --eval-budget $BUDGET \
    --score block --angles 20 5 --num-eval 256 --seed ${SEED} --cem-seed ${SEED}
" 2>&1 | tee logs/horizon_sweep_${CFG}_t${OFF}_seed${SEED}.log
