#!/usr/bin/env bash
#SBATCH --job-name=reacher-smoke
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/reacher_smoke_%j.log
# Reacher harness smoke: oracle (true demo latents) + baseline (flat LeWM, goal
# image) at n=32, LeWM's own reacher protocol (goal_offset 25, budget 50,
# qpos_match 0.05 rad). Neither needs a trained drafter, so this validates the
# ENTIRE eval stack (env + qpos_match termination + callables + encoder +
# SubgoalCostModel injection + CEM) before any training compute is spent.
# Success criterion: baseline SR well above 0 (LeWM's published DMC reacher
# numbers are decent) and oracle >= baseline.
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
H5=$(find /scratch/u6ko/hasoshu.u6ko/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
cd /home/u6ko/hasoshu.u6ko/le-wm
mkdir -p runs/reacher
export MUJOCO_GL=egl

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  export MUJOCO_GL=egl
  echo '=== [1/2] ORACLE n=32 (harness + encoder + injection sanity) ==='
  python -m ffjepa.eval_reacher --subgoal oracle \
    --source local --local-dir encoder_reacher --h5 $H5 --device cuda \
    --num-eval 32 --seed 42 --episode-min 8000
  echo '=== [2/2] BASELINE n=32 (flat LeWM, goal image == LeWM protocol) ==='
  python -m ffjepa.eval_reacher --subgoal baseline \
    --source local --local-dir encoder_reacher --h5 $H5 --device cuda \
    --num-eval 32 --seed 42 --episode-min 8000
" 2>&1 | tee logs/reacher_smoke.log
