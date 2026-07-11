#!/usr/bin/env bash
#SBATCH --job-name=reacher-horizon
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-23
#SBATCH --output=/home/u6ko/hasoshu.u6ko/le-wm/logs/reacher_horizon_%A_%a.log
# Reacher fixed-population horizon sweep -- the PushT Section-"Horizon Scaling"
# + VLWM cross-paper cell transplanted. One episode population (same 128
# (episode,start) pairs, holdout >= 8000) at every offset; per task BOTH
# regimes run:
#   * FF-JEPA regime: budget = 2t (the paper's final-window convention)
#   * VLWM regime:    budget = 50 fixed for every t (budget starvation; skipped
#     at t=25 where 2t == 50 and the two regimes coincide -- the PushT dagger note)
# Arms x offsets x seeds: {s25gdm(FF-JEPA native), s10gdm(scale winner),
# s10spec(tau=.20 k=8)} x {25,50,75,100} x {42,43}.
# Requires trained drafters (run_reacher_train_array.sbatch).
set -eu
SIF=/scratch/u6ko/hasoshu.u6ko/containers/pytorch_2412.sif
VENV=/scratch/u6ko/hasoshu.u6ko/venv3
H5=$(find /scratch/u6ko/hasoshu.u6ko/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
ARMS=(s25gdm s10gdm s10spec)
SUBGOALS=(gdm gdm specaccept)
CKPTS=(gdm_reacher_s25.pt gdm_reacher_s10.pt gdm_reacher_s10.pt)
RHS=(5 2 2)
KS=(50 50 8)
OFFSETS=(25 50 75 100)
SEEDS=(42 43)
A=$((SLURM_ARRAY_TASK_ID / 8))
O=$(((SLURM_ARRAY_TASK_ID / 2) % 4))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 2))]}
NAME=${ARMS[$A]}; SG=${SUBGOALS[$A]}; CKPT=${CKPTS[$A]}; RH=${RHS[$A]}; K=${KS[$A]}
OFF=${OFFSETS[$O]}
cd /home/u6ko/hasoshu.u6ko/le-wm
mkdir -p runs/reacher
EPFILE=reacher_horizon.ep.t${SLURM_ARRAY_TASK_ID}.json
echo "=== arm=$NAME offset=$OFF seed=$SEED (budgets: 2t=$((2*OFF)), vlwm=50) ==="

apptainer exec --nv --bind /scratch:/scratch,/home:/home "$SIF" bash -c "
  source $VENV/bin/activate
  export MUJOCO_GL=egl
  # deterministic (seed 42): every task regenerates identical content, no races
  python batch/build_reacher_horizon_episodes.py --h5 $H5 --out $EPFILE \
    --n 128 --max-offset 100 --episode-min 8000 --seed 42
  echo '--- FF-JEPA regime: budget 2t ---'
  python -m specaccept.envs.reacher.eval --subgoal ${SG} --gdm-ckpt ${CKPT} \
    --accept-tau 0.20 --gdm-steps ${K} \
    --source local --local-dir encoder_reacher --h5 $H5 --device cuda \
    --num-eval 128 --seed ${SEED} --episodes-file $EPFILE \
    --goal-offset ${OFF} --eval-budget $((2*OFF)) \
    --horizon ${RH} --receding-horizon ${RH} \
    --dump-traces runs/reacher/traces_hz_${NAME}_t${OFF}_2t_seed${SEED}.pt
  if [ ${OFF} -ne 25 ]; then
    echo '--- VLWM regime: budget 50 fixed ---'
    python -m specaccept.envs.reacher.eval --subgoal ${SG} --gdm-ckpt ${CKPT} \
      --accept-tau 0.20 --gdm-steps ${K} \
      --source local --local-dir encoder_reacher --h5 $H5 --device cuda \
      --num-eval 128 --seed ${SEED} --episodes-file $EPFILE \
      --goal-offset ${OFF} --eval-budget 50 \
      --horizon ${RH} --receding-horizon ${RH} \
      --dump-traces runs/reacher/traces_hz_${NAME}_t${OFF}_b50_seed${SEED}.pt
  fi
" 2>&1 | tee logs/reacher_horizon_${NAME}_t${OFF}_seed${SEED}.log
