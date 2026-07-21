#!/usr/bin/env bash
#SBATCH --job-name=reacher-hz150
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-35
#SBATCH --output=/lustre/home/ha676/le-wm/logs/reacher_hz150job_%A_%a.log
# Extended-horizon sweep: does baseline's slide (91.8@t50 -> 83.2@t100) keep
# going at t125/t150 while the drafter holds? {baseline, s10gdm, s10spec} x
# t {100,125,150} x seeds 42-45, 2t budget. PUSHT LESSON: offset-150
# eligibility shrinks the episode pool, so this is its OWN fixed population
# (max-offset 150) INCLUDING a t100 re-run for continuity; results are a
# separate table, never spliced onto the max-offset-100 sweep. Log prefix
# reacher_hz150_ deliberately does NOT match the aggregator's
# reacher_horizon_* glob (would collide with the population-A t100 cells).
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
ARMS=(baseline s10gdm s10spec)
SUBGOALS=(baseline gdm specaccept)
CKPTS=(none gdm_reacher_s10.pt gdm_reacher_s10.pt)
RHS=(5 2 2)
KS=(50 50 8)
OFFSETS=(100 125 150)
SEEDS=(42 43 44 45)
A=$((SLURM_ARRAY_TASK_ID / 12))
O=$(((SLURM_ARRAY_TASK_ID / 4) % 3))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 4))]}
NAME=${ARMS[$A]}; SG=${SUBGOALS[$A]}; CKPT=${CKPTS[$A]}; RH=${RHS[$A]}; K=${KS[$A]}
OFF=${OFFSETS[$O]}
CKPT_ARG=""
if [ "$CKPT" != "none" ]; then CKPT_ARG="--gdm-ckpt $CKPT"; fi
cd /lustre/home/ha676/le-wm
mkdir -p runs/reacher
export MUJOCO_GL=egl
EPFILE=reacher_horizon150.ep.t${SLURM_ARRAY_TASK_ID}.json
LOG=logs/reacher_hz150_${NAME}_t${OFF}_seed${SEED}.log
echo "=== arm=$NAME offset=$OFF seed=$SEED (2t budget=$((2*OFF)), pop=max150) ==="

source "$VENV/bin/activate"
python batch/build_reacher_horizon_episodes.py --h5 "$H5" --out $EPFILE \
  --n 128 --max-offset 150 --episode-min 8000 --seed 42
rm -f "$LOG"
python -m specaccept.envs.reacher.eval --subgoal ${SG} ${CKPT_ARG} \
  --accept-tau 0.20 --gdm-steps ${K} \
  --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
  --num-eval 128 --seed ${SEED} --episodes-file $EPFILE \
  --goal-offset ${OFF} --eval-budget $((2*OFF)) \
  --horizon ${RH} --receding-horizon ${RH} \
  --dump-traces runs/reacher/traces_hz150_${NAME}_t${OFF}_seed${SEED}.pt \
  2>&1 | tee -a "$LOG"
