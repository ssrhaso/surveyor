#!/usr/bin/env bash
#SBATCH --job-name=spine-t125
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:30:00
#SBATCH --array=0-11
#SBATCH --output=/lustre/home/ha676/le-wm/logs/spine_t125_%A_%a.log
# Table-completion cells for tab:spine(c): the three non-S10 every-step arms at
# the interior offset t=125, on the SAME max-offset-150 fixed population as
# run_reacher_horizon150.sbatch (builder seed 42, episode-min 8000, n=128,
# budget 2t). Registered in docs/spine_t125_completion.md BEFORE submission:
# descriptive only, no bar; no banked cell is re-run. Log names use the hz150
# prefix so the aggregator picks them up alongside the existing cells.
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
ARMS=(s5gdm s15gdm s25gdm)
CKPTS=(gdm_reacher_s5.pt gdm_reacher_s15.pt gdm_reacher_s25.pt)
RHS=(1 3 5)
SEEDS=(42 43 44 45)
A=$((SLURM_ARRAY_TASK_ID / 4))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 4))]}
NAME=${ARMS[$A]}; CKPT=${CKPTS[$A]}; RH=${RHS[$A]}
OFF=125
cd /lustre/home/ha676/le-wm
export MUJOCO_GL=egl
EPFILE=reacher_spine_t125.ep.t${SLURM_ARRAY_TASK_ID}.json
LOG=logs/reacher_hz150_${NAME}_t${OFF}_seed${SEED}.log
echo "=== arm=$NAME offset=$OFF seed=$SEED (2t budget=250, pop=max150) ==="
source "$VENV/bin/activate"
python batch/build_reacher_horizon_episodes.py --h5 "$H5" --out $EPFILE \
  --n 128 --max-offset 150 --episode-min 8000 --seed 42
rm -f "$LOG"
python -m specaccept.envs.reacher.eval --subgoal gdm --gdm-ckpt ${CKPT} \
  --accept-tau 0.20 --gdm-steps 50 \
  --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
  --num-eval 128 --seed ${SEED} --episodes-file $EPFILE \
  --goal-offset ${OFF} --eval-budget 250 \
  --horizon ${RH} --receding-horizon ${RH} \
  2>&1 | tee -a "$LOG"
