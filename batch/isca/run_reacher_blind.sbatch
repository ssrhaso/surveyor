#!/usr/bin/env bash
#SBATCH --job-name=reacher-blind
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --array=0-7
#SBATCH --output=/lustre/home/ha676/le-wm/logs/reacher_blindjob_%A_%a.log
# Blind commitment on Reacher, zero code changes: spec-accept with tau=999
# accepts every waypoint -> consumes the full N=3 block blindly (call ratio
# ~1/3). The PushT matched-k cells showed blind TIES spec in-distribution and
# BEATS it at the k-cliff (63.5 vs 55.6; the "self-test" was false
# rejection under sampler noise). Reacher is where the verifier actually
# fires in-distribution (call ratio 0.62 = real rejections): if those
# rejections are load-bearing, blind loses here; if blind ties here too,
# verification is pure insurance everywhere measured.
#   t {100,150} x seeds 42-45, k=8 (matches the existing hz150 spec cells)
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
OFFSETS=(100 150)
OFF=${OFFSETS[$((SLURM_ARRAY_TASK_ID / 4))]}
SEED=$((42 + SLURM_ARRAY_TASK_ID % 4))
cd /lustre/home/ha676/le-wm
export MUJOCO_GL=egl
EPFILE=reacher_blind.ep150.t${SLURM_ARRAY_TASK_ID}.json
LOG=logs/reacher_blind3_t${OFF}_seed${SEED}.log
echo "=== reacher blind commit-3 (tau=999): t$OFF seed=$SEED (2t, pop=max150) ==="
source "$VENV/bin/activate"
python batch/build_reacher_horizon_episodes.py --h5 "$H5" --out $EPFILE \
  --n 128 --max-offset 150 --episode-min 8000 --seed 42
rm -f "$LOG"
python -m specaccept.envs.reacher.eval --subgoal specaccept \
  --gdm-ckpt gdm_reacher_s10.pt --accept-tau 999 --gdm-steps 8 \
  --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
  --num-eval 128 --seed ${SEED} --episodes-file $EPFILE \
  --goal-offset ${OFF} --eval-budget $((2*OFF)) \
  --horizon 2 --receding-horizon 2 \
  2>&1 | tee -a "$LOG"
