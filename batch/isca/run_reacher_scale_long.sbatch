#!/usr/bin/env bash
#SBATCH --job-name=reacher-slong
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-23
#SBATCH --output=/lustre/home/ha676/le-wm/logs/reacher_slongjob_%A_%a.log
# The scale sweep AT LONG HORIZON -- the hole the paper's own logic requires
# filled: S is the one tuned knob (tau=0.2, k=8 frozen globally), so S must be
# swept in the regime the headline claims live in, not only at t=25.
# S in {5,15,25} x t in {100,150} x seeds 42-45 on the max-offset-150 fixed
# population (S=10 already measured there at n=4: t100=94.5, t150=95.1).
# Loop matched per scale: RH = S/5. Every-step k=50 (the SR-reference config).
#   idx = s*8 + o*4 + seed_idx
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
ID=$SLURM_ARRAY_TASK_ID
SCALES=(5 15 25)
RHS=(1 3 5)
OFFSETS=(100 150)
S=${SCALES[$((ID / 8))]}
RH=${RHS[$((ID / 8))]}
OFF=${OFFSETS[$(((ID / 4) % 2))]}
SEED=$((42 + ID % 4))
cd /lustre/home/ha676/le-wm
mkdir -p runs/reacher
export MUJOCO_GL=egl
source "$VENV/bin/activate"
EPFILE=reacher_slong.ep150.t${ID}.json
python batch/build_reacher_horizon_episodes.py --h5 "$H5" --out $EPFILE \
  --n 128 --max-offset 150 --episode-min 8000 --seed 42
echo "=== scale-at-horizon: S=$S RH=$RH t=$OFF seed=$SEED (2t budget) ==="
python -m specaccept.envs.reacher.eval --subgoal gdm \
  --gdm-ckpt gdm_reacher_s${S}.pt --gdm-steps 50 \
  --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
  --num-eval 128 --seed $SEED --episodes-file $EPFILE \
  --goal-offset $OFF --eval-budget $((2*OFF)) \
  --horizon $RH --receding-horizon $RH \
  2>&1 | tee logs/reacher_s${S}_t${OFF}_seed${SEED}.log
