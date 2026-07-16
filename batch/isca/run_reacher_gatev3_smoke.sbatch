#!/usr/bin/env bash
#SBATCH --job-name=gatev3-smoke
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/gatev3_smoke_%j.log
# Plumbing smoke for HorizonGatedSource (gate v3) before the confirmation
# array: 16 episodes, t25, gated arm end-to-end. Pass = the [horizon-gate]
# summary prints a sane fire-rate (expected high at t25), fired episodes
# make zero drafter calls, and SR is in a plausible band. No science here.
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
cd /lustre/home/ha676/le-wm
mkdir -p runs/reacher logs
export MUJOCO_GL=egl
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
python batch/build_reacher_horizon_episodes.py --h5 "$H5" \
  --out reacher_gatev3_smoke.ep.json --n 16 --max-offset 100 --episode-min 8000 --seed 777
python -m specaccept.envs.reacher.eval --subgoal horizon_gated \
  --gdm-ckpt gdm_reacher_s10.pt --gdm-steps 8 --accept-tau 0.20 \
  --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
  --num-eval 16 --seed 42 --episodes-file reacher_gatev3_smoke.ep.json \
  --goal-offset 25 --eval-budget 50 --horizon 2 --receding-horizon 2 \
  --dump-traces runs/reacher/traces_gatev3_smoke.pt \
  2>&1 | tee logs/gatev3_smoke.log
