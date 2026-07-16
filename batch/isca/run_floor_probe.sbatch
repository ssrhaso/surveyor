#!/usr/bin/env bash
#SBATCH --job-name=floor-probe
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=/lustre/home/ha676/le-wm/logs/floor_probe_%A_%a.log
# The tau-derivation test (2026-07-13 review): measure the encoder's own
# resolution floor + the sampler's per-k dispersion, both in the accept test's
# units (||a-b||/||a||). If tau*=0.20 sits on the criterion floor and the
# k-cliff is where dispersion crosses it, tau/k become DERIVED quantities,
# not swept constants. Offline: no CEM, no closed loop, ~minutes per env.
#   0 = pusht   1 = reacher
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
cd /lustre/home/ha676/le-wm
mkdir -p Results
export SDL_VIDEODRIVER=dummy MUJOCO_GL=egl
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
  echo "=== floor probe: pusht ==="
  python -m specaccept.probes.probe_floor --env pusht --h5 "$H5" \
    --source pretrained --gdm-ckpt gdm_stride10.pt --device cuda \
    --stride 10 --tau 0.20 --seed 42 --ks 1 2 3 4 5 6 8 12 16 50 \
    --json-out Results/floor_pusht.json \
    2>&1 | tee logs/floor_pusht.log
else
  H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
  echo "=== floor probe: reacher ($H5) ==="
  python -m specaccept.probes.probe_floor --env reacher --h5 "$H5" \
    --source local --local-dir encoder_reacher --gdm-ckpt gdm_reacher_s10.pt \
    --device cuda --stride 10 --tau 0.20 --episode-min 8000 --seed 42 --ks 1 2 3 4 5 6 8 12 16 50 \
    --json-out Results/floor_reacher.json \
    2>&1 | tee logs/floor_reacher.log
fi
