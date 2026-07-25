#!/usr/bin/env bash
#SBATCH --job-name=vj2-cstar2x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=05:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/vj2_cstar2x_%j.log
# ============ C*-V2 LADDER EXTENSION: 3 -> 20 EPISODES ========================
# The calibrated-certificate claim (token metric + self-floor rises with
# offset, SNR 4-6x) currently rests on 3 episodes - the thinnest evidence
# base of anything the paper will cite. Extend to 20 episodes (declared
# NOW: droid_ep000..droid_ep019, no selection). Report: per-episode
# calibrated ladders + the pooled rise (t-cal at far offsets minus t-cal at
# t<=2) with its spread across episodes.
# ==============================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache PYTHONUNBUFFERED=1
source /lustre/home/ha676/le-wm/.venv/bin/activate
cd /lustre/home/ha676/le-wm/vjepa2
mkdir -p ../logs
EPS=""
for i in $(seq -w 0 19); do EPS="$EPS /lustre/home/ha676/data/droid_eps/droid_ep0${i}.npz"; done
python -m specaccept_vjepa2.probe_cstar2 \
  --traj $EPS --device cuda --cem-samples 200 --cem-steps 5 \
  --offsets 1 2 4 8 16 32 64 \
  2>&1 | tee ../logs/vj2_cstar2_ext.log
echo "=== CSTAR2 EXTENSION DONE ==="
