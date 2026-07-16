#!/usr/bin/env bash
#SBATCH --job-name=pusht-v4-route
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_v4route_%j.log
# ================ GATE V4 ON PUSHT -- ROUTING PASS (PRE-REGISTERED) ===========
# PushT is already fully won by PURE spec-accept (k=3 derived, tau=0.20, S=10):
# 99.6/97.9/98.8/97.3/98.0 vs flat 98.1/60.4/42.6/12.7/2.7 at t25-t150 (s5
# spine, seeds 42-43). The router is therefore NOT needed to beat LeWM here --
# this run is the unified-policy SAFETY test: the SAME frozen v4 rule
# (c*_2<=tau -> flat2; elif c*_5<=tau -> flat5; else spec) must not BREAK
# spec's dominance by routing episodes to flat arms that collapse at long t
# (flat=2.7% at t150 -- the c*-optimism worst case). One method, all envs, no
# per-env switches: that claim requires this cell.
# tau READ from Results/floor_pusht.json (derived ON pusht). Populations: the
# existing certified s5 spine files (routing has never seen them). Latents via
# the pixel path (no pusht dense-pt): encoder quentinll/lewm-pusht.
# ==============================================================================
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
cd /lustre/home/ha676/le-wm
mkdir -p logs Results
source "$VENV/bin/activate"
export SDL_VIDEODRIVER=dummy
for OFF in 25 50 75 100 150; do
  POP=pusht.episodes150s5as${OFF}.json
  if [ "$OFF" -eq 150 ]; then POP=pusht.episodes150s5.json; fi
  echo "=== pusht v4 route t=$OFF pop=$POP ==="
  python -m specaccept.probes.route_horizon \
    --episodes-file "$POP" --offsets "$OFF" \
    --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-pusht \
    --floor-json Results/floor_pusht.json --device cuda --cem-batch 128 \
    --out-prefix pusht_gatev4_route \
    --json-out Results/pusht_gatev4_route_t${OFF}.json
done
echo "=== pusht v4 routing pass complete ==="
