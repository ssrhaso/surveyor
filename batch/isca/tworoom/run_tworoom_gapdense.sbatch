#!/usr/bin/env bash
#SBATCH --job-name=tworoom-gd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/tworoom_gd_%j.log
# ============ TWOROOM: STRIDE-1 DENSE ENCODE FOR THE GAP FIGURE ===============
# The five-substrate gap figure needs TwoRoom's EQUIV side (stride-1
# adjacent-frame noise) which the S=10 subgoal file cannot provide. Encode
# 300 head episodes densely, then run gap_stat with equiv=1 / hop=10.
# Expected (from the voracle behavioral evidence): equiv bulk ~ hop bulk ~
# sqrt(2) -> the figure's "saturated at every scale" row, now measured
# directly rather than inferred.
# ==============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
if [ ! -f subgoals_tworoom_s1.pt ]; then
  python -m specaccept.envs.tworoom.build_subgoals \
    --h5 /lustre/home/ha676/data/tworoom/tworoom.h5 \
    --out subgoals_tworoom_s1.pt --source pretrained \
    --encoder-id quentinll/lewm-tworooms --device cuda \
    --stride 1 --max-episodes 300 --sample-mode head \
    2>&1 | tee logs/tworoom_build_s1.log
fi
python -m specaccept.probes.gap_stat --name tworoom \
  --subgoals-pt subgoals_tworoom_s1.pt --equiv-stride 1 --hop-stride 10 \
  --json-out gap_tworoom_dense.json
echo "=== TWOROOM GAP DENSE DONE ==="
