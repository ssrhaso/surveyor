#!/usr/bin/env bash
#SBATCH --job-name=pusht-timing-u
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_timingU_%A_%a.log
# ============ EFFICIENCY COLUMN: measured wall-clock incl. UNIFIED ============
# Companion to the original run_pusht_timing.sbatch (k50/frozen cells, kept
# untouched): --time-instrument cells so the paper's efficiency table reports
# MEASURED drafter-ms vs CEM-ms per episode for the final arms. Key numbers:
# the unified arm's retire-check overhead (one extra flat CEM per replan on
# still-drafting episodes; the declared <=2x planning cost, attributed to
# t_drafter since the check runs inside source.current) and the retired
# episodes' zero-drafter fast path.
#   0 baseline RH5 t100   1 speck3 t100   2 unified t100 (all-spec regime)
#   3 unified t25 (~50% fire regime: measures the cheap path)
# s5 spine populations, seed 42, n=256, certified budgets.
# ==============================================================================
set -euo pipefail
VENV=/lustre/home/ha676/le-wm/.venv
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
cd /lustre/home/ha676/le-wm
mkdir -p logs
export SDL_VIDEODRIVER=dummy
source "$VENV/bin/activate"
ID=$SLURM_ARRAY_TASK_ID
if   [ $ID -eq 0 ]; then NAME=flat5_t100;   OFF=100; ARM="--subgoal baseline --horizon 5 --receding-horizon 5"
elif [ $ID -eq 1 ]; then NAME=speck3_t100;  OFF=100; ARM="--subgoal specaccept --gdm-ckpt gdm_stride10.pt --accept-tau 0.20 --gdm-steps 3 --horizon 2 --receding-horizon 2"
elif [ $ID -eq 2 ]; then NAME=unified_t100; OFF=100; ARM="--subgoal unified --gdm-ckpt gdm_stride10.pt --accept-tau 0.20 --gdm-steps 3 --horizon 2 --receding-horizon 2"
else                     NAME=unified_t25;  OFF=25;  ARM="--subgoal unified --gdm-ckpt gdm_stride10.pt --accept-tau 0.20 --gdm-steps 3 --horizon 2 --receding-horizon 2"
fi
EPFILE=pusht.episodes150s5as${OFF}.json
echo "=== pusht timing-unified arm=$NAME t=$OFF seed=42 (time-instrument) ==="
python -m specaccept.envs.pusht.eval $ARM \
  --source pretrained --h5 "$H5" --device cuda \
  --mode long --episodes-file $EPFILE --goal-offset ${OFF} --eval-budget $((2*OFF)) \
  --num-eval 256 --score block --angles 20 --seed 42 --cem-seed 42 \
  --time-instrument \
  2>&1 | tee logs/pusht_timingU_${NAME}.log
