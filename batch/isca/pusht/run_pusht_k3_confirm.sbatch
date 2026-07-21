#!/usr/bin/env bash
#SBATCH --job-name=pusht-k3
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_k3job_%A_%a.log
# THE DERIVATION-CONFIRMATION CELL (floor probe, 2026-07-13): the PushT sampler
# unconverges abruptly between k=3 and k=2 (bias-to-converged-mean 0.11 -> 1.04,
# dispersion 0.08 -> 0.65), so the rule "smallest k with a converged sampler"
# names k=3; a configuration the log-spaced sweep NEVER RAN (it bracketed the
# knee at [2,4]). Prediction: every-step k=3 lands on the k>=4 plateau (~63-64,
# not the ~55 cliff), and spec tau=.2 k=3 matches spec k=4 at ~25% fewer NFE.
# If this hits, k is derived, not swept.
#   0-3 every-step k=3 seeds 42-45     4-7 spec tau=.2 k=3 seeds 42-45   (t100)
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
ID=$SLURM_ARRAY_TASK_ID
SEED=$((42 + ID % 4))
cd /lustre/home/ha676/le-wm
export SDL_VIDEODRIVER=dummy
if [ $ID -le 3 ]; then NAME=gdmk3; ARM="--subgoal gdm"; else NAME=speck3; ARM="--subgoal specaccept --accept-tau 0.20"; fi
echo "=== pusht k=3 confirmation: $NAME seed=$SEED (t100) ==="
python -m specaccept.envs.pusht.eval $ARM --gdm-ckpt gdm_stride10.pt --gdm-steps 3 \
  --source pretrained --h5 "$H5" --device cuda \
  --mode long --episodes-file pusht.episodes150as100.json --goal-offset 100 --eval-budget 200 \
  --num-eval 256 --score block --angles 20 5 --seed $SEED --cem-seed $SEED \
  --horizon 2 --receding-horizon 2 \
  2>&1 | tee logs/pusht_${NAME}_t100_seed${SEED}.log
