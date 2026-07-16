#!/usr/bin/env bash
#SBATCH --job-name=pusht-specd
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-23
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_specdjob_%A_%a.log
# FINAL dev-env batch: re-stamp every PushT headline/spine spec-accept cell at
# the fully DERIVED config (tau=0.20 = criterion floor, k=3 = sampler
# convergence point; k=3 confirmed on the t100 anchor 2026-07-13: es 63.3,
# spec 64.5 at cr .63 = 1.9 NFE/replan). Invocations mirror the originals
# byte-for-byte except --gdm-steps 8 -> 3. Log prefix *k3 so the k=8 originals
# are preserved (they are the "in-band alternative" row, never overwritten).
#   0-9   success5 spine spec: t{25,50,75,100,150} x seeds 42-43
#   10-19 unfiltered spine spec: same grid
#   20-23 short-horizon replication spec: seeds 42-45
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
ID=$SLURM_ARRAY_TASK_ID
cd /lustre/home/ha676/le-wm
mkdir -p runs/pusht
export SDL_VIDEODRIVER=dummy
source "$VENV/bin/activate"
ARM="--subgoal specaccept --gdm-ckpt gdm_stride10.pt --accept-tau 0.20 --gdm-steps 3 --horizon 2 --receding-horizon 2"
OFFSETS=(25 50 75 100 150)
if [ $ID -le 19 ]; then
  OFF=${OFFSETS[$(((ID % 10) / 2))]}
  SEED=$((42 + ID % 2))
  if [ $ID -le 9 ]; then POP=s5; if [ $OFF -eq 150 ]; then EPFILE=pusht.episodes150s5.json; else EPFILE=pusht.episodes150s5as${OFF}.json; fi
  else POP=unf; if [ $OFF -eq 150 ]; then EPFILE=pusht.episodes150.json; else EPFILE=pusht.episodes150as${OFF}.json; fi; fi
  LOGN=$([ "$POP" = s5 ] && echo hzs5 || echo hz)
  echo "=== derived-config spec (k=3): pop=$POP t$OFF seed=$SEED ==="
  python -m specaccept.envs.pusht.eval $ARM \
    --source pretrained --h5 "$H5" --device cuda \
    --mode long --episodes-file $EPFILE --goal-offset $OFF --eval-budget $((2*OFF)) \
    --num-eval 256 --score block --angles 20 5 --seed $SEED --cem-seed $SEED \
    2>&1 | tee logs/pusht_${LOGN}_s10speck3_t${OFF}_seed${SEED}.log
else
  SEED=$((42 + ID - 20))
  echo "=== derived-config spec (k=3): short repl seed=$SEED ==="
  python -m specaccept.envs.pusht.eval $ARM \
    --source pretrained --h5 "$H5" --device cuda \
    --mode short --eval-filter success5 --num-eval 256 \
    --score block --angles 20 5 --seed $SEED --cem-seed $SEED \
    2>&1 | tee logs/pusht_repl_s10speck3_seed${SEED}.log
fi
