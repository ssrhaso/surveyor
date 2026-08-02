#!/usr/bin/env bash
#SBATCH --job-name=taufill
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-17
#SBATCH --output=/lustre/home/ha676/le-wm/logs/taufill_%A_%a.log
# tab:taugrid table-completion cells (docs/taugrid_completion.md, registered
# BEFORE submission). Each cell = its row's original battery config with only
# tau changed. Descriptive, no bar; no banked cell re-run.
#  0-5   pusht t100 k4:  tau {.15,.15,.25,.25,.40,.40}, seed 42+(ID%2)
#  6-9   pusht val t150 k8: tau {.10,.10,.40,.40}, seed 42+(ID%2)
# 10-17  reacher t100 k4: tau .15 (10-13) / .25 (14-17), seed 42+(ID%4)
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
source "$VENV/bin/activate"
cd /lustre/home/ha676/le-wm
ID=$SLURM_ARRAY_TASK_ID
if [ $ID -le 9 ]; then
  H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
  export SDL_VIDEODRIVER=dummy
  SEED=$((42 + ID % 2))
  if [ $ID -le 5 ]; then
    TAUS=(0.15 0.15 0.25 0.25 0.40 0.40); TAU=${TAUS[$ID]}
    echo "=== taufill A: pusht t100 k4 tau=$TAU seed=$SEED ==="
    python -m specaccept.envs.pusht.eval --subgoal specaccept --accept-tau $TAU \
      --gdm-ckpt gdm_stride10.pt --gdm-steps 4 \
      --source pretrained --h5 "$H5" --device cuda \
      --mode long --episodes-file pusht.episodes150as100.json --goal-offset 100 \
      --eval-budget 200 --num-eval 256 --score block --angles 20 5 \
      --seed $SEED --cem-seed $SEED --horizon 2 --receding-horizon 2 \
      2>&1 | tee logs/pusht_taufill_t100_tau${TAU}_k4_seed${SEED}.log
  else
    TAUS=(0.10 0.10 0.40 0.40); TAU=${TAUS[$((ID - 6))]}
    [ -f pusht.episodes150.val.json ] || \
      python batch/isca/pusht/build_pusht_valsplit.py "$H5"
    echo "=== taufill B: pusht val t150 k8 tau=$TAU seed=$SEED ==="
    python -m specaccept.envs.pusht.eval --subgoal specaccept --accept-tau $TAU \
      --gdm-ckpt gdm_stride10.pt --gdm-steps 8 \
      --source pretrained --h5 "$H5" --device cuda \
      --mode long --episodes-file pusht.episodes150.val.json --goal-offset 150 \
      --eval-budget 300 --num-eval 256 --score block --angles 20 5 \
      --seed $SEED --cem-seed $SEED --horizon 2 --receding-horizon 2 \
      2>&1 | tee logs/pusht_valtau${TAU}_seed${SEED}.log
  fi
else
  H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
  export MUJOCO_GL=egl
  SEED=$((42 + ID % 4))
  if [ $ID -le 13 ]; then TAU=0.15; else TAU=0.25; fi
  EPFILE=reacher_taufill.ep.t${ID}.json
  echo "=== taufill C: reacher t100 k4 tau=$TAU seed=$SEED ==="
  python batch/build_reacher_horizon_episodes.py --h5 "$H5" --out $EPFILE \
    --n 128 --max-offset 100 --episode-min 8000 --seed 42
  python -m specaccept.envs.reacher.eval --subgoal specaccept \
    --gdm-ckpt gdm_reacher_s10.pt --accept-tau $TAU --gdm-steps 4 \
    --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
    --num-eval 128 --seed $SEED --episodes-file $EPFILE \
    --goal-offset 100 --eval-budget 200 \
    --horizon 2 --receding-horizon 2 \
    2>&1 | tee logs/reacher_taufill_t100_tau${TAU}_k4_seed${SEED}.log
fi
