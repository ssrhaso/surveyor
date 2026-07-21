#!/usr/bin/env bash
#SBATCH --job-name=klow
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-31
#SBATCH --output=/lustre/home/ha676/le-wm/logs/klowjob_%A_%a.log
# k-cliff cells: k in {1,2} x {every-step, spec tau=.2} x seeds 42-45 at the
# t100 freeze anchor, BOTH envs. Completes the k cost-quality curve
# (50->8->4->2->1): locates the knee the frozen k sits on. One-step diffusion
# sampling normally degrades sharply; if SR survives at k=1 that is itself
# a finding (max-roughness drafts still steer CEM).
#   0-15 reacher, 16-31 pusht ; within env: arm*8 + kidx*4 + seed
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
ID=$SLURM_ARRAY_TASK_ID
KS=(1 2)
REL=$((ID % 16))
K=${KS[$(((REL / 4) % 2))]}
SEED=$((42 + REL % 4))
if [ $((REL / 8)) -eq 0 ]; then ARM="--subgoal gdm"; AN=gdm; else ARM="--subgoal specaccept --accept-tau 0.20"; AN=spec; fi
cd /lustre/home/ha676/le-wm
if [ $ID -le 15 ]; then
  export MUJOCO_GL=egl
  H5=$(find /lustre/home/ha676/data/reacher \( -name "*.h5" -o -name "*.hdf5" \) | head -1)
  EPFILE=reacher_klow.ep100.t${ID}.json
  python batch/build_reacher_horizon_episodes.py --h5 "$H5" --out $EPFILE \
    --n 128 --max-offset 100 --episode-min 8000 --seed 42
  python -m specaccept.envs.reacher.eval $ARM --gdm-ckpt gdm_reacher_s10.pt --gdm-steps $K \
    --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
    --num-eval 128 --seed $SEED --episodes-file $EPFILE --goal-offset 100 --eval-budget 200 \
    --horizon 2 --receding-horizon 2 \
    2>&1 | tee logs/reacher_${AN}k${K}_t100_seed${SEED}.log
else
  export SDL_VIDEODRIVER=dummy
  H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
  python -m specaccept.envs.pusht.eval $ARM --gdm-ckpt gdm_stride10.pt --gdm-steps $K \
    --source pretrained --h5 "$H5" --device cuda \
    --mode long --episodes-file pusht.episodes150as100.json --goal-offset 100 --eval-budget 200 \
    --num-eval 256 --score block --angles 20 5 --seed $SEED --cem-seed $SEED \
    --horizon 2 --receding-horizon 2 \
    2>&1 | tee logs/pusht_${AN}k${K}_t100_seed${SEED}.log
fi
