#!/usr/bin/env bash
#SBATCH --job-name=poster-strips3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --array=0-7
#SBATCH --output=/lustre/home/ha676/le-wm/logs/poster_strips3_%A_%a.log
# ===== POSTER FILMSTRIPS ROUND 3: SEED VARIETY FOR PANEL SELECTION =====
# Same convention as rounds 1-2: n=8 QUALITATIVE captures, no number from
# these runs may be quoted anywhere. Two extra seeds per env's headline spec
# arm so poster panels can be chosen rather than taken as-is.
#   0-1 pusht spec t150 s43/s44   | 2-3 tworoom spec t75 s43/s44
#   4-5 cube champion s43/s44     | 6-7 reacher spec t150 s43/s44
# =======================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
mkdir -p strips logs

ID=$SLURM_ARRAY_TASK_ID
SEED=$((43 + ID % 2))

case $((ID / 2)) in
  0)
    python -m specaccept.envs.pusht.eval --subgoal specaccept \
      --gdm-ckpt gdm_stride10.pt --accept-tau 0.20 --gdm-steps 3 \
      --source pretrained --h5 /lustre/home/ha676/data/pusht/pusht_expert_train.h5 \
      --device cuda --mode long \
      --goal-offset 150 --eval-budget 300 --num-eval 8 --score block --angles 20 \
      --seed "$SEED" --cem-seed "$SEED" --horizon 2 --receding-horizon 2 \
      --dump-strip 8 --dump-strip-out strips \
      2>&1 | tee logs/poster_strip3_pusht_s${SEED}.log
    ;;
  1)
    H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
    python -m specaccept.envs.tworoom.eval --subgoal specpaired \
      --gdm-ckpt gdm_tworoom_s10_gc_paired_e400v.pt --gdm-steps 50 --accept-tau 0.098 \
      --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
      --device cuda --mode long --goal-offset 75 --eval-budget 150 \
      --eval-filter none --episode-min 4000 --goal-from-proprio \
      --num-eval 8 --seed "$SEED" --cem-seed "$SEED" --receding-horizon 2 \
      --dump-strip 8 --dump-strip-out strips \
      2>&1 | tee logs/poster_strip3_tworoom_s${SEED}.log
    ;;
  2)
    python -m specaccept.envs.cube.eval \
      --subgoal specaccept --gdm-ckpt gdm_cube_s10_gc.pt --goal-gate \
      --accept-tau 0.20 --gdm-steps 3 --horizon 2 --receding-horizon 2 \
      --source pretrained --encoder-id quentinll/lewm-cube \
      --h5 /lustre/home/ha676/data/cube/cube_single_expert.h5 --device cuda \
      --start final --goal-offset 150 --eval-budget 300 \
      --num-eval 8 --seed "$SEED" --cem-seed "$SEED" --episode-min 8000 \
      --num-samples 600 --n-steps 30 \
      --dump-strip 8 --dump-strip-out strips \
      2>&1 | tee logs/poster_strip3_cube_s${SEED}.log
    ;;
  3)
    H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
    EPFILE=reacher_strip3.ep.s${SEED}.json
    python batch/build_reacher_horizon_episodes.py --h5 "$H5" --out $EPFILE \
      --n 8 --max-offset 150 --episode-min 8000 --seed "$SEED"
    python -m specaccept.envs.reacher.eval \
      --subgoal specaccept --gdm-ckpt gdm_reacher_s10.pt \
      --accept-tau 0.20 --gdm-steps 8 --horizon 2 --receding-horizon 2 \
      --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
      --num-eval 8 --seed "$SEED" --cem-seed "$SEED" --episodes-file $EPFILE \
      --goal-offset 150 --eval-budget 300 \
      --dump-strip 8 --dump-strip-out strips \
      2>&1 | tee logs/poster_strip3_reacher_s${SEED}.log
    ;;
esac
echo "=== STRIP3 TASK $ID DONE ==="
