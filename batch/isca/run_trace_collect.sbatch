#!/usr/bin/env bash
#SBATCH --job-name=trace-collect
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=05:00:00
#SBATCH --array=0-11
#SBATCH --output=/lustre/home/ha676/le-wm/logs/trace_collect_%A_%a.log
# ========= TRACE COLLECTION FOR VERIFIER CALIBRATION (M1) -- NO QUOTABLE SR =========
# Registered in docs/certification_prereg.md ("R2" + same-day amendment): these
# cells' success rates are NOT quotable (banked versions exist); the runs exist
# solely to produce per-VERIFICATION calibration events (env, rel, accepted,
# z_achieved, target) for the calibration readout. All four envs, headline arms.
#   0-2  pusht  spec t150 seeds 42-44   | 3-5  tworoom spec t75 seeds 42-44
#   6-8  cube   champion  seeds 42-44   | 9-11 reacher spec t150 seeds 42-44
# ===================================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
mkdir -p logs runs/pusht runs/tworoom runs/cube runs/reacher

ID=$SLURM_ARRAY_TASK_ID
SEED=$((42 + ID % 3))

case $((ID / 3)) in
  0)
    LOG=logs/trace_collect_pusht_seed${SEED}.log
    echo "=== trace-collect pusht spec t=150 seed=$SEED ==="
    python -m specaccept.envs.pusht.eval --subgoal specaccept \
      --gdm-ckpt gdm_stride10.pt --accept-tau 0.20 --gdm-steps 3 \
      --source pretrained --h5 /lustre/home/ha676/data/pusht/pusht_expert_train.h5 \
      --device cuda --mode long \
      --goal-offset 150 --eval-budget 300 --num-eval 64 --score block --angles 20 \
      --seed "$SEED" --cem-seed "$SEED" --horizon 2 --receding-horizon 2 \
      --dump-traces runs/pusht/traces_cal_spec_t150_seed${SEED}.pt \
      2>&1 | tee "$LOG"
    ;;
  1)
    H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
    LOG=logs/trace_collect_tworoom_seed${SEED}.log
    echo "=== trace-collect tworoom spec t=75 seed=$SEED ==="
    python -m specaccept.envs.tworoom.eval --subgoal specpaired \
      --gdm-ckpt gdm_tworoom_s10_gc_paired_e400v.pt --gdm-steps 50 --accept-tau 0.098 \
      --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
      --device cuda --mode long --goal-offset 75 --eval-budget 150 \
      --eval-filter none --episode-min 4000 --goal-from-proprio \
      --num-eval 64 --seed "$SEED" --cem-seed "$SEED" --receding-horizon 2 \
      --dump-traces runs/tworoom/traces_cal_spec_t75_seed${SEED}.pt \
      2>&1 | tee "$LOG"
    ;;
  2)
    LOG=logs/trace_collect_cube_seed${SEED}.log
    echo "=== trace-collect cube champion seed=$SEED ==="
    python -m specaccept.envs.cube.eval --subgoal specaccept \
      --gdm-ckpt gdm_cube_s10_gc.pt --goal-gate --accept-tau 0.20 --gdm-steps 3 \
      --horizon 2 --receding-horizon 2 \
      --source pretrained --encoder-id quentinll/lewm-cube \
      --h5 /lustre/home/ha676/data/cube/cube_single_expert.h5 --device cuda \
      --start final --goal-offset 150 --eval-budget 300 \
      --num-eval 64 --seed "$SEED" --cem-seed "$SEED" --episode-min 8000 \
      --num-samples 600 --n-steps 30 \
      --dump-traces runs/cube/traces_cal_champion_seed${SEED}.pt \
      2>&1 | tee "$LOG"
    ;;
  3)
    H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
    EPFILE=reacher_cal.ep.t${SLURM_ARRAY_TASK_ID}.json
    LOG=logs/trace_collect_reacher_seed${SEED}.log
    echo "=== trace-collect reacher spec t=150 seed=$SEED ==="
    python batch/build_reacher_horizon_episodes.py --h5 "$H5" --out $EPFILE \
      --n 64 --max-offset 150 --episode-min 8000 --seed 42
    python -m specaccept.envs.reacher.eval --subgoal specaccept \
      --gdm-ckpt gdm_reacher_s10.pt --accept-tau 0.20 --gdm-steps 8 \
      --horizon 2 --receding-horizon 2 \
      --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
      --num-eval 64 --seed "$SEED" --cem-seed "$SEED" --episodes-file $EPFILE \
      --goal-offset 150 --eval-budget 300 \
      --dump-traces runs/reacher/traces_cal_spec_t150_seed${SEED}.pt \
      2>&1 | tee "$LOG"
    ;;
esac
echo "=== DONE trace-collect task $ID ==="
