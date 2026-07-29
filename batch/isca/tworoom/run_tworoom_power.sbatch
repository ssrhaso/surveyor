#!/usr/bin/env bash
#SBATCH --job-name=2room-power
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --array=0-23
#SBATCH --output=/lustre/home/ha676/le-wm/logs/2room_power_%A_%a.log
# ============ TWOROOM: POWER ON THE ONE CELL THAT BEATS FLAT ============
# t=75 RH=2 is the only cell where anything clears flat. At six seeds:
#   flat                  43.75  sd 9.00
#   gdm192                47.66  sd 6.90
#   pair, no verification 49.22  sd 11.55   <- +5.5pp over flat
#   pair + verification   45.31  sd 7.19
# With sd 11.55 and n=6 the standard error is about 4.7, so +5.5pp is barely
# one SE. Every margin in this project that was quoted at n<=2 has since
# evaporated (+18.8 -> +1.6, +10.9 -> -2.1), so this one gets powered before
# it is described as anything.
#
# Runs seeds 48-53, which COMBINE with the existing 42-47 to give twelve seeds
# per arm. Same protocol, same drafters, so the two sets pool directly.
#
# Includes the flagship (snap + progress + gate) so that if 2304480 shows it
# winning at six seeds, the confirmation seeds already exist.
# =======================================================================
set -euo pipefail
export TORCH_HOME=/lustre/home/ha676/torch_cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
cd /lustre/home/ha676/le-wm
source .venv/bin/activate
H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
GDM576=gdm_tworoom_s10_gc_paired_e400v.pt
BANK=subgoals_tworoom_s10_paired.pt
SPEC=(--gdm-ckpt "$GDM576" --gdm-steps 50 --accept-tau 0.098)

ID=$SLURM_ARRAY_TASK_ID
SEED=$((48 + ID % 6))
OFF=75; RH=2; BUD=150; MODE=long

case $((ID / 6)) in
  0) ARM=baseline;   NAME=flat;           EXTRA=() ;;
  1) ARM=specpaired; NAME=pair_everystep; EXTRA=("${SPEC[@]}" --accept-tau 0.0) ;;
  2) ARM=specpaired; NAME=pair_t098;      EXTRA=("${SPEC[@]}") ;;
  3) ARM=specpaired; NAME=flagship;       EXTRA=("${SPEC[@]}" --snap-bank "$BANK" --snap-progress --goal-gate) ;;
esac

LOG=logs/2room_power_${NAME}_t${OFF}_rh${RH}_seed${SEED}.log
echo "=== arm=$NAME t=$OFF RH=$RH seed=$SEED ==="
python -m specaccept.envs.tworoom.eval --subgoal "$ARM" \
  --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \
  --device cuda --mode "$MODE" --goal-offset "$OFF" --eval-budget "$BUD" \
  --start final --eval-filter success --require-cross-room --episode-min 4000 \
  --num-eval 64 --seed "$SEED" --cem-seed "$SEED" --receding-horizon "$RH" \
  "${EXTRA[@]}" 2>&1 | tee "$LOG"
echo "=== DONE arm=$NAME seed=$SEED ==="
