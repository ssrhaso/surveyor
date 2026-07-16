#!/usr/bin/env bash
#SBATCH --job-name=cube-verifval
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cubeVV_%A_%a.log
# ===================== PRE-REGISTERED (next array, HELD) ======================
# Written 2026-07-15 BEFORE the gate array (2273854) and the dspark ceiling probe
# land, so it is not post-hoc. Purpose: separate "drafter quality" from
# "verification adds value" -- the paper-relevant split -- at a NON-degenerate tau.
#
# Both arms use the SAME gc15 drafter + goal-gate, differing only in tau:
#   arm0 open-loop  : tau=0.00 -> verifier never accepts -> every-step re-draft,
#                     gated. This is drafting WITHOUT speculative acceptance.
#   arm1 spec-floor : tau=0.76 (the cube criterion floor) -> verifier CAN accept,
#                     coasts on verified blocks. This is spec-accept operating in
#                     a real regime (unlike tau=0.20, which the data shows still
#                     coasts ~25% but was chosen off-floor by the scope guard).
# Read-off: arm1 >= arm0 => verification/coasting adds value at the floor tau
# (fewer diffusion calls at equal-or-better SR). arm1 ~ arm0 => the gate carries
# the method and the verifier is inert on cube. Fixed refs: flat 70.9, oracle 82.4,
# and the gate array's gc15_gate@tau0.20 cell.
#
# CONDITIONAL EXTENSION (do NOT run yet): if the dspark probe A3 shows the m+1
# posterior is ~unimodal AND a trained deterministic regressor clears the
# pre-registered bake-off bar (rel_err <= 1.3x oracle-floor AND <= 0.6x GDM),
# add two mirror arms with the regressor checkpoint (open-loop + spec-floor,
# gate on). That regressor arm is the one expected to clear flat toward oracle;
# these diffusion arms establish the verification-value axis it will reuse.
# =============================================================================
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
mkdir -p runs/cube logs
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
TAUS=(0.00 0.76)
NAMES=(gc15_open gc15_specfloor)
A=$((SLURM_ARRAY_TASK_ID / 4))
SEED=$((42 + SLURM_ARRAY_TASK_ID % 4))
TAU=${TAUS[$A]}; NAME=${NAMES[$A]}
echo "=== cube-verifval arm=$NAME tau=$TAU seed=$SEED (gc15 + goal-gate, certified) ==="
python -m specaccept.envs.cube.eval \
  --subgoal specaccept --gdm-ckpt gdm_cube_s15_gc.pt --goal-gate \
  --accept-tau "$TAU" --gdm-steps 8 --horizon 2 --receding-horizon 2 \
  --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
  --start final --goal-offset 150 --eval-budget 300 \
  --num-eval 128 --seed "$SEED" --cem-seed "$SEED" --episode-min 8000 \
  --num-samples 600 --n-steps 30 \
  --dump-traces runs/cube/traces_vv_${NAME}_seed${SEED}.pt \
  2>&1 | tee logs/cubeVV_${NAME}_seed${SEED}.log
