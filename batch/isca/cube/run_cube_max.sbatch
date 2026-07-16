#!/usr/bin/env bash
#SBATCH --job-name=cube-max
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-27%8
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cubeX_%A_%a.log
# ================= CHAMPION NEIGHBORHOOD SWEEP (post-2273854) =================
# Baseline champion: gc15 + goal-gate, k=8, tau=0.20, RH=2 -> 79.30% (flat 70.9,
# oracle 82.4). This array probes every cheap axis around it, EXPLORATORY (a
# frozen-bar confirm on fresh populations follows once the best config is known):
#   arm0 gc10_gate   : gdm_cube_s10_gc.pt existed but was never gate-tested;
#                      S-curve is sharp (5:67.2, 15:79.3, 25:68.2) -> locate peak
#   arm1 gc15_rh5    : gate's flat branch currently plans at RH=2, but flat's
#                      strongest cube config is RH=5 (70.9 vs ~66) -- the exact
#                      reacher-t50 structural miss, tested here as H=5/RH=5
#   arm2 gc15_k4     : k matched to the certified spec arm (27.3 ran k=4);
#                      champion ran k=8 -- is the win k-sensitive? (cheaper too)
#   arm3 gc15_k3     : derive-k's cliff said k=3 is the minimum resolvable draft
#   arm4 gc15_t15    : tau could NOT be derived on cube (criterion floor
#   arm5 gc15_t30    :   OFF-FLOOR) so 0.20 is a transfer -- bracket it
#   arm6 gc15_s4649  : champion config verbatim, fresh seeds 46-49 -> 8-seed CI
# All at the certified protocol: offset150 budget300 num_samples=600 n_steps=30.
# ==============================================================================
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
mkdir -p runs/cube logs
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
ARMS=(gc10_gate gc15_rh5 gc15_k4 gc15_k3 gc15_t15 gc15_t30 gc15_s4649)
CKPTS=(gdm_cube_s10_gc.pt gdm_cube_s15_gc.pt gdm_cube_s15_gc.pt gdm_cube_s15_gc.pt gdm_cube_s15_gc.pt gdm_cube_s15_gc.pt gdm_cube_s15_gc.pt)
KS=(8 8 4 3 8 8 8)
TAUS=(0.20 0.20 0.20 0.20 0.15 0.30 0.20)
RHS=(2 5 2 2 2 2 2)
SEEDBASE=(42 42 42 42 42 42 46)
A=$((SLURM_ARRAY_TASK_ID / 4))
SEED=$((SEEDBASE[A] + SLURM_ARRAY_TASK_ID % 4))
NAME=${ARMS[$A]}; CK=${CKPTS[$A]}; K=${KS[$A]}; TAU=${TAUS[$A]}; RH=${RHS[$A]}
echo "=== cube-max arm=$NAME ckpt=$CK k=$K tau=$TAU rh=$RH seed=$SEED ==="
python -m specaccept.envs.cube.eval \
  --subgoal specaccept --gdm-ckpt "$CK" --goal-gate \
  --accept-tau "$TAU" --gdm-steps "$K" --horizon "$RH" --receding-horizon "$RH" \
  --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
  --start final --goal-offset 150 --eval-budget 300 \
  --num-eval 128 --seed "$SEED" --cem-seed "$SEED" --episode-min 8000 \
  --num-samples 600 --n-steps 30 \
  2>&1 | tee logs/cubeX_${NAME}_seed${SEED}.log
