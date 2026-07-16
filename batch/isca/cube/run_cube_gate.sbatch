#!/usr/bin/env bash
#SBATCH --job-name=cube-gate
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-19%8
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cubeG_%A_%a.log
# ======================= OVERSHOOT / ARRIVAL-GATE TEST =======================
# Theory (pre-registered here): cube's drafter tax (spec 27.3 / gdm 33.0 vs flat
# 70.9) is the SAME failure as short-horizon Reacher t25 (spec 59.6 vs flat 96.9):
# the drafter can only emit forward motion, so once the goal is near it overshoots
# past it. Certified cube IS a goal-near task (cube arrives ~step 90, goal frame
# ~step 200, 109/201 idle). --goal-gate (SpecAcceptSubgoalSource.goal_gate) serves
# a waypoint only if it REDUCES latent distance to goal, else serves the goal --
# the exact anti-overshoot fix, already worth ~+32pp on Reacher t25 (59.6->~92).
# The cube final battery ran spec WITHOUT the gate -> that 27.3 is un-fixed.
#
# Disambiguation vs the competing "goal not inferable" hypothesis:
#   arm0 gff10_gate  : goal-FREE s10 + gate   -> does gating alone rescue goal-free?
#   arm1 gc15_nogate : goal-cond s15, NO gate -> does goal-conditioning alone fix it?
#   arm2 gc15_gate   : goal-cond s15 + gate   -> the combined win candidate (clean
#                       gate-isolation vs arm1, identical everything else)
#   arm3 gc5_gate    : goal-cond s5  + gate   -> S-sweep
#   arm4 gc25_gate   : goal-cond s25 + gate   -> S-sweep
# Fixed references (already measured, seeds 42-45, same protocol): flat=70.9,
# oracle=82.4, un-gated goal-free spec=27.3, un-gated gdm=33.0.
# READ-OFF: gate >> no-gate at fixed conditioning => overshoot confirmed (theory).
# gc-nogate already ~flat => it was target-visibility, not overshoot. Traces dumped
# for probe_failure_anatomy (expect diverged/unreachable buckets if overshoot).
# =============================================================================
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
mkdir -p runs/cube logs
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
ARMS=(gff10_gate gc15_nogate gc15_gate gc5_gate gc25_gate)
CKPTS=(gdm_cube_s10.pt gdm_cube_s15_gc.pt gdm_cube_s15_gc.pt gdm_cube_s5_gc.pt gdm_cube_s25_gc.pt)
GATES=(1 0 1 1 1)
A=$((SLURM_ARRAY_TASK_ID / 4))
SEED=$((42 + SLURM_ARRAY_TASK_ID % 4))
NAME=${ARMS[$A]}; CK=${CKPTS[$A]}; G=${GATES[$A]}
GATEFLAG=""; if [ "$G" -eq 1 ]; then GATEFLAG="--goal-gate"; fi
echo "=== cube-gate arm=$NAME ckpt=$CK gate=$G seed=$SEED (certified: 600 samples, offset150 budget300) ==="
python -m specaccept.envs.cube.eval \
  --subgoal specaccept --gdm-ckpt "$CK" $GATEFLAG \
  --accept-tau 0.20 --gdm-steps 8 --horizon 2 --receding-horizon 2 \
  --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
  --start final --goal-offset 150 --eval-budget 300 \
  --num-eval 128 --seed "$SEED" --cem-seed "$SEED" --episode-min 8000 \
  --num-samples 600 --n-steps 30 \
  --dump-traces runs/cube/traces_gate_${NAME}_seed${SEED}.pt \
  2>&1 | tee logs/cubeG_${NAME}_seed${SEED}.log
