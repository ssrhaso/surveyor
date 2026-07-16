#!/usr/bin/env bash
#SBATCH --job-name=gspec-pure
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:30:00
#SBATCH --array=0-15
#SBATCH --output=/lustre/home/ha676/le-wm/logs/gspec_%A_%a.log
# ================== PURE SPEC-ACCEPT, NO ROUTER ("just our method") ==========
# Question (user, 2026-07-15): cube is won by spec-accept alone (gc + goal-gate,
# S=10 -> 78.9 vs flat 70.9, every episode on the method, no episode routing).
# Can the SAME single arm beat LeWM at every reacher horizon too -- making the
# v4 router an optimization rather than a correctness requirement?
# Arm: specaccept + --goal-gate, gc drafter gdm_reacher_s10.pt, k=8, tau=0.20,
# RH=2, S=10 -- the exact cube-winning mechanism, nothing env-tuned.
# Populations: the SAME builder-seed-888 files the v4 eval is running on, so
# flat2/flat5/spec references come for free from job 2274091 (same pop, same
# seeds, same budgets). t x seeds = {25,50,100,150} x {42-45}, budget=2t.
# Known prior evidence: goal-gated spec at t25 previously read ~90.6-93.8 vs
# flat2 96.9 (exploratory, old pop) -- i.e. the waypoint gate recovers most but
# maybe not all of the overshoot tax against a near-ceiling baseline. This run
# settles it on a fresh population; reported as-is either way.
# ==============================================================================
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
OFFSETS=(25 50 100 150)
O=$((SLURM_ARRAY_TASK_ID / 4))
SEED=$((42 + SLURM_ARRAY_TASK_ID % 4))
OFF=${OFFSETS[$O]}
MAXOFF=100; if [ "$OFF" -eq 150 ]; then MAXOFF=150; fi
cd /lustre/home/ha676/le-wm
mkdir -p logs
export MUJOCO_GL=egl
source "$VENV/bin/activate"
EPFILE=reacher_gatev4.ep${MAXOFF}.s888.json
echo "=== gspec-pure t=$OFF seed=$SEED pop=$EPFILE (budget=$((2*OFF))) ==="
python -m specaccept.envs.reacher.eval --subgoal specaccept \
  --gdm-ckpt gdm_reacher_s10.pt --gdm-steps 8 --goal-gate \
  --accept-tau 0.20 \
  --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
  --num-eval 128 --seed ${SEED} --episodes-file $EPFILE \
  --goal-offset ${OFF} --eval-budget $((2*OFF)) \
  --horizon 2 --receding-horizon 2 \
  2>&1 | tee logs/gspec_t${OFF}_seed${SEED}.log
