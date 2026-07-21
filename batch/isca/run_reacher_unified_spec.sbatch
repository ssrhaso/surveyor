#!/usr/bin/env bash
#SBATCH --job-name=unif-reacher
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:30:00
#SBATCH --array=0-15%8
#SBATCH --output=/lustre/home/ha676/le-wm/logs/unif_reacher_%A_%a.log
# ============ UNIFIED METHOD (c*-retire spec): REACHER SPEC LEG =============
# PRE-REGISTERED. The claimable method is ONE rule at every scope: draft only
# while the planner certifies the goal is out of reach (c* > tau); retire the
# drafter one-way the moment c* <= tau (CstarRetireSource; replaces the latent
# goal-gate, which breaks at long range). Episode-level commit = the v4 router
# (same instrument, first replan); this array re-runs the composite's SPEC LEG
# under the unified source on the SAME seed-999 populations as the v4c confirm,
# so composite_unified(t, seed) = succ(r_flat2) + succ(r_flat5) + succ(THIS)
# over 128, with the flat legs shared verbatim from job 2276034 (their
# execution is identical under the unified method by construction).
# BARS (frozen; 4-seed means): composite_unified >= max(flat2, flat5) at every
# t. Reported: vs the v4c plain-spec composite (does retire beat plain spec?)
# and vs full-pop spec. tau=0.20, k=8, S=10 frozen. Reported as-is.
# ==============================================================================
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
OFFSETS=(25 50 100 150)
O=$((SLURM_ARRAY_TASK_ID / 4))
SEED=$((42 + SLURM_ARRAY_TASK_ID % 4))
OFF=${OFFSETS[$O]}
cd /lustre/home/ha676/le-wm
mkdir -p logs
export MUJOCO_GL=egl
source "$VENV/bin/activate"
EPFILE=reacher_gatev4c_route_t${OFF}_spec.json
N=$(python -c "import json;print(len(json.load(open('$EPFILE'))['episodes']))")
if [ "$N" -eq 0 ]; then
  echo "=== unif-reacher t=$OFF seed=$SEED: EMPTY SUBSET; SR = (0/0), skipped ==="
  exit 0
fi
echo "=== unif-reacher t=$OFF seed=$SEED n=$N (unified c*-retire, budget=$((2*OFF))) ==="
python -m specaccept.envs.reacher.eval --subgoal unified \
  --gdm-ckpt gdm_reacher_s10.pt --gdm-steps 8 --accept-tau 0.20 \
  --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
  --num-eval $N --seed ${SEED} --episodes-file $EPFILE \
  --goal-offset ${OFF} --eval-budget $((2*OFF)) \
  --horizon 2 --receding-horizon 2 \
  2>&1 | tee logs/unif_reacher_t${OFF}_seed${SEED}.log
