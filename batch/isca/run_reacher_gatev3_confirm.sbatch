#!/usr/bin/env bash
#SBATCH --job-name=gatev3-confirm
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:30:00
#SBATCH --array=0-63
#SBATCH --output=/lustre/home/ha676/le-wm/logs/gatev3_confirm_%A_%a.log
# ============================ PRE-REGISTERED =============================
# Gate v3 closed-loop confirmation (written BEFORE any confirmation data):
# episode-level c* gate (HorizonGatedSource): one flat RH=2 CEM plan at
# episode start; fire iff c* <= tau=0.20 (the verifier's frozen threshold)
# -> whole episode flat on the goal latent; else whole episode spec-accept
# (tau=0.20, k=8, S=10). FRESH populations (builder seed 777 -- never used
# before; all prior analysis used seed 42) so the offline post-mortem
# cannot have leaked into these cells.
#
# Arms x offsets x seeds: {flat2, flat5, spec, gated} x {25,50,100,150} x
# {42,43,44,45}; n=128/cell (matches every existing table; deviation from
# the original Phase-2 note of n=512, declared here); budget=2t; pop
# max100 for t<=100, max150 for t=150.
#
# SUCCESS CONDITION (both must hold, means over 4 seeds, per cell):
#   t25 & t50 : gated SR >= max(flat2, flat5) - 1.0pp
#   t100 & t150: gated SR >= spec - 1.0pp
# Expected fire-rates (from the post-mortem; reported, not conditions):
# ~0.95 @ t25 falling to ~0.5 @ t150. Plumbing check: fired episodes make
# zero drafter calls (see [horizon-gate] line per log). tau and the gate
# rule are frozen; failure is reported as-is, no post-hoc amendment.
# =========================================================================
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=$(find /lustre/home/ha676/data/reacher \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
ARMS=(flat2 flat5 spec gated)
SUBGOALS=(baseline baseline specaccept horizon_gated)
RHS=(2 5 2 2)
OFFSETS=(25 50 100 150)
A=$((SLURM_ARRAY_TASK_ID / 16))
O=$(((SLURM_ARRAY_TASK_ID / 4) % 4))
SEED=$((42 + SLURM_ARRAY_TASK_ID % 4))
NAME=${ARMS[$A]}; SG=${SUBGOALS[$A]}; RH=${RHS[$A]}
OFF=${OFFSETS[$O]}
MAXOFF=100; if [ "$OFF" -eq 150 ]; then MAXOFF=150; fi
CKPT_ARG=""
if [ "$SG" != "baseline" ]; then CKPT_ARG="--gdm-ckpt gdm_reacher_s10.pt --gdm-steps 8"; fi
cd /lustre/home/ha676/le-wm
mkdir -p runs/reacher logs
export MUJOCO_GL=egl
source "$VENV/bin/activate"
EPFILE=reacher_gatev3.ep${MAXOFF}.t${SLURM_ARRAY_TASK_ID}.json
python batch/build_reacher_horizon_episodes.py --h5 "$H5" --out $EPFILE \
  --n 128 --max-offset $MAXOFF --episode-min 8000 --seed 777
echo "=== gatev3-confirm arm=$NAME t=$OFF seed=$SEED (pop=max$MAXOFF seed777, budget=$((2*OFF))) ==="
python -m specaccept.envs.reacher.eval --subgoal ${SG} ${CKPT_ARG} \
  --accept-tau 0.20 \
  --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
  --num-eval 128 --seed ${SEED} --episodes-file $EPFILE \
  --goal-offset ${OFF} --eval-budget $((2*OFF)) \
  --horizon ${RH} --receding-horizon ${RH} \
  --dump-traces runs/reacher/traces_gatev3_${NAME}_t${OFF}_seed${SEED}.pt \
  2>&1 | tee logs/gatev3_${NAME}_t${OFF}_seed${SEED}.log
