#!/usr/bin/env bash
#SBATCH --job-name=pusht-gamma
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-63
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_gammajob_%A_%a.log
# Spread dose-response (gamma sweep): the paper's mechanistic claim; drafter
# conditional SPREAD is load-bearing; currently rests on three categorical
# ablations (refinement/regressor/blind-commit), all at reduced/zero spread,
# none above gamma=1. This sweep converts the claim into a curve, plus the
# joint offline/closed-loop plot (rel_err is monotone-worse in gamma by
# construction; if SR rises where rel_err falls, offline fidelity
# anti-correlates with control; measured on ONE continuous lever).
#
# PRE-REGISTERED GATES (before any result was seen):
#   PASS: SR(gamma) non-monotone w/ interior optimum (>=2pp above gamma=1, or
#         gamma=1 optimal with >=3pp falloff both sides).
#   PARTIAL: SR monotone-decreasing from gamma=0 -> soften "spread is
#         load-bearing" to "mean-collapsing operators damage control".
#   KILL: SR flat on gamma in [0.5,1.5] -> the three ablations failed for
#         unrelated reasons; report as such.
#   CONFOUND: gamma=0 should land near the deterministic-regressor deficit
#         (box battery: -5/-12pp); if wildly off, the knob is suspect; the
#         regressor arm may need porting to this exact protocol to adjudicate.
#
# GAMMA(8: 0 .25 .5 .75 1.0 1.25 1.5 2.0) x S(2: 10,25) x SEED(4: 42-45)
#   idx = g*8 + s*4 + seed_idx
# Protocol: t150 fixed population, budget 300, n=256, every-step gdm, loop
# matched to scale (S=10 -> RH2, S=25 -> RH5). In-job offline diag at the same
# gamma provides the joint-plot x-axis from the same cell.
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
ID=$SLURM_ARRAY_TASK_ID
GAMMAS=(0 0.25 0.5 0.75 1.0 1.25 1.5 2.0)
G=${GAMMAS[$((ID / 8))]}
S=$(( (ID / 4) % 2 == 0 ? 10 : 25 ))
SEED=$((42 + ID % 4))
if [ $S -eq 10 ]; then CKPT=gdm_stride10.pt; RH=2; else CKPT=gdm_stride25.pt; RH=5; fi
cd /lustre/home/ha676/le-wm
mkdir -p runs/pusht
export SDL_VIDEODRIVER=dummy
source "$VENV/bin/activate"
echo "=== gamma-sweep: gamma=$G S=$S RH=$RH seed=$SEED (t150, budget 300, n=256) ==="
python -m specaccept.envs.pusht.eval --subgoal gdm --gdm-ckpt $CKPT \
  --gdm-noise-scale $G \
  --source pretrained --h5 "$H5" --device cuda \
  --mode long --episodes-file pusht.episodes150.json --goal-offset 150 --eval-budget 300 \
  --num-eval 256 --score block --angles 20 5 --seed $SEED --cem-seed $SEED \
  --horizon $RH --receding-horizon $RH \
  2>&1 | tee logs/pusht_gamma${G}_s${S}_seed${SEED}.log
if [ $SEED -eq 42 ]; then
  echo "=== offline diag at gamma=$G (joint-plot x-axis; seed-42 task only) ==="
  python -m specaccept.diag_gdm --gdm-ckpt $CKPT --subgoals subgoals_dense_full.pt \
    --gdm-noise-scale $G --subgoal-step $S --stride $S --h5 "$H5" \
    --device cuda --source pretrained \
    2>&1 | tee logs/pusht_gammadiag${G}_s${S}.log
fi
