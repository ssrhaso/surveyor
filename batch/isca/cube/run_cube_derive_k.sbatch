#!/usr/bin/env bash
#SBATCH --job-name=cube-derive-k
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cube_dk_%A_%a.log
# RE-DERIVE k PER DRAFTER. k is the SAMPLER'S convergence point -- a property of the
# drafter, not the environment. k=4 was derived on the GOAL-FREE gdm_cube_s10.pt; the
# new GOAL-CONDITIONED drafters (S=5/10/15/25) are different models with different
# samplers, so k must be re-derived for each. Reusing k=4 across them would violate
# the method's own rule (the paper claims k is DERIVED, not carried over).
#
# NOTE on tau: the criterion-floor rule is OUT OF SCOPE on cube (floor 0.79-0.97 vs a
# normal temporal floor of 0.087 -- the success criterion scores only the cube while
# the encoder sees the whole arm). Cube therefore uses the TRANSFERRED default
# tau=0.20, reported as a measured scope boundary, NOT as a derived value.
#   S = {5,10,15,25}[idx]  (goal-cond ckpts)
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
cd /lustre/home/ha676/le-wm
mkdir -p Results
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
SS=(5 10 15 25)
S=${SS[$SLURM_ARRAY_TASK_ID]}
CKPT=gdm_cube_s${S}_gc.pt
if [ ! -f "$CKPT" ]; then echo "MISSING $CKPT -- drafter not trained yet"; exit 1; fi
echo "=== derive k for $CKPT (sampler convergence; tau stays transferred 0.20) ==="
python -m specaccept.probes.probe_floor --env cube --h5 "$H5" \
  --source pretrained --encoder-id quentinll/lewm-cube --gdm-ckpt "$CKPT" \
  --device cuda --stride ${S} --cube-tol 0.04 --tau 0.20 --seed 42 \
  --n-anchors 512 --pair-pool 120000 --ks 1 2 3 4 5 6 8 12 16 50 \
  --json-out Results/floor_cube_gc_s${S}.json \
  2>&1 | tee logs/cube_derive_k_s${S}.log
