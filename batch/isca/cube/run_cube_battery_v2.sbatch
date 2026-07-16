#!/usr/bin/env bash
#SBATCH --job-name=cube-batt2
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-29
#SBATCH --output=/lustre/home/ha676/le-wm/logs/cube_batt2job_%A_%a.log
# CORRECTED cube battery. v1 was invalid: cube expert episodes are multi-segment
# pick-and-place, so "goal = final frame" made t=25/t=100 VACUOUS (100% of episodes
# already solved at the start -> every arm trivially 100%). v2 uses a fixed
# population of (episode,start) pairs with goal = start+t and a non-vacuity filter
# (cube must move > 0.08m at EVERY offset), so horizon is the only variable and no
# cell is degenerate.
# tau: the criterion-floor rule is OUT OF SCOPE on cube (floor ~9x temporal even with
# full-scene matching), so spec uses the transferred dev-env default tau=0.20 with the
# in-scope derived k=4. Arms: flat baseline (RH5), s10 goal-free every-step, s10 spec.
#   ARM = idx//10 (0 base,1 gdm,2 spec); OFF = (idx//2)%5; SEED = 42 + idx%2
set -eu
source /lustre/home/ha676/le-wm/.venv/bin/activate
H5=/lustre/home/ha676/data/cube/cube_single_expert.h5
EPFILE=/lustre/home/ha676/le-wm/cube_horizon.ep.json
TAU=0.20; K=4
cd /lustre/home/ha676/le-wm
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
ID=$SLURM_ARRAY_TASK_ID
ARMS=(baseline s10gdm s10spec)
OFFSETS=(25 50 75 100 150)
A=$((ID / 10)); OFF=${OFFSETS[$(((ID / 2) % 5))]}; SEED=$((42 + ID % 2))
NAME=${ARMS[$A]}
case $NAME in
  baseline) ARM="--subgoal baseline --horizon 5 --receding-horizon 5" ;;
  s10gdm)   ARM="--subgoal gdm --gdm-ckpt gdm_cube_s10.pt --gdm-steps 50 --horizon 2 --receding-horizon 2" ;;
  s10spec)  ARM="--subgoal specaccept --gdm-ckpt gdm_cube_s10.pt --accept-tau ${TAU} --gdm-steps ${K} --horizon 2 --receding-horizon 2" ;;
esac
echo "=== cube v2: arm=$NAME t=$OFF seed=$SEED (fixed pop, goal=start+t, tau=$TAU k=$K) ==="
python -m specaccept.envs.cube.eval $ARM \
  --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" --device cuda \
  --episodes-file "$EPFILE" --goal-offset $OFF --eval-budget $((2*OFF)) \
  --num-eval 128 --seed $SEED --cem-seed $SEED \
  2>&1 | tee logs/cubev2_${NAME}_t${OFF}_seed${SEED}.log
