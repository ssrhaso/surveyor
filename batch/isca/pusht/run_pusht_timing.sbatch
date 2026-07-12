#!/usr/bin/env bash
#SBATCH --job-name=pusht-timing
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_timingjob_%A_%a.log
# Wall-clock instrument (audit #5): NFE is not wall-clock, and spec-accept's
# verification (encoder pass + L2) is added work the every-step baseline
# never does. --time-instrument accumulates t_drafter/t_cem with CUDA syncs
# at each boundary. Controlled protocol (t150 test pop), single seed --
# wall-clock/step does not vary by seed the way SR does. ISCA port of
# run_timing_instrument.sbatch + the k=8 every-step twin (new).
#   0: s25 every-step (RH5, k=50)   1: s10 every-step (RH2, k=50)
#   2: s10 spec tau=.2 k=8 (RH2)    3: s10 every-step k=8 (RH2, the mirror)
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
ID=$SLURM_ARRAY_TASK_ID
cd /lustre/home/ha676/le-wm
export SDL_VIDEODRIVER=dummy
source "$VENV/bin/activate"
case $ID in
  0) NAME=s25_k50;  ARM="--subgoal gdm --gdm-ckpt gdm_stride25.pt --gdm-steps 50 --horizon 5 --receding-horizon 5" ;;
  1) NAME=s10_k50;  ARM="--subgoal gdm --gdm-ckpt gdm_stride10.pt --gdm-steps 50 --horizon 2 --receding-horizon 2" ;;
  2) NAME=s10_spec; ARM="--subgoal specaccept --gdm-ckpt gdm_stride10.pt --accept-tau 0.20 --gdm-steps 8 --horizon 2 --receding-horizon 2" ;;
  3) NAME=s10_k8;   ARM="--subgoal gdm --gdm-ckpt gdm_stride10.pt --gdm-steps 8 --horizon 2 --receding-horizon 2" ;;
esac
echo "=== timing arm=$NAME (t150, budget 300, n=256, seed 42) ==="
python -m specaccept.envs.pusht.eval $ARM \
  --source pretrained --h5 "$H5" --device cuda \
  --mode long --episodes-file pusht.episodes150.json --goal-offset 150 --eval-budget 300 \
  --num-eval 256 --score block --angles 20 5 --seed 42 --cem-seed 42 \
  --time-instrument \
  2>&1 | tee logs/pusht_timing_${NAME}.log
