#!/usr/bin/env bash
# Faithful FF-JEPA GDM: LeWM's verified recipe + paper's GDM spec + dense data.
# Reuses subgoals_dense_full.pt (built overnight). Run:
#   cd ~/le-wm && tmux new -s recipe   then:  bash run_recipe.sh 2>&1 | tee recipe.log
set -e
cd ~/le-wm
export STABLEWM_HOME=$HOME/.stable-wm
export SDL_VIDEODRIVER=dummy
H5=$HOME/.stable-wm/datasets/pusht_expert_train.h5

echo "=== [1/5] train GDM: LeWM recipe + dense (batch128/lr5e-5/wd1e-3/clip1.0/warmup_cosine/20ep, no EMA) ==="
python -m ffjepa.train_gdm --subgoals subgoals_dense_full.pt --out gdm_faithful.pt --device cuda --mask 5 --subgoal-step 25 --batch-size 128 --lr 5e-5 --weight-decay 1e-3 --grad-clip 1.0 --lr-schedule warmup_cosine --epochs 20 --ema-decay 0 --amp

echo "=== [2/5] diagnostic GATE (Probe B was 0.72; want it toward 0.9) ==="
python -m ffjepa.diag_gdm --gdm-ckpt gdm_faithful.pt --subgoals subgoals_dense_full.pt --subgoal-step 25 --h5 "$H5" --device cuda

echo "=== [3/5] SHORT eval n=256 20deg block ==="
python -m ffjepa.eval_ffjepa --subgoal gdm --gdm-ckpt gdm_faithful.pt --h5 "$H5" --device cuda --score block --angles 20 --num-eval 256 --mode short

echo "=== [4/5] LONG eval n=256 20deg block ==="
python -m ffjepa.eval_ffjepa --subgoal gdm --gdm-ckpt gdm_faithful.pt --h5 "$H5" --device cuda --score block --angles 20 --num-eval 256 --mode long

echo "=== [5/5] RIGOR: grid-only e200 LONG at n=256 (only had n=32) ==="
python -m ffjepa.eval_ffjepa --subgoal gdm --gdm-ckpt gdm_pusht_e200.pt --h5 "$H5" --device cuda --score block --angles 20 --num-eval 256 --mode long

echo "=== DONE ==="
