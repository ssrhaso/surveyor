#!/usr/bin/env bash
#SBATCH --job-name=pusht-sanity
#SBATCH --partition=gpu
#SBATCH --account=research_project-deepmind
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/home/ha676/le-wm/logs/pusht_sanity_%j.log
# PushT stack sanity on ISCA (gates the whole chain): pretrained-encoder
# auto-download via swm hub (the path proven on the original PushT box),
# box2d env import, headless render, h5 schema.
set -eu
VENV=/lustre/home/ha676/le-wm/.venv
H5=/lustre/home/ha676/data/pusht/pusht_expert_train.h5
cd /lustre/home/ha676/le-wm
export SDL_VIDEODRIVER=dummy
source "$VENV/bin/activate"
python - <<PY
import torch, h5py
from specaccept.encoder import load_lewm
print("=== [1/3] pretrained encoder (swm hub auto-download) ===")
m = load_lewm(source="pretrained", device="cuda")
print("encoder OK:", sum(p.numel() for p in m.parameters())/1e6, "M params, cuda:", next(m.parameters()).is_cuda)
print("=== [2/3] PushT env (box2d) ===")
import stable_worldmodel  # noqa
from stable_worldmodel.envs.pusht.env import PushT
e = PushT(render_mode="rgb_array"); e.reset(seed=0)
img = e.render(); print("env OK, render:", getattr(img, "shape", None))
print("=== [3/3] h5 schema ===")
with h5py.File("$H5", "r") as f:
    print("keys:", sorted(f.keys())[:12])
    print("frames:", f[list(f.keys())[0]].shape if f.keys() else "?")
print("PUSHT-SANITY-OK")
PY
