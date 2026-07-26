"""Generalized DINO-WM latent encoder: any released env checkpoint (pusht,
point_maze, wall_single) through the trained model's own encode path. Env var
MODEL_NAME picks the checkpoint dir under checkpoints/outputs/. Outputs
lat_<name>_<split>_<i>.npz (tokens = pooled (T,384) fp16) for gap_stat."""
import os
import sys

REPO = "/lustre/home/ha676/dino_wm"
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
import torch
from omegaconf import OmegaConf
from pathlib import Path
import hydra

torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

from plan import load_model  # noqa: E402
from preprocessor import Preprocessor  # noqa: E402

NAME = os.environ["MODEL_NAME"]
OUT = Path(f"/lustre/home/ha676/data/dinowm/{NAME}_lat")
OUT.mkdir(parents=True, exist_ok=True)
CHUNK = 64
MAX_PER_SPLIT = int(os.environ.get("MAX_PER_SPLIT", 400))

dev = "cuda" if torch.cuda.is_available() else "cpu"
MP = f"/lustre/home/ha676/data/dinowm/checkpoints/outputs/{NAME}/"
cfg = OmegaConf.load(MP + "hydra.yaml")
model = load_model(Path(MP) / "checkpoints" / "model_latest.pth", cfg,
                   cfg.num_action_repeat, dev)
model.eval()

_, dsets = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist,
                            num_pred=cfg.num_pred, frameskip=cfg.frameskip)

for split in ("train", "valid"):
    dset = dsets[split]
    pre = Preprocessor(action_mean=dset.action_mean, action_std=dset.action_std,
                       state_mean=dset.state_mean, state_std=dset.state_std,
                       proprio_mean=dset.proprio_mean, proprio_std=dset.proprio_std,
                       transform=dset.transform)
    n = min(len(dset), MAX_PER_SPLIT)
    print(f"[{NAME}/{split}] {len(dset)} trajs, encoding {n}", flush=True)
    for i in range(n):
        dst = OUT / f"lat_{NAME}_{split}_{i:04d}.npz"
        if dst.exists():
            continue
        obs, act, state, *_ = dset[i]
        # dset[i] visual is ALREADY transformed (T,C,H,W) float — the exact
        # tensors the model trained on; only proprio needs normalizing.
        vis = torch.as_tensor(np.asarray(obs["visual"]))[None].float().to(dev)
        pro = pre.normalize_proprios(
            torch.as_tensor(np.asarray(obs["proprio"]))).float()[None].to(dev)
        pooled = []
        with torch.no_grad():
            for s in range(0, vis.shape[1], CHUNK):
                z = model.encode_obs({"visual": vis[:, s:s + CHUNK],
                                      "proprio": pro[:, s:s + CHUNK]})
                pooled.append(z["visual"][0].mean(dim=1).half().cpu())
        np.savez(dst, tokens=torch.cat(pooled).numpy(),
                 state=np.asarray(state, dtype=np.float32))
        if i % 100 == 0:
            print(f"[{NAME}/{split}] {i}/{n} T={vis.shape[1]}", flush=True)
print("ENCODE DONE")
