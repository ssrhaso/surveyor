"""Encode the pusht_noise dataset through the trained DINO-WM's OWN encode path
(Preprocessor.transform_obs -> VWorldModel.encode_obs), dumping per-episode
latents for the gap statistic and later drafter training.

Outputs to /lustre/home/ha676/data/dinowm/pusht_lat/:
  lat_pusht_<split>_<i>.npz   tokens=(T,384) fp16 POOLED visual patch mean,
                              proprio=(T,dp) fp16 normalized, state=(T,Ds) fp32
  tokens/tok_pusht_<split>_<i>.npy  (T,P,384) fp16 full grids (subset, for
                              token-space analysis)
Raw-frame granularity (frameskip applied nowhere here): 1 model step = 5 raw
frames, goal_H=5 model steps = 25 raw frames = the natural subgoal hop.
"""
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

torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")  # importable (cache patched)

from plan import load_model  # noqa: E402
from preprocessor import Preprocessor  # noqa: E402

OUT = Path("/lustre/home/ha676/data/dinowm/pusht_lat")
TOK = OUT / "tokens"
GRID = OUT / "grids"    # (T, 257, 384) fp16: 256 visual tokens + 1 proprio
OUT.mkdir(parents=True, exist_ok=True)  # token (raw normalized 2-d proprio in
TOK.mkdir(exist_ok=True)                # dims :2, zero-padded), drafter food in
GRID.mkdir(exist_ok=True)               # the vjepa2 train_drafter npy format
N_TOKEN_EPS = {"train": 100, "valid": 60}   # full-grid subset per split
CHUNK = 64

dev = "cuda" if torch.cuda.is_available() else "cpu"
MP = "/lustre/home/ha676/data/dinowm/checkpoints/outputs/pusht/"
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
    n = len(dset)
    print(f"[{split}] {n} trajectories", flush=True)
    for i in range(n):
        dst = OUT / f"lat_pusht_{split}_{i:04d}.npz"
        gdst = GRID / f"grid_pusht_{split}_{i:04d}.npy"
        if dst.exists() and gdst.exists():
            continue
        obs, act, state, *_ = dset[i]
        # dset[i] visual is ALREADY transformed (T,C,H,W) float, the exact
        # tensors the model trained on; only proprio needs normalizing.
        vis = torch.as_tensor(np.asarray(obs["visual"]))[None].float().to(dev)
        pro = pre.normalize_proprios(
            torch.as_tensor(np.asarray(obs["proprio"]))).float()[None].to(dev)
        pooled, grids = [], []
        with torch.no_grad():
            for s in range(0, vis.shape[1], CHUNK):
                z = model.encode_obs({"visual": vis[:, s:s + CHUNK],
                                      "proprio": pro[:, s:s + CHUNK]})
                v = z["visual"][0]             # (t, P, 384)
                pooled.append(v.mean(dim=1).half().cpu())
                ptok = torch.zeros(v.shape[0], 1, v.shape[-1], device=v.device)
                ptok[:, 0, :pro.shape[-1]] = pro[0, s:s + CHUNK]
                grids.append(torch.cat([v, ptok], dim=1).half().cpu())
        if not dst.exists():
            np.savez(dst,
                     tokens=torch.cat(pooled).numpy(),
                     proprio=pro[0].half().cpu().numpy(),
                     state=np.asarray(state, dtype=np.float32))
        np.save(gdst, torch.cat(grids).numpy())
        if i % 100 == 0:
            print(f"[{split}] {i}/{n} T={vis.shape[1]}", flush=True)
print("ENCODE DONE")
