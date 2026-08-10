"""TwoRoom pixels through RAW frozen DINOv2 (ViT-S/14, imagenet norm) -> pooled
per-frame latents -> gap-stat food.

LeWM's ViT-tiny is SATURATED on TwoRoom, its gap inverted at every scale, so
this asks whether a generic frozen encoder restores it. The h5 layout is FLAT,
pixels (F,224,224,3) uint8 plus an episode-boundary array, introspected below.
Needs hdf5plugin for the compression filter."""
import os
from pathlib import Path

import hdf5plugin  # noqa: F401  (registers HDF5 compression filters)
import h5py
import numpy as np
import torch
from torchvision import transforms

H5 = os.environ.get("TWOROOM_H5",
                    os.path.expanduser("~/data/tworoom/tworoom.h5"))
OUT = Path(os.environ.get("DINOWM_DATA",
                          os.path.expanduser("~/data/dinowm"))) / "tworoom_dino_lat"
OUT.mkdir(parents=True, exist_ok=True)
MAX_EPS = 200
dev = "cuda" if torch.cuda.is_available() else "cpu"

enc = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(dev).eval()
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

with h5py.File(H5, "r") as f:
    all_ds = []
    f.visititems(lambda n, o: all_ds.append((n, o.shape, str(o.dtype)))
                 if isinstance(o, h5py.Dataset) else None)
    print("datasets:", all_ds)

    pix_name = next(n for n, s, d in all_ds if d == "uint8" and len(s) == 4)
    pix = f[pix_name]
    F_total = pix.shape[0]

    # LeWM h5 schema: explicit ep_offset (starts) + ep_len per episode
    ep_off = np.asarray(f["ep_offset"][:])
    ep_len = np.asarray(f["ep_len"][:])
    n_eps = len(ep_off)
    print(f"{n_eps} episodes, {F_total} frames")

    for i in range(min(n_eps, MAX_EPS)):
        dst = OUT / f"lat_tworoom_{i:04d}.npz"
        if dst.exists():
            continue
        frames = pix[ep_off[i]:ep_off[i] + ep_len[i]]
        x = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        pooled = []
        with torch.no_grad():
            for s in range(0, x.shape[0], 128):
                xb = norm(x[s:s + 128]).to(dev)
                z = enc.forward_features(xb)["x_norm_patchtokens"]
                pooled.append(z.mean(dim=1).half().cpu())
        np.savez(dst, tokens=torch.cat(pooled).numpy())
        if i % 25 == 0:
            print(f"{i}/{min(n_eps, MAX_EPS)} T={x.shape[0]}", flush=True)
print("TWOROOM DINO ENCODE DONE")
