"""Batch-encode fetched episodes into token latents for drafter training.

For each probe-format episode npz (observations (1,T,H,W,3), states (1,T,7))
writes lat_<name>.npz with
  tokens (T, tokens_per_frame, D) float16   layer-normed V-JEPA 2 reps
  states (T, 7) float32
The `tokens` key is exactly what train_drafter.py consumes.

  python -m specaccept_vjepa2.encode_episodes --episodes droid_eps/droid_ep*.npz \
      --out-dir droid_lat --device cuda
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from .wm import VJEPA2WM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="vjepa2_ac_vit_giant")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--format", choices=["npz", "npy"], default="npz",
                    help="npy = raw fp16 arrays (lat_X.npy + lat_X.states.npy), "
                         "mmap-able for large-scale training (npz cannot mmap)")
    args = ap.parse_args()

    files = sorted(sum([glob.glob(p) for p in args.episodes], []))
    assert files, f"no files match {args.episodes}"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    wm = VJEPA2WM(model_name=args.model, device=args.device)
    for i, f in enumerate(files):
        stem = f"lat_{Path(f).stem}"
        dst = out / (stem + (".npy" if args.format == "npy" else ".npz"))
        if dst.exists():
            print(f"[skip] {dst.name} exists")
            continue
        ep = np.load(f)
        tokens = wm.encode_frames(ep["observations"][0], chunk=args.chunk)
        tok16 = tokens.cpu().to(torch.float16).numpy()
        states = ep["states"][0].astype(np.float32)
        if args.format == "npy":
            np.save(out / (stem + ".states.npy"), states)
            np.save(dst, tok16)
        else:
            np.savez_compressed(dst, tokens=tok16, states=states)
        print(f"[{i + 1}/{len(files)}] {dst.name}: T={tokens.shape[0]}")
    print("[done]")


if __name__ == "__main__":
    main()
