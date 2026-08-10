"""c*-v2 instrument probe: can the certificate be rescued on DROID?

Stage-1 finding: pooled c* is flat at ~0.4 across all offsets, the AC rollout's
prediction-error floor swamping the reachability signal. Two rescue candidates,
measured here as full ladders per episode:
  1. TOKEN metric: mean L1 over tokens, upstream's own CEM energy, on the
     suspicion that pooling is what destroys the spatial signal.
  2. SELF-FLOOR CALIBRATION: c*_cal(t) = c*(t) - c*(self), where c*(self)
     targets the CURRENT frame's own tokens ("plan to stay put"), subtracting
     out the per-state model floor.
The two combine into 4 ladders: pooled, pooled-cal, token, token-cal. A usable
v2 instrument must rise with offset, with spread clearly above its own jitter.

  python -m specaccept.vjepa2.probe_cstar2 --traj droid_eps/droid_ep000.npz \
      --device cuda --offsets 1 2 4 8 16 32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .planner import cstar
from .wm import VJEPA2WM, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", nargs="+", required=True, help="probe-format npz episodes")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="vjepa2_ac_vit_giant")
    ap.add_argument("--offsets", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--cem-samples", type=int, default=200)
    ap.add_argument("--cem-steps", type=int, default=5)
    ap.add_argument("--cem-rollout", type=int, default=2)
    args = ap.parse_args()

    cem_args = dict(samples=args.cem_samples, cem_steps=args.cem_steps,
                    rollout=args.cem_rollout)
    wm = VJEPA2WM(model_name=args.model, device=args.device)

    for path in args.traj:
        traj = np.load(path)
        frames = traj["observations"][0]
        states = torch.tensor(traj["states"][0]).float()
        T = len(frames)
        tokens = wm.encode_frames(frames)
        states = states.to(tokens.device)
        z0, s0 = tokens[0:1], states[0].view(1, 1, 7)

        # per-state model floor: plan toward the current frame itself
        floor_p, _ = cstar(wm, z0, s0, z0, cem_args=cem_args, metric="pooled")
        floor_t, _ = cstar(wm, z0, s0, z0, cem_args=cem_args, metric="token")

        print(f"\n==== C*-V2 LADDERS: {Path(path).name} (T={T}) ====")
        print(f"  self-floor: pooled={floor_p:.4f}  token={floor_t:.4f}")
        print(f"  {'t':>4} | {'pooled':>7} {'p-cal':>7} | {'token':>7} {'t-cal':>7} | "
              f"{'raw pooled dist':>15}")
        offs = [t for t in args.offsets if 0 < t < T]
        if T - 1 not in offs:
            offs.append(T - 1)
        for t in offs:
            tgt = tokens[t:t + 1]
            cp, _ = cstar(wm, z0, s0, tgt, cem_args=cem_args, metric="pooled")
            ct, _ = cstar(wm, z0, s0, tgt, cem_args=cem_args, metric="token")
            p0, pt = pool(z0), pool(tgt)
            raw = float((p0 - pt).norm() / p0.norm().clamp_min(1e-8))
            print(f"  {t:>4} | {cp:7.4f} {cp - floor_p:+7.4f} | "
                  f"{ct:7.4f} {ct - floor_t:+7.4f} | {raw:15.4f}")
        print("  (v2 verdict wants a column that RISES with t, spread >> jitter; "
              "flat everywhere = certificate stays dead on this substrate)")


if __name__ == "__main__":
    main()
