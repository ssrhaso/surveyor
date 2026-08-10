"""Offline transplant probe on upstream's bundled Franka trajectory.

No robot, no simulator: everything runs against the recorded episode that ships
with the clone (notebooks/franka_example_traj.npz). Four skippable stages:

  1. ANCHOR    upstream CEM from frame 0 toward the final frame, compared to
               the ground-truth first action (the notebook's own sanity check).
               Must look sane before anything else is trusted.
  2. C*        the certificate vs goal offset: c*(frame0 -> frame_t) over a
               ladder of offsets, printed next to the raw pooled encoder
               distance, the gate-v1 instrument that saturates where c* did
               not. The substrate's regime map.
  3. FLOOR     rel pooled distance between frames S apart, for several S: the
               criterion-floor shape that says where tau must sit on this
               substrate (transfer hypothesis 0.20).
  4. PLUMBING  certified Surveyor teacher-forced along the recorded
               trajectory with the data-free lerp drafter. Achieved latents
               come from the recording while the source runs
               verify/serve/redraft/retire exactly as it would closed-loop,
               exercising every code path without a trained TokenGDM.

The first run downloads the ViT-g AC weights (several GB) via torch.hub. CPU
works at the default tiny CEM budgets, the notebook's own CPU setting; raise
--cem-samples/--cem-steps on a GPU.

  python -m surveyor.vjepa2.probe_offline --device cpu --dump-latents lat_franka.npz
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch

from .planner import cstar, flat_plan
from .sources import CstarRetireTokenSource, LerpBlockDrafter
from .wm import UPSTREAM, VJEPA2WM, ensure_upstream_on_path, pool


def rel_pooled(a: torch.Tensor, b: torch.Tensor) -> float:
    pa, pb = pool(a), pool(b)
    return float((pa - pb).norm() / pa.norm().clamp_min(1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default=str(UPSTREAM / "notebooks" / "franka_example_traj.npz"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default="vjepa2_ac_vit_giant")
    ap.add_argument("--dump-latents", default=None, metavar="OUT.npz")
    ap.add_argument("--offsets", type=int, nargs="+", default=None,
                    help="goal offsets for the c* ladder (default: powers of 2 up to T-1)")
    ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 5, 10])
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-future", type=int, default=3)
    ap.add_argument("--replan-stride", type=int, default=2,
                    help="frames between teacher-forced replans in stage 4")
    ap.add_argument("--cem-samples", type=int, default=25)
    ap.add_argument("--cem-steps", type=int, default=2)
    ap.add_argument("--cem-rollout", type=int, default=2)
    for st in ("anchor", "cstar", "floor", "plumbing"):
        ap.add_argument(f"--skip-{st}", action="store_true")
    args = ap.parse_args()

    cem_args = dict(samples=args.cem_samples, cem_steps=args.cem_steps,
                    rollout=args.cem_rollout)

    traj = np.load(args.traj)
    frames = traj["observations"][0]                    # (T, H, W, C)
    states = torch.tensor(traj["states"][0]).float()    # (T, 7)
    T = len(frames)
    print(f"[traj] {Path(args.traj).name}: {T} frames, states {tuple(states.shape)}")

    print(f"[wm] loading {args.model} on {args.device} (first run downloads weights)...")
    wm = VJEPA2WM(model_name=args.model, device=args.device)
    print(f"[wm] tokens_per_frame={wm.tokens_per_frame} embed_dim={wm.embed_dim}")

    print("[encode] encoding all frames...")
    tokens = wm.encode_frames(frames)                   # (T, T_tok, D)
    states = states.to(tokens.device)
    if args.dump_latents:
        np.savez_compressed(args.dump_latents,
                            tokens=tokens.cpu().to(torch.float16).numpy(),
                            states=states.cpu().numpy())
        print(f"[encode] dumped -> {args.dump_latents}")

    z0 = tokens[0:1]
    s0 = states[0].view(1, 1, 7)
    goal = tokens[-1:]

    if not args.skip_anchor:
        plan = flat_plan(wm, z0, s0, goal, cem_args=cem_args)
        H = plan.shape[1]
        disp_plan = plan[0, :, :3].sum(0).float().cpu()
        disp_true = (states[min(H, T - 1)] - states[0])[:3].float().cpu()
        disp_goal = (states[-1] - states[0])[:3].float().cpu()

        def _cos(a, b):
            return float(torch.nn.functional.cosine_similarity(
                a.view(1, -1), b.view(1, -1)))

        print("\n==== ANCHOR (upstream CEM, frame0 -> final frame) ====")
        print(f"  planned xyz displacement (H={H})  = "
              f"({disp_plan[0]:+.4f}, {disp_plan[1]:+.4f}, {disp_plan[2]:+.4f})")
        print(f"  true displacement over {min(H, T - 1)} frames = "
              f"({disp_true[0]:+.4f}, {disp_true[1]:+.4f}, {disp_true[2]:+.4f})"
              f"  |.|={float(disp_true.norm()):.4f}")
        print(f"  start->goal displacement          = "
              f"({disp_goal[0]:+.4f}, {disp_goal[1]:+.4f}, {disp_goal[2]:+.4f})"
              f"  |.|={float(disp_goal.norm()):.4f}")
        print(f"  cosine(plan, true-over-H) = {_cos(disp_plan, disp_true):+.3f}   "
              f"cosine(plan, start->goal) = {_cos(disp_plan, disp_goal):+.3f}")
        print("  (sanity, not a bar: tiny CEM budget; cosines are meaningless "
              "when |.| above is near zero)")

    if not args.skip_cstar:
        offs = [t for t in (args.offsets or (1, 2, 4, 8, 16, 32, 64)) if 0 < t < T]
        offs.append(T - 1)
        print("\n==== C* LADDER (frame0 -> frame_t) ====")
        print(f"  {'t':>4} | {'c* (certificate)':>17} | {'pooled enc dist':>15}")
        for t in sorted(set(offs)):
            c, _ = cstar(wm, z0, s0, tokens[t:t + 1], cem_args=cem_args)
            d = rel_pooled(z0, tokens[t:t + 1])
            print(f"  {t:>4} | {c:>17.4f} | {d:>15.4f}")
        print(f"  (tau transfer hypothesis: {args.tau}; c* should separate near from far "
              f"where raw distance saturates)")

    if not args.skip_floor:
        print("\n==== CRITERION-FLOOR SHAPE (rel pooled dist between frames S apart) ====")
        for S in args.strides:
            if S >= T:
                continue
            ds = [rel_pooled(tokens[m:m + 1], tokens[m + S:m + S + 1])
                  for m in range(0, T - S)]
            ds = np.array(ds)
            print(f"  S={S:>3}: p10={np.percentile(ds, 10):.4f} "
                  f"p50={np.percentile(ds, 50):.4f} p90={np.percentile(ds, 90):.4f}  "
                  f"(n={len(ds)})")
        print(f"  (tau={args.tau} must sit above the p50 single-replan floor to accept "
              f"true progress; if not, tau is re-derived, per the paper's rule)")

    if not args.skip_plumbing:
        print("\n==== PLUMBING (certified Surveyor, teacher-forced, lerp drafter) ====")
        drafter = LerpBlockDrafter(n_future=args.n_future)
        cstar_fn = partial(cstar, wm, cem_args=cem_args)
        src = CstarRetireTokenSource(drafter, cstar_fn, n_envs=1,
                                     device=str(tokens.device),
                                     tau=args.tau, k=args.k)
        for m in range(0, T, args.replan_stride):
            targets = src.current(tokens[m:m + 1], states[m].view(1, 1, 7), [0], goal)
            assert targets[0] is not None and targets[0].shape == goal[0].shape
        print(f"  {src.stats()}")
        print(f"  replans={src._replans[0]} retire_replan={src.retire_replan[0]} "
              f"c*_first={src.c_first[0]:.4f} c*_last={src.c_last[0]:.4f}")
        print("  (expected shape: c* falls as the recorded episode approaches its "
              "own final frame; retire fires late or not at all on a short clip)")

    print("\n[done]")


if __name__ == "__main__":
    main()
