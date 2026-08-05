"""Per-position fidelity probe for the N-position subgoal block: does sampled
quality degrade with block position (suffix decay)?

Samples full blocks with the real diffusion sampler and reports the diag_gdm
metrics broken out by position, from an existing episodes-file.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from specaccept import encoder
from specaccept.diag_gdm import metrics, show
from specaccept.drafter import load_gdm_planner


def parse_args():
    p = argparse.ArgumentParser(description="Per-position (m+1/m+2/m+3) sampled fidelity probe")
    p.add_argument("--gdm-ckpt", required=True)
    p.add_argument("--h5", required=True)
    p.add_argument("--episodes-file", required=True,
                   help="JSON with (episode_idx, start_step) pairs; episodes must have "
                        "ep_len > start_step + n_future*stride (e.g. subset_longeval.episodes150.json)")
    p.add_argument("--source", choices=["pretrained", "local"], default="local")
    p.add_argument("--encoder-id", default="quentinll/lewm-pusht")
    p.add_argument("--local-dir", default=None)
    p.add_argument("--swm-src", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--stride", type=int, default=25, help="steps between subgoal positions")
    p.add_argument("--n-probe", type=int, default=256)
    p.add_argument("--gdm-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    planner = load_gdm_planner(args.gdm_ckpt, device=args.device)
    N = planner.cfg.n_future
    print(f"[gdm] N={N} WG={planner.cfg.wg} T={planner.diffusion.timesteps} "
          f"sampler={planner.diffusion.sampler} param={planner.diffusion.parameterization}")

    with open(args.episodes_file) as f:
        payload = json.load(f)
    pairs = payload["episodes"][:args.n_probe]
    print(f"[episodes-file] {args.episodes_file}: using {len(pairs)} pairs "
          f"(goal_offset={payload.get('goal_offset')}, note={payload.get('note', '-')})")

    model = encoder.load_lewm(source=args.source, encoder_id=args.encoder_id,
                              local_dir=args.local_dir, swm_src=args.swm_src, device=args.device)

    import h5py
    with h5py.File(args.h5, "r") as f:
        ep_off = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
        pixels = f["pixels"]
        cond_rows, target_rows = [], [[] for _ in range(N)]
        for ep, st in pairs:
            base = int(ep_off[ep]); last = base + int(ep_len[ep]) - 1
            cond_rows.append(base + int(st))
            for k in range(N):
                row = base + int(st) + (k + 1) * args.stride
                assert row <= last, (
                    f"episode {ep} too short for position {k+1} at stride {args.stride} "
                    f"(row {row} > last {last}); use an episodes-file with longer episodes")
                target_rows[k].append(row)

        all_rows = np.array(cond_rows + [r for tk in target_rows for r in tk])
        order = np.argsort(all_rows, kind="stable")
        lat = encoder.encode_frames(model, pixels[all_rows[order]], device=args.device)
        out = torch.empty_like(lat)
        out[order] = lat

    M = len(pairs)
    z_cond = out[:M].to(args.device)
    z_true = [out[M + k * M: M + (k + 1) * M].to(args.device) for k in range(N)]

    gen = torch.Generator(device=planner.device).manual_seed(args.seed)
    pred_seq = planner.sample_sequence(z_cond, n_steps=args.gdm_steps, generator=gen)  # (M, N, D)

    print(f"\n[SUFFIX DECAY PROBE] n={M}, real DDIM/DDPM sampling (not teacher-forced), "
          f"per-position ground truth from the h5 at stride {args.stride}")
    for k in range(N):
        m = metrics(pred_seq[:, k], z_true[k], z_cond)
        show(f"position {k} (subgoal m+{k+1}, +{(k+1)*args.stride} steps)", m)

    print("\n[read-off] rel_err/cos_move degrading monotonically 0->1->2 with a widening "
          "gap to noop_err = real suffix decay.")


if __name__ == "__main__":
    main()
