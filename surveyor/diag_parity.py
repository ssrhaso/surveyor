"""Parity diagnostic (no CEM): latent parity between the two encode paths
and cost parity between SubgoalCostModel and LeWM.get_cost on identical
inputs. Fast; run where LeWM.get_cost is available.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

from surveyor import encoder
from surveyor.sources import SubgoalCostModel
from surveyor.envs.pusht.eval import img_transform, img_transform_fallback, sample_short


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--source", default="pretrained")
    ap.add_argument("--encoder-id", default="quentinll/lewm-pusht")
    ap.add_argument("--local-dir", default=None)
    ap.add_argument("--swm-src", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-eval", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    encoder._ensure_swm_importable(args.swm_src)
    import h5py
    from torchvision import tv_tensors

    dev = args.device
    model = encoder.load_lewm(source=args.source, encoder_id=args.encoder_id,
                              local_dir=args.local_dir, swm_src=args.swm_src, device=dev)
    try:
        tf = img_transform()
        print("[tf] stable_pretraining ImageNet stats")
    except Exception:
        tf = img_transform_fallback()
        print("[tf] fallback ImageNet stats")

    # decisive: do the harness (spt) stats match encode_frames' hardcoded stats?
    print(f"[stats] encode_frames mean={encoder._IMAGENET_MEAN.flatten().tolist()} "
          f"std={encoder._IMAGENET_STD.flatten().tolist()}")
    try:
        import stable_pretraining as spt
        print(f"[stats] spt ImageNet = {spt.data.dataset_stats.ImageNet}")
    except Exception as e:
        print(f"[stats] spt not importable here ({e!r}); harness uses fallback (== encode_frames)")

    @torch.no_grad()
    def harness_px(frame_uint8):  # (224,224,3)uint8 -> (3,224,224) like the harness
        return tf(tv_tensors.Image(np.transpose(frame_uint8, (2, 0, 1)))).to(dev)

    @torch.no_grad()
    def harness_encode(frame_uint8):  # how the harness encodes info['goal']
        px = harness_px(frame_uint8).view(1, 1, 3, 224, 224)
        return model.encode({"pixels": px})["emb"][:, 0]  # (1,192)

    # (0) sampling-identity check vs baseline eval.py
    with h5py.File(args.h5, "r") as f:
        episode_idx = f["episode_idx"][:]
        step_idx = f["step_idx"][:]
        ep_len = f["ep_len"][:]
    max_start_per_row = ep_len[episode_idx] - 25 - 1
    valid_indices = np.nonzero(step_idx <= max_start_per_row)[0]
    g = np.random.default_rng(args.seed)
    chosen = g.choice(len(valid_indices) - 1, size=args.num_eval, replace=False)
    rows = np.sort(valid_indices[chosen])
    print(f"[SAMPLING] valid starting points = {len(valid_indices)} (baseline printed 1869611)")
    print(f"[SAMPLING] first rows = {rows[:10].tolist()}  (baseline: 201657 209211 221572 305420 429219 ...)")

    eps, starts = sample_short(args.h5, args.num_eval, 25, args.seed)
    print(f"[diag] {len(eps)} eps; eps[:5]={eps[:5]} starts[:5]={starts[:5]}")

    # (1) latent parity over all eval goal frames
    with h5py.File(args.h5, "r") as f:
        ep_off = f["ep_offset"][:]
        pixels = f["pixels"]
        diffs = []
        first = {}
        for i, (ep, st) in enumerate(zip(eps, starts, strict=True)):
            gf = pixels[int(ep_off[ep]) + st + 25]
            z_oracle = encoder.encode_frames(model, gf[None], device=dev)[0]  # (192,)
            z_harness = harness_encode(gf)[0]                                  # (192,)
            diffs.append((z_oracle - z_harness).abs().max().item())
            if i == 0:
                first = dict(ep=ep, st=st, gf=gf,
                             init=pixels[int(ep_off[ep]) + st],
                             z_harness=z_harness, z_oracle=z_oracle)
    diffs = np.array(diffs)
    print("[LATENT parity] encode_frames vs harness img_transform+E:")
    print(f"   worst|diff|={diffs.max():.3e} mean={diffs.mean():.3e} "
          f"#>1e-4={(diffs > 1e-4).sum()} #>1e-3={(diffs > 1e-3).sum()} (of {len(diffs)})")

    # (2) cost parity on episode 0 (B=1,S=8)
    B, S, H, ADIM = 1, 8, 5, 10
    px_init = harness_px(first["init"]).view(1, 1, 1, 3, 224, 224).expand(B, S, 1, 3, 224, 224).contiguous()
    px_goal = harness_px(first["gf"]).view(1, 1, 1, 3, 224, 224).expand(B, S, 1, 3, 224, 224).contiguous()
    action = torch.zeros(B, S, 1, 2, device=dev)
    g = torch.Generator(device=dev).manual_seed(0)
    candidates = torch.randn(B, S, H, ADIM, generator=g, device=dev)

    try:
        cost_box = model.get_cost({"pixels": px_init.clone(), "goal": px_goal.clone(),
                                   "action": action.clone()}, candidates.clone())
        sub = first["z_harness"].view(1, 1, 192).expand(B, S, 192).contiguous()
        cost_mine = SubgoalCostModel(model).get_cost(
            {"pixels": px_init.clone(), SubgoalCostModel.SUBGOAL_KEY: sub}, candidates.clone())
        cb, cm = cost_box.flatten().float(), cost_mine.flatten().float()
        order_match = bool((torch.argsort(cb) == torch.argsort(cm)).all())
        print("[COST parity] installed LeWM.get_cost vs SubgoalCostModel (same goal latent):")
        print(f"   cost_box[:4] ={[round(x,4) for x in cb[:4].tolist()]}")
        print(f"   cost_mine[:4]={[round(x,4) for x in cm[:4].tolist()]}")
        print(f"   max|diff|={(cb - cm).abs().max().item():.3e}  argsort_match={order_match}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("[COST parity] installed get_cost raised (expected on the broken source checkout):", repr(e))


if __name__ == "__main__":
    main()
