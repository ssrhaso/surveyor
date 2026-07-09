"""TASK C - train the GDM latent diffusion planner.

Trains a conditional DDPM (DiT backbone, WG=1, predicts N=3 future subgoals,
denoising score-matching) on the stride-25 subgoal latents in subgoals_pusht.pt
(Task A output). The frozen LeWM is NOT needed here - we operate purely on the
cached latents.

(condition, target) pairs per episode (stride-25 latent sequence z[0..n_sg-1]):
  condition = z[m]                                     (WG=1)
  target    = [z[m+1], z[m+2], z[m+3]]                 (N=3)
End-of-episode rule (--window-rule):
  clamp (default): future indices clamped to the last subgoal (n_sg-1), so near
      the goal the targets repeat the goal latent - mirrors build_oracle_table's
      frame clamping and the inference behavior. m in [0, n_sg-2].
  full           : require a complete N-window; m in [0, n_sg-1-N]. Drops short
      episodes (mean n_sg ~ 5.3), so far fewer pairs.

Latents live near the sqrt(192) sphere; the DM trains on PER-DIM STANDARDIZED
latents (mean/std over the masked training subset). The stats are saved in the
checkpoint and inverted at inference so injected subgoals return to native
E-space (||z||~13.9). Encoder/predictor stay frozen; only GDM trains.

Train set: --mask 5 (default, in_5deg subset ~ 9,094 eps ~ paper's 8,318) or
--mask 20 (all 12,042 kept episodes).

CPU smoke (tiny model, 1 epoch - logic only; real train on the box):
  python -m ffjepa.train_gdm --subgoals subgoals_pusht.pt --out /tmp/gdm_smoke.pt \
      --device cpu --epochs 1 --hidden 64 --depth 2 --heads 4 --timesteps 50

Box (A100), full training (~20 epochs, ~50M-param DiT):
  cd ~/le-wm && python -m ffjepa.train_gdm --subgoals subgoals_pusht.pt \
      --out gdm_pusht.pt --device cuda --mask 5 --epochs 20

WARNING - the argparse DEFAULTS here (lr 1e-4, batch 256, no grad-clip, no
lr-schedule, no --amp, ema_decay 0.999) do NOT reproduce the faithful baseline
`gdm_faithful.pt`. The faithful recipe = LeWM's config (batch 128, lr 5e-5,
wd 1e-3, grad-clip 1.0, warmup_cosine, bf16 --amp, EMA OFF) + dense data
(subgoals_dense_full.pt, --subgoal-step 25). The CANONICAL faithful invocation
lives in run_recipe.sh (eps) / run_vpred.sh (v); reproduce from there, not the
bare defaults. Two training-set knobs are also live and unspecified by the paper:
--mask {5,20} (filter tolerance; 5=in_5deg~9,094~paper count, 20=all~12,042~our
20 deg eval criterion - a single-variable A/B) and --window-rule.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from ffjepa.gdm_model import (GDM, GDMConfig, GaussianDiffusion, save_gdm,
                              count_params)


def parse_args():
    p = argparse.ArgumentParser(description="FF-JEPA Task C: train GDM diffusion planner")
    p.add_argument("--subgoals", required=True, help="Task A output (subgoals_pusht.pt)")
    p.add_argument("--out", required=True, help="output checkpoint path")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    # train subset
    p.add_argument("--mask", choices=["5", "20", "custom"], default="5",
                   help="5 = in_5deg subset (~paper's 8,318); 20 = all kept episodes; "
                        "custom = load a per-episode boolean array from --custom-mask-file")
    p.add_argument("--custom-mask-file", default=None,
                   help="path to a .npy boolean array, one entry per episode in the "
                        "subgoals file (same order as blob['episode_idx']), for --mask custom")
    p.add_argument("--window-rule", choices=["clamp", "full"], default="clamp")
    p.add_argument("--subgoal-step", type=int, default=1,
                   help="index gap between successive subgoals in the latent array. "
                        "1 for a stride-25 subgoal file (default); for a DENSE file "
                        "(build_subgoals --stride 1) use 25 so targets are 25 frames "
                        "ahead while conditioning on EVERY frame (sliding conditions).")
    # model / diffusion (defaults -> ~50M params, paper-sized)
    p.add_argument("--latent-dim", type=int, default=192)
    p.add_argument("--n-future", type=int, default=3)   # N
    p.add_argument("--wg", type=int, default=1)         # context window
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--depth", type=int, default=11)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--goal-cond", action="store_true",
                   help="ABLATION: also condition the GDM on the episode goal latent "
                        "(NOT FF-JEPA, which is goal-free). Goal = episode's last subgoal.")
    p.add_argument("--goal-rule", choices=["final", "window"], default="final",
                   help="goal_cond only. final = episode's last subgoal (PushT/TwoRoom "
                        "expert demos, which END at the task target). window = hindsight "
                        "relabeling for RANDOM-policy data (Reacher): goal = z[min(m+G, "
                        "last)] with G = --goal-gap, and the target indices are CLAMPED "
                        "at the goal index, so the drafted block terminates at (and then "
                        "repeats) the goal - the same clamp semantics the PushT drafter "
                        "gets from expert episodes ending at the target.")
    p.add_argument("--goal-gap", type=int, default=None,
                   help="G for --goal-rule window, in FILE-INDEX units (= frames for a "
                        "dense stride-1 file). Match the eval protocol's goal_offset "
                        "(Reacher paper protocol: 25). Default = n_future * subgoal_step "
                        "(the drafted window's end).")
    p.add_argument("--timesteps", type=int, default=1000)
    # diffusion parameterization / schedule / loss weighting (flag-gated A/B levers;
    # defaults reproduce gdm_faithful.pt exactly: eps / linear / no Min-SNR)
    p.add_argument("--parameterization", choices=["eps", "v"], default="eps",
                   help="eps = predict noise (faithful default); v = predict velocity "
                        "v=sqrt(acp)*eps-sqrt(1-acp)*x0 (better-conditioned at high t)")
    p.add_argument("--schedule", choices=["linear", "cosine"], default="linear",
                   help="linear beta (default) or cosine bar-alpha (Nichol & Dhariwal)")
    p.add_argument("--min-snr-gamma", type=float, default=0.0,
                   help="Min-SNR-gamma loss weighting (Hang 2023); 0 = plain MSE")
    p.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim",
                   help="inference sampler baked into ckpt: ddim (default, --gdm-steps) or "
                        "ddpm ancestral over all T steps (LDP/Diffusion-Policy default)")
    p.add_argument("--normalization", choices=["standardize", "minmax"], default="standardize",
                   help="latent norm: standardize (per-dim mean0/std1, default) or "
                        "minmax (per-dim scale to [-1,1], the LDP/Diffusion-Policy convention)")
    # optimization
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=0.0,
                   help="max grad-norm (0=off); LeWM recipe uses 1.0")
    p.add_argument("--lr-schedule", choices=["none", "warmup_cosine"], default="none",
                   help="warmup_cosine = LeWM recipe: linear warmup then cosine anneal to 0")
    p.add_argument("--warmup-frac", type=float, default=0.01,
                   help="fraction of total steps for linear warmup (LeWM ~1 pct)")
    p.add_argument("--amp", action="store_true",
                   help="bf16 autocast forward (LeWM uses precision bf16; ~2x faster). "
                        "fp32 master weights kept, no grad scaler needed.")
    p.add_argument("--ema-decay", type=float, default=0.999, help="EMA of weights (0 disables)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=500,
                   help="print running loss every N iters (each print forces a GPU sync)")
    return p.parse_args()


def build_pairs(latents, lengths, offsets, mask, n_future, window_rule, step=1,
                goal_rule="final", goal_gap=None):
    """Build (condition, target) index pairs across all masked episodes.

    condition = z[m]; target = [z[m+1*step], ..., z[m+N*step]]. With step=1 on a
    stride-25 subgoal file this is the stride-aligned scheme. With step=25 on a
    DENSE file (every frame encoded) the condition is ANY frame and the targets
    are the subgoals 25/50/75 frames ahead -> sliding conditions, full phase
    coverage (what the closed-loop eval actually feeds).

    goal_rule (used only when the caller trains goal_cond):
      final  = per-episode goal = z[last] (expert demos ending at the target).
      window = hindsight window goal for random-policy data: goal index
               gi = min(m + goal_gap, last), and target indices are clamped at
               gi, so the block runs TO the goal and then repeats it (the clamp
               semantics PushT gets for free from demos ending at the target).

    Returns conds (M,D), targets (M,N,D) as float32 tensors (native E-space).
    """
    D = latents.shape[1]
    gap = goal_gap if goal_gap is not None else n_future * step
    conds, targets, goals = [], [], []
    n_eps_used = 0
    for i in range(len(lengths)):
        if not mask[i]:
            continue
        n_eps_used += 1
        off, L = int(offsets[i]), int(lengths[i])
        z = latents[off:off + L]                       # (L, D)
        last = L - 1
        z_goal = z[last]                               # episode goal = final subgoal latent
        if window_rule == "full":
            m_max = last - n_future * step              # need z[m+N*step] to exist
        else:  # clamp future indices to the last subgoal (the goal)
            m_max = L - 2                               # need at least one future frame
        for m in range(0, m_max + 1):
            conds.append(z[m])
            if goal_rule == "window":
                gi = min(m + gap, last)
                idx = [min(m + k * step, gi) for k in range(1, n_future + 1)]
                goals.append(z[gi])
            else:
                idx = [min(m + k * step, last) for k in range(1, n_future + 1)]
                goals.append(z_goal)                    # (D,), used only if goal_cond
            targets.append(z[idx])                      # (N, D)
    if not conds:
        raise RuntimeError("no (condition,target) pairs built - check mask/window-rule/step")
    conds = torch.stack(conds).float()                  # (M, D)
    targets = torch.stack(targets).float()              # (M, N, D)
    goals = torch.stack(goals).float()                  # (M, D)
    assert conds.shape[1] == D and targets.shape[1] == n_future
    return conds, targets, goals, n_eps_used


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()
    print(f"[cfg] {vars(args)}")

    blob = torch.load(args.subgoals, map_location="cpu", weights_only=False)
    latents = blob["latents"].float()                   # (sum_nsg, D)
    lengths = blob["lengths"].numpy()
    offsets = blob["offsets"].numpy()
    in_5deg = blob["in_5deg"].numpy().astype(bool)
    D = int(blob.get("latent_dim", latents.shape[1]))
    assert D == args.latent_dim, f"latent_dim mismatch {D} vs {args.latent_dim}"

    if args.mask == "custom":
        if not args.custom_mask_file:
            raise ValueError("--mask custom requires --custom-mask-file")
        mask = np.load(args.custom_mask_file).astype(bool)
        assert len(mask) == len(lengths), (
            f"custom mask length {len(mask)} != {len(lengths)} episodes in subgoals file")
    else:
        mask = in_5deg if args.mask == "5" else np.ones(len(lengths), dtype=bool)
    file_stride = int(blob.get("stride", 25))
    conds, targets, goals, n_eps = build_pairs(latents, lengths, offsets, mask,
                                        args.n_future, args.window_rule, step=args.subgoal_step,
                                        goal_rule=args.goal_rule, goal_gap=args.goal_gap)
    M = conds.shape[0]
    print(f"[data] mask={args.mask}deg: {int(mask.sum())} episodes used; "
          f"{M} (cond,target) pairs; window_rule={args.window_rule}; "
          f"file_stride={file_stride} subgoal_step={args.subgoal_step} "
          f"=> subgoal horizon = {file_stride * args.subgoal_step} frames"
          f"{f'  [GOAL-COND ablation, goal_rule={args.goal_rule} gap={args.goal_gap}]' if args.goal_cond else ''}")

    # normalization stats over the pool of training latents actually used
    # (conditions + targets together - same subgoal distribution). stat_a/stat_b
    # are (mean,std) for standardize or (min,max) for minmax; inverted at inference.
    pool = torch.cat([conds, targets.reshape(-1, D)], dim=0)
    if args.normalization == "standardize":
        stat_a = pool.mean(dim=0)
        stat_b = pool.std(dim=0).clamp_min(1e-6)
        def _norm(z):
            return (z - stat_a) / stat_b
        print(f"[norm] native ||z|| mean={pool.norm(dim=1).mean():.2f} "
              f"(sqrt(D)={np.sqrt(D):.2f}); standardize per-dim "
              f"(mean|.|={stat_a.abs().mean():.3f}, std mean={stat_b.mean():.3f})")
    else:  # minmax -> [-1,1]
        stat_a = pool.min(dim=0).values
        stat_b = pool.max(dim=0).values
        rng = (stat_b - stat_a).clamp_min(1e-6)
        def _norm(z):
            return 2.0 * (z - stat_a) / rng - 1.0
        print(f"[norm] native ||z|| mean={pool.norm(dim=1).mean():.2f} "
              f"(sqrt(D)={np.sqrt(D):.2f}); minmax->[-1,1] per-dim "
              f"(range mean={rng.mean():.3f})")

    conds_s = _norm(conds)
    targets_s = _norm(targets)
    goals_s = _norm(goals)

    device = args.device
    # keep the whole (small) training set resident on-device and index it with a
    # per-epoch random permutation. At batch 8 the DataLoader + per-batch host->
    # device copy dominate wall-clock; this removes that overhead entirely.
    conds_s = conds_s.to(device)
    targets_s = targets_s.to(device)
    goals_s = goals_s.to(device) if args.goal_cond else None  # only resident if used
    M = conds_s.shape[0]

    cfg = GDMConfig(latent_dim=D, n_future=args.n_future, wg=args.wg,
                    hidden=args.hidden, depth=args.depth, heads=args.heads,
                    mlp_ratio=args.mlp_ratio, goal_cond=args.goal_cond)
    model = GDM(cfg).to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps, schedule=args.schedule,
                                  parameterization=args.parameterization,
                                  min_snr_gamma=args.min_snr_gamma, sampler=args.sampler,
                                  device=device)
    n_head = count_params(model)
    print(f"[model] GDM (DiT) head params = {n_head/1e6:.2f}M "
          f"(hidden={args.hidden} depth={args.depth} heads={args.heads}); "
          f"diffusion T={args.timesteps} param={args.parameterization} "
          f"schedule={args.schedule} sampler={args.sampler} "
          f"norm={args.normalization} min_snr_gamma={args.min_snr_gamma}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # LR schedule. LeWM uses linear warmup (~1% of total steps) then cosine anneal to
    # eta_min=0 (train.py + stable_pretraining LinearWarmupCosineAnnealingLR), stepped
    # per optimizer step. Default "none" preserves the old constant-LR behavior.
    scheduler = None
    if args.lr_schedule == "warmup_cosine":
        steps_per_epoch = (M + args.batch_size - 1) // args.batch_size
        total_steps = max(1, args.epochs * steps_per_epoch)
        warmup_steps = max(1, int(args.warmup_frac * total_steps))

        def _lr_mult(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps                     # linear 0 -> 1
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))   # cosine 1 -> 0

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_mult)
        print(f"[sched] warmup_cosine: total_steps={total_steps} warmup={warmup_steps} "
              f"peak_lr={args.lr:.1e} -> 0 (eta_min)")
    # EMA of weights. Cache live references to the float tensors once and update
    # them with fused _foreach ops; the old per-iter `model.state_dict()` loop
    # launched ~2 kernels per tensor (hundreds/iter), which dwarfed the tiny
    # batch-8 forward/backward. GDM has no non-float buffers, so float-only EMA
    # is exact (math identical to ema*decay + param*(1-decay)).
    ema = None
    if args.ema_decay and args.ema_decay > 0:
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema_float = [ema[k] for k, v in model.state_dict().items() if v.dtype.is_floating_point]
        model_float = [v.detach() for v in model.state_dict().values() if v.dtype.is_floating_point]

    n_iter = 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(M, device=device)
        run_loss_t = torch.zeros((), device=device)  # accumulate on-device; sync only at log time
        n_batch = 0
        for b in range(0, M, args.batch_size):
            idx = perm[b:b + args.batch_size]
            cond_b = conds_s[idx]
            tgt_b = targets_s[idx]
            goal_b = goals_s[idx] if args.goal_cond else None
            if args.amp:
                with torch.autocast(device_type=args.device.split(":")[0],
                                    dtype=torch.bfloat16):
                    loss = diffusion.p_losses(model, tgt_b, cond_b, goal=goal_b)
            else:
                loss = diffusion.p_losses(model, tgt_b, cond_b, goal=goal_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            n_iter += 1
            if ema is not None:
                with torch.no_grad():
                    torch._foreach_mul_(ema_float, args.ema_decay)
                    torch._foreach_add_(ema_float, model_float, alpha=1 - args.ema_decay)
            run_loss_t += loss.detach()
            n_batch += 1
            # intra-epoch progress so the loss curve is visible during long epochs
            # (each epoch is ~M/batch iters; per-epoch-only logging looks "stuck").
            # .item() forces a GPU sync, so only do it at the log cadence.
            if n_iter % args.log_every == 0:
                el = time.time() - t0
                print(f"  [iter {n_iter}] loss={loss.item():.5f} "
                      f"(run {run_loss_t.item()/max(n_batch,1):.5f}, {el:.0f}s, "
                      f"{1000*el/n_iter:.0f}ms/it)", flush=True)
        print(f"[epoch {epoch+1:02d}/{args.epochs}] loss={run_loss_t.item()/max(n_batch,1):.5f} "
              f"iters={n_iter} ({time.time()-t0:.0f}s)", flush=True)

    # save EMA weights if used (better samples), else raw
    if ema is not None:
        model.load_state_dict(ema)
    model.eval()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_gdm(args.out, model, diffusion, stat_a, stat_b, normalization=args.normalization, extra={
        "mask": args.mask, "window_rule": args.window_rule,
        "goal_rule": args.goal_rule, "goal_gap": args.goal_gap,
        "file_stride": file_stride, "subgoal_step": args.subgoal_step,
        "subgoal_horizon_frames": file_stride * args.subgoal_step,
        "n_episodes": int(n_eps), "n_pairs": int(M), "n_iter": int(n_iter),
        "epochs": args.epochs, "lr": args.lr, "ema_decay": args.ema_decay,
        "seed": args.seed, "head_params": int(n_head),
        "parameterization": args.parameterization, "schedule": args.schedule,
        "sampler": args.sampler, "normalization": args.normalization,
        "min_snr_gamma": args.min_snr_gamma, "subgoals_src": str(args.subgoals),
    })
    print(f"[save] {args.out}  ({Path(args.out).stat().st_size/1e6:.1f} MB)")
    print(f"[done] {time.time()-t0:.0f}s; {n_iter} iters over {args.epochs} epochs")


if __name__ == "__main__":
    main()
