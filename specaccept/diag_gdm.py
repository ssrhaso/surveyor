"""Diagnose GDM subgoal-prediction quality WITHOUT running CEM (~2 min on GPU).

Tests the hypothesis that the gap (GDM short 56% block vs oracle ~90%) is a
phase / sliding-condition problem: we trained conditioning on stride-25-ALIGNED
subgoals (z_{sg,m} -> z_{sg,m+1}), but at eval the closed-loop conditions on
E(current frame) at an ARBITRARY phase. If the model predicts well for
stride-aligned conditions (in-distribution) but poorly for arbitrary-phase
conditions (eval-distribution), that confirms we need sliding-condition training.

Two probes, both comparing the GDM's predicted next subgoal to the TRUE next
latent (the thing the oracle injects):

  A. IN-DISTRIBUTION  - condition = z_{sg,m} straight from subgoals_pusht.pt
                        (a stride-aligned training condition); true = z_{sg,m+1}.
  B. EVAL-DISTRIBUTION - condition = E(h5 frame at an arbitrary eval start);
                        true = E(h5 frame at start+25) (== the oracle subgoal[1]).

Metrics per probe:
  rel_err   = ||z_pred - z_true|| / ||z_true||         (lower better)
  cos_pred  = cos(z_pred, z_true)                       (higher better)
  cos_move  = cos(z_pred - z_cond, z_true - z_cond)     (did it move the RIGHT way?)
  noop_err  = ||z_cond - z_true|| / ||z_true||          (baseline: predict no change)
  collapse  = mean_d std_d(z_pred) / mean_d std_d(z_true) over the batch
              (<<1 => mode collapse / mean prediction)

Read-off:
  * A good, B bad           -> phase/OOD: rebuild with sliding conditions, retrain.
  * A and B both bad         -> capacity/undertraining: train far longer / more data.
  * collapse << 1            -> model predicts ~the mean subgoal regardless of input.

Run on the box:
  STABLEWM_HOME=$HOME/.stable-wm SDL_VIDEODRIVER=dummy python -m specaccept.diag_gdm \
    --gdm-ckpt gdm_pusht.pt --subgoals subgoals_pusht.pt \
    --h5 $HOME/.stable-wm/datasets/pusht_expert_train.h5 --device cuda
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from specaccept import encoder
from specaccept.drafter import load_gdm_planner


def parse_args():
    p = argparse.ArgumentParser(description="GDM subgoal-prediction diagnostic (no CEM)")
    p.add_argument("--gdm-ckpt", required=True)
    p.add_argument("--subgoals", required=True)
    p.add_argument("--h5", required=True)
    p.add_argument("--source", choices=["pretrained", "local"], default="pretrained")
    p.add_argument("--encoder-id", default="quentinll/lewm-pusht")
    p.add_argument("--local-dir", default=None)
    p.add_argument("--swm-src", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mask", choices=["5", "20"], default="5")
    p.add_argument("--subgoal-step", type=int, default=1,
                   help="index gap between subgoals in the latent array; match train "
                        "(1 for stride-25 file, 25 for a dense --stride 1 file)")
    p.add_argument("--n-probe", type=int, default=256, help="conditions per probe")
    p.add_argument("--gdm-steps", type=int, default=50)
    p.add_argument("--gdm-noise-scale", type=float, default=1.0,
                   help="gamma: sampler noise scale (see eval drivers); offline arm "
                        "of the gamma dose-response joint plot")
    p.add_argument("--tf-steps", type=int, nargs="+", default=[50, 200, 400, 600, 800],
                   help="diffusion timesteps for the teacher-forced denoising probe")
    p.add_argument("--stride", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def metrics(z_pred, z_true, z_cond):
    """All tensors (M, D). Returns a dict of scalar diagnostics."""
    def rel(a, b):
        return (torch.norm(a - b, dim=1) / torch.norm(b, dim=1).clamp_min(1e-8)).mean().item()
    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a, b, dim=1).mean().item()
    pred_std = z_pred.std(dim=0).mean().item()
    true_std = z_true.std(dim=0).mean().item()
    return {
        "rel_err": rel(z_pred, z_true),
        "cos_pred": cos(z_pred, z_true),
        "cos_move": cos(z_pred - z_cond, z_true - z_cond),
        "noop_err": rel(z_cond, z_true),
        "collapse": pred_std / max(true_std, 1e-8),
        "||z_pred||": z_pred.norm(dim=1).mean().item(),
        "||z_true||": z_true.norm(dim=1).mean().item(),
    }


def show(tag, m):
    print(f"\n[{tag}]")
    print(f"  rel_err  = {m['rel_err']:.4f}   (vs no-op baseline {m['noop_err']:.4f}; "
          f"want << no-op)")
    print(f"  cos_pred = {m['cos_pred']:.4f}   cos_move = {m['cos_move']:.4f}   "
          f"(1.0 = perfect direction)")
    print(f"  collapse = {m['collapse']:.3f}   (<<1 => mode collapse)")
    print(f"  ||z_pred|| = {m['||z_pred||']:.2f}   ||z_true|| = {m['||z_true||']:.2f}   "
          f"(native ~13.9)")


@torch.no_grad()
def teacher_forced(planner, cond_native, tgt_native, ts, gen_fn):
    """Teacher-forced denoising: given the TRUE target, add known noise at a
    fixed t, predict the model output, reconstruct x0_hat via the ACTIVE
    parameterization, and measure recovery. Isolates the model+schedule math
    from the sampling chain.

    cond_native: (M,D); tgt_native: (M,N,D) (the true future subgoal sequence).
    Reports, per t: pred_mse (model-space residual: eps_mse for eps, v_mse for v;
    -> 0 = perfect, predict-zero baseline = 1.0) and x0 rel_err over the full
    N-seq AND the immediate-next subgoal [:,0].

    GATE A caveat: for the v parameterization the x0 <- (x_t, v) map uses bounded
    coefficients, so a flat-low x0_relerr across ALL t is TRUE BY CONSTRUCTION
    once the v_mse is small -- it only confirms the math is wired correctly, NOT
    that sampled fidelity improved. The arbiter for "did it help" is GATE B
    (SAMPLED Probe B rel_err / cos_move below), never this curve.
    """
    diff = planner.diffusion
    cond_s = planner.standardize(cond_native)              # (M,D)
    x0 = planner.standardize(tgt_native)                   # (M,N,D)
    M = x0.shape[0]
    pname = f"{diff.parameterization}_mse"
    print(f"\n[C: TEACHER-FORCED DENOISING (true target + known noise @ t)] "
          f"[param={diff.parameterization} schedule={diff.schedule}]")
    print("   pred_mse -> 0 = perfect regression; x0_relerr = the latent CEM sees.")
    print("   eps-param: x0_relerr RISES with t (1/sqrt(acp) amplification, intrinsic).")
    print("   v-param:   x0_relerr stays flat-low across t BY CONSTRUCTION (GATE A only).")
    print(f"   {'t':>5} {pname:>9} {'x0_relerr(seq)':>15} {'x0_relerr(next)':>16}")
    for t_val in ts:
        t = torch.full((M,), int(t_val), device=x0.device, dtype=torch.long)
        noise = torch.randn(x0.shape, device=x0.device, generator=gen_fn())
        x_t = diff.q_sample(x0, t, noise)
        model_out = planner.model(x_t, cond_s, t)
        target = diff.target(x0, noise, t)
        pred_mse = torch.nn.functional.mse_loss(model_out, target).item()
        x0_hat, _ = diff.pred_from_model(model_out, x_t, t)
        relerr_seq = (torch.norm((x0_hat - x0).reshape(M, -1), dim=1)
                      / torch.norm(x0.reshape(M, -1), dim=1).clamp_min(1e-8)).mean().item()
        relerr_next = (torch.norm(x0_hat[:, 0] - x0[:, 0], dim=1)
                       / torch.norm(x0[:, 0], dim=1).clamp_min(1e-8)).mean().item()
        print(f"   {int(t_val):>5} {pred_mse:>9.4f} {relerr_seq:>15.4f} {relerr_next:>16.4f}")


def sample_short(h5, num_eval, goal_offset, seed):
    import h5py
    with h5py.File(h5, "r") as f:
        episode_idx = f["episode_idx"][:]; step_idx = f["step_idx"][:]; ep_len = f["ep_len"][:]
    ep_len_per_row = ep_len[episode_idx]
    valid = np.nonzero(step_idx <= ep_len_per_row - goal_offset - 1)[0]
    g = np.random.default_rng(seed)
    rows = np.sort(valid[g.choice(len(valid) - 1, size=num_eval, replace=False)])
    return episode_idx[rows].tolist(), step_idx[rows].tolist()


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    planner = load_gdm_planner(args.gdm_ckpt, device=args.device)
    planner.noise_scale = args.gdm_noise_scale
    print(f"[gdm] N={planner.cfg.n_future} WG={planner.cfg.wg} "
          f"T={planner.diffusion.timesteps} "
          f"sampler={planner.diffusion.sampler}"
          f"{'/' + str(args.gdm_steps) + 'steps' if planner.diffusion.sampler == 'ddim' else '/allT'} "
          f"param={planner.diffusion.parameterization} "
          f"schedule={planner.diffusion.schedule} "
          f"norm={planner.normalization} "
          f"min_snr_gamma={planner.diffusion.min_snr_gamma} "
          f"head={sum(p.numel() for p in planner.model.parameters())/1e6:.1f}M")
    print("[GATE] A = teacher-forced x0_relerr (math check only). "
          "B = SAMPLED Probe B rel_err/cos_move (the fidelity arbiter).")
    gen = lambda: torch.Generator(device=planner.device).manual_seed(args.seed)

    # ---- Probe A: in-distribution (stride-aligned training conditions) ----
    blob = torch.load(args.subgoals, map_location="cpu", weights_only=False)
    latents = blob["latents"].float()
    lengths = blob["lengths"].numpy(); offsets = blob["offsets"].numpy()
    mask = blob["in_5deg"].numpy().astype(bool) if args.mask == "5" else np.ones(len(lengths), bool)
    rng = np.random.default_rng(args.seed)
    N = planner.cfg.n_future
    conds_a, tgts_a = [], []
    eps = np.nonzero(mask & (lengths >= 2))[0]
    while len(conds_a) < args.n_probe:
        i = int(rng.choice(eps))
        off, L = int(offsets[i]), int(lengths[i])
        last = L - 1
        m = int(rng.integers(0, L - 1))                   # m in [0, L-2]
        idx = [off + min(m + k * args.subgoal_step, last)  # clamp (== train_gdm)
               for k in range(1, N + 1)]
        conds_a.append(latents[off + m]); tgts_a.append(latents[idx])
    cond_a = torch.stack(conds_a).to(args.device)          # (M, D)
    tgt_a = torch.stack(tgts_a).to(args.device)            # (M, N, D)
    true_a = tgt_a[:, 0]                                    # immediate-next subgoal
    pred_a = planner.sample_next(cond_a, n_steps=args.gdm_steps, generator=gen())
    show("A: IN-DISTRIBUTION (stride-aligned condition)", metrics(pred_a, true_a, cond_a))

    # ---- Probe C: teacher-forced denoising (model+math sanity, no sampling) ----
    # tf-steps default is T=1000-scaled; rescale to this model's actual T (e.g. T=100
    # for the LDP config) so we never index past the schedule arrays.
    T = planner.diffusion.timesteps
    tf_steps = [t for t in args.tf_steps if 0 <= t < T]
    if len(tf_steps) < len(args.tf_steps):
        tf_steps = sorted({max(1, int(f * (T - 1))) for f in (0.05, 0.2, 0.4, 0.6, 0.9)})
        print(f"[diag] tf-steps out of range for T={T}; rescaled to {tf_steps}")
    teacher_forced(planner, cond_a, tgt_a, tf_steps, gen)

    # ---- Probe B: eval-distribution (arbitrary-phase condition from h5) ----
    import h5py
    model = encoder.load_lewm(source=args.source, encoder_id=args.encoder_id,
                              local_dir=args.local_dir, swm_src=args.swm_src, device=args.device)
    episodes_idx, start_steps = sample_short(args.h5, args.n_probe, args.stride, args.seed)
    with h5py.File(args.h5, "r") as f:
        ep_off = f["ep_offset"][:]; ep_len = f["ep_len"][:]; pixels = f["pixels"]
        cond_rows, true_rows, fin_rows = [], [], []
        for ep, st in zip(episodes_idx, start_steps):
            base = int(ep_off[ep]); last = base + int(ep_len[ep]) - 1
            cond_rows.append(base + int(st))
            true_rows.append(min(base + int(st) + args.stride, last))
            fin_rows.append(last)
        allr = np.array(cond_rows + true_rows)
        order = np.argsort(allr, kind="stable")
        lat = encoder.encode_frames(model, pixels[allr[order]], device=args.device)
        out = torch.empty_like(lat); out[order] = lat
        # h5py fancy indexing needs STRICTLY increasing rows; fin_rows has
        # duplicates when the same episode is drawn twice (different starts).
        fin_rows = np.array(fin_rows)
        uniq, inv = np.unique(fin_rows, return_inverse=True)
        finals = f["state"][uniq][inv]
    cond_b = out[:args.n_probe].to(args.device)
    true_b = out[args.n_probe:].to(args.device)
    pred_b = planner.sample_next(cond_b, n_steps=args.gdm_steps, generator=gen())
    show("B: EVAL-DISTRIBUTION (arbitrary-phase E(frame) condition)", metrics(pred_b, true_b, cond_b))

    # ---- Probe B STRATIFIED by the Task-A success filter ----
    # sample_short draws from ALL episodes, but the GDM trained ONLY on successful
    # ones: a FAILED demo's future is exactly what a success-trained planner should
    # NOT predict, so the mixed mean conflates off-manifold phase with episode mix.
    # If B-success ~= Probe A and B-failed << it, the plateau is CONTAMINATION,
    # not a diffusion/phase problem (H8) -- fix the EVAL set, not the model.
    ok = np.array([encoder.eval_state_tol(encoder.canonical_target_for(s), s, 20.0)
                   for s in finals], dtype=bool)
    okt = torch.from_numpy(ok)
    if ok.any():
        show(f"B-success: episode passed the 20deg filter (n={int(ok.sum())}/{len(ok)})",
             metrics(pred_b[okt], true_b[okt], cond_b[okt]))
    if (~ok).any():
        show(f"B-failed: FAILED demos, off the training population (n={int((~ok).sum())}/{len(ok)})",
             metrics(pred_b[~okt], true_b[~okt], cond_b[~okt]))

    print("\n==== verdict ====")
    print("GATE B (SAMPLED Probe B rel_err/cos_move) is the fidelity arbiter for EVERY")
    print("'did it help' call below. Probe C (teacher-forced x0_relerr) is GATE A: a MATH")
    print("wiring check only, read differently per parameterization:")
    if planner.diffusion.parameterization == "v":
        print("  [v-param] Probe C x0_relerr is flat-low across ALL t BY CONSTRUCTION (bounded")
        print("    sqrt(1-acp)<=1 coeff). This says NOTHING about fidelity - do NOT read a flat")
        print("    v x0 curve as 'the denoiser is now well-conditioned'. Judge ONLY on Probe B.")
    else:
        print("  [eps-param] Probe C x0_relerr RISES with t (1/sqrt(acp) amplification, intrinsic);")
        print("    low-t eps_mse~0 but mid/high-t x0 poor => amplification, not necessarily a bug.")
        print("    x0_relerr large even at LOW t => schedule/inversion BUG (fix before training).")
    print("A good + B bad  => phase/sliding-condition issue (rebuild dataset, retrain).")
    print("A ~= B-success >> B-failed => eval-set CONTAMINATION (H8): the probe/eval mix")
    print("    ~34-37% FAILED demos the success-trained GDM rightly mispredicts.")
    print("    Fix the EVAL episode set (--eval-filter success20), not the model.")
    print("A and B bad     => capacity/undertraining (train longer / more data).")
    print("collapse << 1   => model predicts ~mean subgoal regardless of input.")


if __name__ == "__main__":
    main()
