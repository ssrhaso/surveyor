"""CEM cost-signal SNR probe (stride-spacing round): is the cost landscape a
fine-stride CEM sees informative, or flat relative to noise?

This is the MECHANISTIC form of the degeneracy risk (DIRECTION_stride_spacing.md
1): at stride 5 the subgoal sits 5 env-steps out, and the worry is that CEM's
cost -- terminal L2^2 of the frozen-predictor rollout to the subgoal -- barely
depends on the chosen action block, so CEM ranks noise. Measured IN THE MODEL
CEM actually optimizes (frozen LeWM predictor rollout, exactly the
SubgoalCostModel math), with REAL demo action blocks as candidates (drawn from
other windows/episodes), which sidesteps action-space scale guesswork and asks
the operative question: can the cost function tell apart realistic candidate
plans at this horizon?

Per window t (end-anchored runway, matching the anchor convention) and per
stride S in {5, 25}: subgoal = TRUE E(t+S); candidates = the window's own true
action block + K decoy blocks from random other success episodes (the SAME
decoy windows serve both strides: S=5 uses the first 5 actions, S=25 all 25)
+ an all-zeros "hold" block. CEM loop geometry matches Stage 3 by construction
(horizon = S/5 predictor steps, action_block=5, H=1 history).

Metrics (mean over windows):
  true_pct   : fraction of decoys costlier than the TRUE continuation block --
               the cost function's ability to rank the right plan highly.
               ~0.5 = chance = degenerate; high = informative.
  cv         : std/mean of decoy costs (cost-landscape dynamic range).
  best_gap   : (median - min)/median of decoy costs (how much better the best
               candidate looks than a typical one -- what CEM's elite picks up).
  act_sens   : normalized per-dim std of the predicted TERMINAL latent across
               candidates (do different plans even reach different predicted
               states at this horizon? compare to the Stage-0 S=1 encoder
               floor ~0.074 and S=5 realized displacement ~0.35).
  hold_pct   : fraction of decoys costlier than doing nothing (zeros block).

Read-off: if S=5 true_pct ~ 0.5 and cv/act_sens collapse vs S=25, the fine-
stride cost signal is degenerate IN-MODEL -> mechanism for a Stage 3 loss. If
S=5 ranks the true block as well as S=25 does, the degeneracy concern is dead
in-model and Stage 3 measures control, not cost noise.

CPU-friendly (predictor is small); runs on the laptop against the local full h5:
    python -m ffjepa.probes.probe_cost_snr \
        --h5 ../drift_probe/expert/pusht_expert_train.h5 \
        --source local --local-dir ../drift_probe/model \
        --swm-src ../lewm-investigation/stable-worldmodel \
        --device cpu --n-windows 64 --n-decoys 128
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ffjepa import lewm_io

ACTION_BLOCK = 5
STRIDES = (5, 25)
RUNWAY = 150  # end-anchored windows: start = ep_len-1-RUNWAY (anchor convention)


def parse_args():
    p = argparse.ArgumentParser(description="CEM cost-signal SNR at fine vs native stride")
    p.add_argument("--h5", required=True)
    p.add_argument("--source", choices=["pretrained", "local"], default="local")
    p.add_argument("--encoder-id", default="quentinll/lewm-pusht")
    p.add_argument("--local-dir", default=None)
    p.add_argument("--swm-src", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-windows", type=int, default=64)
    p.add_argument("--n-decoys", type=int, default=128)
    p.add_argument("--batch-windows", type=int, default=8)
    p.add_argument("--angle-headline", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json-out", default=None)
    return p.parse_args()


@torch.no_grad()
def rollout_terminal(model, z0, blocks, device):
    """Terminal predicted latent for each (window, candidate) pair.

    z0: (B, D) initial latents; blocks: (B, S, T, 10) action-block sequences.
    Returns (B, S, D). Mirrors SubgoalCostModel.get_cost's use of LeWM.rollout
    (H=1 history frame, emb pre-injected so no pixel re-encode)."""
    B, S, T, A = blocks.shape
    info = {
        "pixels": torch.zeros(B, 1, 1, 1, 1, 1, device=device),  # only .size(2)==H is read
        "emb": z0[:, None, None, :].expand(B, S, 1, z0.shape[1]).contiguous(),
    }
    info = model.rollout(info, blocks.to(device))
    return info["predicted_emb"][..., -1, :]  # (B, S, D)


@torch.no_grad()
def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    import h5py

    model = lewm_io.load_lewm(source=args.source, encoder_id=args.encoder_id,
                              local_dir=args.local_dir, swm_src=args.swm_src,
                              device=args.device)

    with h5py.File(args.h5, "r") as f:
        ep_off = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
        state = f["state"]
        pixels = f["pixels"]
        actions = f["action"]

        # success-filtered episodes long enough for an end-anchored 150 runway
        scan = rng.permutation(len(ep_off))
        kept = []
        for e in scan:
            off, L = int(ep_off[e]), int(ep_len[e])
            if L < RUNWAY + 2:
                continue
            final = state[off + L - 1].astype(np.float64)
            target = lewm_io.canonical_target_for(final)
            if lewm_io.eval_state_tol(target, final, args.angle_headline):
                kept.append((int(e), off, L))
            if len(kept) >= args.n_windows + 64:  # extra eps for the decoy pool
                break
        assert len(kept) > args.n_windows, "not enough long successful episodes"
        windows = kept[:args.n_windows]
        decoy_pool = kept  # decoys may come from any kept episode (incl. others' windows)
        print(f"[filter] {len(kept)} successful@{args.angle_headline:g}deg episodes "
              f"with runway>{RUNWAY}; {len(windows)} probe windows, decoy pool {len(decoy_pool)}")

        # encode E(t), E(t+5), E(t+25) per window (t end-anchored)
        starts = {e: off + (L - 1 - RUNWAY) for e, off, L in windows}
        rows = []
        for e, off, L in windows:
            t = starts[e]
            rows += [t, t + 5, t + 25]
        rows = np.array(rows)
        order = np.argsort(rows, kind="stable")
        lat = lewm_io.encode_frames(model, pixels[rows[order]], device=args.device)
        out = torch.empty_like(lat)
        out[order] = lat
        z_cond = out[0::3].to(args.device)
        z_sg = {5: out[1::3].to(args.device), 25: out[2::3].to(args.device)}

        # action blocks: true continuation per window + shared-decoy convention
        # (each decoy is a 25-step window; S=5 uses its first block, S=25 all 5)
        def read_blocks(row0, n_steps):
            a = actions[row0:row0 + n_steps].astype(np.float32)      # (n_steps, 2)
            return a.reshape(n_steps // ACTION_BLOCK, ACTION_BLOCK * 2)  # (T, 10)

        true_blocks, decoy_blocks = [], []
        for e, off, L in windows:
            true_blocks.append(read_blocks(starts[e], 25))
            dec = []
            while len(dec) < args.n_decoys:
                de, doff, dL = decoy_pool[rng.integers(len(decoy_pool))]
                dt = int(rng.integers(0, dL - 25))
                if de == e and abs(doff + dt - starts[e]) < 25:
                    continue  # don't sample the true window as its own decoy
                dec.append(read_blocks(doff + dt, 25))
            decoy_blocks.append(np.stack(dec))
        true_blocks = torch.from_numpy(np.stack(true_blocks))          # (B, 5, 10)
        decoy_blocks = torch.from_numpy(np.stack(decoy_blocks))       # (B, K, 5, 10)

    B, K = len(windows), args.n_decoys
    hold = torch.zeros(B, 1, 5, 10)
    # candidate axis: [true, hold, decoys...]
    cands25 = torch.cat([true_blocks[:, None], hold, decoy_blocks], dim=1)  # (B, K+2, 5, 10)

    results = {}
    for S in STRIDES:
        T = S // ACTION_BLOCK
        cands = cands25[:, :, :T]                                     # first T blocks
        costs, terms = [], []
        for i in range(0, B, args.batch_windows):
            sl = slice(i, i + args.batch_windows)
            zT = rollout_terminal(model, z_cond[sl], cands[sl], args.device)  # (b, K+2, D)
            c = ((zT - z_sg[S][sl].unsqueeze(1)) ** 2).sum(-1)               # (b, K+2)
            costs.append(c); terms.append(zT)
        cost = torch.cat(costs)   # (B, K+2)
        zT = torch.cat(terms)     # (B, K+2, D)

        c_true, c_hold, c_dec = cost[:, 0], cost[:, 1], cost[:, 2:]
        true_pct = (c_dec > c_true.unsqueeze(1)).float().mean(1)      # (B,)
        hold_pct = (c_dec > c_hold.unsqueeze(1)).float().mean(1)
        cv = c_dec.std(1) / c_dec.mean(1)
        med = c_dec.median(1).values
        best_gap = (med - c_dec.min(1).values) / med
        act_sens = zT[:, 2:].std(dim=1).mean(1) / (zT[:, 2:].norm(dim=2).mean(1)
                                                   / np.sqrt(zT.shape[2]))

        r = {k: (float(v.mean()), float(v.std())) for k, v in
             dict(true_pct=true_pct, cv=cv, best_gap=best_gap,
                  act_sens=act_sens, hold_pct=hold_pct).items()}
        results[S] = r
        print(f"\n[S={S}] horizon={T} block(s), n={B} windows, {K} decoys "
              f"(mean +/- sd over windows)")
        for k, (m, s) in r.items():
            print(f"  {k:>9} = {m:.4f} +/- {s:.4f}")

    print("\n[read-off] degenerate fine stride would show: true_pct(5) ~ 0.5 with "
          "cv/act_sens collapsed vs S=25. true_pct(5) ~ true_pct(25) with healthy "
          "cv => the 5-step cost signal is informative in-model; Stage 3 then "
          "measures control granularity, not cost noise.")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"args": vars(args), "n_windows": B,
                       "per_stride": {str(k): v for k, v in results.items()}}, fh, indent=2)
        print(f"[save] {args.json_out}")


if __name__ == "__main__":
    main()
