"""Train GC-IDM (arXiv 2605.08732) on cached dense LeWM latents.

Consumes the stride-1 latent caches already built for this project
(subgoals_dense_full.pt / subgoals_reacher_dense.pt / subgoals_cube_dense.pt),
joined to the raw actions in the episode h5. The caches store one latent per
frame with per-episode offsets/lengths and an episode_idx back-reference, so
latent row (offsets[e] + t) corresponds to h5 row (ep_offset[episode_idx[e]] + t).

Training tuples follow the paper exactly: sample (t, h) with h ~ U[1, H_max] and
form (z_t, z_{t+h}, a_t), regressing the action under MSE. Actions are
standardised with the same StandardScaler the rest of the stack uses, and the
scaler is saved alongside the weights so serving lands in raw action units.

  python -m specaccept.train_gcidm --latents subgoals_dense_full.pt \
      --h5 /path/pusht_expert_train.h5 --out gcidm_pusht.pt --h-max 50
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from specaccept.gcidm import GCIDM, count_params, save_gcidm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latents", required=True, help="stride-1 dense latent cache (.pt)")
    p.add_argument("--h5", required=True, help="episode h5 holding the actions")
    p.add_argument("--out", required=True)
    p.add_argument("--h-max", type=int, default=50,
                   help="paper sets H_max = evaluation budget T (their protocol: 50)")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=50,
                   help="paper Appendix F: 50 epochs, batch 1024. Steps are derived "
                        "from the training-frame count so 'epoch' means one pass.")
    p.add_argument("--steps", type=int, default=None,
                   help="override the epoch-derived step count (debug only)")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="held-out episodes, so the reported loss is not a training loss")
    p.add_argument("--episode-min", type=int, default=None,
                   help="restrict to episodes >= this index (held-out floor, e.g. cube 8000)")
    p.add_argument("--episode-max", type=int, default=None)
    p.add_argument("--exclude-episodes", nargs="*", default=None,
                   help="eval population JSONs whose episodes must NOT be trained on. "
                        "Reacher/Cube get a clean holdout from --episode-min alone, but "
                        "the PushT populations span the whole index range, so without "
                        "this the baseline would be trained on its own test episodes and "
                        "any win it posted would be meaningless.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=2000)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    d = torch.load(args.latents, map_location="cpu", weights_only=False)
    lat = d["latents"]                                   # (rows, 192) stride-1
    lengths = d["lengths"].numpy().astype(np.int64)
    offsets = d["offsets"].numpy().astype(np.int64)
    ep_idx = d["episode_idx"].numpy().astype(np.int64)
    assert int(d["stride"]) == 1, f"need a stride-1 dense cache, got stride={d['stride']}"
    latent_dim = int(d["latent_dim"])

    import h5py
    with h5py.File(args.h5, "r") as f:
        actions = f["action"][:].astype(np.float32)      # (rows, adim)
        h5_off = f["ep_offset"][:].astype(np.int64)
        h5_len = f["ep_len"][:].astype(np.int64)
    action_dim = actions.shape[1]

    # every cached episode must have exactly as many latents as the h5 has frames
    bad = [i for i in range(len(ep_idx)) if lengths[i] != h5_len[ep_idx[i]]]
    if bad:
        raise ValueError(f"{len(bad)} episodes disagree between cache and h5 "
                         f"(first: cache ep {bad[0]} len {lengths[bad[0]]} vs h5 "
                         f"{h5_len[ep_idx[bad[0]]]})")

    keep = np.ones(len(ep_idx), dtype=bool)
    if args.episode_min is not None:
        keep &= ep_idx >= args.episode_min
    if args.episode_max is not None:
        keep &= ep_idx < args.episode_max
    n_excluded = 0
    if args.exclude_episodes:
        banned = set()
        for path in args.exclude_episodes:
            with open(path) as fh:
                banned.update(int(pair[0]) for pair in json.load(fh)["episodes"])
        drop = np.isin(ep_idx, np.fromiter(banned, dtype=np.int64, count=len(banned)))
        n_excluded = int((keep & drop).sum())
        keep &= ~drop
        print(f"[holdout] {len(banned)} eval episodes listed across "
              f"{len(args.exclude_episodes)} file(s); {n_excluded} removed from training")
    # need at least one usable (t, t+h) pair
    keep &= lengths > 1
    usable = np.nonzero(keep)[0]
    perm = rng.permutation(len(usable))
    n_val = max(1, int(round(args.val_frac * len(usable))))
    val_eps, train_eps = usable[perm[:n_val]], usable[perm[n_val:]]
    print(f"[data] {len(usable)} usable episodes -> {len(train_eps)} train / {len(val_eps)} val; "
          f"latents {tuple(lat.shape)}, action_dim={action_dim}")

    # fit the action scaler on the TRAIN split only, so val cannot leak in
    from sklearn import preprocessing
    tr_rows = np.concatenate([
        np.arange(h5_off[ep_idx[e]], h5_off[ep_idx[e]] + lengths[e]) for e in train_eps])
    scaler = preprocessing.StandardScaler()
    fit_src = actions[tr_rows]
    scaler.fit(fit_src[~np.isnan(fit_src).any(axis=1)])
    act_n = torch.from_numpy(
        ((actions - scaler.mean_) / scaler.scale_).astype(np.float32))

    dev = args.device
    lat = lat.to(dev)
    act_n = act_n.to(dev)

    model = GCIDM(latent_dim=latent_dim, action_dim=action_dim, hidden=args.hidden,
                  depth=args.depth, dropout=args.dropout, h_max=args.h_max).to(dev)
    print(f"[model] GC-IDM {count_params(model)/1e6:.2f}M params "
          f"(paper: ~1.5M), H_max={args.h_max}, hidden={args.hidden}, depth={args.depth}")
    n_train_frames = int(sum(lengths[e] for e in train_eps))
    total_steps = (args.steps if args.steps is not None
                   else max(1, args.epochs * n_train_frames // args.batch_size))
    print(f"[sched] {args.epochs} epochs over {n_train_frames} train frames "
          f"at batch {args.batch_size} -> {total_steps} steps")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # paper: cosine annealing to lr/100
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps,
                                                       eta_min=args.lr / 100.0)

    lat_off = torch.from_numpy(offsets).to(dev)
    h5_base = torch.from_numpy(h5_off[ep_idx]).to(dev)
    ep_lens = torch.from_numpy(lengths).to(dev)

    def sample_batch(pool, gen):
        """(z_t, z_{t+h}, h_norm, a_t) with h ~ U[1, H_max], t uniform in range."""
        e = pool[torch.randint(len(pool), (args.batch_size,), generator=gen, device=dev)]
        L = ep_lens[e]
        # h is capped by what the episode can supply, then t by what h leaves
        h_cap = torch.clamp(L - 1, max=args.h_max)
        h = (torch.rand(args.batch_size, generator=gen, device=dev) * h_cap).long() + 1
        h = torch.minimum(h, h_cap)
        t_max = L - 1 - h
        t = (torch.rand(args.batch_size, generator=gen, device=dev)
             * (t_max + 1).float()).long()
        t = torch.minimum(t, t_max)
        z_t = lat[lat_off[e] + t]
        z_g = lat[lat_off[e] + t + h]
        a_t = act_n[h5_base[e] + t]
        h_norm = torch.clamp(h.float(), max=float(args.h_max)) / float(args.h_max)
        return z_t, z_g, h_norm, a_t

    gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
    vgen = torch.Generator(device=dev); vgen.manual_seed(args.seed + 1)
    train_pool = torch.from_numpy(train_eps).to(dev)
    val_pool = torch.from_numpy(val_eps).to(dev)

    t0 = time.time()
    hist = []
    for step in range(1, total_steps + 1):
        model.train()
        z_t, z_g, h_norm, a_t = sample_batch(train_pool, gen)
        loss = torch.nn.functional.mse_loss(model(z_t, z_g, h_norm), a_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.log_every == 0 or step == total_steps:
            model.eval()
            with torch.no_grad():
                # held-out episodes, averaged over a few batches
                vl = float(np.mean([
                    torch.nn.functional.mse_loss(
                        model(zb, gb, hb), ab).item()
                    for zb, gb, hb, ab in (sample_batch(val_pool, vgen)
                                           for _ in range(8))]))
            hist.append({"step": step, "train_mse": loss.item(), "val_mse": vl})
            print(f"[{step:6d}/{total_steps}] train {loss.item():.5f}  val {vl:.5f}  "
                  f"({time.time()-t0:.0f}s)")

    save_gcidm(args.out, model, action_scaler=scaler, meta={
        "paper": "arXiv 2605.08732 (GC-IDM)",
        "latents": args.latents, "h5": args.h5,
        "h_max": args.h_max, "steps": total_steps, "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "weight_decay": args.weight_decay, "seed": args.seed,
        "episode_min": args.episode_min, "episode_max": args.episode_max,
        "exclude_episodes": args.exclude_episodes,
        "n_eval_episodes_excluded": n_excluded,
        "n_train_eps": int(len(train_eps)), "n_val_eps": int(len(val_eps)),
        "params": count_params(model), "history": hist,
        "train_seconds": time.time() - t0,
    })
    print(f"[done] {time.time()-t0:.0f}s -> {args.out}")
    print(json.dumps(hist[-1] if hist else {}, indent=2))


if __name__ == "__main__":
    main()
