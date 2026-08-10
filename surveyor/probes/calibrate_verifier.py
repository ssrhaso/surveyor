"""Verifier calibration against ground-truth state (certification prereg M1).

Self-contained per env:
  1. sample dataset frames plus their recorded ground-truth state from the h5
     and encode them through the SAME path the verifier reads (native LeWM
     latents, or the pooled DINOv2 half for tworoom's paired verifier);
  2. fit a ridge probe latent -> state and report held-out R^2 against the
     frozen 0.90 quality gate on the criterion-relevant dims, declaring the
     env's row unavailable below the gate rather than fudging it;
  3. load the calibration event lists dumped by the trace-collection runs, one
     (env, rel, accepted, z_achieved, target) per VERIFICATION event, and read
     the probe-state distance between the achieved state and the pursued
     waypoint's decoded state;
  4. report the frozen readouts: accept-vs-reject separation, Spearman
     rho(rel, state distance), and false-accept / false-reject rates at the
     env's own success-criterion radius.

Definitions are frozen in docs/certification_prereg.md and may not change after
the first number is seen. Caveat recorded there: probing a DRAFTED target
assumes near-manifold drafts (LeWM drafters pass the norm and faithfulness
gates), so probe outputs on drafts are sanity-checked against state bounds.
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import torch

from surveyor import encoder

# per-env config: state column, criterion-relevant dims of that column, the
# success-criterion radius in state units, and which latent space the verifier
# reads. dims=None -> all dims. Criterion distance = L2 over `dims` except
# reacher, whose qpos_match criterion is max-abs per joint.
ENV = {
    "pusht": dict(state_col="state", dims=(2, 3), radius=20.0, metric="l2",
                  space="native", encoder_id="quentinll/lewm-pusht"),
    "reacher": dict(state_col="qpos", dims=None, radius=0.05, metric="maxabs",
                    space="native", encoder_id="quentinll/lewm-reacher"),
    "cube": dict(state_col="privileged_block_0_pos", dims=None, radius=0.04,
                 metric="l2", space="native", encoder_id="quentinll/lewm-cube"),
    "tworoom": dict(state_col="state", dims=(0, 1), radius=16.0, metric="l2",
                    space="dino", encoder_id="quentinll/lewm-tworooms"),
}


def parse_args():
    p = argparse.ArgumentParser(description="verifier calibration vs ground truth")
    p.add_argument("--env", required=True, choices=sorted(ENV))
    p.add_argument("--h5", required=True)
    p.add_argument("--traces", required=True,
                   help="glob of trace .pt files carrying the 'cal' event list")
    p.add_argument("--tau", type=float, required=True,
                   help="the arm's accept threshold (replay check only)")
    p.add_argument("--out", required=True, help="output JSON")
    p.add_argument("--source", choices=["pretrained", "local"], default="pretrained")
    p.add_argument("--encoder-id", default=None, help="override env default")
    p.add_argument("--local-dir", default=None)
    p.add_argument("--swm-src", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-probe", type=int, default=4000,
                   help="dataset frames sampled for probe training")
    p.add_argument("--probe", choices=["ridge", "mlp"], default="ridge",
                   help="probe class; mlp = sensitivity row (certification "
                        "prereg M1 amendment): same frozen definitions, "
                        "reported alongside the ridge row")
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--circular", action="store_true",
                   help="M1 reacher amendment: probe targets [sin,cos] per "
                        "angle dim, decode via atan2, criterion distance = "
                        "wrapped angular difference (registered before "
                        "computing; see certification prereg)")
    p.add_argument("--r2-gate", type=float, default=0.90,
                   help="frozen probe-quality gate on criterion dims")
    p.add_argument("--dump-events", default=None,
                   help="08-01 extraction addendum: write per-event "
                        "(rel, accepted, state_dist) CSV for the radius-"
                        "sensitivity and operating-curve readouts "
                        "(descriptive; tau and all decisions unchanged)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def crit_dist(sa, sb, dims, metric, circular=False):
    a = sa if dims is None else sa[..., list(dims)]
    b = sb if dims is None else sb[..., list(dims)]
    if circular:
        d = np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))   # wrapped diff
    else:
        d = np.abs(a - b)
    return d.max(axis=-1) if metric == "maxabs" else np.linalg.norm(a - b, axis=-1)


def fit_ridge(X, Y, lam):
    """closed-form ridge with bias; returns predict(X)->Y."""
    Xb = np.concatenate([X, np.ones((X.shape[0], 1), X.dtype)], axis=1)
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1], dtype=X.dtype)
    W = np.linalg.solve(A, Xb.T @ Y)
    return lambda Z: np.concatenate(
        [Z, np.ones((Z.shape[0], 1), Z.dtype)], axis=1) @ W


def main():
    args = parse_args()
    cfg = ENV[args.env]
    rng = np.random.default_rng(args.seed)

    # ---- 1. probe food: sampled dataset frames + ground-truth state --------
    try:
        import hdf5plugin  # noqa: F401  (registers blosc/zstd filters for pixels)
    except ImportError:
        pass
    import h5py
    with h5py.File(args.h5, "r") as f:
        if cfg["state_col"] not in f:
            raise SystemExit(f"[cal] state col '{cfg['state_col']}' not in h5; "
                             f"available: {list(f.keys())}")
        n_rows = f["pixels"].shape[0]
        rows = np.sort(rng.choice(n_rows, size=min(args.n_probe, n_rows),
                                  replace=False))
        frames = f["pixels"][rows]
        state = f[cfg["state_col"]][rows].astype(np.float64)
    print(f"[cal] {args.env}: {len(rows)} probe frames, state dim {state.shape[1:]}",
          flush=True)

    # ---- 2. encode through the verifier's own space ------------------------
    if cfg["space"] == "native":
        model = encoder.load_lewm(source=args.source,
                                  encoder_id=args.encoder_id or cfg["encoder_id"],
                                  local_dir=args.local_dir, swm_src=args.swm_src,
                                  device=args.device)
        lat = encoder.encode_frames(model, frames, device=args.device)
    else:  # tworoom: the DINOv2 pooled half the paired verifier reads
        from surveyor import paired as sp
        dino = None
        for name in ("load_dinov2", "load_dino"):
            if hasattr(sp, name):
                dino = getattr(sp, name)(device=args.device)
                break
        if dino is None:
            dino = torch.hub.load("facebookresearch/dinov2",
                                  "dinov2_vits14").to(args.device).eval()
        lat = sp.encode_frames_dino(dino, frames, device=args.device)
    X = lat.numpy().astype(np.float64)

    # ---- 3. ridge probe + frozen quality gate ------------------------------
    # circular amendment: regress [sin, cos] per angle dim; decode via atan2
    d_state = state.shape[1]
    if args.circular:
        target = np.concatenate([np.sin(state), np.cos(state)], axis=1)
    else:
        target = state

    n = X.shape[0]
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]
    if args.probe == "mlp":
        from sklearn.neural_network import MLPRegressor
        mlp = MLPRegressor(hidden_layer_sizes=(256,), max_iter=500,
                           random_state=args.seed)
        mlp.fit(X[tr], target[tr])
        predict_t = mlp.predict
    else:
        predict_t = fit_ridge(X[tr], target[tr], args.ridge)

    def predict(Z):
        p = predict_t(Z)
        if args.circular:
            return np.arctan2(p[:, :d_state], p[:, d_state:])
        return p

    pred_te = predict_t(X[te])
    if args.circular:
        # gate applies to the transformed targets, all dims (prereg amendment)
        dims_r2 = tuple(range(target.shape[1]))
    else:
        dims_r2 = cfg["dims"] if cfg["dims"] is not None else tuple(range(state.shape[1]))
    ss_res = ((target[te][:, dims_r2] - pred_te[:, dims_r2]) ** 2).sum(axis=0)
    ss_tot = ((target[te][:, dims_r2] - target[te][:, dims_r2].mean(axis=0)) ** 2).sum(axis=0)
    r2 = float(1.0 - (ss_res.sum() / max(ss_tot.sum(), 1e-12)))
    print(f"[cal] probe R^2 (criterion dims) = {r2:.4f} (gate {args.r2_gate})",
          flush=True)
    gated_out = r2 < args.r2_gate

    # ---- 4. calibration events ---------------------------------------------
    events = []   # (rel, accepted, z_now, tgt)
    files = sorted(glob.glob(args.traces))
    if not files:
        raise SystemExit(f"[cal] no trace files match {args.traces}")
    for fn in files:
        rec = torch.load(fn, map_location="cpu", weights_only=False)
        recs = ([rec] if "cal" in rec
                else list(rec.get("records", {}).values()))   # pusht wraps per score/angle
        got = 0
        for r in recs:
            for (i, rel, acc, z_now, tgt) in r.get("cal", []):
                events.append((float(rel), bool(acc),
                               z_now.numpy().astype(np.float64),
                               tgt.numpy().astype(np.float64)))
                got += 1
        print(f"[cal] {fn}: {got} verification events", flush=True)
    if not events:
        raise SystemExit("[cal] traces carry no 'cal' events (legacy schema?)")

    rels = np.array([e[0] for e in events])
    accs = np.array([e[1] for e in events])
    Zn = np.stack([e[2] for e in events])
    Zt = np.stack([e[3] for e in events])
    Sn, St = predict(Zn), predict(Zt)
    # drafted-target sanity: decoded states within the dataset's state bounds
    lo, hi = state.min(axis=0), state.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    frac_oob = float(((St < lo - 0.1 * span) | (St > hi + 0.1 * span)).any(axis=1).mean())
    d = crit_dist(Sn, St, cfg["dims"], cfg["metric"], circular=args.circular)

    if args.dump_events:
        with open(args.dump_events, "w") as fo:
            fo.write("rel,accepted,state_dist\n")
            for rel_i, acc_i, d_i in zip(rels, accs, d):
                fo.write(f"{rel_i:.6f},{int(acc_i)},{d_i:.6f}\n")
        print(f"[cal] dumped {len(rels)} events -> {args.dump_events}", flush=True)

    # replay check: recorded decisions must equal rel <= tau
    mism = int((accs != (rels <= args.tau)).sum())

    # ---- 5. frozen readouts -------------------------------------------------
    from scipy.stats import spearmanr
    rho = float(spearmanr(rels, d).statistic)
    R = float(cfg["radius"])
    fa = float((d[accs] > R).mean()) if accs.any() else float("nan")
    fr = float((d[~accs] < R).mean()) if (~accs).any() else float("nan")

    def q(x):
        return {} if len(x) == 0 else {
            "p10": float(np.percentile(x, 10)), "p50": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90))}

    out = {
        "env": args.env, "tau": args.tau, "probe": args.probe,
        "circular": bool(args.circular),
        "n_events": int(len(events)),
        "n_accept": int(accs.sum()), "n_reject": int((~accs).sum()),
        "probe_r2_criterion_dims": r2, "probe_gate": args.r2_gate,
        "row_unavailable_probe_below_gate": bool(gated_out),
        "replay_mismatches": mism,
        "drafted_target_frac_out_of_bounds": frac_oob,
        "criterion_radius": R, "metric": cfg["metric"],
        "state_dist_accept": q(d[accs]), "state_dist_reject": q(d[~accs]),
        "spearman_rel_vs_state_dist": rho,
        "false_accept_rate": fa, "false_reject_rate": fr,
        "traces": files,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)
    print(f"[cal] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
