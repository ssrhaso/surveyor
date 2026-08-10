"""Foundation-scale calibration row (certification prereg M1-FM).

Dataset-pair operating characteristic of the transplant verifier's accept
statistic (mean-pooled V-JEPA 2 tokens, rel L2, tau=0.20) against RECORDED
DROID proprio ground truth. No closed loop, no probe error: both frames of
every pair are real frames with recorded states. Definitions and predictions
frozen in docs/certification_prereg.md ("M1-FM") before any number was
computed; the row must be labeled dataset-pair wherever quoted (the LeWM M1
rows replay deployed serving decisions; this scores the same statistic over
recorded-frame pairs).

  python -m surveyor.vjepa2.calibrate_droid_pairs \
      --lat-dir /lustre/home/ha676/data/droid2k_lat \
      --out-json Results/calibration/cal_vjepa2_droid.json \
      --out-csv Results/calibration/events_vjepa2_droid.csv
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def pooled(path):
    """(T,256,D) fp16 npy/npz -> (T,D) float32 token-mean."""
    if path.endswith(".npz"):
        tok = np.load(path)["tokens"]
    else:
        tok = np.load(path, mmap_mode="r")
    return np.asarray(tok, dtype=np.float32).mean(axis=1)


def fit_ridge(X, Y, lam=1.0):
    Xb = np.concatenate([X, np.ones((X.shape[0], 1), X.dtype)], axis=1)
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1], dtype=X.dtype)
    W = np.linalg.solve(A, Xb.T @ Y)
    return lambda Z: np.concatenate(
        [Z, np.ones((Z.shape[0], 1), Z.dtype)], axis=1) @ W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--n-eps", type=int, default=400)
    ap.add_argument("--pairs-per-ep", type=int, default=50)
    ap.add_argument("--n-cross", type=int, default=5000)
    ap.add_argument("--tau", type=float, default=0.20)
    ap.add_argument("--hmax", type=int, default=32)
    ap.add_argument("--probe-frames", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    lat_files = sorted(f for f in glob.glob(str(Path(args.lat_dir) / "lat_*.np*"))
                       if not f.endswith(".states.npy"))
    assert lat_files, f"no lat_*.npy under {args.lat_dir}"
    pick = rng.choice(len(lat_files), size=min(args.n_eps, len(lat_files)),
                      replace=False)
    print(f"[fm] {len(lat_files)} episodes available, sampling {len(pick)}",
          flush=True)

    pools, states = [], []
    for k, idx in enumerate(pick):
        f = lat_files[idx]
        pools.append(pooled(f))
        states.append(np.load(f)["states"] if f.endswith(".npz")
                      else np.load(f.replace(".npy", ".states.npy")))
        if (k + 1) % 50 == 0:
            print(f"[fm] pooled {k + 1}/{len(pick)}", flush=True)

    # ---- within-episode pairs: (i, i+Delta), Delta ~ U{1..hmax} -----------
    rels, dists, ees = [], [], []
    for p, s in zip(pools, states):
        T = p.shape[0]
        for _ in range(args.pairs_per_ep):
            d = int(rng.integers(1, args.hmax + 1))
            if T - 1 - d < 0:
                d = T - 1
            i = int(rng.integers(0, max(T - d, 1)))
            j = min(i + d, T - 1)
            rel = float(np.linalg.norm(p[i] - p[j])
                        / max(np.linalg.norm(p[i]), 1e-8))
            rels.append(rel)
            dists.append(float(np.linalg.norm(s[i, :3] - s[j, :3])))
            ees.append((i, j))
    rels = np.array(rels)
    dists = np.array(dists)
    accs = rels <= args.tau

    # ---- cross-episode pairs: accept rate only ----------------------------
    cross_rels = []
    for _ in range(args.n_cross):
        a, b = rng.choice(len(pools), size=2, replace=False)
        i = int(rng.integers(0, pools[a].shape[0]))
        j = int(rng.integers(0, pools[b].shape[0]))
        cross_rels.append(float(np.linalg.norm(pools[a][i] - pools[b][j])
                                / max(np.linalg.norm(pools[a][i]), 1e-8)))
    cross_rels = np.array(cross_rels)

    # ---- decodability row: ridge pooled -> state, R^2 on EE dims ----------
    allp = np.concatenate(pools)
    alls = np.concatenate(states)
    sub = rng.choice(len(allp), size=min(args.probe_frames, len(allp)),
                     replace=False)
    X, Y = allp[sub].astype(np.float64), alls[sub].astype(np.float64)
    perm = rng.permutation(len(X))
    cut = int(0.8 * len(X))
    tr, te = perm[:cut], perm[cut:]
    pred = fit_ridge(X[tr], Y[tr])(X[te])
    ss_res = ((Y[te][:, :3] - pred[:, :3]) ** 2).sum()
    ss_tot = ((Y[te][:, :3] - Y[te][:, :3].mean(axis=0)) ** 2).sum()
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))

    # ---- frozen readouts ---------------------------------------------------
    from scipy.stats import spearmanr
    rho = float(spearmanr(rels, dists).statistic)
    grid = [0.02, 0.04, 0.08, 0.16]

    def q(x):
        return {} if len(x) == 0 else {
            "p10": float(np.percentile(x, 10)), "p50": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90))}

    out = {
        "row": "vjepa2-droid dataset-pair (M1-FM; NOT a deployed-decision replay)",
        "tau": args.tau, "n_within": int(len(rels)), "n_cross": int(len(cross_rels)),
        "n_episodes": int(len(pick)), "hmax": args.hmax,
        "accept_rate_within": float(accs.mean()),
        "accept_rate_cross": float((cross_rels <= args.tau).mean()),
        "spearman_rel_vs_ee_dist": rho,
        "probe_r2_ee_dims": r2, "probe_gate": 0.90,
        "fa_at_r": {str(r): float((dists[accs] > r).mean()) if accs.any() else None
                    for r in grid},
        "fr_at_r": {str(r): float((dists[~accs] < r).mean()) if (~accs).any() else None
                    for r in grid},
        "ee_dist_accept": q(dists[accs]), "ee_dist_reject": q(dists[~accs]),
        "rel_within": q(rels), "rel_cross": q(cross_rels),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    with open(args.out_csv, "w") as f:
        f.write("rel,accepted,ee_dist\n")
        for r_, a_, d_ in zip(rels, accs, dists):
            f.write(f"{r_:.6f},{int(a_)},{d_:.6f}\n")
    print(json.dumps(out, indent=2), flush=True)
    print(f"[fm] saved {args.out_json} + {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
