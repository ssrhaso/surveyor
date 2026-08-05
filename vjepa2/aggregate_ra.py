"""Aggregate the overnight rollout-agreement fleet and score it against the
pre-registered bars frozen in run_rollout32_v1.sbatch."""
import glob
import re
from collections import defaultdict

import numpy as np


def load(pattern):
    data = defaultdict(list)      # (budget, arm) -> rows
    for f in sorted(glob.glob(pattern)):
        b = int(re.search(r"_b(\d+)_", f).group(1))
        for ep, m, arm, cos, fin, rmse, secs in np.load(f, allow_pickle=True)["rows"]:
            data[(b, str(arm))].append(
                (str(ep), int(m), float(cos), float(fin), float(rmse), float(secs)))
    return data


def table(data, title):
    print(f"\n==== {title} ====")
    print(f"{'budget':>6} {'arm':>6} | {'disp_cos':>8} {'final_err':>9} {'rmse':>7} {'s/anchor':>8} | n")
    for (b, arm) in sorted(data):
        v = data[(b, arm)]
        print(f"{b:>6} {arm:>6} | {np.mean([x[2] for x in v]):>8.3f} "
              f"{np.mean([x[3] for x in v]):>9.4f} {np.mean([x[4] for x in v]):>7.4f} "
              f"{np.mean([x[5] for x in v]):>8.1f} | {len(v)}")


def keyed(data, b, arm):
    return {(x[0], x[1]): x for x in data.get((b, arm), [])}


def ci(d):
    d = np.array(d, dtype=float)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(len(d))
    return m, m - 1.96 * se, m + 1.96 * se


v1 = load("ra_v1_t32_b*_lat_*.npz")
table(v1, "t=32 v1 drafter")

s25, f400, f25 = keyed(v1, 25, "spec"), keyed(v1, 400, "flat"), keyed(v1, 25, "flat")
common = sorted(set(s25) & set(f400) & set(f25))
m1 = ci([s25[k][2] - f400[k][2] for k in common])
m2 = ci([s25[k][2] - f25[k][2] for k in common])
print(f"\nB1 spec@25 - flat@400: {m1[0]:+.3f} [95% {m1[1]:+.3f},{m1[2]:+.3f}] "
      f"(bar >= -0.02) -> {'PASS' if m1[0] >= -0.02 else 'FAIL'}  (n={len(common)})")
print(f"B2 spec@25 - flat@25 : {m2[0]:+.3f} [95% {m2[1]:+.3f},{m2[2]:+.3f}] "
      f"(bar > 0) -> {'PASS' if m2[0] > 0 else 'FAIL'}")
for b in (25, 100, 400):
    l, fl = keyed(v1, b, "lerp"), keyed(v1, b, "flat")
    com = sorted(set(l) & set(fl))
    d = np.mean([l[k][2] - fl[k][2] for k in com])
    print(f"B3 lerp-flat @b{b}: {d:+.3f} (bar <= 0) -> {'PASS' if d <= 0 else 'FAIL'}")
for b in (25, 100, 400):
    r, fl, sp = keyed(v1, b, "route"), keyed(v1, b, "flat"), keyed(v1, b, "spec")
    com = sorted(set(r) & set(fl) & set(sp))
    d = np.mean([r[k][2] - max(fl[k][2], sp[k][2]) for k in com])
    n_flat = sum(1 for k in com if abs(r[k][2] - fl[k][2]) < 1e-9)
    print(f"B4 route-max(f,s) @b{b}: {d:+.3f} (bar >= -0.02) -> "
          f"{'PASS' if d >= -0.02 else 'FAIL'}  (routed-to-flat {n_flat}/{len(com)})")

v3 = load("ra_v3_t32_b*_lat_*.npz")
table(v3, "t=32 v3 drafter (spec arm only)")
for b in (25, 100, 400):
    a, c = keyed(v1, b, "spec"), keyed(v3, b, "spec")
    com = sorted(set(a) & set(c))
    if com:
        m = ci([c[k][2] - a[k][2] for k in com])
        print(f"v3-v1 paired spec @b{b}: {m[0]:+.3f} [95% {m[1]:+.3f},{m[2]:+.3f}] (n={len(com)})")

t16 = load("ra_v1_t16_b*_task*.npz")
table(t16, "t=16 v1 (crossover slice)")
for b in (25, 400):
    sp, fl = keyed(t16, b, "spec"), keyed(t16, b, "flat")
    com = sorted(set(sp) & set(fl))
    m = ci([sp[k][2] - fl[k][2] for k in com])
    print(f"t16 spec-flat @b{b}: {m[0]:+.3f} [95% {m[1]:+.3f},{m[2]:+.3f}] (n={len(com)})")
