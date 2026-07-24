import csv, re, statistics as st
from collections import defaultdict

rows = list(csv.DictReader(open("isca_logs/parsed.csv")))
# descriptive files only (contain _seed<NN>.log); jobid files are duplicates
groups = defaultdict(list)
for r in rows:
    m = re.match(r"^(.+)_seed(\d+)\.log$", r["file"])
    if not m:
        continue
    groups[m.group(1)].append(r)

def fmean(vals):
    vals = [float(v) for v in vals if v not in (None, "", "None")]
    if not vals: return None, None, 0
    return (st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0), len(vals))

print(f"{'arm':44s} {'n':>2s} {'SR20':>12s} {'SR5':>12s} {'SRq':>12s} {'cr':>6s} {'nev':>5s} {'filt':>8s}")
for k in sorted(groups):
    g = groups[k]
    out = [f"{k:44s} {len(g):2d}"]
    for field in ("sr20", "sr5", "srq"):
        m, s, n = fmean([r[field] for r in g])
        out.append(f"{m:6.2f}+-{s:4.2f}" if m is not None else " " * 12)
    m, s, n = fmean([r["call_ratio"] for r in g])
    out.append(f"{m:6.3f}" if m is not None else "      ")
    out.append(f"{g[0]['n'] or '':>5s}")
    out.append(f"{(g[0]['eval_filter'] or ''):>8s}")
    print(" ".join(out))
