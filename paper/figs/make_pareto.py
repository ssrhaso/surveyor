"""SR-vs-drafter-compute Pareto figure (two panels, one fixed population each).

All points are ISCA-certified cells at the matched t=100 anchor, n=512/cell
(4 seeds x 128/256): PushT on the unfiltered episodes150as100 population
(harder-population regime; internally consistent), Reacher on the
max-offset-100 fixed population. x = drafter NFE per replan (call_ratio x k
for spec-accept; k for every-step), log2 scale. y = SR (20deg / joint
criterion). The Pareto frontier is drawn over each panel's points.

Usage:  python make_pareto.py  -> pareto.pdf / pareto.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, NFE/replan, SR, kind)  kind: es=every-step, sa=spec-accept
PUSHT = [
    ("es k=1", 1.0, 55.9, "es"), ("es k=2", 2.0, 55.0, "es"),
    ("es k=4", 4.0, 63.3, "es"), ("es k=8", 8.0, 64.1, "es"),
    ("es k=50", 50.0, 64.6, "es"),
    ("sa k=1", 1.00 * 1, 54.7, "sa"), ("sa k=2", 1.00 * 2, 55.6, "sa"),
    ("sa k=4", 0.575 * 4, 65.0, "sa"), ("sa k=8", 0.561 * 8, 63.3, "sa"),
    ("sa t.1 k=4", 0.873 * 4, 64.1, "sa"), ("sa t.3 k=4", 0.472 * 4, 65.0, "sa"),
]
REACHER = [
    ("es k=1", 1.0, 95.1, "es"), ("es k=2", 2.0, 95.3, "es"),
    ("es k=4", 4.0, 96.1, "es"), ("es k=8", 8.0, 94.9, "es"),
    ("es k=50", 50.0, 90.8, "es"),
    ("sa k=1", 1.00 * 1, 95.9, "sa"), ("sa k=2", 1.00 * 2, 95.9, "sa"),
    ("sa t.2 k=4", 0.627 * 4, 95.5, "sa"), ("sa t.1 k=4", 0.837 * 4, 95.9, "sa"),
    ("sa t.3 k=4", 0.517 * 4, 94.1, "sa"), ("sa t.4 k=4", 0.462 * 4, 90.2, "sa"),
    ("sa t.2 k=8", 0.609 * 8, 93.4, "sa"), ("sa t.2 k=16", 0.637 * 16, 92.6, "sa"),
]


def pareto(points):
    """Upper-left frontier: max SR at min NFE."""
    pts = sorted((x, y) for _, x, y, _ in points)
    front, best = [], -1
    for x, y in pts:
        if y > best:
            front.append((x, y))
            best = y
    # frontier as staircase sorted by x
    return front


fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharex=True)
for ax, data, title in ((axes[0], PUSHT, "PushT (t=100, fixed population)"),
                        (axes[1], REACHER, "Reacher (t=100, fixed population)")):
    es = [(x, y) for _, x, y, k in data if k == "es"]
    sa = [(x, y) for _, x, y, k in data if k == "sa"]
    ax.scatter(*zip(*es), marker="s", s=42, facecolor="none",
               edgecolor="#444444", linewidth=1.4, label="every-step", zorder=3)
    ax.scatter(*zip(*sa), marker="o", s=42, color="#b03030",
               label=r"spec-accept ($\tau$, $k$ varied)", zorder=4)
    fr = pareto(data)
    ax.plot(*zip(*sorted(fr)), color="#b03030", alpha=0.35, linewidth=2, zorder=2)
    # annotate the frozen config and the legacy reference
    for lbl, x, y, k in data:
        if lbl in ("sa t.2 k=4", "sa k=4", "es k=50", "es k=4"):
            ax.annotate(lbl.replace("sa ", "spec ").replace("es ", "every-step "),
                        (x, y), textcoords="offset points", xytext=(4, -11),
                        fontsize=7, color="#333333")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("drafter NFE per replan (log scale)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.5)
axes[0].set_ylabel(r"success rate (\%)" if plt.rcParams.get("text.usetex")
                   else "success rate (%)")
axes[0].legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig("pareto.pdf")
fig.savefig("pareto.png", dpi=200)
print("wrote pareto.pdf / pareto.png")
