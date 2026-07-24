"""The cost figure: measured planning time, success, and time-per-completed-task,
all three environments, 4 arms x 5 seeds (CUDA-synced, instrumented policy;
flat measured via the lerp frac=1.0 tautology, whose SR reproduces the flat
reference on every environment; the built-in equivalence check).

Cells: PushT t=100 (s5as100, n=256, budget 200), the all-drafting regime and
the WORST case for certified overhead; Reacher t=100 (pop-999, n=128, budget
200); Cube certified protocol (offset 150, budget 300, 600-sample CEM, n=128).
Seeds 42-46 (three Cube cells lost to node timeouts; those means carry 3-4
seeds, SEs shown). Per-episode time includes early termination on success:
the deployment-honest accounting, which is why low-SR arms (whose episodes
grind to budget exhaustion) are expensive per episode AND per success.

Usage:  python make_cost.py  -> cost.pdf / cost.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# env -> arm -> (time_s_per_ep, time_se, SR, sr_se)   [aggregate_timing.py, 2026-07-16]
DATA = {
    "PushT ($t{=}100$)": {
        "LeWM flat":          (4.338, 0.021, 12.7, 0.5),
        "FF-JEPA ($k{=}50$)": (2.387, 0.012, 95.2, 0.5),
        "spec-accept":        (1.889, 0.023, 98.4, 0.5),
        "certified (ours)":   (2.942, 0.025, 97.7, 0.3),
    },
    "Reacher ($t{=}100$)": {
        "LeWM flat":          (2.112, 0.021, 79.5, 0.5),
        "FF-JEPA ($k{=}50$)": (2.514, 0.013, 88.3, 0.7),
        "spec-accept":        (2.198, 0.033, 89.7, 1.1),
        "certified (ours)":   (1.616, 0.024, 91.7, 0.8),
    },
    "OGB-Cube (certified)": {
        "LeWM flat":          (4.758, 0.191, 68.8, 1.8),
        "FF-JEPA ($k{=}50$)": (8.219, 0.135, 29.2, 1.4),
        "spec-accept":        (2.980, 0.094, 79.7, 0.8),
        "certified (ours)":   (3.734, 0.294, 82.3, 1.8),
    },
}
COLOR = {"LeWM flat": "#888888", "FF-JEPA ($k{=}50$)": "#5a7fb0",
         "spec-accept": "#b03030", "certified (ours)": "#b03030"}
HATCH = {"spec-accept": "//"}

envs = list(DATA)
arms = list(DATA[envs[0]])[::-1]  # bottom-to-top: flat first at top after reverse
fig, axes = plt.subplots(len(envs), 3, figsize=(10.6, 6.2))

for r, env in enumerate(envs):
    d = DATA[env]
    times = [d[a][0] for a in arms]
    terr = [d[a][1] for a in arms]
    srs = [d[a][2] for a in arms]
    serr = [d[a][3] for a in arms]
    tps = [t / (s / 100.0) for t, s in zip(times, srs)]
    cols = [COLOR[a] for a in arms]
    for c, (vals, errs, xlabel, fmt) in enumerate([
            (times, terr, "planning time / episode (s)", "{:.2f}"),
            (srs, serr, "success rate (%)", "{:.1f}"),
            (tps, None, "time / completed task (s)", "{:.1f}")]):
        ax = axes[r][c]
        bars = ax.barh(arms, vals, xerr=errs, color=cols, height=0.62,
                       edgecolor="white", linewidth=0.6,
                       error_kw=dict(lw=0.9, capsize=2, ecolor="#333333"))
        for b, a in zip(bars, arms):
            if a in HATCH:
                b.set_hatch(HATCH[a])
        for b, v in zip(bars, vals):
            inside = v > 0.60 * max(vals)
            ax.annotate(fmt.format(v), (v, b.get_y() + b.get_height() / 2),
                        xytext=(-22 if inside else 14, 0),
                        textcoords="offset points", va="center",
                        ha="right" if inside else "left", fontsize=8,
                        fontweight="bold",
                        color="white" if inside else "#333333")
        ax.set_xlim(0, max(vals) * 1.30)
        ax.grid(axis="x", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=7.5)
        if c > 0:
            ax.set_yticklabels([])
        if r == len(envs) - 1:
            ax.set_xlabel(xlabel, fontsize=8.5)
        if c == 0:
            ax.set_ylabel(env, fontsize=9)

axes[0][0].set_title("the cost (measured)", fontsize=9.5)
axes[0][1].set_title("what it buys", fontsize=9.5)
axes[0][2].set_title("cost per completed task", fontsize=9.5)
fig.tight_layout()
fig.savefig("cost.pdf")
fig.savefig("cost.png", dpi=200)
print("wrote cost.pdf / cost.png")
