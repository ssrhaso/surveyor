"""TwoRoom crossover figure (poster/paper panel).

Flat vs certified spec-accept over goal distance t, at protocols anchored to
LeWM's Fig. 6 (t=25 = their published protocol verbatim; t>=40 = the anchored
long protocol). Best arm per (method, t); error bars = SE over seeds
(12 seeds everywhere except the declared t=60 tie, which stays at 6; t=40/50
extension seeds 48-53 from job 2306764, precision-only, band unchanged). The
shaded band marks the
measured crossover, which landed inside the frozen prediction (goal exits the
25-step planning window plus one serving stride).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"     # categorical slots 1-2 (validated)
INK, MUTED = "#333333", "#8a8a8a"

def m_se(vals):
    v = np.asarray(vals, float)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

T = [25, 40, 50, 60, 75]
flat = {
    25: [89.06, 78.12, 76.56, 89.06, 87.50, 89.06, 73.44, 78.12, 84.38, 84.38, 90.62, 84.38],
    40: [67.19, 71.88, 65.62, 59.38, 62.50, 65.62,
         57.81, 67.19, 70.31, 57.81, 68.75, 67.19],
    50: [65.62, 57.81, 54.69, 62.50, 60.94, 65.62,
         62.50, 56.25, 59.38, 60.94, 48.44, 64.06],
    60: [56.25, 57.81, 45.31, 50.00, 57.81, 65.62],
    75: [37.50, 42.19, 50.00, 29.69, 53.12, 43.75, 42.19, 39.06, 34.38, 45.31, 40.62, 42.19],
}
spec = {
    25: [78.12, 73.44, 75.00, 78.12, 82.81, 75.00],
    40: [65.62, 64.06, 53.12, 59.38, 50.00, 70.31,
         60.94, 71.88, 67.19, 57.81, 56.25, 62.50],
    50: [67.19, 67.19, 59.38, 60.94, 71.88, 60.94,
         53.12, 71.88, 60.94, 73.44, 57.81, 54.69],
    60: [56.25, 50.00, 50.00, 51.56, 60.94, 65.62],
    75: [57.81, 54.69, 65.62, 43.75, 59.38, 51.56, 59.38, 59.38, 54.69, 62.50, 60.94, 54.69],
}

fm, fs = zip(*[m_se(flat[t]) for t in T])
sm, ss = zip(*[m_se(spec[t]) for t in T])

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.axvspan(40, 50, color="#e8e8e8", zorder=0)
ax.text(45, 92, "measured\ncrossover", ha="center", va="top", fontsize=8.5,
        color=MUTED)
ax.axvline(25, color=MUTED, lw=0.8, ls=":")
ax.text(25.6, 30.5, "one planning window\n(LeWM's own protocol)", fontsize=8,
        color=MUTED, va="bottom")

ax.errorbar(T, fm, yerr=fs, color=BLUE, lw=2, marker="o", ms=5,
            capsize=2.5, zorder=3)
ax.errorbar(T, sm, yerr=ss, color=ORANGE, lw=2, marker="o", ms=5,
            capsize=2.5, zorder=3)
ax.annotate("flat planning", (T[-1], fm[-1]), xytext=(76.5, 39.5),
            color=BLUE, fontsize=10, fontweight="bold")
ax.annotate("certified\nspec-accept", (T[-1], sm[-1]), xytext=(76.5, 55.5),
            color=ORANGE, fontsize=10, fontweight="bold")
ax.annotate("+15.4pp\n(12 seeds,\nevery seed)", (75, 49.3), xytext=(66.5, 45.5),
            fontsize=8.5, color=INK)
ax.annotate("LeWM's published 87", (25, 87), xytext=(27, 86.2), fontsize=8,
            color=MUTED)
ax.plot([25], [87], marker="*", ms=11, color=MUTED, ls="none")

ax.set_xlim(22, 84)
ax.set_ylim(25, 95)
ax.set_xticks(T)
ax.set_xlabel("goal distance t (env steps)", fontsize=10)
ax.set_ylabel("success rate (%)", fontsize=10)
ax.set_title("TwoRoom on LeWM: the crossover the window rule predicted",
             fontsize=11, color=INK)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(MUTED)
ax.tick_params(colors=INK, labelsize=9)
ax.grid(axis="y", color="#ececec", lw=0.7, zorder=0)

fig.tight_layout()
out = r"C:\Users\hasaa\Desktop\LEWM\le-wm\paper\figures\fig_tworoom_crossover.png"
fig.savefig(out, dpi=220)
print("saved", out)
