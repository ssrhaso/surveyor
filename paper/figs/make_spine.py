"""The spine figure (two panels): success vs goal distance, flat family vs drafted
arms, on the fixed-population horizon sweeps of tab:spine.

Left: PushT (clean collapse). Right: Reacher (the crossover). Reacher uses two
fixed populations by construction (max-offset-100 for t<=100, max-offset-150 for
t>=125); we plot pop-A solid and the pop-B extension dashed, with both t=100
values marked; the honest way to show the full arc without splicing.

Usage:  python make_spine.py  -> spine.pdf / spine.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = [25, 50, 75, 100, 150]
# --- PushT (success-filtered fixed population) ---
PUSHT = {
    "flat (RH5)":        ([25, 50, 75, 100, 150], [98.1, 60.4, 42.6, 12.7, 2.7]),
    "FF-JEPA ($S{=}25$)":([25, 50, 75, 100, 150], [99.0, 94.1, 95.9, 95.7, 93.9]),
    "ours every-step":   ([25, 50, 75, 100, 150], [99.6, 99.0, 98.1, 98.1, 98.4]),
    "ours spec-accept":  ([25, 50, 75, 100, 150], [99.6, 97.9, 98.8, 97.3, 98.1]),
}
# --- Reacher: pop-A (t<=100) solid, pop-B (t>=100) dashed extension ---
REACH_A = {  # max-offset-100
    "flat (RH5)":      ([25, 50, 75, 100], [83.2, 91.8, 89.1, 83.2]),
    "ours every-step": ([25, 50, 75, 100], [59.8, 79.7, 90.2, 90.8]),
}
REACH_B = {  # max-offset-150 (extension)
    "flat (RH5)":      ([100, 125, 150], [83.2, 81.4, 76.0]),
    "ours every-step": ([100, 125, 150], [94.5, 93.2, 95.1]),
}
STYLE = {
    "flat (RH5)":         dict(color="#888888", marker="s", lw=1.8),
    "FF-JEPA ($S{=}25$)": dict(color="#5a7fb0", marker="^", lw=1.6),
    "ours every-step":    dict(color="#b03030", marker="o", lw=2.0),
    "ours spec-accept":   dict(color="#b03030", marker="o", lw=2.0, ls="--", mfc="none"),
}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.5), sharey=True)

for name, (xs, ys) in PUSHT.items():
    axL.plot(xs, ys, label=name, **STYLE[name])
axL.set_title("PushT (fixed population)", fontsize=10)
axL.annotate("flat collapses\n$98.1\\to2.7$", (150, 2.7), textcoords="offset points",
             xytext=(-6, 18), fontsize=7.5, color="#666666", ha="right")

for name, (xs, ys) in REACH_A.items():
    axR.plot(xs, ys, label=name, **STYLE[name])
for name, (xs, ys) in REACH_B.items():
    st = dict(STYLE[name]); st["ls"] = ":"; st.pop("label", None)
    axR.plot(xs, ys, **st)
axR.set_title("Reacher (crossover; dotted $=$ extended pop.)", fontsize=10)
axR.axvline(100, color="#cccccc", lw=0.8, zorder=0)
axR.annotate("crossover", (75, 89), textcoords="offset points", xytext=(-2, 8),
             fontsize=7.5, color="#666666")

for ax in (axL, axR):
    ax.set_xlabel("goal distance $t$ (env steps)")
    ax.set_ylim(0, 103)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7.5, loc="lower left")
axL.set_ylabel("success rate (\\%)" if plt.rcParams.get("text.usetex")
               else "success rate (%)")
fig.tight_layout()
fig.savefig("spine.pdf")
fig.savefig("spine.png", dpi=200)
print("wrote spine.pdf / spine.png")
