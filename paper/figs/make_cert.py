"""Certification figures (sec:results-cert), two multi-panel files.

cert_decision: what the accept decision buys, measured three ways.
  [ignition | SR(p) PushT | SR(p) Reacher] -- the corruption sweep's
  call-ratio ignition at the derived tau, then the pre-registered
  rejection-rate dose-response with the verifier's operating point starred.
cert_cal: what the decisions mean physically.
  [FA(r) radius-sensitivity, all four envs | Reacher operating scatter
  (rel vs true wrapped angular distance, tau and criterion marked)].

Data: banked verdicts (docs/randreject_prereg.md, docs/certification_prereg.md)
and the per-event CSVs in Results/calibration/events_*.csv.
Usage:  python make_cert.py  -> cert_decision.pdf/png, cert_cal.pdf/png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAL = ROOT / "Results" / "calibration"

BLUE, VERM, TEAL, PURP = "#0072B2", "#D55E00", "#009E73", "#8C5AA0"
GRAY = "#888888"

# ---------------- figure 1: the decision, three ways ----------------
fig, (axI, axP, axR) = plt.subplots(1, 3, figsize=(9.8, 2.75))

# ignition (R1 verdict): verified call_ratio vs sigma; blind at its 1/N floor
SIG = [0.0, 0.1, 0.2, 0.4]
axI.axvspan(0.10, 0.40, color="#f2e8dc", zorder=0)             # [tau/2, 2tau]
axI.axvline(0.20, color=GRAY, lw=0.9, ls="--", zorder=1)
axI.plot(SIG, [0.59, 0.66, 1.00, 1.00], color=BLUE, marker="o", lw=2.0,
         label="verified ($\\tau{=}0.20$)")
axI.plot(SIG, [0.34, 0.34, 0.34, 0.34], color=GRAY, marker="s", lw=1.8,
         ls=":", label="blind ($\\tau{=}\\infty$)")
axI.annotate("derived $\\tau$", (0.20, 0.44), fontsize=7.5, color="#666666",
             ha="center")
axI.set_xlabel("draft corruption $\\sigma$ (rel. norm)")
axI.set_ylabel("call ratio (re-drafts / boundaries)")
axI.set_ylim(0.25, 1.05)
axI.set_title("rejection ignites at the derived $\\tau$", fontsize=10)
axI.legend(fontsize=7.5, loc="upper left", frameon=False)

# SR(p) dose-response (randreject prereg verdicts); verifier starred
def sr_panel(ax, coin_x, coin_y, star_xy, blind=None, title=""):
    ax.plot(coin_x, coin_y, color=VERM, marker="o", lw=2.0, label="random reject at rate $p$")
    ax.plot(*star_xy, marker="*", ms=15, color=BLUE, ls="none",
            label="verifier (measured rate)", zorder=5)
    if blind is not None:
        ax.plot(*blind, marker="s", ms=6, mfc="none", color=GRAY, ls="none",
                label="blind commit-3 (banked $-18$pp)")
    ax.set_xlabel("per-event rejection probability $p$")
    ax.set_xlim(-0.04, 0.62)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title(title, fontsize=10)

sr_panel(axP, [0.125, 0.25, 0.4937], [96.97, 96.39, 98.05], (0.4937, 98.14),
         blind=([0.0], [80.1]), title="PushT $t{=}150$: step at zero")
axP.set_ylabel("success rate (%)")
axP.set_ylim(76, 101)
axP.legend(fontsize=7, loc="lower right", frameon=False)

sr_panel(axR, [0.0, 0.125, 0.25, 0.4902], [84.57, 85.74, 90.43, 90.33],
         (0.4902, 91.11), title="Reacher $t{=}150$: graded, knee ${\\sim}0.25$")
axR.set_ylim(76, 101)

fig.tight_layout()
fig.savefig("cert_decision.pdf")
fig.savefig("cert_decision.png", dpi=200)
plt.close(fig)

# ---------------- figure 2: calibration ----------------
ENVS = [  # (name, csv, criterion radius, color, marker)
    ("Cube", "events_cube.csv", 0.04, TEAL, "^"),
    ("PushT", "events_pusht.csv", 20.0, BLUE, "o"),
    ("Reacher", "events_reacher.csv", 0.05, VERM, "s"),
    ("Two-Room", "events_tworoom.csv", 16.0, PURP, "D"),
]

def load(csvname):
    rows = list(csv.DictReader(open(CAL / csvname)))
    rel = np.array([float(r["rel"]) for r in rows])
    acc = np.array([r["accepted"] == "1" for r in rows])
    d = np.array([float(r["state_dist"]) for r in rows])
    return rel, acc, d

fig, (axF, axS) = plt.subplots(1, 2, figsize=(9.4, 3.2))

mult = np.linspace(0.5, 3.0, 26)
for name, f, R, c, m in ENVS:
    rel, acc, d = load(f)
    fa = [(d[acc] > k * R).mean() for k in mult]
    axF.plot(mult, fa, color=c, marker=m, ms=3.5, lw=1.8, markevery=5)
    k = {"Cube": 0.62, "PushT": 0.85, "Reacher": 1.5, "Two-Room": 2.2}[name]
    fk = float((d[acc] > k * R).mean())
    axF.annotate(name, (k, fk), textcoords="offset points",
                 xytext=(5, 4) if name in ("Reacher", "PushT") else (4, 5),
                 fontsize=7.5, color=c)
axF.axvline(1.0, color=GRAY, lw=0.9, ls="--")
axF.annotate("task's own\nsuccess radius", (1.0, 0.78), fontsize=7.5,
             color="#666666", ha="left", xytext=(1.06, 0.78))
axF.set_xlabel("readout radius $r$ (multiples of the criterion radius)")
axF.set_ylabel("false-accept rate at $\\tau$")
axF.set_ylim(-0.03, 1.0)
axF.grid(alpha=0.25, lw=0.5)
axF.set_title("false accepts are near-boundary", fontsize=10)

rel, acc, d = load("events_reacher.csv")
n = len(rel)
sub = np.random.default_rng(0).choice(n, size=min(1500, n), replace=False)
for mask, c, lab in [(acc[sub], BLUE, "accepted"), (~acc[sub], VERM, "rejected")]:
    axS.scatter(rel[sub][mask], d[sub][mask], s=6, alpha=0.35, color=c,
                label=lab, edgecolors="none")
axS.axvline(0.20, color=GRAY, lw=0.9, ls="--")
axS.axhline(0.05, color=GRAY, lw=0.9, ls=":")
axS.annotate("derived $\\tau$", (0.215, 1.0), fontsize=7.5, color="#666666")
axS.annotate("criterion radius", (0.62, 0.055), fontsize=7.5, color="#666666")
axS.set_xlim(0, 0.8)
axS.set_ylim(0, 1.3)
axS.set_xlabel("verifier statistic (rel. latent distance)")
axS.set_ylabel("true angular distance (rad)")
axS.set_title("Reacher: the accept test tracks truth ($\\rho{=}0.978$)",
              fontsize=10)
axS.legend(fontsize=7.5, loc="upper left", frameon=False, markerscale=2.2)
axS.grid(alpha=0.25, lw=0.5)

fig.tight_layout()
fig.savefig("cert_cal.pdf")
fig.savefig("cert_cal.png", dpi=200)
plt.close(fig)
print("wrote cert_decision.{pdf,png} cert_cal.{pdf,png}")
