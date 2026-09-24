"""Figure: success rate against goal distance, executor alone versus the full SURVEYOR layer.

Writes, from the same harvest files build_table1.py reads (rb*_cells.csv, looked up through build_table1.load_cells /
cell, so nothing is typed by hand), into the directory given as the first argument (default out/figures):

    fig_sr_distance.tex       one drop-in figure block (pgfplots groupplot, five panels: CEM, GC-IDM, iCEM, MPPI, Gradient)
    fig_sr_distance_data.csv  every plotted coordinate with its source arm

Per panel four series: PushT executor alone, PushT full layer, Reacher executor alone, Reacher
full layer. Colour is arm identity (armNeutral = executor alone, armOurs = full layer), line
style and marker are the environment (PushT solid with circles, Reacher dashed with triangles).
Error bars are the standard error over the evaluation seeds. The caption's numbers (where the
full layer is below the executor alone, the lowest PushT value of the full layer) are computed
from the plotted cells:

    python reproduce/build_fig_sr_distance.py [out_dir]
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_table1 as b  # noqa: E402
import build_table1_latex as bl  # noqa: E402

FIG_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out", "figures")
TEX = os.path.join(FIG_DIR, "fig_sr_distance.tex")
CSV = os.path.join(FIG_DIR, "fig_sr_distance_data.csv")

# (block name in build_table1.ROWS, panel title), in panel order
BLOCKS = [("CEM (LeWM)", "CEM"), ("GC-IDM (official code, s42)", "GC-IDM"), ("iCEM", "iCEM"),
          ("MPPI", "MPPI"), ("Gradient (AdamW)", "Gradient")]
ENVS = ["pusht", "reacher"]
ENV_NAME = {"pusht": "PushT", "reacher": "Reacher"}
ARM_LABEL = {"alone": "executor alone", "full": "SURVEYOR (full)"}
LEGEND = {"alone": "vanilla", "full": "+ \\method{}"}
LEGEND_NAME = "leg:sr-distance"

# series style: colour = arm, line/marker = environment
ARM_STYLE = {"alone": "armNeutral", "full": "armOurs"}
ENV_STYLE = {"pusht": "solid, mark=*", "reacher": "dashed, mark=triangle*, mark options={solid}"}

# Beyond t = 150 (batteries W and W2): every held-out episode that is long enough, so the populations are smaller
# (PushT 197 and 50 episodes; Reacher 128 pairs) and the points are drawn hollow. Search executors only: GC-IDM has no
# checkpoint for those step budgets. A distance is plotted only if all its cells have eight seeds.
LONG_T = {"pusht": [175, 200], "reacher": [175, 195]}
LONG_PREFIX = {"CEM": "", "iCEM": "icem_", "MPPI": "mppi_", "Gradient": "gd_"}

# layout (cm): five equal axis boxes; total width must stay below \linewidth = 13.97 cm
AXIS_W, AXIS_H, HSEP = 2.28, 2.2, 0.28


def block_specs():
    """block name -> (alone spec, full spec) for the five plotted executors"""
    blocks = dict(bl.block_iter(b.ROWS))
    out = {}
    for name, _ in BLOCKS:
        rows = blocks[name]
        alone = [spec for key, spec, full, variant in rows if key.startswith("alone")]
        full = [spec for key, spec, full, variant in rows if full]
        if len(alone) != 1 or len(full) != 1:
            raise SystemExit(f"block {name!r}: expected one alone row and one full row, got {len(alone)}/{len(full)}")
        out[name] = (alone[0], full[0])
    return out


def source_arm(cells, env, spec, t, got):
    """which arm of a (possibly multi-arm) spec produced the cell build_table1.cell returned"""
    arms = spec if isinstance(spec, list) else [spec]
    for a in arms:
        if b.cell(cells, env, a, t) == got:
            return a
    return "|".join(arms)


def long_cell(cells, env, disp, arm, t):
    """battery W / W2 cell beyond t = 150: alone = the better of the two horizons, full = the own-tolerance full row"""
    pre = LONG_PREFIX[disp]
    names = [pre + "flat_rh5", pre + "flat_rh2"] if arm == "alone" else [pre + "router_tauown"]
    got = [(cells.get(("rbW", env, a, t)), a) for a in names]
    got = [(c, a) for c, a in got if c is not None and c[1] >= 8]
    return max(got, key=lambda ca: ca[0][0]) if len(got) == len(names) else None


def long_ts(cells, env):
    return [t for t in LONG_T[env]
            if all(long_cell(cells, env, d, arm, t) for d in LONG_PREFIX for arm in ("alone", "full"))]


def collect(cells):
    """rows of the coordinate table: executor, env, arm, t, sr, se, n_seeds, source_arm"""
    specs = block_specs()
    rows = []
    for name, disp in BLOCKS:
        alone, full = specs[name]
        for env in ENVS:
            for arm, spec in (("alone", alone), ("full", full)):
                for t in b.T[env]:
                    c = b.cell(cells, env, spec.get(env), t)
                    if c is None:
                        raise SystemExit(f"missing cell: {disp} {env} {arm} t={t} (arm {spec.get(env)})")
                    sr, n, _, se = c
                    rows.append({"executor": disp, "env": env, "arm": arm, "t": t, "sr": sr, "se": se,
                                 "n_seeds": n, "source_arm": source_arm(cells, env, spec.get(env), t, c)})
                for t in (long_ts(cells, env) if disp in LONG_PREFIX else []):
                    (sr, n, _, se), a = long_cell(cells, env, disp, arm, t)
                    rows.append({"executor": disp, "env": env, "arm": arm, "t": t, "sr": sr, "se": se,
                                 "n_seeds": n, "source_arm": "rbW:" + a})
    return rows


def lookup(rows):
    return {(r["executor"], r["env"], r["arm"], r["t"]): r for r in rows}


def f1(x):
    return f"{x:.1f}"


def t_list(ts):
    """'$t{=}25$ and $t{=}50$' / '$t{=}25$, $t{=}50$ and $t{=}75$'"""
    items = [f"$t{{=}}{t}$" for t in ts]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def caption(rows):
    """caption prose whose numbers are read off the plotted cells"""
    d = lookup(rows)
    execs = [disp for _, disp in BLOCKS]
    for e in execs:      # "falls with distance": first minus last PushT value of the executor alone
        if d[(e, "pusht", "alone", 25)]["sr"] <= d[(e, "pusht", "alone", 150)]["sr"]:
            raise SystemExit(f"caption claim broken: {e} alone does not fall with distance on PushT; reword the caption")
    fmin, fmin_e, fmin_t = min((d[(e, "pusht", "full", t)]["sr"], e, t) for e in execs for t in b.T["pusht"])
    floor = int(fmin)      # "stays above floor%": read off the plotted cells (89 at the own tolerance: iCEM, t=50, 89.7)
    if floor < 85:
        raise SystemExit(f"caption claim broken: full layer on PushT reaches {fmin} ({fmin_e}, t={fmin_t}); reword the caption")
    # where the full layer is below the executor alone, grouped by executor and environment
    below = {}
    for e in execs:
        for env in ENVS:
            ts_e = sorted(t for (ee, en, arm, t) in d if ee == e and en == env and arm == "full")
            diffs = [(t, d[(e, env, "alone", t)]["sr"] - d[(e, env, "full", t)]["sr"]) for t in ts_e]
            # a deficit counts when it exceeds two pooled standard errors, the criterion of the tables' amber shading
            def pooled(t):
                return ((d[(e, env, "alone", t)]["se"] or 0.0) ** 2 + (d[(e, env, "full", t)]["se"] or 0.0) ** 2) ** 0.5
            diffs = [(t, x) for t, x in diffs if x > 0 and x > 2 * pooled(t)]
            if diffs:
                below.setdefault(e, {})[env] = diffs
    if not below:
        below_txt = "it is never below the vanilla executor."
    else:
        parts = []
        for e, envs in below.items():
            sub = []
            for env, diffs in envs.items():
                n_t = len([1 for (ee, en, arm, t) in d if ee == e and en == env and arm == "full"])
                where = "at every distance" if len(diffs) == n_t else "at " + t_list([t for t, _ in diffs])
                mags = [x for _, x in diffs]
                by = f"by {f1(mags[0])} points" if len(mags) == 1 else (
                    f"by at most {f1(max(mags))} points" if max(mags) < 2.0 else f"by {f1(min(mags))} to {f1(max(mags))} points")
                sub.append(f"{by} on {ENV_NAME[env]} {where}")
            parts.append(f"over {e}, " + (sub[0] if len(sub) == 1 else " and ".join(sub)))
        only = "only " if len(below) == 1 else ""
        below_txt = f"beyond two pooled standard errors it falls below the vanilla executor {only}" + "; ".join(parts) + "."
    long_pts = sorted({(r["env"], r["t"]) for r in rows if r["t"] > 150})
    long_txt = ""
    if long_pts:
        # kept to one short sentence: the conclusion must still end on page 9; the details are in the appendix
        long_txt = " Hollow markers, beyond $t{=}150$, use held-out populations, smaller on PushT (\\Cref{app:beyond})."
    # the legend, axis labels and titles carry what is plotted (success against goal distance per
    # executor, executor alone versus the full layer, PushT and Reacher), so the caption states facts
    return (
        "\\textbf{Success against goal distance.} "
        "Each point is the mean over eight evaluation seeds with its standard error. "
        "On PushT the vanilla executor falls with distance whereas the full layer stays above $" + str(floor) + "\\%$ "
        "throughout; " + below_txt + long_txt)


def coords(rows, executor, env, arm, ts=None):
    d = lookup(rows)
    out = []
    for t in (ts or b.T[env]):
        r = d[(executor, env, arm, t)]
        se = r["se"] if r["se"] is not None else 0.0
        note = ""
        if r["se"] is None:
            note += " %% TODO(data): no SE for this cell"
        if r["n_seeds"] < 8:
            note += f" %% [{r['n_seeds']} seeds]"
        out.append(f"    ({t},{r['sr']:.2f}) +- (0,{se:.2f}){note}")
    return "\n".join(out)


def figure(rows):
    n = len(BLOCKS)
    mid = n // 2
    tmax = max(r["t"] for r in rows)
    xaxis = ("  xmin=25, xmax=150, enlarge x limits={abs=9},", "  xtick={25,50,75,100,150}, xticklabels={25,50,{},100,150},")
    if tmax > 150:   # 25 to 200 in the same box: label 25, 100 and 200 only, or neighbouring labels would touch
        xaxis = ("  xmin=25, xmax=200, enlarge x limits={abs=12},",
                 "  xtick={25,50,75,100,150,200}, xticklabels={25,{},{},100,{},200},")
    L = ["% generated by reproduce/build_fig_sr_distance.py from reproduce/rb*_cells.csv through",
         "% build_table1.load_cells / cell; do not edit by hand. Every coordinate with its source arm is in",
         "% figures/fig_sr_distance_data.csv. Needs pgfplots (compat=1.16), the groupplots library and the",
         "% preamble colours armNeutral / armOurs and the \\method{} macro.",
         "\\begin{figure}[t]",
         "\\centering",
         "\\begin{tikzpicture}",
         "\\begin{groupplot}[",
         f"  group style={{group size={n} by 1, horizontal sep={HSEP}cm,",
         "               ylabels at=edge left, yticklabels at=edge left},",
         f"  scale only axis, width={AXIS_W}cm, height={AXIS_H}cm,",
         xaxis[0],
         "  ymin=0, ymax=100, enlarge y limits={upper, abs=3},",
         xaxis[1],
         "  ytick={0,25,50,75,100},",
         "  ylabel={success (\\%)},",
         "  axis line style={black!40},",
         "  ymajorgrids, grid style={black!10},",
         "  tick label style={font=\\footnotesize},",
         "  label style={font=\\small},",
         "  title style={font=\\small, yshift=-2pt},",
         "  every axis plot/.append style={thick, mark size=1.8pt},",
         "  error bars/error bar style={line width=0.4pt},",
         "  error bars/error mark options={rotate=90, mark size=0.9pt, line width=0.4pt},",
         # the four entry texts alone are 352 pt wide at \footnotesize against a 397 pt line, so one
         # row cannot hold them: two rows, PushT above Reacher, executor alone left of the full layer
         "  legend columns=2,",
         "  legend style={draw=black!25, font=\\footnotesize, inner ysep=1.5pt, nodes={inner ysep=1pt},",
         "                /tikz/every even column/.append style={column sep=0.8em}},",
         "  legend image code/.code={\\draw[#1, mark repeat=2, mark phase=2]",
         "                            plot coordinates {(0cm,0cm) (0.2cm,0cm) (0.4cm,0cm)};},",
         "]"]
    for i, (name, disp) in enumerate(BLOCKS):
        opts = [f"title={{{disp}}}"]
        if i == mid:
            opts += ["xlabel={goal distance $t$}", f"legend to name={LEGEND_NAME}"]
        L.append(f"\\nextgroupplot[{', '.join(opts)}]")
        for env in ENVS:
            for arm in ("alone", "full"):
                L.append(f"% {disp}, {ENV_NAME[env]}, {ARM_LABEL[arm]}")
                L.append(f"\\addplot[{ARM_STYLE[arm]}, {ENV_STYLE[env]}, error bars/.cd, y dir=both, y explicit]")
                L.append("  coordinates {")
                L.append(coords(rows, disp, env, arm))
                L.append("  };")
                if i == mid:
                    L.append(f"\\addlegendentry{{{ENV_NAME[env]}, {LEGEND[arm]}}}")
                ts_long = sorted(r["t"] for r in rows if r["executor"] == disp and r["env"] == env
                                 and r["arm"] == arm and r["t"] > 150)
                if ts_long:   # continuation beyond t = 150: same line, hollow markers, no legend entry
                    hollow = ENV_STYLE[env] + ", mark options={fill=white}"
                    if "mark options" in ENV_STYLE[env]:
                        hollow = ENV_STYLE[env].replace("mark options={solid}", "mark options={solid, fill=white}")
                    L.append(f"\\addplot[{ARM_STYLE[arm]}, {hollow}, forget plot, error bars/.cd, y dir=both, y explicit]")
                    L.append("  coordinates {")
                    L.append(coords(rows, disp, env, arm, [150] + ts_long))
                    L.append("  };")
    L += ["\\end{groupplot}",
          "\\end{tikzpicture}",
          "\\par\\vspace{2pt}",
          f"\\ref{{{LEGEND_NAME}}}",
          f"\\caption{{{caption(rows)}}}",
          "\\label{fig:sr-distance}",
          "\\end{figure}"]
    return "\n".join(L) + "\n"


def main():
    cells = b.load_cells()
    rows = collect(cells)
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["executor", "env", "arm_label", "t", "sr", "se", "n_seeds", "source_arm"])
        for r in rows:
            w.writerow([r["executor"], r["env"], ARM_LABEL[r["arm"]], r["t"], f"{r['sr']:.2f}",
                        "" if r["se"] is None else f"{r['se']:.2f}", r["n_seeds"], r["source_arm"]])
    with open(TEX, "w", encoding="utf-8", newline="\n") as f:
        f.write(figure(rows))
    short = [r for r in rows if r["n_seeds"] < 8]
    print(f"wrote {os.path.abspath(TEX)} and {os.path.abspath(CSV)}: {len(BLOCKS)} panels, "
          f"{len(BLOCKS) * len(ENVS) * 2} series, {len(rows)} points"
          + (f"; {len(short)} cells with fewer than 8 seeds" if short else "; every cell has 8 seeds"))


if __name__ == "__main__":
    main()
