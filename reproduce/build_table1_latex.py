"""Emit the success and cost tables as LaTeX from the same harvest files build_table1.py reads.

Writes, into the directory given as the first argument (default out/tables):

    table1_main.tex        Table 1: per executor X, the rows "X" and "X + SURVEYOR": mean success per environment,
                           seconds per episode at each environment's longest distance, call ratio at PushT t=150
    table1_long.tex        Table 2: the same pairs, mean success over the short (t <= 50) and the long (t >= 75) distances
    table1_ablation.tex    appendix: the components added one at a time (every-step, blind chain, accept rule, full)
    table1_range.tex       appendix: the shortest and the longest distance of each environment, every configuration
    table1_full_<env>.tex  appendix: every cell with its SE over evaluation seeds, one table per environment
    table1_cost_pr.tex, table1_cost_ct.tex   appendix: SR, call ratio and seconds per episode at the longest distance
    table1_leflow.tex      appendix: LeFlow as the executor
    table1_gcidm_ours.tex  appendix: our GC-IDM reimplementation (replication check)
    table1_main_aggregates.tex, table1_ladder.tex, table1_env.tex, table1_cost.tex: other layouts, not in the paper

Every number comes from rb*_cells.csv and rb*_sacct.txt through build_table1; nothing is typed by hand. Requires
booktabs, multirow and colortbl with the colours winbg, losebg and tiebg in the preamble:

    python reproduce/build_table1_latex.py [out_dir]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_table1 as b  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(b.HERE, "out", "tables")
ENV_NAME = {"pusht": "PushT", "reacher": "Reacher", "cube": "Cube", "tworoom": "Two-Room"}

# main-text ladder: substrate alone, every-step, blind chain, accept rule, full. The S=25,
# blind commit-2 and arbiter-variant rows stay in the appendix tables.
MAIN_LABELS = {
    "alone": "vanilla",
    "drafter only, every-step": "+ drafter (every-step)",
    "drafter only, every-step, matched S=10": "+ drafter (every-step)",
    "drafter + blind chain": "+ drafter, blind chain",
    "drafter + accept rule": "+ drafter, accept rule",
    "full: drafter + accept rule + arbiter": "\\method{} (full)",
}


def tex(s):
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


# appendix row labels: short spellings so the per-environment tables fit \textwidth
SHORT = {
    "alone, strongest flat": "vanilla",
    "alone": "vanilla",
    "alone (flat RH5)": "vanilla",
    "drafter only, every-step, matched S=10": "drafter, every-step (S=10)",
    "drafter only, every-step, S=25": "drafter, every-step (S=25)",
    "drafter only, every-step": "drafter, every-step",
    "drafter + blind commit-2": "blind chain (2 waypoints)",
    "drafter + blind chain": "blind chain",
    "drafter + accept rule": "accept rule",
    "full: drafter + accept rule + arbiter": "full (accept rule + arbiter)",
    # arbiter variants (indented under the full row, so "full," is implied); no abbreviations
    "full, matched probe": "executor as probe",
    "full, arbiter rt 0.10": "$\\tau_{\\mathrm{r}}{=}0.10$",
    "full, matched probe + rt 0.10": "executor as probe, $\\tau_{\\mathrm{r}}{=}0.10$",
    "full, every-step under the arbiter": "every-step under the arbiter",
    "full, plan-free router": "plan-free router",
    "alone (RH 25)": "vanilla (twenty-five-step horizon)",
    "LeFlow alone (RH 10)": "vanilla (ten-step horizon)",
}
SHORT_BLOCK = {"GC-IDM (official code, s42)": "GC-IDM", "GC-IDM (our reimplementation)": "GC-IDM (ours)"}


def short(key):
    # mapped spellings are hand-written LaTeX (they may carry math), so only unmapped keys are escaped
    return SHORT[key] if key in SHORT else tex(key)


def short_block(name):
    return tex(SHORT_BLOCK.get(name, name))


def split_label(raw):
    """(block name or None, ladder key, is_full, is_variant)"""
    import re
    m = re.match(r"^\*\*(.+?)\*\*(.*)$", raw)
    if m:
        bold, rest = m.group(1).strip(), m.group(2).strip()
        if bold.startswith("SURVEYOR"):
            inner = re.match(r"SURVEYOR \((.*)\) over (.*)$", bold)
            return None, inner.group(1), True, False
        rest = rest.strip(", ")
        return bold, (rest if rest else "alone"), False, False
    inner = re.match(r"SURVEYOR \((.*)\) over (.*)$", raw)
    if inner:
        return None, inner.group(1), False, inner.group(1).startswith("full,")
    return None, raw, False, False


def fmt_cell(c, se=True):
    if c is None:
        return "--"
    sr, n, _, s = c
    out = f"{sr:.1f}"
    if se and s is not None:
        out += f"\\,{{\\scriptsize$\\pm${s:.1f}}}"
    if n < 8:
        out += f"$^{{[{n}]}}$"
    return out


def block_iter(rows):
    """yield (block name, [(ladder key, spec, is_full, is_variant), ...])"""
    blocks = []
    for label, spec in rows:
        block, key, full, variant = split_label(label)
        if block:
            blocks.append((block, []))
        blocks[-1][1].append((key, spec, full, variant))
    return blocks


def sec_per_episode(cellsd, cost, env, spec, t):
    """scheduler wall-clock seconds per episode for the row's arm(s) in env at t, or None"""
    c = b.row_cost(cellsd, cost, env, spec, t)
    return c[0] if c else None


def main_table(cellsd, cost, ladder=False):
    """Table 1: executor alone versus the full layer. With ladder=True the intermediate
    consumption policies (every-step, blind chain, accept rule) are included as well."""
    lines = ["% generated by reproduce/build_table1_latex.py; do not edit by hand",
             "\\begin{tabular}{@{}llrrrrrrr@{}}", "\\toprule",
             "Executor & Configuration & \\multicolumn{2}{c}{PushT + Reacher} & \\multicolumn{2}{c}{all 17 settings} & \\multicolumn{3}{c}{at $t{=}150$} \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-9}",
             " & & mean & worst & mean & worst & call ratio & \\multicolumn{2}{c}{seconds per episode} \\\\",
             " & & & & & & PushT & PushT & Reacher \\\\", "\\midrule"]
    for block, rows in block_iter(b.ROWS):
        kept = []
        for key, spec, full, variant in rows:
            if variant:
                continue
            if key.startswith("alone"):
                name = "executor alone"
            elif key in MAIN_LABELS and (ladder or full):
                name = MAIN_LABELS[key]
            else:
                continue
            cs = {(e, t): b.cell(cellsd, e, spec.get(e), t) for e in b.ENVS for t in b.T[e]}
            pr = b.mean_worst([cs[(e, t)] for e in ("pusht", "reacher") for t in b.T[e]])
            al = b.mean_worst(list(cs.values()))
            c150 = cs[("pusht", 150)]
            if name.startswith("+ drafter (every-step)"):
                cr = "1.00"        # by definition: one drafter call per replanning step
            elif name == "executor alone":
                cr = "0"           # by definition: the executor alone never calls the drafter
            else:
                cr = "--" if c150 is None or c150[2] is None else f"{c150[2]:.2f}"
            sp = [sec_per_episode(cellsd, cost, e, spec.get(e), 150) for e in ("pusht", "reacher")]
            sp = ["--" if v is None else f"{v:.1f}" for v in sp]
            kept.append((name, full, pr, al, cr, sp))
        if not kept:
            continue
        first = True
        for name, full, pr, al, cr, sp in kept:
            lead = f"\\multirow{{{len(kept)}}}{{*}}{{{short_block(block)}}}" if first else ""
            first = False
            vals = [pr[0], pr[1], al[0], al[1], cr] + sp
            if full:
                vals = [f"\\textbf{{{v}}}" for v in vals]
                name = f"\\textbf{{{name}}}"
            lines.append(f"{lead} & {name} & " + " & ".join(vals) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


ABLATION_STEPS = [("every-step", ("drafter only, every-step", "drafter only, every-step, matched S=10")),
                  ("blind chain", ("drafter + blind chain",)),
                  ("accept rule", ("drafter + accept rule",)),
                  ("full", ("full: drafter + accept rule + arbiter",))]


def ablation_table(cellsd):
    """Table 2: the ladder as an ablation. One row per executor; for each consumption step the
    mean / worst success over the nine PushT and Reacher settings, then the call ratio at PushT
    t=150 (every-step is 1.00 by definition)."""
    n = len(ABLATION_STEPS)
    lines = ["% generated by reproduce/build_table1_latex.py; do not edit by hand",
             "\\begin{tabular}{@{}l" + "r" * n + "r" * n + "@{}}", "\\toprule",
             f"Executor & \\multicolumn{{{n}}}{{c}}{{success over PushT + Reacher, mean / worst}} & "
             f"\\multicolumn{{{n}}}{{c}}{{call ratio at PushT $t{{=}}150$}} \\\\",
             f"\\cmidrule(lr){{2-{1 + n}}}\\cmidrule(lr){{{2 + n}-{1 + 2 * n}}}",
             " & " + " & ".join(s for s, _ in ABLATION_STEPS) + " & " + " & ".join(s for s, _ in ABLATION_STEPS) + " \\\\",
             "\\midrule"]
    for block, rows in block_iter(b.ROWS):
        bykey = {key: (spec, full) for key, spec, full, variant in rows if not variant}
        srs, crs = [], []
        for step, keys in ABLATION_STEPS:
            spec = next((bykey[k][0] for k in keys if k in bykey), None)
            if spec is None:
                srs.append("--"); crs.append("--")
                continue
            cs = [b.cell(cellsd, e, spec.get(e), t) for e in ("pusht", "reacher") for t in b.T[e]]
            mw = b.mean_worst(cs)
            sr = f"{mw[0]} / {mw[1]}"
            c150 = b.cell(cellsd, "pusht", spec.get("pusht"), 150)
            cr = "1.00" if step == "every-step" else ("--" if c150 is None or c150[2] is None else f"{c150[2]:.2f}")
            if step == "full":
                sr, cr = f"\\textbf{{{sr}}}", f"\\textbf{{{cr}}}"
            srs.append(sr); crs.append(cr)
        if all(v == "--" for v in srs):
            continue
        lines.append(f"{short_block(block)} & " + " & ".join(srs) + " & " + " & ".join(crs) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def env_table(cellsd, env, rows, se=True, stacked=False):
    """one environment's ladder. stacked=True is for a float that stacks the four environments: the first header
    cell then names the environment, which the float's caption cannot do"""
    ts = b.T[env]
    cols = "@{}ll" + "r" * len(ts) + "@{}"
    lines = ["% generated by reproduce/build_table1_latex.py; do not edit by hand",
             f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
             (ENV_NAME[env] if stacked else "Executor") + " & Configuration & " + " & ".join(f"$t={t}$" for t in ts) + " \\\\",
             "\\midrule"]
    # battery TR: once its Two-Room full rows are in, the Two-Room "executor as probe" rows are dropped: they ran with
    # the default retirement tolerance, the accept rule's DINOv2-space 0.127, and never acted (battery TR2 re-runs them).
    # The rt0.10 rows set 0.10 explicitly and stay valid.
    tr_in = b.TR_ON and any(k[0] == b.TR_PREF for k in cellsd)
    for block, brows in block_iter(rows):
        kept = []
        for key, spec, full, variant in brows:
            arm_env = spec.get(env) if isinstance(spec, dict) else None
            if (env == "tworoom" and variant and tr_in and isinstance(arm_env, str) and arm_env.endswith("_mp")
                    and not any(k[0] == "rbTR2" and k[2] == arm_env + "_tauown_lewmarb" for k in cellsd)):
                continue
            cs = [b.cell(cellsd, env, spec.get(env), t) for t in ts]
            if all(c is None for c in cs):
                continue
            kept.append((key, full, variant, cs))
        if not kept:
            continue
        first = True
        for key, full, variant, cs in kept:
            lead = f"\\multirow{{{len(kept)}}}{{*}}{{{short_block(block)}}}" if first else ""
            first = False
            name = short(key)
            if variant:
                name = f"\\quad {name}"
            vals = [fmt_cell(c, se) for c in cs]
            if full:
                name = f"\\textbf{{{name}}}"
                vals = [f"\\textbf{{{v}}}" for v in vals]
            lines.append(f"{lead} & {name} & " + " & ".join(vals) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def cost_table(cellsd, cost, rows, envs=None):
    envs = envs or b.ENVS
    lines = ["% generated by reproduce/build_table1_latex.py; do not edit by hand",
             "\\begin{tabular}{@{}ll" + "rrr" * len(envs) + "@{}}", "\\toprule",
             "Executor & Configuration & " + " & ".join(
                 f"\\multicolumn{{3}}{{c}}{{{ENV_NAME[e]} $t={b.T_COST[e]}$}}" for e in envs) + " \\\\",
             "".join(f"\\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(envs))),
             " & & " + " & ".join("SR & call ratio & s/ep" for _ in envs) + " \\\\", "\\midrule"]
    for block, brows in block_iter(rows):
        kept = []
        for key, spec, full, variant in brows:
            if variant:
                continue
            vals = []
            any_ = False
            for e in envs:
                t = b.T_COST[e]
                c = b.cell(cellsd, e, spec.get(e), t)
                rc = b.row_cost(cellsd, cost, e, spec.get(e), t)
                any_ = any_ or c is not None
                if c is not None and key.startswith("drafter only, every-step"):
                    cr = "1.00"      # by definition: one drafter call per replanning step
                elif c is not None and key.startswith("alone"):
                    cr = "0"         # by definition: the executor alone never calls the drafter
                else:
                    cr = "--" if c is None or c[2] is None else f"{c[2]:.2f}"
                vals += [fmt_cell(c, se=False), cr, "--" if not rc else f"{rc[0]:.1f}"]
            if any_:
                kept.append((key, full, vals))
        if not kept:
            continue
        first = True
        for key, full, vals in kept:
            lead = f"\\multirow{{{len(kept)}}}{{*}}{{{short_block(block)}}}" if first else ""
            first = False
            name = short(key)
            if full:
                name = f"\\textbf{{{name}}}"
            lines.append(f"{lead} & {name} & " + " & ".join(vals) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


X_NAME = {"CEM (LeWM)": "CEM", "GC-IDM (official code, s42)": "GC-IDM", "Gradient (AdamW)": "Gradient"}
# citation key printed beside each executor in Table 1 (the AdamW gradient solver is the one served by
# stable-worldmodel's planning interface)
X_CITE = {"CEM": "cem", "GC-IDM": "gcidm", "iCEM": "pinneri2021icem", "MPPI": "williams2017mppi", "Gradient": "maes2026stable"}
BEST_BG = "winbg!55"   # light green of the 'better of the pair' cells in Table 1
# in Tables 1 and 2 a pair whose printed values are equal is shaded light amber (no bold); needs \definecolor{tiebg}
# in the preamble
TIE_BG = "tiebg!60"   # a tied pair: shaded on the SURVEYOR row only


def main_table_env(cellsd, cost):
    """Table 1: per executor X, the rows "X" and "X + SURVEYOR" with no Configuration column, one success
    column per environment (mean over that environment's goal distances), the seconds per episode at each environment's
    longest distance, then the call ratio at PushT t=150."""
    envs = " & ".join(ENV_NAME[e] for e in b.ENVS)
    lines = ["% generated by reproduce/build_table1_latex.py; do not edit by hand",
             "% Marking: in each pair the better value is light green and bold whenever the printed",
             "% values differ (higher success, lower seconds). The call ratio is not marked (the vanilla executor never calls the",
             "% drafter). Executors carry their citation.",
             "\\begin{tabular}{@{}l" + "r" * 9 + "@{}}", "\\toprule",
             "Executor & \\multicolumn{4}{c}{mean success rate (\\%)} & \\multicolumn{4}{c}{seconds per episode}"
             " & call ratio \\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-10}",
             f" & {envs} & {envs} & PushT \\\\", "\\midrule"]

    def best(v):
        return f"\\cellcolor{{{BEST_BG}}}\\textbf{{{v}}}"

    def tie(v):
        return f"\\cellcolor{{{TIE_BG}}}{v}"

    for block, rows in block_iter(b.ROWS):
        x = X_NAME.get(block, block)
        pair = {}
        for key, spec, full, variant in rows:
            if variant or not (key.startswith("alone") or full):
                continue
            means = []
            for e in b.ENVS:
                cs = [b.cell(cellsd, e, spec.get(e), t) for t in b.T[e]]
                if any(c is None for c in cs):
                    raise SystemExit(f"missing cell: {block} {key} {e}")
                se = (sum((c[3] or 0.0) ** 2 for c in cs) ** 0.5) / len(cs)
                means.append((sum(c[0] for c in cs) / len(cs), se))
            c150 = b.cell(cellsd, "pusht", spec.get("pusht"), 150)
            cr = "0" if not full else ("--" if c150 is None or c150[2] is None else f"{c150[2]:.2f}")
            # seconds at each environment's own longest distance
            sp = [sec_per_episode(cellsd, cost, e, spec.get(e), b.T_COST[e]) for e in b.ENVS]
            if any(v is None for v in sp):
                raise SystemExit(f"missing timing: {block} {key}")
            pair["full" if full else "van"] = (means, sp, cr)
        assert set(pair) == {"van", "full"}, (block, list(pair))
        for who in ("van", "full"):
            other = "full" if who == "van" else "van"
            means, sp, cr = pair[who]
            omeans, osp, _ = pair[other]
            vals = []
            # a tie is shaded on the SURVEYOR row only
            for (v, _), (ov, _) in zip(means, omeans):
                txt = f"{v:.1f}"
                vals.append((tie(txt) if who == "full" else txt) if txt == f"{ov:.1f}" else (best(txt) if v > ov else txt))
            for v, ov in zip(sp, osp):
                txt = f"{v:.1f}"
                vals.append((tie(txt) if who == "full" else txt) if txt == f"{ov:.1f}" else (best(txt) if v < ov else txt))
            vals.append(cr)
            name = tex(x)
            name = f"\\textbf{{{name} + \\method{{}}}}" if who == "full" else f"{name}~\\citep{{{X_CITE[x]}}}"
            lines.append(f"{name} & " + " & ".join(vals) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


SHORT_T = 50   # short range: t <= SHORT_T (t = 25, 50); long range: every longer distance (t >= 75)


# Table 2 shows only X and X + SURVEYOR; SHOW_FFJ = True adds the X + FF-JEPA row (every-step drafting, S=10) between
# them.
SHOW_FFJ = False


def long_range_table(cellsd):
    """Table 2, companion of Table 1: the same pairs "X" / "X + SURVEYOR", with two columns per environment,
    the mean success over the short distances (t <= SHORT_T) and over the long ones. Marking as in Table 1."""
    groups = [(e, [t for t in b.T[e] if t <= SHORT_T]) for e in b.ENVS] + [(e, [t for t in b.T[e] if t > SHORT_T]) for e in b.ENVS]
    groups = sorted(groups, key=lambda g: (b.ENVS.index(g[0]), g[1][0]))   # env by env: short, then long
    n = len(groups)
    lines = ["% generated by reproduce/build_table1_latex.py; do not edit by hand",
             f"% Short range = goal distances t <= {SHORT_T} (t = 25, 50); long range = t >= 75 (Reacher 100, 150; Two-Room 75).",
             ("% Rows per executor: X, X + FF-JEPA (every-step drafting, S=10), X + SURVEYOR; the highest printed value of each block"
              if SHOW_FFJ else "% Marking as in table1_main.tex: the better value of a pair is light green and bold whenever the printed values"),
             ("% is light green and bold (none when all three print the same)." if SHOW_FFJ else "% differ."),
             # no trailing @{}: the last column is shaded, and colortbl fills \tabcolsep beyond it, past the rules
             "\\begin{tabular}{@{}l" + "r" * n + "}", "\\toprule",
             "Executor & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{ENV_NAME[e]}}}" for e in b.ENVS) + " \\\\",
             "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(b.ENVS))),
             " & " + " & ".join("short & long" for _ in b.ENVS) + " \\\\", "\\midrule"]
    for block, rows in block_iter(b.ROWS):
        x = X_NAME.get(block, block)
        # the optional third row, "X + FF-JEPA" = every-step drafting with the same drafter at the same spacing (S=10; CEM's
        # S=25 row is FF-JEPA's default spacing and weaker, so it is not the one shown). With it, the highest printed value of
        # each three-row block is bold on green (none when all three print the same).
        trio = {}
        for key, spec, full, variant in rows:
            every = key.startswith("drafter only, every-step") and "S=25" not in key
            if variant or not (key.startswith("alone") or full or every):
                continue
            means = []
            for e, ts in groups:
                cs = [b.cell(cellsd, e, spec.get(e), t) for t in ts]
                if any(c is None for c in cs):
                    raise SystemExit(f"missing cell: {block} {key} {e}")
                se = (sum((c[3] or 0.0) ** 2 for c in cs) ** 0.5) / len(cs)
                means.append((sum(c[0] for c in cs) / len(cs), se))
            trio["full" if full else ("ffj" if every else "van")] = means
        assert set(trio) == {"van", "ffj", "full"}, (block, list(trio))
        printed = {who: [f"{v:.1f}" for v, _ in trio[who]] for who in trio}
        shown = ("van", "ffj", "full") if SHOW_FFJ else ("van", "full")
        for who in shown:
            vals = []
            for j, txt in enumerate(printed[who]):
                col = [float(printed[w][j]) for w in shown]
                mark = float(txt) == max(col) and len(set(col)) > 1
                tied = len(set(col)) == 1 and who == "full"   # a tie is shaded on the SURVEYOR row only
                vals.append(f"\\cellcolor{{{BEST_BG}}}\\textbf{{{txt}}}" if mark
                            else (f"\\cellcolor{{{TIE_BG}}}{txt}" if tied else txt))
            name = tex(x)
            name = {"van": name, "ffj": f"{name} + FF-JEPA", "full": f"\\textbf{{{name} + \\method{{}}}}"}[who]
            lines.append(f"{name} & " + " & ".join(vals) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


TIE = 1.0   # points: a per-environment mean within this of the block's best is shaded light green


def env_mean_table(cellsd):
    """main-text companion: per executor and ladder step, mean SR over each environment's goal
    distances; best in each block and column is dark green + bold, ties (within TIE) light green"""
    envs = b.ENVS
    lines = ["% generated by reproduce/build_table1_latex.py; do not edit by hand",
             "\\begin{tabular}{@{}ll" + "r" * len(envs) + "@{}}", "\\toprule",
             "Executor & Configuration & " + " & ".join(ENV_NAME[e] for e in envs) + " \\\\",
             " & & " + " & ".join(f"$t{{=}}{min(b.T[e])}$--${max(b.T[e])}$" for e in envs) + " \\\\", "\\midrule"]
    for block, rows in block_iter(b.ROWS):
        kept = []
        for key, spec, full, variant in rows:
            if variant:
                continue
            if key.startswith("alone"):
                name = "vanilla"
            elif key in MAIN_LABELS:
                name = MAIN_LABELS[key]
            else:
                continue
            vals = []
            for e in envs:
                cs = [b.cell(cellsd, e, spec.get(e), t) for t in b.T[e]]
                vals.append(None if any(c is None for c in cs) else sum(c[0] for c in cs) / len(cs))
            kept.append((name, full, vals))
        if not kept:
            continue
        best = [max((v[i] for _, _, v in kept if v[i] is not None), default=None) for i in range(len(envs))]
        first = True
        for name, full, vals in kept:
            lead = f"\\multirow{{{len(kept)}}}{{*}}{{{short_block(block)}}}" if first else ""
            first = False
            cells = []
            for i, v in enumerate(vals):
                if v is None:
                    cells.append("--")
                elif best[i] is not None and v >= best[i] - 1e-9:
                    cells.append(f"\\cellcolor{{winbg}}\\textbf{{{v:.1f}}}")
                elif best[i] is not None and v >= best[i] - TIE:
                    cells.append(f"\\cellcolor{{winbg2}}{v:.1f}")
                else:
                    cells.append(f"{v:.1f}")
            if full:
                name = f"\\textbf{{{name}}}"
            lines.append(f"{lead} & {name} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def shade(vals):
    """vals: list of (value, se) or None per row of one block/column -> list of LaTeX cell strings.
    Dark green + bold: the best value when its lead over the next best exceeds two pooled SEs.
    Light green: every value within that margin of the best (including the best when it is not a
    clear winner). Plain otherwise."""
    present = [(i, v, s or 0.0) for i, vs in enumerate(vals) if vs is not None for v, s in [vs]]
    if not present:
        return ["--"] * len(vals)
    ranked = sorted(present, key=lambda x: -x[1])
    ib, vb, sb = ranked[0]
    if len(ranked) > 1:
        _, v2, s2 = ranked[1]
        res = 2.0 * (sb ** 2 + s2 ** 2) ** 0.5
        clear = vb - v2 > res
    else:
        res, clear = 0.0, True
    out = []
    for i, vs in enumerate(vals):
        if vs is None:
            out.append("--")
            continue
        v, s = vs
        if i == ib and clear:
            out.append(f"\\cellcolor{{winbg}}\\textbf{{{v:.1f}}}")
        else:
            out.append(f"{v:.1f}")   # no tie shading: a column without a green cell has no clear winner
    return out


def shade_vs_alone(vals, is_full):
    """vals: list of (value, se) or None per ladder row of one block/column; is_full: which row is
    the full layer. The first row is the executor alone. The full row is green when it beats the
    executor alone by more than two pooled SEs and amber when it loses by more; the column maximum
    (every row tied at the maximum) is bold; nothing else is marked."""
    out = []
    alone = vals[0]
    present = [v for v, _ in (x for x in vals if x is not None)]
    vmax = max(present) if present else None
    for i, vs in enumerate(vals):
        if vs is None:
            out.append("--")
            continue
        v, s = vs
        txt = f"\\textbf{{{v:.1f}}}" if vmax is not None and abs(v - vmax) < 1e-9 else f"{v:.1f}"
        if is_full[i] and alone is not None:
            va, sa = alone
            res = 2.0 * ((s or 0.0) ** 2 + (sa or 0.0) ** 2) ** 0.5
            if v - va > res:
                txt = "\\cellcolor{winbg}" + txt
            elif va - v > res:
                txt = "\\cellcolor{losebg}" + txt
        out.append(txt)
    return out


def range_table(cellsd):
    """main-text companion: SR at the shortest and the longest goal distance of each environment,
    per executor and ladder step, shaded per block and column by shade()"""
    envs = b.ENVS
    cols = [(e, t) for e in envs for t in (min(b.T[e]), max(b.T[e]))]
    lines = ["% generated by reproduce/build_table1_latex.py; do not edit by hand",
             "\\begin{tabular}{@{}ll" + "rr" * len(envs) + "@{}}", "\\toprule",
             "Executor & Configuration & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{ENV_NAME[e]}}}" for e in envs) + " \\\\",
             "".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(envs))),
             " & & " + " & ".join(f"$t{{=}}{t}$" for _, t in cols) + " \\\\", "\\midrule"]
    for block, rows in block_iter(b.ROWS):
        kept = []
        for key, spec, full, variant in rows:
            if variant:
                continue
            if key.startswith("alone"):
                name = "vanilla"
            elif key in MAIN_LABELS:
                name = MAIN_LABELS[key]
            else:
                continue
            vals = []
            for e, t in cols:
                c = b.cell(cellsd, e, spec.get(e), t)
                vals.append(None if c is None else (c[0], c[3]))
            kept.append((name, full, vals))
        if not kept:
            continue
        shaded = [shade_vs_alone([v[j] for _, _, v in kept], [f for _, f, _ in kept]) for j in range(len(cols))]
        first = True
        for r, (name, full, _) in enumerate(kept):
            lead = f"\\multirow{{{len(kept)}}}{{*}}{{{short_block(block)}}}" if first else ""
            first = False
            if full:
                name = f"\\textbf{{{name}}}"
            lines.append(f"{lead} & {name} & " + " & ".join(shaded[j][r] for j in range(len(cols))) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    cellsd = b.load_cells()
    cost = b.load_cost()
    os.makedirs(OUT, exist_ok=True)
    written = []

    def put(name, text):
        p = os.path.join(OUT, name)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        written.append(name)

    put("table1_main.tex", main_table_env(cellsd, cost))
    put("table1_long.tex", long_range_table(cellsd))
    put("table1_main_aggregates.tex", main_table(cellsd, cost))     # an earlier layout; not in the paper
    put("table1_ladder.tex", main_table(cellsd, cost, ladder=True))
    put("table1_ablation.tex", ablation_table(cellsd))
    put("table1_env.tex", env_mean_table(cellsd))
    put("table1_range.tex", range_table(cellsd))
    for env in b.ENVS:
        put(f"table1_full_{env}.tex", env_table(cellsd, env, b.ROWS))
    put("table1_cost.tex", cost_table(cellsd, cost, b.ROWS))
    put("table1_cost_pr.tex", cost_table(cellsd, cost, b.ROWS, envs=["pusht", "reacher"]))
    put("table1_cost_ct.tex", cost_table(cellsd, cost, b.ROWS, envs=["cube", "tworoom"]))
    gap = "\\par\\vspace{6pt}\n"      # as in table_dseeds.tex: the stacked environments are separate tabulars
    put("table1_leflow.tex", gap.join(env_table(cellsd, env, b.LEFLOW_ROWS, stacked=True) for env in b.ENVS))
    put("table1_gcidm_ours.tex", gap.join(env_table(cellsd, env, b.APPENDIX_ROWS, stacked=True) for env in b.ENVS))
    print(f"wrote {len(written)} files to {os.path.abspath(OUT)}: " + ", ".join(written))
