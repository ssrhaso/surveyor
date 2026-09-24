"""Table 1 lookup layer: success cells, call ratios and wall-clock per table row, from the harvested batteries.

Reads, from this directory, rb*_cells.csv (SR, SE over evaluation seeds, call ratio, n seeds), the manifests
battery_*.tsv and the scheduler dumps rb*_sacct.txt (wall-clock per array task). The other builders import it; run on its
own it writes a markdown draft of the success and cost tables (table1_draft.md, into the directory given as the first
argument, default out/). Cells are looked up across batteries in PREFS order (Cube: batteries K and later); the first
full-seed cell wins, otherwise the cell with the most seeds is shown with [n]. Nothing is typed by hand:

    python reproduce/build_table1.py [out_dir]
"""
import csv
import datetime as dt
import glob
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PREFS = ["rbA", "rbB", "rbD", "rbE", "rbF", "rbG", "rbH", "rbI", "rbI2", "rbJ", "rbK", "rbKg", "rbL", "rbM", "rbN",
         "rbO", "rbP", "rbR", "rbS"]   # R: ten-step alone rows; S: arbiter controls; O: gradient-solver arbiter variants; P: GC-IDM arbiter variants
# A row that reads the tolerance is reported at its environment's own measured value (calibration/). Those runs carry
# the arm suffix _tauown (batteries T, U, V and V2) and are used once all eight seeds of a cell are in.
# SURVEYOR_TAU=served rebuilds the uniform-0.20 tables, which the appendix keeps as the transferred-tolerance comparison.
OWN_PREFS = [] if os.environ.get("SURVEYOR_TAU") == "served" else ["rbT", "rbU", "rbV", "rbV2"]
OWN_SUFFIX = "_tauown"
# wall-clock of an own-tolerance row comes from its A100 replay on cluster A (battery Ua), never from cluster B
OWN_COST_SUFFIX = "_tauown_a100"
# Battery TR: on Two-Room the full layer of the earlier batteries read the arbiter's LeWM-space probe against the accept
# rule's DINOv2-space tolerance and never acted. TR re-ran those rows with the probe read against a tolerance measured in
# LeWM space (calibration/gap_tworoom_lewm_TR_fullscene.json); its rows replace the earlier Two-Room full-layer rows
# (cells and A100 seconds). SURVEYOR_TR=off rebuilds the earlier ones.
TR_ON = os.environ.get("SURVEYOR_TR") != "off"
TR_PREF, TR_SUFFIX = "rbTR", "_tauown_lewmarb"
# SURVEYOR_TIMING=gh200 reads the cluster B timing set (battery Ug, arm suffix _gh200) and nothing else: one seconds column
# never mixes machines. The default is the A100 set.
GH200 = os.environ.get("SURVEYOR_TIMING") == "gh200"
# Cube cells come only from batteries on the fixed non-trivial populations (cube_c2222; K and later): the
# `--start final` Cube rows of A/H/I/J are solved at step 0 at t=25/100 and 37% trivial at t=50.
ENV_PREFS = {"cube": ["rbK", "rbKg", "rbL", "rbM", "rbN", "rbO", "rbP", "rbR"]}
ENVS = ["pusht", "reacher", "cube", "tworoom"]
T = {"pusht": [25, 50, 75, 100, 150], "reacher": [25, 50, 100, 150], "cube": [25, 50, 75, 100, 150],
     "tworoom": [25, 50, 75]}
T_COST = {"pusht": 150, "reacher": 150, "cube": 150, "tworoom": 75}
NCOL = sum(len(v) for v in T.values()) + 4   # + mean/worst over PushT+Reacher, mean/worst over all

# SLURM array job -> manifest whose idx column the array index addresses
JOBS = {"2382688": "battery_A.tsv", "2388190": "battery_A.tsv", "2388058": "battery_D.tsv",
        "2388068": "battery_E.tsv", "2388179": "battery_G.tsv", "2388299": "battery_F.tsv",
        "2390665": "battery_H.tsv", "2390666": "battery_H2.tsv", "2391665": "battery_H2.tsv",
        "2390661": "battery_I.tsv", "2382800": "battery_B.tsv", "2390683": "battery_I2.tsv",
        "2393809": "battery_K.tsv", "2393810": "battery_K.tsv",
        # L/M/N: cluster A arrays only (their cluster B (GH200) remainder is not comparable wall-clock)
        "2394065": "battery_L.tsv", "2407867": "battery_M.tsv", "2407849": "battery_N.tsv",
        "2405789": "battery_Kg.tsv",
        # the five re-run seeds of L/M/N and battery Q, the gradient block's cost rows
        # replayed on the A100 partition so that its wall-clock is comparable with every other block
        "2410999": "battery_L.tsv", "2411000": "battery_M.tsv", "2411001": "battery_N.tsv",
        "2411002": "battery_Q.tsv",
        # battery Q's Table 1 rows (Reacher t=150, manifest rows 80-95) brought forward as their own array
        "2412415": "battery_Q.tsv",
        # battery Ua, the A100 replay of the own-tolerance full rows at t=150 (PushT, Reacher)
        "2415420": "battery_Ua.tsv",
        # Ub = own-tolerance accept / full rows of the cost tables, Uc = Two-Room at 0.127 (A100 replays)
        "2415546": "battery_Ub.tsv", "2416121": "battery_Uc.tsv",
        # battery Ud, the A100 timing of the one "alone" arm that wins its cell and had run on cluster B only
        # (MPPI ten-step, Cube t=150), plus the missing eighth seed of MPPI alone on Two-Room t=75 (row 8, its own array)
        "2416880": "battery_Ud.tsv", "2416888": "battery_Ud.tsv",
        # battery Ue: A100 timing of the shortest-distance rows that had none
        "2417157": "battery_Ue.tsv", "2417158": "battery_Ue.tsv",
        # battery Ug: the shortest-distance rows on cluster B (GH200s), a second, self-contained timing set
        "6720059": "battery_Ug.tsv", "6720532": "battery_Ug.tsv", "6720533": "battery_Ug.tsv",
        # battery TR (cluster A, A100): smoke row 96, the 120-row array, and its sweeper pass over unbanked rows
        "2506709": "battery_TR.tsv", "2506711": "battery_TR.tsv", "2506817": "battery_TR.tsv",
        "2507389": "battery_TR2.tsv"}


def S(*arms):
    """search-executor spelling of a row: (pusht, reacher, cube, tworoom) arms"""
    return dict(zip(ENVS, arms))


def block(name, alone, every, blind2, blindk, accept, full, alone_label="alone"):
    return [(f"**{name}** {alone_label}", alone),
            (f"SURVEYOR (drafter only, every-step) over {name}", every),
            (f"SURVEYOR (drafter + blind commit-2) over {name}", blind2),
            (f"SURVEYOR (drafter + blind chain) over {name}", blindk),
            (f"SURVEYOR (drafter + accept rule) over {name}", accept),
            (f"**SURVEYOR (full: drafter + accept rule + arbiter) over {name}**", full)]


ROWS = []
ROWS += [("**CEM (LeWM)** alone, strongest flat", S(["flat_rh5", "flat_rh2"], ["flat_rh5", "flat_rh2"], ["flat_rh5", "flat_rh2"], ["flat_rh5", "flat_rh2"])),
         ("SURVEYOR (drafter only, every-step, matched S=10) over CEM", S("ffjepa_s10k3", "ffjepa_s10k8", "ffjepa_s10k3", "pair_every")),
         # S=25 is the FF-JEPA default spacing; only PushT and Reacher have a stride-25 drafter checkpoint
         ("SURVEYOR (drafter only, every-step, S=25) over CEM", S("ffjepa_s25", "ffjepa_s25", None, None)),
         # Two-Room's pair_blind runs at --accept-tau 999 (every draft consumed whole), i.e. the blind chain
         ("SURVEYOR (drafter + blind commit-2) over CEM", S("blind2", "blind2", None, None)),
         ("SURVEYOR (drafter + blind chain) over CEM", S("blindk", "blindk", "blindk", "pair_blind")),
         ("SURVEYOR (drafter + accept rule) over CEM", S("surveyor", "surveyor", "surveyor", "pair_accept")),
         ("**SURVEYOR (full: drafter + accept rule + arbiter) over CEM**", S("router", "router", "router", "router")),
         # battery L/M: deployed arbiter (retire threshold 0.10 instead of the verifier tau)
         ("SURVEYOR (full, arbiter rt 0.10) over CEM", S(*["router_rt0.10"] * 4)),
         # battery S: arbiter controls (every-step drafting under the arbiter; plan-free router)
         ("SURVEYOR (full, every-step under the arbiter) over CEM", S("router_every", "router_every", None, None)),
         ("SURVEYOR (full, plan-free router) over CEM", S("router_lat", "router_lat", None, None))]
# Table 1 uses the authors' released GC-IDM code (training seed 42, the same for every row);
# our reimplementation moves to the appendix table as a replication check.
# Cube accept-rule rows are the gate-matched battery-Kg reruns (no --goal-gate, like the other rows).
ROWS += block("GC-IDM (official code, s42)", S(*["gcidm_off_s42"] * 4), S(*["gcevery_off_s42"] * 4), S(None, None, None, None),
              S(*["gcblind_off_s42"] * 4), S("gcsurv_off_s42", "gcsurv_off_s42", "gcsurv_off_s42_nogate", "gcsurv_off_s42"),
              S(*["gcrouter_off_s42"] * 4))
# battery P: the arbiter's two settings on the amortized executor; the matched probe is GC-IDM's own forecast,
# which routes at the first boundary and retires later
ROWS += [("SURVEYOR (full, matched probe) over GC-IDM (official code, s42)", S(*["gcrouter_off_s42_mp"] * 4)),
         ("SURVEYOR (full, arbiter rt 0.10) over GC-IDM (official code, s42)", S(*["gcrouter_off_s42_rt0.10"] * 4)),
         ("SURVEYOR (full, matched probe + rt 0.10) over GC-IDM (official code, s42)", S(*["gcrouter_off_s42_mp_rt0.10"] * 4))]
# LeFlow (batteries F/G/I/J/K) is not in Table 1; its block is an appendix table (table1_leflow.tex).
LEFLOW_ROWS = [("**LeFlow** alone (RH 25)", S(*["leflow"] * 4)),
               ("LeFlow alone (RH 10)", S(*["leflow_rh10"] * 4)),
               ("SURVEYOR (drafter only, every-step) over LeFlow", S(*["leflow_every"] * 4)),
               ("SURVEYOR (drafter + blind chain) over LeFlow", S(*["leflow_blind"] * 4)),
               ("SURVEYOR (drafter + accept rule) over LeFlow", S(*["leflow_accept"] * 4)),
               ("**SURVEYOR (full: drafter + accept rule + arbiter) over LeFlow**", S(*["leflow_router"] * 4))]
for sol in ("icem", "mppi"):
    ROWS += block(sol.upper() if sol == "mppi" else "iCEM",
                  # alone = the better of receding horizons 5 and 2 per cell (battery R adds the 10-step rows)
                  S([f"{sol}_flat_rh5", f"{sol}_flat_rh2"], [f"{sol}_flat_rh5", f"{sol}_flat_rh2"], [f"{sol}_flat_rh5", f"{sol}_flat_rh2"], f"{sol}_flat_rh2"),
                  S(f"{sol}_ffjepa_s10k3", f"{sol}_ffjepa_s10k8", f"{sol}_ffjepa_s10k3", f"{sol}_pair_every"),
                  S(f"{sol}_blind2", f"{sol}_blind2", None, None),
                  S(f"{sol}_blindk", f"{sol}_blindk", f"{sol}_blindk", f"{sol}_pair_blind"),
                  S(f"{sol}_surveyor", f"{sol}_surveyor", f"{sol}_surveyor", f"{sol}_pair_accept"),
                  S(f"{sol}_router", f"{sol}_router", f"{sol}_router", f"{sol}_router"),
                  alone_label="alone (flat RH5)")
    # battery L/M: the arbiter's two knobs on the sampling solvers (matched c* probe; retire 0.10; both)
    ROWS += [(f"SURVEYOR (full, matched probe) over {sol}", S(*[f"{sol}_router_mp"] * 4)),
             (f"SURVEYOR (full, arbiter rt 0.10) over {sol}", S(*[f"{sol}_router_rt0.10"] * 4)),
             (f"SURVEYOR (full, matched probe + rt 0.10) over {sol}", S(*[f"{sol}_router_mp_rt0.10"] * 4))]
    if sol == "icem":
        ROWS += [("SURVEYOR (full, every-step under the arbiter) over icem", S("icem_router_every", "icem_router_every", None, None)),
                 ("SURVEYOR (full, plan-free router) over icem", S("icem_router_lat", "icem_router_lat", None, None))]
# battery N: LeWM's gradient solver (AdamW, config/eval/solver/adam.yaml) as the search executor
ROWS += block("Gradient (AdamW)", S(["gd_flat_rh5", "gd_flat_rh2"], ["gd_flat_rh5", "gd_flat_rh2"], ["gd_flat_rh5", "gd_flat_rh2"], "gd_flat_rh2"),
              S("gd_ffjepa_s10k3", "gd_ffjepa_s10k8", "gd_ffjepa_s10k3", "gd_pair_every"),
              S("gd_blind2", "gd_blind2", None, None),
              S("gd_blindk", "gd_blindk", "gd_blindk", "gd_pair_blind"),
              S("gd_surveyor", "gd_surveyor", "gd_surveyor", "gd_pair_accept"),
              S(*["gd_router"] * 4), alone_label="alone (flat RH5)")
ROWS += [("SURVEYOR (full, matched probe) over Gradient (AdamW)", S(*["gd_router_mp"] * 4)),
         # battery O: the stricter retirement tolerance, alone and with the matched probe
         ("SURVEYOR (full, arbiter rt 0.10) over Gradient (AdamW)", S(*["gd_router_rt0.10"] * 4)),
         ("SURVEYOR (full, matched probe + rt 0.10) over Gradient (AdamW)", S(*["gd_router_mp_rt0.10"] * 4))]

APPENDIX_ROWS = block("GC-IDM (our reimplementation)", S(*["gcidm"] * 4), S(*["gcidm_every"] * 4), S(None, None, None, None),
                      S(*["gcidm_blind"] * 4), S("gcidm_surv", "gcidm_surv", "gcidm_surv_nogate", "gcidm_surv"), S(*["gcidm_router"] * 4))


def load_cells():
    cells = {}
    for fn in sorted(glob.glob(os.path.join(HERE, "rb*_cells.csv"))):
        pref = os.path.basename(fn).split("_")[0]
        with open(fn) as f:
            for r in csv.DictReader(f):
                cr = float(r["call_ratio"]) if r.get("call_ratio") not in (None, "", "nan") else None
                se = float(r["sr_se_pct"]) if r.get("sr_se_pct") not in (None, "", "nan") else None
                cells[(pref, r["env"], r["arm"], int(r["t"]))] = (float(r["sr_mean_pct"]), int(r["n_seeds"]), cr, se)
    return cells


def get(cells, env, arm, t):
    """first full-seed cell in the env's battery order, else the one with most seeds;
    the own-tolerance run of the arm wins when it exists with all eight seeds"""
    if env == "tworoom" and TR_ON:
        # battery TR: the full layer; battery TR2: the executor-as-probe variant (arms *_mp), same LeWM-space arbiter
        for p in (TR_PREF, "rbTR2"):
            c = cells.get((p, env, arm + TR_SUFFIX, t))
            if c is not None and c[1] >= 8:
                return c
    for p in OWN_PREFS:
        c = cells.get((p, env, arm + OWN_SUFFIX, t))
        if c is not None and c[1] >= 8:
            return c
    best = None
    for p in ENV_PREFS.get(env, PREFS):
        c = cells.get((p, env, arm, t))
        if c is None:
            continue
        if c[1] >= 8:
            return c
        if best is None or c[1] > best[1]:
            best = c
    return best


def cost_of(cells, cost, env, arm, t):
    """(seconds per episode, n) of an arm. An arm that has an own-tolerance run is timed by that
    run's A100 replay (battery Ua / Ub) and by nothing else: its 0.20 timing belongs to a different configuration."""
    if env == "tworoom" and TR_ON and not GH200 and (TR_PREF, env, arm + TR_SUFFIX, t) in cells:
        return cost.get((env, arm + TR_SUFFIX, t))   # battery TR ran on cluster A (A100s), one row per job
    if GH200:
        own = any((p, env, arm + OWN_SUFFIX, t) in cells for p in OWN_PREFS)
        return cost.get((env, arm + (OWN_SUFFIX if own else "") + "_gh200", t))
    if any((p, env, arm + OWN_SUFFIX, t) in cells for p in OWN_PREFS):
        return cost.get((env, arm + OWN_COST_SUFFIX, t))
    # an arm that ran on cluster B only is timed by its A100 replay when it has one (battery Ud, arm suffix _a100)
    return cost.get((env, arm, t)) or cost.get((env, arm + "_a100", t))


def row_cost(cells, cost, env, spec, t):
    """(seconds per episode, n) of a table row. A row that is the better of several arms per cell (the "alone" rows: two
    horizons) is timed by the arm whose success rate it prints, so that success and seconds describe the same run."""
    arms = [a for a in (spec if isinstance(spec, list) else [spec]) if a]
    got = [(get(cells, env, a, t), a) for a in arms]
    got = [(g, a) for g, a in got if g is not None]
    if not got:
        return None
    winner = max(got, key=lambda x: x[0][0])[1]
    c = cost_of(cells, cost, env, winner, t)
    if c is None and len(arms) > 1:
        print(f"warn: {env} {winner} t={t} wins its row but has no A100 timing", file=sys.stderr)
    return c


def cell(cells, env, spec, t):
    if spec is None or t not in T[env]:
        return None
    arms = spec if isinstance(spec, list) else [spec]
    got = [get(cells, env, a, t) for a in arms]
    got = [g for g in got if g is not None]
    return max(got, key=lambda g: g[0]) if got else None


def fmt(c):
    if c is None:
        return "--"
    sr, n, _, se = c
    return f"{sr:.1f}" + (f"±{se:.1f}" if se is not None else "") + (f" [{n}]" if n < 8 else "")


def mean_worst(cs):
    if not cs or any(c is None for c in cs):
        return ["--", "--"]
    srs = [c[0] for c in cs]
    return [f"{sum(srs) / len(srs):.1f}", f"{min(srs):.1f}"]


def elapsed_seconds(s):
    if re.fullmatch(r"\d+", s):
        return int(s)
    d, _, hms = s.rpartition("-")
    h, m, sec = hms.split(":")
    return (int(d) if d else 0) * 86400 + int(h) * 3600 + int(m) * 60 + int(sec)


def load_cost():
    """(env, arm, t) -> mean SLURM seconds per episode over completed seeds"""
    manifests = {}
    for m in set(JOBS.values()):
        fn = os.path.join(HERE, m)
        if not os.path.exists(fn):
            continue
        rows = {}
        with open(fn) as f:
            f.readline()
            for line in f:
                p = line.rstrip("\n").split("\t", 6)
                if len(p) == 7:
                    ne = re.search(r"--num-eval (\d+)", p[6])
                    rows[int(p[0])] = (p[1], p[2], int(p[3]), int(ne.group(1)) if ne else None)
        manifests[m] = rows
    by_row = defaultdict(list)            # (manifest, row) -> elapsed seconds of every COMPLETED task that ran it
    seen = set()
    for fn in glob.glob(os.path.join(HERE, "rb*_sacct.txt")):
        with open(fn) as f:
            for line in f:
                p = line.rstrip("\n").split("|")
                if len(p) < 3 or not re.match(r"\d+_\d+$", p[0]) or p[0] in seen:
                    continue
                job, idx = p[0].split("_")
                state = [x for x in p if x in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "RUNNING")]
                el = [x for x in p[1:] if re.fullmatch(r"(\d+-)?\d+:\d\d:\d\d|\d+", x)]
                if job not in JOBS or JOBS[job] not in manifests or not state or state[0] != "COMPLETED" or not el:
                    continue
                seen.add(p[0])                # only a COMPLETED line claims the task: a stale RUNNING line must not hide it
                r = manifests[JOBS[job]].get(int(idx))
                if r is None or r[3] is None:
                    print(f"warn: {p[0]} has no manifest row / num-eval", file=sys.stderr)
                    continue
                by_row[(JOBS[job], int(idx))].append((elapsed_seconds(el[0]), r))
    per = defaultdict(list)
    for tasks in by_row.values():
        # A row submitted in two arrays (battery Q's rows 80-95 were brought forward as array 2412415 and met again by
        # 2411002) exits within seconds the second time because its log is already banked, and SLURM still reports
        # COMPLETED. Such a task timed nothing: drop any task shorter than a quarter of the row's longest one.
        longest = max(e for e, _ in tasks)
        for e, r in tasks:
            if e >= 0.25 * longest:
                per[(r[0], r[1], r[2])].append(e / r[3])
    return {k: (sum(v) / len(v), len(v)) for k, v in per.items()}


def sr_table(cells, rows):
    head = "| Substrate / row | " + " | ".join(
        (f"{e.capitalize() if e != 'tworoom' else '2-Room'} {t}" if i == 0 else str(t))
        for e in ENVS for i, t in enumerate(T[e])) + " | mean PR | worst PR | mean all | worst all |"
    out = [head, "|---|" + "---|" * NCOL]
    for label, spec in rows:
        cs = {(e, t): cell(cells, e, spec.get(e), t) for e in ENVS for t in T[e]}
        vals = [fmt(cs[(e, t)]) for e in ENVS for t in T[e]]
        if all(v == "--" for v in vals):
            continue                      # no such arm on any substrate yet
        vals += mean_worst([cs[(e, t)] for e in ("pusht", "reacher") for t in T[e]])
        vals += mean_worst(list(cs.values()))
        out.append(f"| {label} | " + " | ".join(vals) + " |")
    return out


def main():
    cells = load_cells()
    cost = load_cost()
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out")
    os.makedirs(out_dir, exist_ok=True)
    out = ["# Table 1 draft (generated by build_table1.py)", "",
           "One block per executor X (CEM, iCEM, MPPI, gradient, GC-IDM), the components added one at a time: "
           "`X alone | drafter only (every-step) | + blind | + accept rule | full (+ arbiter)`.",
           "`mean PR` / `worst PR` = mean and minimum over the nine PushT and Reacher cells every executor has;",
           "`mean all` / `worst all` = over all 17 cells (PushT 5, Reacher 4, Cube 5, Two-Room 3), shown only when the row has every cell.",
           "One training seed per component (drafter 42, official GC-IDM 42, LeFlow 42, official LeWM checkpoint), identical in every row.",
           "Cells: mean ± SE over evaluation seeds 42-49 (evaluation randomness only); `[n]` marks cells with fewer than 8 seeds; `--` = not run.",
           "Episodes per seed: PushT 256, Reacher 128, Cube 128 (fixed pair files in episodes/); Two-Room 64 (pairs drawn per seed).",
           "Budget 2t everywhere. Two-Room t=100 is infeasible (episodes are at most 101 steps).",
           "SR in %. `cr` = drafter calls per replan relative to every-step (1.00). s/ep = SLURM",
           "wall-clock seconds per episode (includes model load; successful episodes end early).",
           "Cube: batteries K and later only (fixed non-trivial populations, block moves >= 8 cm between start and goal).",
           "Sources: " + ", ".join(sorted({k[0] for k in cells})) + ".", "",
           "## Success rate", ""]
    out += sr_table(cells, ROWS)
    out += ["", "## Cost at the longest goal distance (PushT 150, Reacher 150, Cube 150, Two-Room 75)", "",
            "| Substrate / row | " + " | ".join(f"{e} SR | cr | s/ep" for e in ("PushT", "Reacher", "Cube", "2-Room")) + " |",
            "|---|" + "---|" * 12]
    for label, spec in ROWS:
        if all(cell(cells, e, spec.get(e), t) is None for e in ENVS for t in T[e]):
            continue
        vals = []
        for e in ENVS:
            t = T_COST[e]
            c = cell(cells, e, spec.get(e), t)
            rc = row_cost(cells, cost, e, spec.get(e), t)
            vals += [fmt(c), "--" if c is None or c[2] is None else f"{c[2]:.2f}",
                     "--" if not rc else f"{rc[0]:.2f}" + (f" [{rc[1]}]" if rc[1] < 8 else "")]
        out.append(f"| {label} | " + " | ".join(vals) + " |")
    out += ["", "## Appendix: GC-IDM, our reimplementation (replication check for the official-code block)", ""]
    out += sr_table(cells, APPENDIX_ROWS)
    out += ["", "## Appendix: LeFlow", ""]
    out += sr_table(cells, LEFLOW_ROWS)
    fn = os.path.join(out_dir, "table1_draft.md")
    with open(fn, "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {fn}: {len(ROWS)} rows + {len(APPENDIX_ROWS)} appendix rows, {len(cells)} cells read, {len(cost)} cost cells")


if __name__ == "__main__":
    main()
