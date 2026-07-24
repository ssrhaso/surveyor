import re, os, csv, sys, json
from collections import defaultdict

LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "isca_logs", "logs")

rows = []
for fn in sorted(os.listdir(LOGDIR)):
    if not fn.endswith(".log"):
        continue
    try:
        text = open(os.path.join(LOGDIR, fn), encoding="utf-8", errors="replace").read()
    except Exception as e:
        print("ERR", fn, e); continue

    header = None
    m = re.search(r"^=== (.+?) ===\s*$", text, re.M)
    if m:
        header = m.group(1).strip()

    evalline = None
    m = re.search(r"^\[eval\] (.+)$", text, re.M)
    if m:
        evalline = m.group(1)

    epfile = None
    m = re.search(r"^\[episodes-file\] (\S+):", text, re.M)
    if m:
        epfile = m.group(1)

    # last spec-accept call ratio line
    cr = None; k_steps = None; tau = None
    for m in re.finditer(r"\[specaccept\] tau=([\d.]+) gdm_steps=(\d+).*?call_ratio=([\d.]+)", text):
        tau, k_steps, cr = m.group(1), m.group(2), m.group(3)

    # all eval summary blocks (usually one, at end)
    for m in re.finditer(
        r"==== FF-JEPA eval summary( \(Reacher qpos_match\))? ====\s*\n(.+?)\n((?:\s+.+\n?)+)",
        text):
        is_reacher = bool(m.group(1))
        meta = m.group(2)
        body = m.group(3)
        d = dict(re.findall(r"(\w+)=(\S+)", meta))
        sr20 = sr5 = srq = None
        mm = re.search(r"20deg\s*:\s*([\d.]+)%\s*\((\d+)/(\d+)\)", body)
        if mm: sr20 = mm.group(1)
        mm = re.search(r"\b5deg\s*:\s*([\d.]+)%\s*\((\d+)/(\d+)\)", body)
        if mm: sr5 = mm.group(1)
        mm = re.search(r"SR = ([\d.]+)%\s*\((\d+)/(\d+)\)", body)
        if mm: srq = mm.group(1)

        offset = d.get("goal_offset")
        budget = d.get("budget")
        if evalline:
            dd = dict(re.findall(r"(\w+)=(\S+)", evalline))
            offset = offset or dd.get("goal_offset")
            budget = budget or dd.get("budget")

        rows.append(dict(
            file=fn, header=header, env=("reacher" if is_reacher else "pusht"),
            subgoal=d.get("subgoal"), mode=d.get("mode"), n=d.get("num_eval"),
            seed=d.get("seed"), S=d.get("S"), eval_filter=d.get("eval_filter"),
            offset=offset, budget=budget, epfile=epfile,
            sr20=sr20, sr5=sr5, srq=srq, call_ratio=cr, tau=tau, k=k_steps,
        ))

w = csv.DictWriter(open(os.path.join(os.path.dirname(LOGDIR), "parsed.csv"), "w", newline=""),
                   fieldnames=list(rows[0].keys()))
w.writeheader()
for r in rows: w.writerow(r)
print(f"{len(rows)} summary blocks parsed from {LOGDIR}")

# quick family census of parsed rows
fam = defaultdict(int)
for r in rows:
    fam[re.sub(r"(_?\d+)?(_seed\d+)?\.log$", "", r["file"]).rstrip("_0123456789")] += 1
for k in sorted(fam): print(f"{fam[k]:4d}  {k}")
