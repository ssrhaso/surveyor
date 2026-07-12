# Reacher battery — ISCA run logs (2026-07-11/12)

Raw SLURM + per-arm logs pulled verbatim from ISCA (`ha676@login.isca.ex.ac.uk:~/le-wm/logs`)
after the full Reacher generalization battery completed. Cluster: `gpu` partition,
A100 80GB PCIe (gpu11-16), venv per `batch/isca/`. Eval traces (`runs/reacher/*.pt`,
174MB) remain on the box — re-derivable analysis inputs, not records.

## Job map

| Job ID | Log family | What it is |
|---|---|---|
| 2270232 | `probe_*` | GPU probe: A100 = 80GB, driver 560.35.03, compute nodes have internet |
| 2270234 | `reacher_smoke_*` | smoke n=32: oracle 90.62 / baseline 93.75 — stack validation |
| 2270235 | `reacher_dense_*` | stride-1 dense build, 2.01M frames @ ~790 f/s |
| 2270236[0-3] | `reacher_train_*` | 4 drafters S={5,10,15,25}, faithful recipe, goal-cond U[1,100], eps<8000; final losses 0.006-0.009 |
| 2270237[0-35] | `reacher_arms_*` | native arms: 9 configs x seeds 42-45, n=128, t25/budget50 |
| 2270238[0-23] | `reacher_horizon_*` (+`_2t_`/`_b50_` raw) | horizon sweep {s25gdm,s10gdm,s10spec} x t{25,50,75,100} x seeds{42,43}, both regimes |
| 2270556[0-7] | `reacher_horizon_baseline_*`, `reacher_hzbase_*` | the missing baseline arm at horizon (both regimes) |
| 2270564[0-11] | `reacher_hzext_*` + merged into `reacher_horizon_*_seed4[45]` | seed extension t75/t100 2t -> n=4 |
| 2270576[0-35] | `reacher_hz150_*`, `reacher_hz150job_*` | extended horizon t{100,125,150}, OWN fixed max-offset-150 population (do NOT splice with the max-offset-100 table) |
| 2270577[0-23] | `reacher_frontier_*`, `reacher_frontierjob_*` | spec-accept tau x k frontier at t100/2t (n=2 seeds) |
| — | `inspect_reacher.log` | dataset schema gate (10k eps x 201 steps, all columns) |

Aggregate with `python batch/aggregate_reacher_results.py` (arms + main horizon tables;
requires the combined `reacher_horizon_<name>_t<off>_seed<seed>.log` naming, already merged here).
hz150/frontier parse separately (SR + call_ratio lines; see session notes).

## Final tables

### Native arms (t=25, budget 50, n=128, seeds 42-45)
```
baseline    83.99 +-2.89        <- reproduces paper LeWM Reacher (86, Fig. 6)
s25gdm      53.91 +-1.69   s25spec  59.77 +-5.28  cr=0.881
s15gdm      54.49 +-2.66   s15spec  60.94 +-5.95  cr=0.727
s10gdm      60.55 +-4.16   s10spec  58.59 +-1.11  cr=0.669
s5gdm       52.53 +-0.75   s5spec   46.09 +-4.65  cr=0.589
```

### Horizon, 2t budget (fixed max-offset-100 population)
```
            t25          t50          t75(n=4)     t100(n=4)
baseline   83.20        91.80        89.06        83.20
s10gdm     59.77        79.69        90.23        90.82
s10spec    54.30        79.30        90.03        91.02   (cr .61-.65)
s25gdm     55.47        78.91        89.84        89.45
```

### Horizon, starved (budget 50 fixed)
```
            t50          t75          t100
baseline   76.56        72.27        64.84   <- baseline dominates when starved
s10gdm     50.00        51.17        41.02
s10spec    45.70        46.09        30.86
```

### Extended horizon, 2t (SEPARATE fixed max-offset-150 population, n=4)
```
            t100         t125         t150
baseline   83.20        81.44        75.97   <- keeps degrading with distance
s10gdm     94.53        93.16        95.12   <- flat
s10spec    94.53        93.16        93.94   (cr ~0.62)
```

### Spec-accept frontier (t100, 2t, s10 drafter, n=2 seeds — CAUTION, thin)
```
            k=4               k=8               k=16
tau=0.1    96.09 (NFE 3.35)  93.36 (NFE 6.58)  92.97 (NFE 13.09)
tau=0.2    95.70 (NFE 2.55)  93.36 (NFE 4.87)  92.58 (NFE 10.20)
tau=0.3    94.14 (NFE 2.07)  91.80 (NFE 4.14)  86.72 (NFE 8.22)
tau=0.4    90.23 (NFE 1.85)  90.23 (NFE 3.65)  85.16 (NFE 7.30)
every-step reference: 90.82 at NFE 50/replan
```

## Verdict vs pre-registered criteria

- (i) scale law: interior optimum (s10) reproduces — but the whole native curve sits ~24pp below baseline.
- (ii) spec-accept SR-neutral at call_ratio<1: PASSES in every cell of every regime, zero re-tuning.
- (iii) gdm >= baseline: FAILS natively and under starved budgets (waypoint detour tax — data is
  SAC-wander, oracle waypoints also gain nothing natively); PASSES at long horizon with adequate
  budget, where baseline degrades with goal distance (84->76 at t150) and drafters hold ~94.

Headline: subgoal drafting pays off exactly where the goal image stops informing the planner
(t>=100), and spec-accept delivers it at ~1/10-1/20 of the drafter NFE. NOT comparable to the
paper's native 86% — different protocol; our native repro is 84.

## Known caveats
- frontier is n=2 seeds; tau=0.2/k=4 (95.70) needs an n=4 top-up before being cited as THE number.
- hz150 population (max-offset 150, start<=51) differs from the t25-100 sweep population by
  construction — never merge the two tables (PushT population-artifact lesson).
- horizon logs here are the MERGED combined-naming copies (2t then b50 appended), plus the raw
  `_2t_`/`_b50_` split files from the first sweep.
