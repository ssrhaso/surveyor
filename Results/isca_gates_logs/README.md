# PushT restage + overnight gates + Reacher audit: ISCA run logs (2026-07-12/13)

Raw SLURM + per-arm logs pulled verbatim from ISCA (`ha676@login.isca.ex.ac.uk:~/le-wm/logs`)
on 2026-07-13 after the full overnight gate battery completed (queue empty, every job
COMPLETED; the lone `pusht-train`/`pusht-pops` FAILED entries have completed retries).
763 files = everything not already archived in `../isca_reacher_logs/` (the 2026-07-11/12
Reacher battery). Descriptive per-arm copies (`*_seed<NN>.log`) duplicate the jobid-named
SLURM files; parse the descriptive ones.

## Job map

| Job ID | Log family | What it is |
|---|---|---|
| 2270687-2270717 | `pusht_dense/pops/sanity/smoke/train` | PushT restage: dense build, populations, S={5,15} + gc drafter training |
| 2270724 | `pusht_repl_*` | Stage B replication anchors, success5 pop: baseline 94.6 / oracle 92.6 / s25 94.8 / s10 99.4 / s10spec 99.3 (cr .74); box-era certified |
| 2270739/2270740 | `pusht_hz_*` | UNFILTERED spine t{25..150} x 2 seeds: baseline 95.5->13.7, s10 87.1->59.6, spec cr .52-.84, s25 87.7->55.9 |
| 2270741/2271135 | `pusht_hzs5_*` | success5 spine respin: baseline 98.1->2.7, s10 99.6->98.4, spec 99.6->98.8 (cr .54-.85), s25 99.0->93.9; reproduces box-era table |
| 2270743 | `pusht_rh[12]_*` | flat replan-rate control: RH1 88.3->4.1, RH2 96.3->9.8 over t25->150 (+ short-protocol RH1/RH2 on success5) |
| 2270817 | `pusht_gamma*` | gamma dose-response x40: S25 49.0->57.9(gamma=1)->4.7(gamma=2); S10 56.6->59.4(gamma=1-1.25)->8.4 |
| 2270819 | `pusht_gdmk*/speck*_t150` | k-anchor t150: es k4/k8 58.4/58.8; spec k4 58.9 (cr .53) |
| 2270820 | `pusht_timing_*` | CUDA-synced timing: drafter 1.02% of episode (k=50), 0.41% (k=8/spec); S10 12.1% faster than S25 |
| 2271077/2271112 | `pusht_gdmk*/speck*_t100`, `klowjob` | k-grid t100 + k-cliff: es 55.9/55.0/63.3/64.1 (k=1/2/4/8); spec k<=2 cr=1.000 (verifier self-test), k4 65.0 (cr .58) |
| 2271078/2271079 | `pusht_s{5,15}_{short,t150}`, `scurve_ev` | S-curve completion: short 98.6/98.4 (s5/s15); t150 55.3/58.3; interior optimum persists at t150 |
| 2271108 | `pusht_valtau0.*` | tau on episode-disjoint val split (seed-777 pop, t150, k=8): 63.5/64.5/63.7/63.7 at tau=.15/.20/.25/.30 (cr .61->.42); tau=0.20 re-selected |
| 2271111 | `pusht_offaxis_*` | tau x k off-axis spot-check t100 k4: tau=.10 64.1 (cr .87), tau=.30 65.0 (cr .47); separable |
| 2271178 | `pusht_gc_*` | gc-ablation last 2x2 cell: short 98.9 (vs goal-free 99.4), t150 60.1 (vs 59.6); conditioning redundant on expert data |
| 2271051 | `reacher_audit2job`, `reacher_lerp*_t100`, `reacher_gdmk4_t100` | lerp oracle at t100 (33.4/43.8/55.3 at frac .25/.5/.75) + es k4 96.1 |
| 2271064 | `reacher_slongjob`, `reacher_s{5,15,25}_t{100,150}` | Reacher scale law at long horizon: 91.2-95.5, flat; optimum is short-range there |
| 2271112 | `reacher_gdmk{1,2}/speck{1,2}_t100` | Reacher k-cliff: es k1/k2 95.1/95.3 (fine); spec k<=2 cr=1.000, SR 95.9 (conservative fail-safe) |
| (07-12 batch) | `reacher_audit/oracle/lerp*_t25/gf/gate/rh/strong` | oracle t25 84.2, lerp t25 95.5/70.7/21.9, goal-free 8.6-11.3, goal-gate 91.6/82.8/80.1, RH sweeps |
| (07-12 batch) | `reacher_frontier_*` (n=4 top-up rows) | frontier: tau=.2/k4 **95.51+-1.17 (cr .627, 2.5 NFE/replan) at n=4**; removes the thin-n=2 caveat; tau=.1/k4 95.90 (cr .84) |

## Populations (NEVER splice)

- `pusht_hzs5_*` / `pusht_repl_*` / `*_short`: success5-filtered population (box-era comparable).
- `pusht_hz_*` / `pusht_rh*_t*` / all t100/t150 gates / gamma / gc_t150 / valtau: UNFILTERED
  populations (harder; absolutes sit ~35-40pp below success5 at long horizon). Within-table
  comparisons only.
- val split (`pusht_valtau*`): independent seed-777 draw, zero episode overlap with eval cells.

## Key certified numbers fed into paper/main_v2.tex (2026-07-13 pass)

- tau-grid table (3 grids, flat SR, monotone calls; tau=0.20 frozen) -> tab:taugrid
- k-curve exact values both envs; k=4 frozen -> sec 5.2
- S-curve 4x4 table (PushT short/t150, Reacher t25/t100-150) -> tab:scurve
- dual-population PushT spine + RH collapse to t150 -> sec 6.2
- timing: drafter 1.0%->0.4%, S10 12% faster -> sec 6.3
- Reacher verifier fail-safe (spec k<=2 cr=1.0, SR unchanged) -> sec 7.1
- full gamma curve both scales -> sec 7.2
- goal-gated verifier negative (82.8/80.1 vs 94.5/93.9) -> sec 8

Aggregation: `parse_logs.py` + `agg.py` (included here; point `LOGDIR` at this
directory); parses the `==== FF-JEPA eval summary ====` blocks + last
`[specaccept] ... call_ratio=` line of every descriptive `*_seed<NN>.log`;
`parsed.csv` is the parsed output over the full 1093-file pull.
