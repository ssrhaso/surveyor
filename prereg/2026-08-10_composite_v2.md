# Pre-registration: Composite-v2 (route amendment, P-COMP-2)

Frozen 2026-08-10, BEFORE submission of the comp2222 jobs. Extends the
composite arc of prereg/2026-08-07_composite_and_executor.md section A;
nothing here alters a banked verdict.

## What changes, and what licenses it

The v1 route (frozen 2026-08-07) assigns PushT->SURVEYOR, Reacher->GC-IDM
their-spec, Cube->GC-IDM their-spec. This amendment changes EXACTLY ONE
assignment:

    Cube -> the SURVEYOR-Base layer over GC-IDM
            (gate-only mode: drafter gdm_cube_s10_gc.pt, k=3, tau=0.20,
             S=10, arrival gate on; executor gcidm_cube_h300.pt; no c*
             probe, no CEM anywhere in the arm -- byte-identical to the
             P-EXEC-8 arm)

V2 RULE, stated generally (no new constants): within an amortized-routed
environment, serve the accept-rule layer over the amortized policy iff the
layer's banked PAIRED margin there is positive; keep the plain policy where
it is at ceiling or the banked layer margin is negative.
  - Cube: banked paired margin +4.9pp (P-EXEC-8, seeds 42-49, falsifier
    fired in the method's favor) -> layer ON.
  - Reacher: GC-IDM alone at ceiling (96-100 banked) and the layer's banked
    short-range cells are far below it (53.9/76.6) -> layer OFF (plain
    GC-IDM), unchanged from v1.
  - PushT: SURVEYOR (CEM, certified) remains the routed arm, unchanged.

DEVIATION, stated openly: this amendment postdates the P-EXEC-8 outcome and
is licensed BY it -- a banked, paired, pre-registered measurement -- not by
any peek at the fresh populations below, which are untouched at freeze time.
The oracle-selection guard of the v1 prereg said misses are reported and the
rule is not edited; the v1 row and its verdict (9/1/0) therefore STAND and
are not restated. V2 is a new, separately-registered composite with its own
verdict.

## Populations (never analyzed)

Builder seed 2222 (disjoint from 777/888/999/1111 and every analysis
population):
  - pusht subset pusht_c2222.h5 (success5 filter, n=256/offset, offsets
    25/50/75/100/150), per-offset episode files;
  - reacher reacher_c2222.ep100/.ep150 (n=128, episode-min 8000);
  - cube per-seed sampling at eval time, seeds 62-69 (disjoint from 42-49
    grid-v2 and 52-59 comp1111), episode-min 8000.
Cells: identical protocols to comp1111 (budgets 2t; cube offset 150 budget
300). Runner-up re-runs mirror comp1111: reacher flat RH2/RH5 at t25/50,
plus -- new -- plain GC-IDM on cube seeds 62-69, so the amendment's margin
is PAIRED on the same fresh populations. n=16 smokes gate the cube layer
tasks and are never quoted.

## Frozen bars

P-COMP-2 (grid bar, mirrors P-COMP-1): on the fresh seed-2222 populations,
the v2 composite is >= every fixed arm (flat RH2/RH5, GC-IDM their-spec,
SURVEYOR, and the layer arm itself) minus 1.0pp, in every cell of the
10-cell grid.
AMBIGUOUS: within (-1.0, -3.0]pp of the best fixed arm in at most 2 cells.
FALSIFIER: > 3pp below the best fixed arm in any cell.

P-COMP-2-CUBE (the amendment's own bar, paired on seeds 62-69): the v2 cube
cell >= plain GC-IDM + 2pp on the identical fresh populations.
AMBIGUOUS: [0, +2)pp.
FALSIFIER (full prominence): below plain GC-IDM paired -> the route
amendment is REJECTED, the v1 composite row stands in the paper, and the
failed amendment is reported in the negatives appendix.

## Reporting rule (frozen)

If P-COMP-2 and P-COMP-2-CUBE both pass: tab:grand's composite row updates
to the v2 cells (populations and rule noted in the dagger note; v1 verdict
still quoted). If the cube bar is ambiguous: both v1 and v2 cube values are
reported, no green. If any falsifier fires: v1 row stands, miss reported.
GC-IDM cells report n_eff = distinct episodes (deterministic). The
composite remains labeled "instrument-routed + banked-margin amendment";
we do NOT claim task-blind per-episode routing across policy classes.

Jobs: batch/isca/run_comp2222_build.sbatch (populations) ->
batch/isca/run_comp2222_arms.sbatch (array 0-31: 0-9 pusht SURVEYOR,
10-14 pusht GC-IDM, 15-18 reacher GC-IDM, 19-22 reacher flat runner-ups,
23 cube plain GC-IDM seeds 62-69, 24-31 cube layer arm one seed/task).

---
---

# BANKED OUTCOMES (2026-08-10/11, jobs 2331170 build + 2331171 arms, all 32
# tasks completed; harvested from logs comp2222_*; folded into
# main_workshop_final.tex same night)

## Cells (20deg where applicable; pooled unrounded counts)

V2 composite by route:
  PushT (SURVEYOR unified, 2 seeds x 256): t25 510/512 = 99.61;
    t50 488/512 = 95.31; t75 489/512 = 95.51; t100 486/512 = 94.92;
    t150 499/512 = 97.46
  Reacher (GC-IDM their-spec, n_eff=128): t25 128/128 = 100.0;
    t50 127/128 = 99.22; t100 123/128 = 96.09; t150 123/128 = 96.09
  Cube (the S layer, seeds 62-69): 121/118/124/118/117/121/119/122 of 128
    -> pooled 960/1024 = 93.75

Fixed-arm comparators on the same fresh populations:
  pusht GC-IDM h=2t: 99.22 / 96.09 / 96.88 / 91.80 / 65.62
  reacher flat runner-ups (2 seeds x 128): RH2 t25 50.00, RH2 t50 76.56,
    RH5 t25 85.94, RH5 t50 92.97
  cube plain GC-IDM (seeds 62-69): 103/114/112/99/112/113/114/110 of 128
    -> pooled 877/1024 = 85.64

## Verdicts

P-COMP-2-CUBE: PASS, decisively. Layer 93.75 vs plain 85.64 paired on the
identical seeds = +8.11pp, positive on 8/8 seeds (bar >= +2). Per-seed
diffs: +14.1/+3.1/+9.4/+14.9/+3.9/+6.3/+3.9/+9.4. This REPLICATES the
P-EXEC-8 margin (+4.9 on seeds 42-49) on a fully disjoint seed set.

P-COMP-2 (grid): 9 PASS / 1 AMBIGUOUS / 0 FAIL. The ambiguous cell is
PushT t=75: composite 95.51 vs same-population GC-IDM 96.88 = -1.37, inside
the declared (-1.0,-3.0] band. Mirrors v1's 9/1/0 (whose ambiguous cell was
PushT t=50 at -2.5). Note pusht t50 this time: 95.31 vs 96.09 = -0.78,
inside the -1.0 bar (pass).

## Reporting actions taken (per the frozen rule)

Both bars passed -> tab:grand Router row updated to the v2 cells
(99.6/95.3/95.5/94.9/97.5 | 100.0/99.2/96.1/96.1 | 93.8); summary columns
recomputed for ALL rows against the new column maxima (Router avg/worst =
98.9/96.4, both green; headline now "holds or ties best in 9 of 10
columns", GC-IDM's one exclusive win = pusht t25); App H ddagger note
rewritten (v2 rule + both verdicts + v1 verdict retained); fig 2 Router
bars updated (in-bar PushT margin now +31.8, the SAME-population margin vs
GC-IDM's fresh 65.62); abstract/intro/S4.3/S5 sentences updated to
"9/1/0 twice, disjoint populations" and "+4.9 replicated at +8.1".
