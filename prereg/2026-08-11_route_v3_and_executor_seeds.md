# Pre-registration: Router v3 (the route repair) + executor training-seed
# robustness (P-ROUTE-3, P-EXEC-9)

Frozen 2026-08-11, BEFORE submission of any cell (ISCA login was unreachable
at freeze time; no job can have started). Both extend banked arcs; neither
may alter a banked verdict.

---

## A. P-ROUTE-3: the route repair -- Two-Room joins the Router

CONTEXT. The v1/v2 routing rule derived its assignments from the SEARCH
stack's collapse statistics, and its fourth-environment prospective test
(P-2R, prereg 2026-08-09) fired its falsifier: GC-IDM saturates Two-Room
(banked 100.0 at t=25, 99.7 at t=75) where flat CEM collapses to 41.7. The
paper's stated lesson (Sec. 5): the map's collapse statistics are
properties of a consumption stack and must be re-measured per executor.
This registration enacts that lesson.

V3 RULE (frozen; a re-derivation, not a per-cell choice): route each
environment by the CANDIDATE EXECUTOR'S OWN collapse statistic on that
environment -- an environment goes to the amortized executor iff GC-IDM's
own goal-distance curve there is flat at the evaluated range; to the
certified layer over it iff the layer's banked paired margin is positive;
to the certified CEM policy where the amortized curve collapses. Applied to
the banked statistics this yields: PushT -> CEM + SURVEYOR (GC-IDM
collapses, 60.6-69.1); Reacher -> GC-IDM (flat curve, at ceiling);
Cube -> GC-IDM + SURVEYOR layer (paired margin +4.9/+8.1, v2); and now
Two-Room -> GC-IDM (its banked curve 100.0 -> 99.7 is flat). The first
three assignments are unchanged from v2 and are NOT re-run here.

CELLS (fresh evaluation seeds 54-65, disjoint from the banked 42-53; the
released hindsight-goal configuration, identical protocols to the banked
P-2R cells; n=64/seed, 12 seeds/cell; checkpoints and serving identical to
the banked runs -- t=25 their protocol on the h=50 checkpoint, t=75
anchored on the h=150 checkpoint; smokes never quoted):
  - Two-Room GC-IDM t=25 (fresh seeds)
  - Two-Room GC-IDM t=75 (fresh seeds)

BARS (frozen):
  P-ROUTE-3-A: the v3 Two-Room t=75 cell >= 51.7 (banked flat 41.7 + 10pp)
    AND >= 94.7 (within 5pp of the banked GC-IDM 99.7).
  P-ROUTE-3-B (direction): t=25 >= t=75 - 5pp (no collapse appears on
    fresh seeds).
AMBIGUOUS: t=75 in [57.0, 94.7) -- above the certified spec arm but
  meaningfully below the banked GC-IDM value.
FALSIFIER (full prominence): t=75 < 57.0 (below the certified spec arm) ->
  the v3 route fails its own confirmation, Two-Room stays OUT of the
  Router, and the repair attempt is reported in the negatives appendix.

REPORTING RULE (frozen): on pass, the Router row gains a Two-Room column
(labeled: the v3 re-derivation and its date; the CEM-stack rows keep
dashes there -- their cells are the App E/heldout arc, different verifier
space) and Sec. 5 reports the loss-then-repair arc as measured; on
ambiguity, both numbers reported, no green; on falsifier, v1..v2 rows
stand unchanged.

---

## B. P-EXEC-9: the flagship's executor training-seed robustness

CONTEXT. The GC-IDM + SURVEYOR flagship (96.1 at PushT t=150; band
96.1-97.3 over four identical runs) serves through the seed-0 GC-IDM
checkpoint. The bare executor's collapse spans 60.6-69.1 across its three
training seeds. Question a reviewer can ask: does the layer's rescue
survive executor training-seed variance?

CELLS: the byte-identical flagship configuration (drafter gdm_stride10.pt,
k=3, tau=0.20, S=10, no CEM; the banked t=150 fixed population
pusht.episodes150s5.json, n=256, budget 300, seed 42), with the executor
checkpoint swapped to gcidm_pusht_h50_s1.pt and gcidm_pusht_h50_s2.pt
(training seeds 1 and 2; both already on ISCA from the 2026-08-08
hardening).

BARS (frozen):
  P-EXEC-9: BOTH cells >= 90 (the original P-EXEC-1 bar, unchanged).
SECONDARY (indicative): both within +-3pp of the seed-0 band's low end
  (i.e., >= 93.1), matching the measured GPU-nondeterminism band.
FALSIFIER (full prominence): any cell < 90 -> the rescue is
  training-seed-dependent; the failing seed is reported alongside that
  seed's own bare-GC-IDM collapse value, and every "across training seeds"
  phrasing in Secs. 4.3/6 is weakened accordingly.

REPORTING RULE (frozen): on pass, the executor exhibit's seed note extends
to "the layer holds >= 90 over all three executor training seeds"; numbers
quoted as measured either way.

Jobs: batch/isca/run_execseeds.sbatch (job 2331545, P-EXEC-9) and
batch/isca/run_route_v3.sbatch (job 2331547, P-ROUTE-3), both submitted
only after this file's freeze.

---
---

# BANKED OUTCOMES (2026-08-10 late; harvested same night, both jobs complete)

## A. P-ROUTE-3 outcome: BOTH BARS PASS -- the route repair is confirmed

Fresh evaluation seeds 54-65 (disjoint from the banked 42-53), n=64/seed:
  t=25 (their protocol, h50):  100.00% x 12 seeds  -> 768/768 = 100.00
  t=75 (anchored, h150):       100.00% x 11 seeds, 98.44% x 1 (seed 60)
                               -> 767/768 = 99.87

VERDICTS:
- P-ROUTE-3-A (t75 >= 51.7 AND >= 94.7): PASS, 99.87 on both counts.
- P-ROUTE-3-B (t25 >= t75 - 5pp): PASS (100.00 vs 94.87 required).
- Falsifier (t75 < 57.0) did NOT fire.
- Replication quality: the banked 42-53 cells read 100.0 / 99.74; the fresh
  54-65 cells read 100.0 / 99.87. Two disjoint 12-seed sets agree to
  0.13pp.

READING: the v3 rule -- route by the CANDIDATE EXECUTOR'S own collapse
statistic rather than the search stack's -- recovers the environment the
v1/v2 rule got wrong. The routing rule's first loss (P-2R, 2026-08-09) is
now a measured loss-then-repair arc: the map was re-derived from the stated
lesson, frozen, and confirmed on never-analyzed seeds.

## B. P-EXEC-9 outcome: PASS on both bars, and conservatively quoted

Flagship arm (specgcidm, PushT t=150 fixed population, n=256, seed 42),
executor training seed swapped:
  seed 1 (gcidm_pusht_h50_s1.pt): 253/256 = 98.83  (5deg 94.53)
  seed 2 (gcidm_pusht_h50_s2.pt): 252/256 = 98.44  (5deg 92.58)
  seed 0 (banked flagship):       246/256 = 96.09

VERDICTS:
- P-EXEC-9 (both >= 90): PASS at 98.83 / 98.44.
- SECONDARY (both >= 93.1): PASS.
- Falsifier did NOT fire.

READING: the certified layer's long-horizon rescue is NOT training-seed
dependent -- it holds 96.1-98.8 across all three executor training seeds,
and seed 0 (the instantiation quoted everywhere in the paper) is the WORST
of the three. This mirrors the bare-GC-IDM replicates, where seed 0 was
also the worst (60.6 vs 62.9/69.1): every quoted margin is conservative in
both directions.

## Reporting actions (per the frozen rules) -- TO APPLY

1. Table 1 gains a Two-Room column: Router row ~99.9, CEM-stack rows dashed
   (their Two-Room cells are the App E held-out arc in a different verifier
   space). Keep the "% of best" summary columns computed over the SAME ten
   cells for cross-row comparability, and say so in the caption.
2. Sec. 5 replaces "the routing rule takes its first loss" with the full
   loss-then-repair arc, both cells quoted.
3. Sec. 4.3 / App H executor-seed note: "the layer holds 96.1-98.8 across
   all three executor training seeds; the quoted 96.1 is the worst."
4. Figure 2 is NOT changed (its axis is t=150; Two-Room's anchored cell is
   t=75).
