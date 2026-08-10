# Pre-registration: instrument-routed composite + GC-IDM as certified executor

Frozen 2026-08-07 (evening), BEFORE any run below was submitted. Context: GC-IDM
(arXiv 2605.08732) entered the baseline set this morning (see
`2026-08-07_baselines.md`, `2026-08-07_gcidm_branches.md`,
`2026-08-07_hmax_tradeoff.md`, P-HMAX-1 refuted). This document freezes the two
structural responses before any cell runs, plus one baseline-stability
measurement and one validity check.

---

## A. Instrument-routed composite (policy-class routing)

CLAIM UNDER TEST: the offline applicability instrument, extended by one
statistic, selects the winning POLICY CLASS per environment without seeing any
closed-loop composite cell.

FROZEN RULE (offline statistics only, all measured before today's grid ran):
deploy certified drafting (SURVEYOR) iff difficulty scales with horizon,
operationalized as BOTH of:

  (i)  the random-policy success floor collapses with goal distance t
       (measured, Results/random_floor.csv: PushT 82.71 -> 0.10 across
       t=25->150; Reacher flat 9-13% at every t; Cube no t-scaling);
  (ii) the amortized policy's own offline goal-distance curve degrades
       (GC-IDM paper Table 2, fixed budget: PushT 94.0 -> 70.0 over offsets
       5-50; Reacher 99.5 -> 99.0; Cube 98.5 -> 93.0).

Otherwise deploy the amortized policy (GC-IDM at its published H_max=T rule).
Both statistics are offline and cheap; neither uses any closed-loop composite
result.

IMPLIED ASSIGNMENTS (written now, before any composite cell):
PushT (all t) -> SURVEYOR. Reacher (all t) -> GC-IDM. Cube -> GC-IDM.

P-COMP-1 (frozen): on fresh confirmation populations (builder seed 1111, never
analyzed), the routed composite is >= every fixed arm (flat RH2/RH5, GC-IDM
their-spec, GC-IDM best-swept, SURVEYOR) minus 1.0pp, in every cell of the
10-cell grid.
AMBIGUOUS: within (-1.0, -3.0]pp of the best fixed arm in at most 2 cells.
FALSIFIER: > 3pp below the best fixed arm in any cell.

HONESTY CLAUSES:
- GC-IDM rows always report n_eff (deterministic policy: n_eff = number of
  distinct episodes, never seeds x episodes).
- The composite is reported as "instrument-routed", rule stated in full. We do
  NOT claim task-blind per-episode routing across policy classes.
- Oracle-selection guard: the assignments above are frozen now. If any later
  measurement contradicts the rule's choice, the miss is REPORTED; the rule is
  not edited.

---

## B. GC-IDM as the executor inside SURVEYOR ("certified speculative execution")

MECHANISM HYPOTHESIS: GC-IDM fails long horizon because amortized inverse
dynamics degrades with goal DISTANCE (its own Table 2), and the drafter's whole
job is manufacturing NEAR goals: S=10 hops. A 10-step hop is mid-distribution
for the H_max=50 checkpoint (h = 10/50 = 0.2, trained on h ~ U[1,50]/50). The
accept rule is executor-agnostic -- it verifies the achieved latent against the
pursued waypoint and never inspects what produced the actions -- so tau, k, S
all carry unchanged. If this works, the consumption layer turns a ~100x cheaper
amortized policy into a long-horizon method: their policy fixes our cost, our
drafter fixes their distance limit, reality remains the verifier.

ARM (frozen configuration; implemented in `surveyor/spec_gcidm.py`, served by
`--subgoal specgcidm`): PushT goal-free drafter S=10, k=3 DDIM (the derived k),
tau=0.20 (the derived tau), N=3 block, accept test every S=10 env steps;
executor = GC-IDM `gcidm_pusht_h50.pt` fed (z_t, w_j, h = steps-to-boundary);
on reject or block exhaustion, re-draft from the achieved latent. No CEM
anywhere in the arm. No new constants.

Executor checkpoint choice is principled, not swept: H_max=50 puts a 10-step
hop in-distribution. ONE declared secondary (reported as a finding, never used
for selection): the same arm with `gcidm_pusht_h200.pt`.

P-EXEC-1 (primary, frozen): PushT t=150, fixed population
`pusht.episodes150s5.json` (n=256), budget 300, score=block, 20deg, seeds
42/43 (seed drives draft sampling): SR >= 90.0%.
  References on this cell: SURVEYOR-Base (CEM executor) 98.1; GC-IDM alone
  best-swept 86.33 / their-spec 60.55; flat 3.9.
AMBIGUOUS: [75, 90) -- decomposition helps the amortized policy but tracking is
lossy; reported as partial support.
FALSIFIER: < 75.0% -- amortized tracking cannot consume drafted plans; reported
as a negative result with the same prominence a win would have received.

P-EXEC-2 (cost, co-primary): zero CEM calls by construction; wall-clock per
episode <= 20% of the CEM-executor spec arm on the same cell (CEM is 94-99% of
wall-clock in the banked cost story). Measured from the arm's own timers.

P-EXEC-3 (diagnostic, secondary): call ratio in [0.35, 0.80] (CEM-executor spec
at this cell ~0.50). Above 0.80 = the tracker leaves the plan and the verifier
reads it correctly (diagnostic, not failure of the verifier); below 0.35 with
low SR = audit for a serving bug before quoting anything.

SECONDARY CELL (declared now): same arm at t=100 (population
`pusht.episodes150s5as100.json` if present, else the t=100 fixed population used
by the banked grid), same seeds, reported alongside.

SMOKE GATE: an n=16 plumbing run precedes the cells; smoke numbers are never
quoted as results.

--- AMENDMENT 2026-08-08 (frozen before submission of the curve cells) ---

P-EXEC-4 (PushT executor curve, completing the exhibit): the executor arm
(config unchanged from B) at t in {25, 50, 75} on the banked fixed populations
(episodes150s5as{t}.json, n=256, seeds 42/43) reads >= GC-IDM (their spec,
H_max=2t, grid-v2 cells: 100.0 / 96.9 / 94.9) minus 2.0pp at every t.
PREDICTION: the executor-arm-minus-GC-IDM gap is monotone nondecreasing in t
(known anchors: +6.3 at t=100, +27.4 vs their-spec at t=150).
FALSIFIER: > 5pp below GC-IDM at any t -> the near-goal-manufacture mechanism
does not explain the t>=100 wins; reported as such.

P-EXEC-5 (Reacher do-no-harm, the layer on their home turf): executor arm
(drafter gdm_reacher_s10.pt goal-conditioned, executor gcidm_reacher_h50.pt,
k=4, tau=0.20, S=10) at t in {25, 50, 100, 150} on the gatev4c s999 fixed
populations (n=128, seeds 42/43) within +/-2pp of plain GC-IDM (their spec,
grid-v2: 99.2 / 100.0 / 95.3 / 96.1... grid-v2 banked row) per cell.
AMBIGUOUS: -2 to -5pp = decomposition tax measured on an amortized-sufficient
task, reported as the layer's cost there.
FALSIFIER: < -5pp anywhere -> the layer actively harms a strong amortized
policy; reported at full prominence.
Rationale for the frozen +/-2 band: waypoints on Reacher are near goals for a
near-goal-competent policy; the accept rule should mostly advance, and c*-free
SURVEYOR-Base has no routing to save it -- this is the honest stress test of
"composes with any executor".

OUTCOME NOTE (2026-08-08, before P-EXEC-6 freeze): P-EXEC-4 PASSED (99.8/98.4/
96.9 vs 100.0/96.9/94.9; gap monotone as predicted). P-EXEC-5 hit its
FALSIFIER at t25/t50 (53.9/76.6 vs 99.2/100.0) and its bands at t100 (-3.9,
ambiguous) and t150 (-0.8, pass). The executor-arm reacher curve tracks the
CEM-executor curve (57.3/78.7/88.2/91.1) within a few pp at every t: the
decomposition tax is executor-independent. Reported at full prominence.

--- AMENDMENT 2 (P-EXEC-6), frozen before submission of the certified cells ---

ARM: specgcidm + --cstar-route. Certificate at both scopes, no new constants:
episode scope = ONE batched flat-CEM probe at the first boundary (arbiter
window horizon 2 x block 5, the banked unified convention), c* <= tau routes
the env to plain GC-IDM (goal served directly, zero drafter calls); replan
scope = the tau arrival gate on drafting envs (Cube's mechanism). Cost: one
CEM solve per episode; execution stays amortised.

P-EXEC-6a (recovery): reacher t25 AND t50 (n=128, seeds 42/43) >= plain
GC-IDM (99.2 / 100.0) - 2pp.
P-EXEC-6b (retention): reacher t100 >= 90.4 and t150 >= 94.3 (bare arm - 1pp).
P-EXEC-6c (range safety): pusht t150 (n=256, seed 42) within +/-1.5pp of the
bare arm's 96.09; expected router fire ~0 there.
MECHANISM PREDICTIONS: router fire rate high at reacher t25/50 (banked c*
AUC 0.85-0.92 for flat reachability, and GC-IDM's window reach contains
flat's), low at reacher t150 first boundary, ~zero at pusht t150.
FALSIFIERS: reacher t25 or t50 < GC-IDM - 5pp -> the c* probe is not
discriminative for the amortised executor's reachability (the V-JEPA c* null,
third architecture); OR pusht t150 < 94.0 -> the router misfires at range.
Either way the certified arm is shelved, reported, and the composite carries
the routed claim. Probe overhead reported as probe_s/episode.

---

## C. H_max seed replicates (baseline stability; measurement, not prediction)

The t=150 H_max sweep behind P-HMAX-1's refutation has ONE training seed per
point and is non-monotone (59.4 / 85.9 / 79.3 / 86.3 / 60.6), so "best swept
86.33" may be selection over training noise. Before quoting it or any margin
against it: train seeds {1, 2} x H_max {100, 150, 200} (their optimizer spec,
same holdout policy), evaluate each on the identical t=150 cell.

DECLARED REPORTING RULE: the paper's "GC-IDM (best swept)" row is the max over
all (H_max, seed) runs -- 11 configurations -- and is labeled as oracle
selection; the "GC-IDM (their spec)" row remains H_max=T at seed 0. Whatever
the numbers read is the row; no falsifier, this is measurement.

---

## D. Two-Room goal-key: mechanism found, decisive run redefined

CORRECTION, same evening, before any run under this doc: this section's first
draft proposed a "benign final-start coincidence" hypothesis. Reading the
already-run goalkey sbatch refutes it -- those arms ran at --start random,
where that hypothesis cannot apply. Retracted; the actual mechanism follows.

MECHANISM (read from the driver, closed against the probe's own numbers):
- `build_process` registers `goal_{col}` for every cached column -- swm's
  derived "column at the goal frame" convention. The serving-path `goal_state`
  is therefore STATE-AT-THE-GOAL-FRAME, not the h5's raw `goal_state` column
  (the episode's fixed target) that the displacement probe measured at
  114.39 px median.
- The probe separately established `state` == `proprio` byte-identical in this
  h5, so derived goal_state == derived goal_proprio exactly. The two "goal
  keys" served identical values; byte-identical outcomes are the expected
  signature of a NAME COLLISION, not of a broken setter.

CONSEQUENCES (declared):
1. Every banked Two-Room cell -- anchor 85.33, +15.4pp t=75, random floor
   22.53 -- ran the HINDSIGHT-goal task ("reach where the demo was t steps
   later"), which is also the task LeWM's released config (goal_proprio)
   defines. All internal comparisons (spec vs flat vs oracle, the crossover)
   remain valid and controlled on that task.
2. The FIXED-TARGET task has never actually been run by us. LeWM Fig. 6's
   Random=0 is consistent with the fixed-target task (probe: 114 px median,
   0.00% pre-solved) and inconsistent with our measured hindsight floor.

DECISIVE RUN (frozen): serve the h5's RAW `goal_state` (fixed target) through
`_set_goal_state` by literal injection (driver flag added for this), random +
flat arms, protocol otherwise unchanged (t=25, budget 50, n=64, eval-filter
none, episode-min 4000), 6 seeds.
P-2ROOM-KEY-1 (frozen): random on the fixed-target task < 5% (the Fig. 6
Random=0 band).
P-2ROOM-KEY-2 (frozen): flat on the fixed-target task in [80, 90] iff their
published 87 is the fixed-target task; < 70 means their figure is better
explained as the hindsight task after all, and the discrepancy moves
elsewhere. Reported either way.
INTERPRETATION RULE (frozen): if P-2ROOM-KEY-1 holds, the paper's Two-Room
section is re-scoped, not retracted: "anchored to LeWM's released
configuration (hindsight goal)", the +15.4pp keeps its internal validity, the
protocol table gains the distinction, and the random-floor row is labeled
hindsight-task. No banked number is deleted; labels change.

---

Files this doc governs: `surveyor/spec_gcidm.py`,
`batch/isca/run_specgcidm_smoke.sbatch`, `batch/isca/run_gcidm_hmax_seeds.sbatch`,
`batch/isca/tworoom/run_tworoom_goalkey_nonfinal.sbatch`.
