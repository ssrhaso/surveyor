# Pre-registration: Cube executor arm (P-EXEC-8) + GC-IDM on Two-Room (P-2R)

Frozen 2026-08-09, BEFORE submission of either job. Both extend banked arcs;
neither may alter a banked verdict. Code wiring added for these runs:
gcidm_executor.py gains a gate-only mode (goal_gate without cstar_route), the cube
driver gains --subgoal specgcidm, the tworoom driver gains --subgoal gcidm.

---

## A. P-EXEC-8: the certified executor arm on Cube (fourth executor env)

ARM (`batch/isca/run_specgcidm_cube.sbatch`): SURVEYOR-Base with the GC-IDM
executor on Cube -- drafter gdm_cube_s10_gc.pt (goal-conditioned, derived
k=3), executor gcidm_cube_h300.pt (their-spec H_max=2t=300 checkpoint),
tau=0.20, S=10, arrival gate ON (the confirmed CEM arm is gated, so its
executor twin must be; gate-only mode, no c* probe, no CEM anywhere).
Certified protocol: offset 150, budget 300, n=128/seed, episode-min 8000,
per-seed sampling, seeds 42-49 -- the IDENTICAL populations of the banked
plain-GC-IDM grid-v2 cells (pooled 89.8). n=16 smoke precedes each cell,
never quoted.

REFERENCES: plain GC-IDM 89.8 (same seeds and populations, paired);
CEM-executor gated SURVEYOR-Base 76.5 (confirmation block, different seed
set -- indicative only, cross-population).

P-EXEC-8 (frozen): the executor arm scores >= 2pp BELOW plain GC-IDM pooled
over the same populations -- the decomposition tax, the regime law's fourth
measurement (predicted from P-EXEC-5 and the regime map: cube difficulty
does not scale with horizon, so amortization suffices and decomposition can
only tax it).
SECONDARY (indicative, no bar): the arm lands within +/-5pp of the CEM-arm's
76.5, extending executor-independence of the tax to a third environment.
AMBIGUOUS: within 2pp of plain GC-IDM either way.
FALSIFIER (full prominence): the executor arm meets or beats plain GC-IDM
pooled -> decomposition does NOT tax a sufficient amortized policy on Cube;
the regime-law claim is weakened accordingly in Sections 4.3 and 5.

---

## B. P-2R: GC-IDM on Two-Room -- the routing rule's prospective test on a
##    fourth environment

This closes the "never run on Two-Room" hole with the instrument's verdict
frozen FIRST. The routing rule's two offline statistics say Two-Room
difficulty SCALES with horizon (flat falls 83.7 at t=25 to 41.7 at t=75; the
crossover exists), so the frozen rule routes Two-Room to the drafting side
and predicts GC-IDM degrades with goal distance, exactly as on PushT.

PIPELINE (`batch/isca/run_gcidm_tworoom.sbatch`, one sequential job):
1. Dense cache: tworoom build_subgoals, stride 1, filters OFF
   (--no-require-cross-room --no-success-filter), episodes limited to the
   first 4000 (--max-episodes 4000 --sample-mode head) so training is
   disjoint from the eval holdout (eval samples episode-min 4000).
2. Train gcidm_tworoom_h50.pt and gcidm_tworoom_h150.pt, their optimizer
   spec verbatim (h_max = 2t for the two cells), training seed 0.
3. Cells, 12 seeds each (42-53, per-seed populations = real replicates,
   matching the banked 12-seed cells):
   - their protocol t=25 (mode short, start random, budget 50), h50 ckpt;
   - anchored t=75 (mode long, budget 150), h150 ckpt.
   n=16 smoke first; smokes never quoted.

REFERENCES (banked, 12-seed): their-protocol t=25 flat 83.7 (published 87);
anchored t=75 flat 41.7, certified spec 57.0, ground-truth-waypoint oracle
56.9 (the measured ceiling).

P-2R-1 (frozen): GC-IDM t=75 (pooled, 12 seeds) scores BELOW the certified
spec arm's 57.0.
P-2R-2 (frozen, direction): GC-IDM t=75 < GC-IDM's own t=25 (monotone
degradation with goal distance, the PushT shape).
DESCRIPTIVE (no bar): GC-IDM at their-protocol t=25 vs flat 83.7 -- GC-IDM
may win short range as it does everywhere; banked either way.
AMBIGUOUS: GC-IDM t=75 in [50.0, 57.0) -- below spec but at the oracle
ceiling's shoulder.
FALSIFIER (full prominence): GC-IDM t=75 >= 57.0 -> a concurrent amortized
policy matches the certified drafting arm at range on Two-Room; the Two-Room
headline gains a mandatory context row and the routing rule takes its first
loss on a fourth environment.

PAPER CONSEQUENCE RULE (frozen): App H's sentence "GC-IDM is never run on
Two-Room (no stride-1 dense cache exists...)" is replaced by the measured
cells whichever way they land; the composite's routing table gains a
Two-Room row only if the routing rule's verdict here is a pass.

---
---

# BANKED OUTCOMES (2026-08-10; harvested from ISCA logs, folded into
# main_workshop_final.tex the same day per the frozen consequence rules)

First submissions 2330529 (cube) / 2330530 (tworoom) were VOIDED before any
result: killed by the ISCA non-GNU-sed CRLF footgun (see isca memory), no
cells produced. Clean re-runs: sgc_cube job 2330690 (array 0-7),
gcidm_2room job 2330691 (cache + h50/h150 training + 24 cells). Smokes ran
first, never quoted.

## A. P-EXEC-8 outcome: PRIMARY REFUTED -- FALSIFIER FIRED, in the method's
##    favor

Per-seed SR (n=128 each, seeds 42-49, identical grid-v2 populations):
  executor arm : 92.19 91.41 98.44 94.53 91.41 95.31 96.88 96.88
                 -> pooled 969/1024 = 94.63
  plain GC-IDM : 89.84 92.97 92.97 90.62 89.06 88.28 85.16 89.06
                 -> pooled 919/1024 = 89.75  (the banked 89.8)
Paired diffs: +2.35 -1.56 +5.47 +3.91 +2.34 +7.03 +11.72 +7.81
  -> mean +4.88pp, positive 7/8 seeds, paired t ~= 3.40.
Call ratio 0.946-0.966, NFE/replan 2.84-2.90, arrival gate firing normally.

VERDICTS:
- P-EXEC-8 primary (arm >= 2pp BELOW plain GC-IDM): REFUTED. The frozen
  falsifier (arm meets or beats plain GC-IDM) FIRED: +4.9pp ABOVE.
- SECONDARY (within +/-5pp of the CEM arm's 76.5): also missed (94.6) --
  executor-independence of the tax does NOT extend to Cube; GC-IDM tracks
  drafted waypoints far better than CEM does there.
- Reading: GC-IDM's own goal-distance degradation at offset 150 (89.8 vs
  its short-range ceiling) means near-goal manufacture buys more than
  decomposition costs -- the Sec 4.3 distance hypothesis confirmed on a
  second env, against the regime-map "amortization suffices => tax only"
  corollary. Regime-law claim weakened in Sections 4.3 and 5 as required.

PAPER ACTIONS TAKEN: tab:grand gains the cube executor cell (94.6+-0.7,
green winner, S-note); Sec 4.3 + do-no-harm paragraph scoped to
"short-range tax"; Sec 5 and App C scoped; abstract/contrib/generality
mention +4.9pp paired "against our own frozen tax prediction".

## B. P-2R outcome: BOTH PREDICTIONS REFUTED -- FALSIFIER FIRED (routing
##    rule's first loss)

Cache/training per spec (stride 1, filters off, train episodes <4000 head;
eval holdout episode-min 4000; h50/h150 ckpts, training seed 0).
  t=25 their protocol (h50):  100.00 x12 seeds  -> 768/768 = 100.0
  t=75 anchored (h150):       100.00 x11, 96.88 (seed 52) -> 766/768 = 99.74
References (banked): flat 83.7 (t25) / 41.7 (t75); certified spec 57.0;
ground-truth-waypoint oracle 56.9.

VERDICTS:
- P-2R-1 (GC-IDM t75 < 57.0): REFUTED -- falsifier fired at full
  prominence: 99.7 >> 57.0, above the "oracle ceiling" too.
- P-2R-2 (monotone degradation with distance): REFUTED (100.0 -> 99.7).
- DESCRIPTIVE t25: GC-IDM 100.0 vs flat 83.7 -- wins short range, as
  everywhere.
- Reading: on near-deterministic navigation with hindsight goals,
  single-step inverse dynamics with per-step re-encoding does not inherit
  the search stack's lookahead limit. The "oracle ceiling" 56.9 is a
  ceiling on CEM-consumed waypoints, not on the task. The routing
  statistic (flat's collapse curve) read the search stack, not the task:
  the routing rule takes its FIRST LOSS, on a fourth environment.

PAPER ACTIONS TAKEN (per the frozen consequence rule): App H "never run"
sentence replaced; App E gains item (vii) with the cells and this reading;
Sec 4.5 scopes every Two-Room claim to the search stack; Sec 5 records the
first loss + "map statistics are per-consumption-stack" boundary;
tab:protocols Two-Room GC-IDM cell updated. Composite gains NO Two-Room
row (verdict = fail; the frozen rule adds one only on a pass).
