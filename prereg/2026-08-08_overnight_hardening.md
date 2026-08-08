# Pre-registration: overnight hardening runs (seed replicates + executor blind control)

Frozen 2026-08-08 (evening), BEFORE submission of either job below. Both jobs
harden claims already written into the workshop paper; neither may be used for
selection, per the reporting rules frozen here.

---

## A. GC-IDM their-spec training-seed replicates (PushT t=25 and t=150)

MOTIVATION: the grey context row and the executor exhibit quote GC-IDM
their-spec at ONE training seed (seed 0). The H_max seed replicates (section C
of 2026-08-07_composite_and_executor.md) measured per-config spread up to 19pp
(H200: 67.6/76.2/86.3) but covered only H_max in {100,150,200}; H=50 (t=25)
and H=300 (t=150, the "collapse to 60.6" cell) remain single-seed. Both ends
of the curve carry paper claims (the 100.0 short-horizon dominance and the
long-horizon collapse the executor rescues), so their seed ranges must be
measured before a reviewer asks.

RUNS (`batch/isca/run_gcidm_seed_replicates.sbatch`): train pusht
H_max in {50, 300} x train seed in {1, 2}, optimizer spec and holdout
excludes unchanged from the anchored port; evaluate each on the identical
banked fixed-pop cell (h50 -> t=25, episodes150s5as25; h300 -> t=150,
episodes150s5; n=256, eval seed 42; deterministic policy, n_eff=256).

REPORTING RULE (frozen; measurement, no falsifier):
- "GC-IDM (their spec)" stays H_max=T at training seed 0 in every table (the
  published instantiation).
- App H gains the 3-seed range for t=25 and t=150.
- If the t=150 spread exceeds 3pp, every prose statement of the collapse and
  of the executor-arm margin quotes the range ("60.6 at their seed; range
  x-y"), including in the abstract if the low end moves materially.
- If any t=150 replicate meets or exceeds the executor arm's 96.1, that is
  stated in the exhibit text at full prominence.
- No number is promoted or demoted by seed selection in either direction.

---

## B. P-EXEC-7: blind-commitment control for the amortized executor (Reacher)

MOTIVATION: the banked PushT t=150 blind control (tau=999) TIES the executor
arm (246/256 both), and the exhibit concedes the verifier's SR contribution
is nil there. Section C of the paper shows the opposite regime for the
CEM executor on exploratory Reacher (blind commitment loses 6.6/4.1pp at
t=100/150). Whether verification remains load-bearing when the executor is
amortized is unmeasured, and the exhibit's honesty clause is incomplete
without it.

ARM (`batch/isca/run_specgcidm_blind_reacher.sbatch`): byte-identical to the
banked P-EXEC-5 executor arm (drafter gdm_reacher_s10.pt goal-conditioned,
k=4, executor gcidm_reacher_h50.pt, S=10, gatev4c s999 fixed populations,
n=128, seeds 42/43) except --accept-tau 999. Cells: t=100 and t=150.
References: the banked tau=0.20 cells, 91.4 (t=100) and 95.3 (t=150).

P-EXEC-7 (frozen): blind commitment loses >= 2.0pp vs the banked tau=0.20
cell at BOTH t=100 and t=150.
AMBIGUOUS: loses at one t only, or by 0 to 2pp at both.
FALSIFIER: ties or wins at both -> verification's success-rate value on
exploratory Reacher is executor-dependent, not general; reported at the same
prominence as a pass, and the exhibit's blind-tie honesty clause is extended
to cover Reacher.

---

## Considered and NOT run tonight (recorded so the absence is auditable)

- Composite PushT t=50 (the ambiguous cell): NOT extended. Extending only the
  cell that came back ambiguous is optional stopping; the banked 9/1/0
  verdict stands as reported.
- Executor arm on Cube: would be the fourth executor environment; requires
  wiring --subgoal specgcidm into the cube driver first. Regime-map
  prediction if run later: tie-or-tax vs plain GC-IDM (amortized-sufficient
  environment). Not launched unattended.
- GC-IDM on Two-Room (hindsight task): the routing rule's two offline
  statistics predict degradation by t=75 (Two-Room difficulty scales with
  horizon; the crossover exists). Requires a stride-1 dense cache build plus
  training; too much new plumbing to run unattended overnight.
