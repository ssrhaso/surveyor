# P-REMATCH: can the P-DRIFT Reacher coin be rate-matched at all?

**Frozen 2026-08-13, before any run below.** Post-hoc-motivated and declared as
such: written immediately after the P-DRIFT stage-2 harvest
(`2026-08-11_autocorrelated_divergence.md`, Outcome section) found the Reacher
coin cells failed their own `rho=0` tie control (coin 93.75 vs test 91.21,
2.54pp, outside the frozen 2.0pp band), with the recorded ~6pp rate
over-realisation running in the coin's favour on that environment.

## What this is, and what it is not

This is **control hygiene, not a rescue**. The accept-test arm is untouched
(stage 1, job 2334493, stands as banked); the only thing re-run is the coin,
at a specified rate corrected so its **realised** call ratio matches the
test's realised call ratio, which is what "rate-matched" was always supposed
to mean. P-TAU's and P-DRIFT's "no further rescue is attempted" clauses bind
the test arm and are not touched here.

The provenance constraint is stated now, before the result: because this run
was motivated *after* seeing stage 2, no outcome of it can be reported as a
pre-registered win for the verifier. The original P-DRIFT-1 verdict
(refuted-as-run on PushT, unscored on Reacher) stands in the record whatever
happens below. What this run decides is only whether the Reacher leg is
*scorable*, and if so, what it says.

## Why the specified rate was wrong, measured

The coin rejects i.i.d. at specified `p`, but realised call ratio adds
structural re-drafts (block exhaustion, episode starts/ends). The effect is
visible in the stage-2 blind arm: `tau=999` realises 0.362-0.373 against the
mechanical `1/N = 0.333`. The stage-2 coins specified `p` equal to the test's
realised ratio and therefore realised ~0.72-0.73 against the test's ~0.66:

| cell | test realised (target) | coin specified | coin realised |
|---|---|---|---|
| rho=0 | 0.6608 | 0.6608 | 0.7283 |
| rho=0.9 | 0.6625 | 0.6625 | 0.7318 |
| rho=0.99 | 0.6570 | 0.6570 | 0.7225 |

Inverting the paper's own `eq:cost` model (`ratio = q/(1-(1-q)^3)`) gives
first-order `q ~ 0.63`, but the model under-predicts the observed inflation
(predicts 0.688 realised at `q=0.6608`; observed 0.728), so the correction is
calibrated empirically rather than trusted from the model.

## Calibration procedure (fixed before any run)

1. **Smoke grid, never scored:** Reacher `t=100`, `sigma=0.1`,
   `rho in {0, 0.9, 0.99}` x `q in {0.570, 0.615}`, evaluation seed **41**
   (deliberately outside the scored 42-45), `n=128`, everything else
   byte-identical to the stage-2 coin cells. Read only the realised call
   ratio.
2. Per `rho`, linearly interpolate/extrapolate the two smoke points to the
   cell's target (the test's realised ratio above), giving one final `q` per
   cell. At most one further smoke iteration if an interpolated `q` falls
   outside [0.570, 0.615] by more than 0.02.
3. **Scored array:** the three coin cells at their final `q`, seeds 42-45,
   `n=128`, byte-identical to stage 2 otherwise. No cell excluded after the
   fact; a run with no SR line is re-run once and otherwise reported missing.

## Frozen predictions

**P-REMATCH-0 (validity gate).** In every scored cell the coin's realised call
ratio (4 seeds pooled) sits within **0.015** of that cell's target. *If any
cell misses, that cell is invalid; if all three miss, the Reacher coin is
declared unmatchable under this construction and the paper's P-DRIFT scoping
to PushT becomes permanent.*

**P-REMATCH-1 (the re-tested control, = P-DRIFT-2 on Reacher).** At `rho=0`
the matched coin ties the test within 2.0pp. *Refuted otherwise, in which case
the Reacher leg remains unscorable and the PushT scoping is permanent.*

**P-REMATCH-2 (the re-tested claim, = P-DRIFT-1 Reacher leg).** At
`rho >= 0.9` the test beats the matched coin by >= 2.0pp at some `rho`.
**Expectation, stated now: refutation.** The stage-2 coin led by 2.5-3.1pp
carrying a ~+0.07 drafting advantage worth roughly +2 to +4pp on this
environment (Extension A dose-response); removing the advantage is expected to
shrink the coin's lead toward a tie, not flip it by >= 2pp.

## Consequence rules (frozen before the run)

* **P-REMATCH-1 holds, 2 refuted.** The Reacher leg becomes scorable and
  refuting: the paper's third-control sentence extends back to both
  environments ("at matched realised rate the coin ties or beats the test at
  every `rho` in both environments"), with the stage-2 mismatch and this
  re-match both disclosed.
* **P-REMATCH-1 holds, 2 holds.** Reported as a post-hoc-motivated positive,
  labelled as such, in this document and the paper's appendix; the deleted
  "autocorrelated divergence" sentence stays deleted; the abstract's control
  summary is not upgraded. (Provenance constraint above.)
* **P-REMATCH-1 refuted (or gate fails).** The Reacher leg is reported
  unmatchable/unscorable; the paper keeps the PushT-only scoping permanently
  and says why in one clause.

## Runner

`batch/isca/run_drift_rematch_smoke.sbatch` (6 tasks), then
`batch/isca/run_drift_rematch.sbatch` (3 tasks) with the calibrated `q`
values filled in and echoed into the log header before scoring.

## Outcome (jobs 2335658/2335670 calibration, 2335673 scored, 2026-08-13)

**Calibration trail** (seed 41, never scored). Two-point smoke:
`q=0.570 -> realised 0.669/0.673/0.664`, `q=0.615 -> 0.695/0.693/0.698` at
`rho=0/0.9/0.99`. Per-rho interpolation to the targets gave
`q = 0.5558/0.5464/0.5607`; the `rho=0.9` value fell more than 0.02 outside
the bracket, triggering the one allowed further iteration, which validated all
three at seed 41 (realised 0.660/0.659/0.659).

**Scored cells** (seeds 42-45, `n=128`, means; realised ratio pooled over
seeds):

| `rho` | coin SR | coin realised | target | gate (0.015) | test SR (stage 1) | test - coin |
|---|---|---|---|---|---|---|
| 0.0 | 92.97 | 0.652 | 0.6608 | pass (0.009) | 91.21 | -1.76 |
| 0.9 | 92.97 | 0.643 | 0.6625 | **FAIL (0.019)** | 91.02 | (-1.95, not scored) |
| 0.99 | 91.41 | 0.654 | 0.6570 | pass (0.003) | 91.80 | +0.39 |

### Verdicts against the frozen predictions

* **P-REMATCH-0: 2 of 3 cells valid.** The `rho=0.9` cell misses the 0.015
  gate by 0.004 (its seed-41 validation read 0.659; seed-to-seed drift) and is
  invalid per the frozen rule. Not all cells missed, so the unmatchable clause
  does not fire.
* **P-REMATCH-1 (the re-tested control): HOLDS.** At `rho=0` the matched coin
  leads by 1.76pp, inside the 2.0pp band. The Reacher comparison is now valid.
* **P-REMATCH-2 (the re-tested claim): REFUTED, as expected.** On the valid
  `rho=0.99` cell the test leads the matched coin by only 0.39pp, far short of
  the 2.0pp bar; a tie. The invalid `rho=0.9` cell reads the same way (coin
  +1.95, reported not scored).

### Consequence applied (the holds-and-refuted rule)

The Reacher leg is scorable and refuting: at matched realised rate the coin
ties or beats the accept test at every valid `rho` in **both** environments.
`sec:results-cert` now says so, with the stage-2 mismatch (the failed `rho=0`
control) disclosed in the same sentence and the full trail in this document.
The provenance constraint was not needed: the expected refutation is what
happened, so nothing here upgrades any claim.
