# P-DRIFT: is verification load-bearing against *correlated* divergence?

**Frozen 2026-08-11, before the code change below was written and before any
cell was run.**

## Why this exists

Two pre-registered controls have now shown the accept test does not carry
closed-loop success at the threshold we serve:

* `2026-08-01_random_rejection.md`: a coin at the test's own measured rate ties
  SURVEYOR-Base on both `t=150` headline cells.
* `2026-08-11_rate_transfer.md`: one *fixed* rate matches it at every horizon in
  both dev environments (P-RATE-1b refuted twice).
* `2026-08-11_tau_per_env.md`: serving each environment at its own derived
  floor recovers up to +6.6pp but still misses the bar (P-TAU-1 refuted).

The paper reports all three. It also contains one unmeasured assertion, in
`sec:results-cert`: under the corruption sweep the call ratio ignited but
success stayed flat, which we explain as *"verification's value is against
autocorrelated divergence, not white noise."* That sentence is a hypothesis we
have never tested, and it is the last place a success role for the accept rule
could survive, because it is the one regime a fixed-rate rejector structurally
cannot serve: a coin rejects at rate `p` whatever reality does, whereas the
test's ignition is conditioned on the realized discrepancy.

This document registers the direct test. If it fails, the accept rule's success
contribution is closed in every regime we can measure, and the paper says so.

## The mechanism under test, and the code change it needs

`--draft-noise sigma` (`sources.py:380-385`) currently displaces every drafted
waypoint by `sigma * ||w||` along a **fresh** unit direction at every draft,
including re-drafts. That is white noise by construction, so consecutive
corruptions cancel in expectation and any consumption policy averages them out
— which is exactly why SR stayed flat.

The change: carry one unit drift vector per environment and evolve it
`d <- rho*d + sqrt(1-rho^2)*u`, renormalised, shared across the `N` waypoints of
a block, so corruption persists across re-drafts instead of resampling.
`rho=0` reduces to the existing draw **exactly** and is the control arm.
New flag `--draft-noise-rho`, default `0.0`.

**Engagement smoke test, run before any cell and with its criterion fixed
here:** at `sigma=0.2`, `rho=0.99` must produce a materially higher
cosine similarity between consecutive drafts' injected displacement than
`rho=0` (target: `>0.9` against `~0` for white noise). If the two are
indistinguishable the flag is inert, and **no array is submitted** — the same
failure that made `run_commit_depth.sbatch` return fifteen mislabelled replicas
of one configuration.

## Design

PushT and Reacher, seed-2222 confirmation populations, seeds 42--45, single
20deg/joint scoring pass, everything byte-identical to the banked P-RATE arms
except the drift.

* `rho in {0, 0.9, 0.99}` crossed with `sigma in {0.1, 0.2}`.
* Three arms per cell, on identical episodes: the `tau=0.20` accept test; an
  i.i.d. coin matched to that cell's realized call ratio; blind commitment at
  the safe depth.
* Horizons: PushT `t=100`, Reacher `t=100`. One mid-range horizon each, chosen
  now, because this tests a mechanism and not a horizon profile.

## Frozen predictions

**P-DRIFT-1 (the claim).** At `rho >= 0.9`, the accept test beats the
rate-matched coin by `>= 2.0`pp in at least one environment. *Refuted if the
gap is under 2.0pp in both.*

**P-DRIFT-2 (the control).** At `rho = 0`, test and coin tie within 2.0pp,
reproducing the banked result. *Refuted otherwise* — and a refutation here
invalidates the comparison, so P-DRIFT-1 is not scored if P-DRIFT-2 fails.

**P-DRIFT-3 (the dose).** The test-minus-coin margin is monotone
non-decreasing in `rho` within each environment at fixed `sigma`.

## Consequence rules (frozen before the run)

* **P-DRIFT-1 holds.** The paper states, as measured, that verification's
  success value appears against correlated divergence and is absent in
  distribution — the abstract and `sec:results-cert` gain one scoped positive
  claim, still with the in-distribution refutations intact. The existing
  "autocorrelated divergence" sentence stops being an assertion and cites this.
* **P-DRIFT-1 refuted.** The accept rule's success contribution is closed in
  every regime we have tested. The sentence *"verification's success value is
  against autocorrelated divergence, not white noise"* is **deleted** from
  `sec:results-cert` rather than left as speculation, and the limitation is
  stated plainly: we found no operating regime in which the test's decisions
  beat a rate-matched coin. **No further rescue is attempted** — this is the
  third and last.
* **P-DRIFT-3 refuted while 1 holds.** Report the effect as present but not
  dose-ordered, and do not claim a mechanism beyond existence.

No cell is excluded after the fact. A run that produces no SR line is re-run
once with the identical command and otherwise reported missing.

## Runner

`batch/isca/run_drift.sbatch` (to be written after the smoke test passes).

## Interpretation recorded before any cell (2026-08-12)

The design above contains one internal conflict, resolved here in writing
*before* the mechanism was implemented and before any cell was run, so the
choice cannot be argued backwards from a result.

It asks for a drift vector "shared across the `N` waypoints of a block", and
also says "`rho=0` reduces to the existing draw **exactly**". Those cannot both
hold: the existing draw is *per waypoint* (`N` independent directions per
block), so a shared vector at `rho=0` is not it.

**Chosen: always shared, `rho=0` being a fresh shared draw.** Under this
reading `rho` is the only quantity that differs between arms, so the
dose-response of **P-DRIFT-3** measures autocorrelation alone. Under the
literal alternative, `rho=0` and `rho>0` would differ in *two* ways at once
(per-waypoint vs shared, and uncorrelated vs correlated), and a positive
P-DRIFT-1 could then be produced by block coherence rather than by
autocorrelation, which is precisely the claim under test.

The cost is that `rho=0` is no longer bit-identical to the banked corruption
sweep. It remains white noise, so **P-DRIFT-2** should still reproduce the
banked tie behaviourally, and P-DRIFT-2 failing is already defined above as
invalidating the comparison. That is the intended check on this decision.

## Engagement smoke test: **PASS** (2026-08-12, before any array)

Run as `surveyor/test_draft_drift_cpu.py`, on the realised displacement
directions rather than inferred from SR. Criterion as fixed above (`sigma=0.2`,
`rho=0.99` cos `>0.9` against `~0`):

| `rho` | mean cos between consecutive drafts |
|-------|-------------------------------------|
| 0.0   | **-0.023** |
| 0.9   | **+0.899** |
| 0.99  | **+0.990** |

Measured cos tracks `rho` to three decimals, as AR(1) predicts. The knob
engages and is monotone, so P-DRIFT-3 can be attributed to the mechanism if it
holds. Two further guards are pinned in the same file: the corruption is one
unit direction per env shared over all `N` waypoints at every `rho`, and at
`sigma=0` no corruption is injected and no drift state is allocated, so every
banked cell in the paper is untouched.

## Stage 1 result, and a scoping decision recorded before stage 2 (2026-08-12)

The design specifies three arms per cell. The other two are not runnable until
the accept arm has run: the coin must be matched to each cell's **realised**
call ratio, which moves with the corruption and cannot be reused from the
uncorrupted cells; and blind commitment is "at the safe depth", which on
Reacher was still being measured by Extension A of the P-RATE document. Stage 1
therefore ran the accept arm alone and harvested the rates stage 2 needs.
Job `2334493`, `t=100`, seeds 42-45, `n=256`/`128`.

| env | `sigma` | `rho=0` | `rho=0.9` | `rho=0.99` |
|-----|---------|---------|-----------|------------|
| PushT | 0.1 | 96.39 (r .674) | 95.99 (r .680) | 95.21 (r .679) |
| PushT | 0.2 | 95.21 (r 1.000) | 94.63 (r 1.000) | 94.82 (r 1.000) |
| Reacher | 0.1 | 91.21 (r .661) | 91.02 (r .663) | 91.80 (r .657) |
| Reacher | 0.2 | 93.94 (r .990) | 95.51 (r .990) | 95.70 (r .990) |

**Scoping decision, fixed here before stage 2 runs: `sigma=0.2` is dropped as
degenerate, and stage 2 runs at `sigma=0.1` only.** At `sigma=0.2` the call
ratio saturates at 1.00/0.99, i.e. total rejection, reproducing the banked
corruption sweep. A test that rejects every boundary consumes nothing and *is*
every-step drafting, and a coin matched at rate 1.0 is byte-identical to it by
construction. Those cells cannot discriminate test from coin whatever the
answer, so scoring them would pad the grid with six guaranteed ties. This is a
statement about the arm being degenerate, not about the result, and it is
recorded before the comparison is run.

**Adverse early signal, stated now rather than after stage 2.** At the
informative `sigma=0.1`, the realised call ratio does **not** move with `rho`
(PushT .674/.680/.679; Reacher .661/.663/.657). The test is not igniting more
under correlated drift than under white noise. If ignition does not respond to
`rho`, it is hard to see how its *decisions* could beat a rate-matched coin,
which is exactly P-DRIFT-1. Absolute SR is also flat to mildly *decreasing* in
`rho` on PushT. Stage 2 is still run, because P-DRIFT-1 is a test-vs-coin
comparison rather than an absolute-SR one, but the expectation going in is
refutation, and that expectation is on record.

**Blind-commitment arm, deviation recorded.** "The safe depth" is `d=2` on
PushT but does **not exist** on Reacher: Extension A of the P-RATE document
measured every depth failing somewhere there. Additionally
`DSparkSubgoalSource` carries no corruption path, so a depth-`d` arm under
injected noise is not available without a further change to a shared class that
holds banked results. Stage 2 therefore uses the codebase's existing blind
idiom, `--accept-tau 999` (accept everything, serve the full block), which is
blind consumption at `d=N` rather than at the safe depth. It remains a genuine
"does not look at reality" control; it is simply not the depth the document
named, and P-DRIFT-1 does not depend on it.

## Stage 2 arm-engagement check, and a caveat recorded before scoring (2026-08-12)

Both arms verified live on the first cells, before any comparison was read.

* **Blind arm** (`--accept-tau 999`): realised call ratio **0.368**, i.e. `1/N`
  at `N=3`. It accepts everything and serves the full block, as intended.
* **Coin arm**: engages, but **is not rate-matched in realised terms**.
  Specified `p=0.6735` (the stage-1 accept arm's realised ratio) produces a
  realised ratio of **0.718--0.739**, about 6pp *above* the arm it controls
  for.

The cause is structural rather than a bug. The coin rejects with probability
`p` at each verification event, but a block is *also* re-drafted on exhaustion,
so the realised ratio exceeds `p`. Inverting the cost model of the paper's own
`eq:cost`, `ratio = q/(1-(1-q)^N)` at `N=3`, the rate needed to *realise* 0.674
is `q ~ 0.645`, not 0.6735.

**Decision: the cells stand as run, and the caveat is reported with them.**
Re-running at `q~0.645` would make stage 2 inconsistent with the banked
`2026-08-01_random_rejection.md` control, which matched the coin to the
measured call ratio under the same convention; consistency with the result this
one extends is worth more than the 6pp.

**Direction of the bias, stated now.** The coin receives *more* drafting than
the test, and on these environments more re-anchoring helps (Extension A of the
P-RATE document: every-step drafting beats the accept rule on Reacher by up to
+9.37pp). So the bias runs **in the coin's favour**. If the test beats the coin,
it does so despite the handicap and P-DRIFT-1 is conservative. If the coin ties
or beats the test, some unquantified part of that is the extra drafting rather
than the decision content, and the refutation must be reported with that
qualification rather than as a clean win for the coin.

## Outcome (stage 2, job 2334555, 12/12 tasks, harvested from logs 2026-08-13)

*Appended during the 2026-08-13 audit: the numbers below were in the paper but
this section had been left empty, against convention. Means over seeds 42--45,
`t=100`, `sigma=0.1`; test = the stage-1 accept arm (job 2334493), same
episodes and seeds.*

| env | arm | `rho=0` | `rho=0.9` | `rho=0.99` |
|-----|-----|---------|-----------|------------|
| PushT (n=256) | test | 96.39 | 95.99 | 95.21 |
| PushT | coin | 96.58 | 96.68 | 97.07 |
| PushT | blind (`tau=999`) | 96.48 | 95.99 | 96.10 |
| Reacher (n=128) | test | 91.21 | 91.02 | 91.80 |
| Reacher | coin | 93.75 | 94.14 | 94.53 |
| Reacher | blind | 80.86 | 81.05 | 81.05 |

Realised coin call ratios: PushT 0.718--0.739 (specified 0.6735/0.6803/0.6790),
Reacher 0.716--0.741 (specified 0.6608/0.6625/0.6570) — the ~6pp over-realisation
recorded in the engagement check above, present in every coin cell.

### Verdicts against the frozen predictions

* **P-DRIFT-2 (the control): HOLDS on PushT, REFUTED on Reacher.** PushT
  `rho=0`: test 96.39 vs coin 96.58, a 0.19pp tie. Reacher `rho=0`: test 91.21
  vs coin 93.75, **2.54pp**, outside the frozen 2.0pp band. Per the frozen
  rule, the Reacher comparison is invalidated and **P-DRIFT-1 is not scored on
  Reacher**. The direction is exactly the recorded bias: the coin over-drafts,
  and on Reacher more drafting helps (Extension A: `d=1` +6.5 to +9.4pp).
* **P-DRIFT-1 (the claim): REFUTED on PushT**, the only environment with a
  valid comparison. The test never beats the coin at any `rho`; it trails by
  0.19/0.69/1.86pp at `rho=0/0.9/0.99`, and it is the **coin** that improves
  with correlation (96.58 -> 97.07). The PushT refutation is *conservative*:
  on PushT extra drafting hurts at this horizon (`d=1` trails the adaptive arm
  by 2.34pp at `t=100`), so the coin's over-realisation biased it downward and
  it won anyway.
* **P-DRIFT-3 (the dose): REFUTED.** The test-minus-coin margin is monotone
  *decreasing* in `rho` on PushT (-0.19/-0.69/-1.86). Moot given P-DRIFT-1.

### Consequences applied

The P-DRIFT-1-refuted rule fires on the PushT evidence: the "verification's
value is against autocorrelated divergence, not white noise" sentence is
deleted from `sec:results-cert`, and the limitation is stated. Applied to
`main_workshop_final.tex` with the scope the data licenses: the claim is
stated **on PushT only**, with the realised-rate caveat reported per the
stage-2 decision above, and the Reacher leg reported as failing its `rho=0`
control and unscored. An incidental observation, reported not claimed: under
injected corruption on Reacher, blind consumption collapses (~81 vs ~91-95 for
test/coin), so *some* re-anchoring policy is load-bearing there; *which*
boundaries re-anchor is not.

### Follow-up registered

`2026-08-13_drift_coin_rematch.md` freezes a realised-rate-matched Reacher
coin re-run (control hygiene, not a rescue: the test arm is untouched) to
decide whether the Reacher leg can be scored at all.
