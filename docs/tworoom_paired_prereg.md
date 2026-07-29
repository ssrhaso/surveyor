# TwoRoom paired-verifier: pre-registration

**Frozen 2026-07-27, before any closed-loop cell of the paired arm existed.**
tau is filled in from `Results/gap_stat/gap_tworoom_paired.json` (job 2300008)
the moment the probe lands and BEFORE the battery is submitted; the bars below
do not depend on its value.

## The problem this addresses

TwoRoom is the one substrate the method has never worked on. The diagnosis is
now specific, and it is NOT that TwoRoom is unplannable:

**Flat CEM planning on LeWM works there.** Raw anchor logs
(`logs/tworoom_anchor_flat_t{25,75}_seed{42,43}.log`, n=64 each, cross-room
successful holdout episodes >= 4000):

| horizon | seed 42 | seed 43 | mean |
|---|---|---|---|
| t=25 | 65.62 | 62.50 | **64.06** |
| t=75 | 34.38 | 20.31 | **27.34** |

(PLAN.md's shorthand "68 -> 31" overstates both ends; these are the logged
numbers.) The 36.7pp collapse with horizon is exactly the regime the method
targets.

**What is broken is the VERIFIER, not the planner.** In LeWM's TwoRoom latent
space (`Results/gap_stat/gap_tworoom_dense.json`):

| pairs | rel L2 p10 / p50 / p90 |
|---|---|
| consecutive frames (equivalent) | 0.374 / **0.876** / 1.393 |
| 10 frames apart (one subgoal hop) | 1.018 / 1.380 / 1.558 |
| different episodes (saturation reference) | 1.341 / **1.456** / 1.621 |

Equivalent frames sit 60 percent of the way to unrelated frames, and equiv p90
(1.393) is ABOVE hop p10 (1.018). No threshold separates "arrived" from "not
arrived". CEM never consults this metric, which is why flat is unharmed; the
accept test consults nothing else, which is why spec-accept cannot operate.

## The intervention

Decouple the two roles. The planner keeps consuming LeWM latents; the verifier
certifies in frozen DINOv2 space, where the same frames are metrically well
behaved (equiv p50 0.081 vs LeWM's 0.876). Both halves of a waypoint must
describe the SAME imagined future, so a single drafter emits
`z = [ z_lewm (192) | z_dino (384) ]` and serving splits it: the LeWM half goes
to the cost model, the DINOv2 half is what the accept test compares
(`specaccept/paired.py`).

This generalises the lens: `learn_readout` already certifies in a learned
readout of the planner's space rather than the space itself. Here the readout
is a frozen general-purpose encoder instead of a trained one.

**tau is derived, not tuned**: `probe_dino_gap.py` measures equiv/hop/cross in
the exact serving space (`encode_frames_dino`, pooled patch tokens, ImageNet
preprocessing) on TRAIN episodes only (< 4000; the eval holdout is untouched),
and tau = the midpoint of the open gap [equiv_criterion p90, hop p10] at S=10.
A tau lifted from the DINO-WM lens checkpoint would be a different space and is
NOT used.

## MEASURED (jobs 2300010 / 2300013, frozen 2026-07-27 before any paired cell)

Pooled DINOv2, the exact serving space, 600 train episodes / 19k frames:

| pairs | p10 / p50 / p90 |
|---|---|
| equiv, criterion (agent within 16 px) | 0.051 / **0.098** / 0.210 |
| equiv, temporal (consecutive frames) | 0.049 / 0.084 / 0.160 |
| hop S=5 | 0.058 / 0.119 / 0.252 |
| hop S=10 (serving) | 0.073 / **0.163** / 0.298 |
| hop S=25 | 0.114 / 0.231 / 0.348 |
| cross (random episodes) | 0.118 / 0.226 / 0.345 |

**The encoder swap worked**: equiv p50 0.098 against a random-pair baseline of
0.226 is real metric structure, versus LeWM's 0.876 against 1.456 (no
structure). This is the measurement that says the diagnosis was right.

**But the gap is CLOSED at every hop**: [equiv p90, hop p10] = [0.210, 0.073]
at S=10. Bulks are separated ~1.7x (0.098 vs 0.163); the TAILS overlap. That is
the reacher/wall shape.

**The prescribed lens does NOT rescue it** (job 2300013, trained at the serving
hop on these exact latents): gap width -0.080 before, -0.027 after. Narrowed,
never opened. This is the second lens failure at a serving hop, after wall h10
(-0.131). The 7/26 lens successes were all at hop 20-25; at hop 10 the hop
population sits too close to equivalence for the lens to separate.

What the lens DOES do is widen the bulk separation
(`Results/gap_stat/lens_floor_tworoom.json`, measured over the same 600
episodes, 47k frames): equiv p50 0.3617, hop10 p50 0.8190, cross p50 1.3202, so
equiv->hop goes from 1.66x in the raw space to **2.26x** in lens space. Because
`tau_derived` is null whenever the gap stays closed, the lens arm's threshold is
derived the same way as the raw one: **tau_lens = criterion floor p50 in lens
space = 0.3617**. The driver now REFUSES to run a lens arm with a null derived
tau unless one is passed explicitly, so this can never be silently invented.

**Why the tails overlap** (mechanism, measured not assumed): a 10-frame hop
does not always move the agent -- demos contain stalls and wall contacts -- so
some hop pairs are genuinely near-identical. Those are not verifier errors:
when a drafted waypoint is already where we stand, accepting it is correct.
The gap statistic's conservative [p90, p10] construction counts them against
us anyway.

**tau, from the paper's stated rule** (band = criterion floor p50 .. 1.3x p90
= [0.098, 0.273]). Two endpoints are run, both fixed in advance, no sweep:
- **tau = 0.20**, the globally frozen transfer default, which lands INSIDE
  TwoRoom's derived band. Primary.
- **tau = 0.098**, the band floor (criterion floor p50). Secondary.
The cube scope guard does not trip here (criterion/temporal = 1.16, far below
the 3x cutoff), so the criterion-floor rule is in scope on this substrate.

**TWO OF OUR OWN RULES DISAGREE, and this battery adjudicates.** The
criterion-floor rule yields a usable tau; the gap statistic says a closed gap
means inapplicable. Recording the conflict now, before the closed loop, so
whichever way it falls is a prospective result and not a retrofit.
Gap-statistic prediction on record: **the fixed-tau paired arms underperform
flat** (the same call the instrument made for wall P3).

## Protocol

Identical to the anchor above, arm-for-arm: cross-room successful holdout
episodes >= 4000, `--start final`, goal-offset t in {25, 75}, eval budget 2t,
n=64, seeds 42-43, CEM seed = eval seed, S=10, N=3, goal-conditioned drafter,
50 DDIM steps.

Four arms, so the claim carries its own controls:

| arm | what it is |
|---|---|
| `baseline` | flat CEM to the goal latent (the anchor) |
| `gdm` | draft every replan, no verification (drafting without certification) |
| `specaccept` tau=0.20 | verify in LeWM space (PREDICTED DEGENERATE-REJECT) |
| `specpaired` tau=0.20 | verify in frozen-DINOv2 space, transfer default (primary) |
| `specpaired` tau=0.098 | same, at the band floor (secondary) |
| `specpaired` + lens | the prescribed mechanism, run even though its gap stayed closed |

## FROZEN BARS

- **T1 (applicability).** `specpaired` mechanics are non-degenerate:
  call_ratio strictly inside (0.05, 0.95). The verifier must actually both
  accept and reject.
- **T2 (control, isolates the verifier).** `specaccept` at the same tau IS
  degenerate: call_ratio >= 0.95, and its SR is within noise of `gdm`. This is
  the measurement proving the paired SPACE is what changed the outcome, not the
  drafter and not the extra parameters.
- **T3 (short horizon).** `specpaired` >= flat - 2pp at t=25.
- **T4 (long horizon, the headline).** `specpaired` >= flat + 5pp at t=75,
  where flat collapses to 27.34.
- **T5 (efficiency).** `specpaired` call_ratio < 1.0, i.e. it reaches its SR
  with strictly fewer drafter calls than `gdm`. Reported as NFE/replan, not
  barred.

## Declared readings of the possible outcomes

Written now so no result can be re-framed after the fact.

- **T3 and T4 pass** -> the method works on TwoRoom, and the enabling claim is
  that the verifier is MODULAR: when the planner's own latent space cannot
  certify, certification transplants to any space with an open gap.
- **T3 passes, T4 fails, and `gdm` also fails to beat flat at t=75** ->
  drafting itself has no headroom on TwoRoom; the honest claim is confined to
  verifier portability (spec-accept becomes applicable and stays SR-neutral at
  reduced drafter cost), and TwoRoom is a bounded row, not a win. This outcome
  must NOT be reported as a TwoRoom win.
- **T3 passes, T4 fails, but `gdm` DOES beat flat at t=75** -> drafting helps
  but the verifier is discarding good waypoints; report as a tau/mechanism
  miss with the call_ratio and rel-L2 distributions as evidence.
- **T1 fails** -> tau is outside the usable band in the serving space; report
  the gap probe and stop. Do not re-tune tau against a closed-loop number.
- **T2 fails** (LeWM-space verifier is NOT degenerate) -> the diagnosis above
  is wrong and the whole framing needs revisiting before any claim is made.

## AMENDMENT: stride saturation, and a second frozen prediction (2026-07-27)

Measured after the first battery, before the repaired-drafter rerun.

The first battery collapsed because both TwoRoom drafters emitted latents about
100x too large (undertrained: 940 gradient steps). That is now fixed (400
epochs, norm ratio 0.934). But repairing it exposed a deeper fault that no
drafter can fix.

`specaccept/probes/probe_stride_saturation.py` measures, per stride S, how far
a frame S ahead sits relative to two UNRELATED frames. The planner scores CEM
candidates by terminal L2 to the served subgoal, so once that ratio approaches
1.0 the subgoal is indistinguishable from noise and drafting cannot help at any
drafter quality. This is a different question from the verification gap, which
asks whether ARRIVAL is detectable.

| substrate | tuned S | saturation at tuned S | largest S with sat <= 0.80 |
|---|---|---|---|
| PushT (method wins) | 10 | 0.526 | 15 |
| Reacher (method wins) | 10 | 0.639 | 15 |
| Cube (method wins) | 15 | 0.624 | **15, an exact hit** |
| **TwoRoom on LeWM** | 10 | **0.988** | **2** |

Every substrate where the method works serves its subgoal at 0.46 to 0.64
saturation. TwoRoom on LeWM at S=10 sits at 0.988: the drafted waypoint is
98.8 percent of the way to being a random latent as far as the planner's cost
function is concerned. Its encoder is already at 0.626 for CONSECUTIVE frames,
so there is no stride that is both informative and far enough ahead to
decompose anything. S=2 is inside the range but spans about 10 px of motion
against a task needing roughly 190 px.

This also supplies a mechanism for a finding the paper currently reports as
purely empirical, the interior optimum in S: the upper arm of that curve is
subgoal saturation. PushT S=25 is at 0.904, Reacher S=25 at 0.909, Cube S=25 at
0.803, and all three were measured worse than their tuned S.

**FROZEN PREDICTION (S1), before the repaired-drafter rerun:** with the norm
fault fixed, TwoRoom on LeWM at S=10 STILL fails to beat flat at either
horizon, because the subgoal is saturated in the planner's space. Passing this
makes the negative mechanistic rather than asserted; failing it refutes the
saturation reading and is the more interesting outcome.

Caveat to state whenever this is used: the 0.80 cutoff is a chosen threshold,
not a derived one. The defensible claim is the ordering and the separation
(three working substrates at 0.46 to 0.64, one failing substrate at 0.99), not
a precise boundary.

## AMENDMENT: anchor CLOSED; anchored-protocol re-run + best-of-k (2026-07-29)

### Anchor ablation 2304691, MEASURED (t=25, three seeds per variant)

| variant | difference peeled | seeds 42/43/44 | mean |
|---|---|---|---|
| V0 | ours as run | 65.62 / 62.50 / 67.19 | 65.10 |
| V1 | drop cross-room filter | identical to V0 | 65.10 |
| V2 | also drop success filter | 89.06 / 75.00 / 75.00 | 79.69 |
| V3 | also drop holdout | 89.06 / 73.44 / 78.12 | 80.21 |
| V4 | also start random | 82.81 / 75.00 / 89.06 | 82.29 |
| V5 | also goal_proprio, n=50 | 86.00 / 82.00 / 88.00 | **85.33** |

**Bar was V5 >= 82: PASSED.** LeWM's published 87 sits inside our V5 seed
range (82-88). The 21pp reproduction gap is protocol, not the checkpoint:
success filter ~14.6pp, start source ~2.1pp, goal source + n ~3.0pp,
cross-room 0pp (vacuous: the filter removed no episode already passing the
success filter -- success is a subset of cross-room in this dataset), holdout
~0.5pp. Two honest notes recorded with it: (1) the success filter was meant
as hygiene and instead selects a strictly HARDER population; (2) with
`goal_state` on unfiltered episodes the goal IMAGE (agent at demo end)
mismatches the scored target, so LeWM's `goal_proprio` is not just their
protocol, it is the coherent one on the unfiltered population.

### The ANCHORED protocol (long horizon), fixed before any cell runs

`--mode long --goal-offset 75 --eval-budget 150 --eval-filter none
--episode-min 4000 --goal-from-proprio --num-eval 64`, CEM seed = eval seed.
Holdout is kept (drafters train on episodes < 4000; measured cost ~0.5pp).
Mode long forces start=final. No cross-room filter. Everything else as the
2026-07-27 protocol.

### Stage A (fired first, no bars -- it SETS the bar): flat RH in {1,2,5}
x seeds 42-47, plus oracle stride-25 RH=1 x seeds 42-47 (the ceiling row).

Predictions recorded before it runs:
- P-A1: anchored flat t=75 reads HIGHER than the filtered-protocol 43-44
  (the unfiltered population was easier at t=25 by ~15pp).
- P-A2 (risk, stated plainly): under goal_proprio the t=75 goal is "where a
  mostly-failed demo wandered to 75 steps later", which may be close to the
  start; decomposition headroom at t=75 may therefore SHRINK relative to the
  filtered protocol. The oracle row adjudicates this before Stage B is read.

**Stage B gate:** Stage B margins are only meaningful if oracle >= strongest
flat + 3pp at six seeds. If the gate fails, the anchored t=75 population has
no decomposition headroom, the outcome is declared a bounded scope row
(the +5.47pp result stands but stays internal to the filtered protocol,
reported next to the anchor attribution table), and no Stage B margin may be
quoted as a win.

### Best-of-k drafting, registered before any run

Mechanism (`--best-of-k`, `specaccept/paired.py`): sample k candidate blocks
per re-draft, serve the one scoring best in the DINOv2 half. Same principle
as the two mechanisms that measured positive (decisions in the structured
space, execution in the planner's space). Two zero-constant scoring rules:
`goal` (final waypoint nearest goal) and `feas` (first waypoint nearest
current state). k* derived OFFLINE by `probe_bok.py`: smallest k in
{2,4,8,16} capturing >= 80% of the k=16 median-score gain over k=1.
Offline kill criteria (either kills it before any closed loop): no k=16 gain
to select on; or selection degrades the LeWM half's fidelity to the true
future (the planner would be served worse waypoints).

### Stage B arms (six seeds each, fired only after Stage A + probe land)

flat(best RH from Stage A) | pair no-verify | pair+verify tau=0.098 |
pair+verify+bok(k*, best offline rule) | pair+verify tau=0.20 (3 seeds,
robustness row, unbarred).

### FROZEN BARS (Stage B)

- **B1 (headline).** pair+verify tau=0.098 >= strongest flat + 4pp at six
  seeds. Extension rule: if it passes, run seeds 48-53 for both arms; the
  headline requires pooled 12-seed margin >= +3pp AND t >= 2.
- **B2 (best-of-k adds value).** pair+verify+bok >= pair+verify + 2pp at six
  seeds; otherwise best-of-k is recorded refuted and does not recur.
- **B3 (verification is cheap).** pair+verify >= pair no-verify - 2pp with
  call_ratio < 1.0; verification must retain SR while cutting drafter calls.

Declared readings: B1 pass -> TwoRoom joins PushT/Reacher as a
horizon-extension win, now at a protocol anchored to LeWM's published number,
with the representational story quotable. B1 fail with pair no-verify beating
flat -> verification discards good waypoints on this population; mechanism
miss, reported with call_ratio and rel-L2 distributions. Both fail (or Stage B
gate fails) -> TwoRoom is a scope row; the honest deliverable is the anchor
attribution table + the measured representational deficit + the filtered-
protocol +5.47pp as an internal result.

## Log

- 2026-07-27: diagnosis re-confirmed from raw logs and gap JSONs; paired
  verifier implemented (`specaccept/paired.py`, opt-in `wants_frames` hook in
  `sources.py`, `--dino-pair` in the TwoRoom builder, `specpaired` arm in the
  TwoRoom driver); prep job 2300008 submitted (gap probe -> paired subgoals ->
  576-d joint drafter). Bars above frozen before it returned.
- 2026-07-27 (later): prep landed (`gdm_tworoom_s10_gc_paired.pt`, 576-d, 20
  epochs, final loss 0.199); gap measured at 250 and again at 600 episodes with
  the same reading; lens job 2300013 landed CLOSED; lens-space criterion floor
  measured (tau_lens 0.3617); paired source dry-run passed on CPU (served shape
  (n,192), lens path, and the `wants_frames` hook confirmed inert for every
  other source). Battery **2300025** submitted: 24 cells, six arms. An earlier
  submission (2300017) was cancelled two cells in -- before any lens cell ran --
  solely to fold in the explicit lens tau; no number from it is used.
- 2026-07-29: anchor ablation 2304691 read (table above); amendment frozen
  BEFORE any anchored-protocol long-horizon cell or best-of-k sample existed.

## STAGE A MEASURED + STAGE B FINALIZED (2026-07-29 afternoon, before any
## Stage B cell ran)

Stage A (2304722, anchored t=75, six seeds/arm): flat RH1 41.67, RH2 **42.71**
(strongest), RH5 38.54; oracle s25 RH1 **57.81**.
- **Stage B gate PASSED**: oracle - best flat = +15.1pp >= +3pp. Headroom at
  the anchored protocol is LARGER than at the filtered one (+8.6).
- **P-A1 REFUTED, recorded**: anchored flat t=75 (42.71) is NOT above the
  filtered-protocol 43.4-44.0; the population effect that lifted t=25 by
  ~15pp does nothing at t=75. The success-filter cost is horizon-dependent.
- P-A2 did not materialize (the gate passing is its direct test).

Best-of-k probe (2304723, filtered drafter): both rules derive **k* = 8**
(sel-score p50 0.22 -> 0.13, >= 80% of the k=16 gain); no kill criterion
trips (selection does not move the LeWM half beyond noise). Probe on the
nofilt drafter (2304733): same reading, k* = 8.
**Rule choice, recorded with reason BEFORE closed loop:** primary bok arm =
`goal`. The `feas` rule (argmin first-hop distance) selects
smallest-first-hop blocks, the stay-put pathology that regressed the V-JEPA 2
v2 residual drafter; its better offline m+1 fidelity (0.137 vs 0.184) does
not outweigh that documented failure mode. `feas` runs as a 3-seed
exploratory row, unbarred.

Nofilt drafter chain (2304733): 39,637 pairs (3.36x), 400 epochs, loss
0.0346, **norm gate PASSED (1.003 lewm / 1.002 dino)**. Enters Stage B as
its own arm (same tau=0.098: the derivation is an encoder/criterion
property, drafter-independent).

### Stage B arms as fired (amendment BEFORE submission; bars concrete now
### that the flat bar exists)

| arm | seeds | bar |
|---|---|---|
| pair no-verify (filtered, tau=0) | 6 | B3 reference |
| pair+verify tau=0.098 (filtered) | 6 | **B1: >= 46.71 (flat 42.71 + 4)** |
| pair+verify + bok k=8 goal (filtered) | 6 | **B2: >= B1 arm + 2pp** |
| pair+verify tau=0.20 (filtered) | 3 | robustness, unbarred |
| pair+verify tau=0.098 (NOFILT drafter) | 6 | **B4: >= B1 arm + 2pp, else null** |
| pair+verify + bok k=8 feas (filtered) | 3 | exploratory, unbarred |

B1 extension rule unchanged (pass -> seeds 48-53 both arms, pooled 12-seed
margin >= +3pp and t >= 2). B3 unchanged (verify within 2pp of no-verify at
call_ratio < 1).

Amendment, logged before any Stage B cell was read: the extension seeds
48-53 (flat RH2 + pair tau=0.098) are fired UNCONDITIONALLY alongside Stage
B rather than after its 6-seed read. This only adds data and removes a
conditional branch; the pooled 12-seed bar applies exactly as written, and
the 6-seed B1 verdict is still read first and reported.

## STAGE B VERDICT (2306077, read 2026-07-29 evening; all six-seed means,
## anchored protocol t=75 RH=2, seeds 42-47 unless noted)

| arm | mean | bar | verdict |
|---|---|---|---|
| flat RH2 (Stage A) | 42.71 | - | reference |
| oracle s25 RH1 (Stage A) | 57.81 | - | ceiling |
| pair no-verify | 56.25 | B3 ref | - |
| pair+verify tau=0.098 | **55.47** | >= 46.71 | **B1 PASSED** |
| pair+verify + bok k=8 goal | **58.33** | >= 57.47 | **B2 PASSED** |
| pair+verify tau=0.20 (3 seeds) | 52.08 | unbarred | robust, below t098 |
| pair+verify NOFILT drafter | 53.65 | >= 57.47 | **B4 FAILED - null** |
| pair+verify + bok k=8 feas (3 seeds) | 32.29 | unbarred | **catastrophic** |

- **B1: +12.8pp over flat, wins on ALL SIX seeds** (per-seed paired diffs
  +20.3/+12.5/+15.6/+14.1/+6.3/+7.8; paired t ~ 5.9). More than double the
  filtered-protocol margin (+5.47). 12-seed extension 2306141 in flight.
- **B2: bok(goal) >= pair on every seed** (+3.1/+3.1/+3.1/0.0/+4.7/+3.1;
  paired t ~ 4.5) and its mean sits AT the oracle ceiling (58.33 vs 57.81;
  oracle is RH1 + static-table staleness, so a soft ceiling).
- **B3 PASSED**: verify within 2pp of no-verify (-0.78) at call_ratio
  0.939-0.968 (advances 22-47 per ~700 replans; verification distance p50
  ~0.21 vs tau 0.098, rejects dominate as expected on this substrate).
- **B4 null, recorded**: 3.36x distribution-matched data does NOT beat the
  success-filtered pool (-1.8pp vs the same arm on the filtered drafter).
  The success filter helps as a TRAINING curator even though it distorts
  EVAL populations.
- **feas rule: the pre-recorded stay-put pathology is confirmed, -23pp vs
  the goal rule** despite BETTER offline m+1 fidelity (0.137 vs 0.184).
  Third demonstration in this program that offline fidelity does not track
  closed-loop value, and the first one predicted in advance from a named
  failure mode.
- tau=0.20 robustness: -3.4pp vs the derived tau=0.098 but still +9.4 over
  flat; the derived-tau prescription holds on this substrate.
