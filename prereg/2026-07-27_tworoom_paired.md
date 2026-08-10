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
(`surveyor/paired.py`).

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
| `surveyor` tau=0.20 | verify in LeWM space (PREDICTED DEGENERATE-REJECT) |
| `specpaired` tau=0.20 | verify in frozen-DINOv2 space, transfer default (primary) |
| `specpaired` tau=0.098 | same, at the band floor (secondary) |
| `specpaired` + lens | the prescribed mechanism, run even though its gap stayed closed |

## FROZEN BARS

- **T1 (applicability).** `specpaired` mechanics are non-degenerate:
  call_ratio strictly inside (0.05, 0.95). The verifier must actually both
  accept and reject.
- **T2 (control, isolates the verifier).** `surveyor` at the same tau IS
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

`surveyor/probes/probe_stride_saturation.py` measures, per stride S, how far
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

Mechanism (`--best-of-k`, `surveyor/paired.py`): sample k candidate blocks
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
  verifier implemented (`surveyor/paired.py`, opt-in `wants_frames` hook in
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

## 12-SEED POOL (extension 2306141, read 2026-07-30 morning): HEADLINE BANKED

flat RH2 (seeds 42-53): 41.67. pair+verify tau=0.098 (seeds 42-53): **57.03**.
**Margin +15.4pp, positive on ALL TWELVE seeds** (paired t ~ 10.9, unpaired
t ~ 6.2) vs the frozen bar (>= +3pp, t >= 2): **PASSED**. The extension
seeds were STRONGER for spec (58.6) and slightly weaker for flat (40.6), so
no regression-to-mean shrinkage (the failure mode of both retracted July
margins). 57.03 is statistically AT the oracle ceiling (57.81, RH1 static
table). TwoRoom is the third horizon-extension win, at the anchored
protocol.

## THEIR-PROTOCOL BATTERY (registered 2026-07-30 night, BEFORE any cell ran):
## spec-accept AT LeWM's OWN published TwoRoom setup

Mission (user): replicate LeWM's Fig. 6 TwoRoom number (DONE: V5 = 85.33,
their 87 in seed range) and now field the best certified spec-accept AT
THAT PROTOCOL. Protocol (paper App. F.1/D, config verbatim): start sampled
randomly from a dataset trajectory, goal = the state 25 steps later in the
SAME trajectory (goal_proprio), budget 50, CEM 300/30/30 var 1.0, horizon 5
blocks, RH = full 5-block execution; our additions: eval-filter none,
episode-min 4000 (holdout hygiene, ~0.5pp), n=64, seeds 42-47.

Paper fact folded in (Tab. 3): agent position is PERFECTLY probeable from
LeWM's TwoRoom latent (MLP MSE 0.000, r 1.000) -- the information is
present, the METRIC is what our gap probe shows broken. Their own probing
table is evidence for our mechanism split (information vs metric).

NEW MECHANISM, registered before running: the PROXIMITY ROUTER
(--route-goal-hop, paired.py). At each replan, if the goal itself verifies
within one hop in the DINOv2 half (rel L2 <= hop-S10 p50 = 0.163, the gap
probe's measured hop scale -- derived, not tuned), serve the goal directly;
an intermediate waypoint cannot help inside one serving stride, and the
oracle measured decomposition NEGATIVE at short horizon. c*-retire
transplanted to the verification space; unlike the null arrival gate it is
re-evaluated every replan (routing, not retirement).

Arms (t=25, six seeds each): flat RH5 (their config) | flat RH2 (sweep) |
oracle s10 RH5 (short-horizon decomposition ceiling) | pair+verify t098 RH2
(no router: the honest exposure of the oracle logic) | +bok8-goal RH2 |
+bok8+router RH2 | +bok8+router RH5 (their exact serving config).

FROZEN BARS AND PREDICTIONS:
- **P25-1 (prediction, recorded): router-OFF spec arms LOSE to flat at
  t=25** (the oracle logic; if oracle-s10 itself reads below flat, pure
  decomposition cannot win here and the router is the only path).
- **B25-1 (primary, non-inferiority): best router arm >= best flat - 2pp.**
- **B25-2 (aspirational): best router arm > best flat** (any positive
  margin; powered follow-up before quoting if it lands within noise).
- Reading either way: combined with the banked +15.4pp at t=75, the paper
  claim becomes "ties LeWM's own short-horizon protocol, beats it decisively
  at range" -- the Reacher shape, at a protocol anchored to their Fig. 6.

### VERDICT (2306523, all 42 cells COMPLETED same night, read 2026-07-30)

flat RH5 (their config) **84.89** | flat RH2 80.99 | oracle-s10 **63.54** |
pair RH2 70.31 | +bok 72.40 | +bok+router RH2 69.27 | +bok+router RH5 77.08.

- **P25-1 CONFIRMED** (router-off spec loses by 12-15pp).
- **Oracle -21pp vs flat**: at their protocol the goal sits INSIDE one
  planning window (25 steps = their horizon exactly); even ground-truth
  waypoints are counterproductive there. No drafter can fix this; it is the
  regime law's window leg, measured at the anchored protocol.
- **B25-1 FAILED** (77.08 vs bar 82.89; the router recovered 70->77 but not
  parity). B25-2 failed. Recorded as-is. Mechanism notes: RH5 >> RH2 for
  every arm at t=25 (replanning itself taxes the short game); router helped
  at RH5, slightly hurt at RH2.
- **RESULTING CLAIM (no further compute): the WINDOW RULE composite.**
  goal_offset is a known task parameter in this protocol, so the regime
  law's first leg applies with zero constants: goal within one planning
  window -> plan flat (84.9, ties their published 87 within noise); beyond
  -> certified spec-accept (57.0 vs flat 41.7 at t=75, +15.4pp, 12 seeds).
  TwoRoom is now measured on BOTH sides of the crossover at protocols
  anchored to LeWM's Fig. 6; the deployed policy never loses to flat.

## CROSSOVER SWEEP (registered 2026-07-30 night, before running; DESCRIPTIVE,
## no bars -- it maps the composite's switch point)

t in {40, 50, 60} at the anchored long protocol (eval-filter none,
episode-min 4000, goal_proprio, budget 2t, n=64, seeds 42-47); arms = flat
RH2, flat RH5, spec(pair+verify t098) RH2, spec RH5. Endpoints already
banked: t=25 flat 84.9 vs spec 77.1; t=75 flat 41.7 vs spec 57.0.
**Frozen prediction (falsifiable, from the window rule): flat still ahead
at t=40; spec ahead by t in [50, 60]; crossover where the goal exits the
~25-step planning window plus one serving stride.** Also fired: bok-arm
extension seeds 48-53 at t=75 (12-seed backing for the B2 sentence) and
their-protocol flat RH5 seeds 48-53 (12-seed symmetry for the composite
table).

### CROSSOVER + EXTENSIONS VERDICT (2306577, read 2026-07-31 morning)

Best-arm means (6 seeds): t=40 flat 65.36 vs spec 60.42 (flat +4.9); t=50
flat 61.20 vs spec 64.59 (spec +3.4); t=60 flat 55.47 vs spec 55.73 (tie,
inside noise); endpoints banked t=25 -7.8 / t=75 +15.4.
**FROZEN PREDICTION CONFIRMED: flat ahead at t=40, spec ahead at t=50; the
crossover sits in [40, 50], where the goal exits the ~25-step planning
window plus one serving stride.** t=60's tie is a wobble within seed noise
on a monotone-trend curve; reported as measured. (Protocol note: the t=25
point is the their-protocol start-random battery; t>=40 are mode-long
start-final; both unfiltered + goal_proprio.)

- **bok 12-seed pool: 58.85 vs pair 57.03 = +1.8pp, paired t ~ 1.1.** The
  pre-registered 6-seed B2 bar passed (+2.86 >= +2), but the extension
  weakens the margin to within noise. HONEST STATUS: best-of-k stays an
  optional add-on with a passed bar and a non-significant 12-seed margin;
  it does NOT enter the headline sentence.
- their-protocol flat RH5 12-seed: **83.72** (composite table symmetric).
- COMPOSITE REFINEMENT (correcting the "zero constants" phrasing above): the
  switch is placed at the MEASURED crossover, one measured constant with the
  same epistemic status as tau and S -- placing it at the window edge would
  route t=40 to spec and lose 4.9pp. Paper text updated to match.

## COMPUTE-MATCHED FLAT CONTROL (registered 2026-07-31 before running):
## the fairness control every other headline env has, applied to TwoRoom

flat RH2 at the anchored t=75 protocol with DOUBLED CEM samples (300 -> 600,
the paper's convention: a strict upper bound on the drafter's overhead), six
seeds 42-47. **Frozen prediction: no material rescue -- flat+2xCEM stays
well below certified spec's 57.03; the t=75 collapse is a lookahead limit,
not an optimization limit** (same call as PushT/Reacher, which held 3/4).
If flat+2x reaches within 3pp of 57.03 the headline must be re-framed as
partially compute-driven and the margin re-stated against this control.

### VERDICT (2306686, read 2026-07-31): flat+2xCEM = 48.44
(40.62/43.75/56.25/45.31/57.81/46.88). The doubled budget DOES rescue
+6.8pp over plain flat (41.67) -- the "no material rescue" prediction was
too strong, as it was for Reacher RH2, and is reported as-is. But certified
spec stays **+8.6pp above the control** (57.03 vs 48.44; per-seed paired
t ~ 2.5, 5/6 seeds positive) while using LESS planning compute than the 2x
upper bound. The re-frame trigger (within 3pp) does NOT trip. Reading: the
t=75 collapse is partly optimization, mostly lookahead; the headline margin
is not compute-driven. THIS CLOSES THE LAST CONTROL IN THE PROGRAM.

## TIMING ACCOUNTING (2306731, read 2026-07-31): overhead ~2% of wall-clock

Instrumented policy (lerp-1.0 tautology for flat, the fig:cost convention),
t=75 anchored, 3 seeds each, n=64, 150 timed replans per run:
  flat-instr: drafter 19.7-19.9s, cem 400-437s (drafter_frac 4.3-4.7%),
              total 420-457s; SR 40.6/40.6/46.9 (tautology reproduces flat
              RH2's 42.71 -- instrumentation validated again)
  spec:       drafter 28.5-28.8s, cem 399-436s (drafter_frac 6.2-6.7%),
              total 428-465s; SR 57.8/54.7/65.6 (consistent with 57.03)
Per-episode: ~6.9s both arms; the drafter + DINOv2 verifier add ~140ms
(+2%). CEM is ~94% of wall-clock on this stack, so the near-every-step
call_ratio (0.95) is cheap in seconds: **the +15.4pp costs ~2% wall-clock.**
Completes the fourth env's fig:cost column.

## PUSHT C1/C2 VERIFY-SPACE CONTROL: VERDICT (2306565 -> 2306566, read
## 2026-07-31 morning)

tau_dino derived at runtime: criterion floor p50 = 0.1182 (tau*=0.20 is
floor-consistent in DINO space too, band [0.118, 0.276]). Six seeds each,
one shared paired drafter, t=150:
  C1 (verify LeWM half, tau=0.20):  **63.54**, call_ratio 0.660,
      advances 450 (the verifier genuinely fires both ways)
  C2 (verify DINO half, tau=0.1182): **62.24**, call_ratio 1.000,
      advances 0 (degenerate-reject: closed-loop distances p50 0.184 sit
      above the offline floor)
**Declared reading |C1-C2| <= 3pp holds: SR is space-indifferent where the
own gap is open. But the mechanics add the real answer to "why not DINOv2
everywhere": the external space loses the ENTIRE call-savings benefit on
this substrate (cr 1.00 vs 0.66) at equal SR. The rule stands with a
measured reason: verify in the stack's own space when its gap is open (no
extra model, real efficiency); transplant only when it is not.**

## k*=2 CONFIRM: REFUTED (2306515, read 2026-07-30 night)

k=2 arm at the anchored t=75 protocol: 34.4/21.9/37.5/21.9/32.8/39.1 =
**31.25** vs bar "within 3pp of 57.03" -> the k-from-bias derivation rule is
**refuted closed-loop**, k stays 50. Mechanism visible in the probe's own
numbers: dispersion at k=2 is 0.049 vs 0.127 at k=50 -- the sampler
COLLAPSES TO ITS MEAN at low step counts (mode-averaging, the DROID-lerp
poison), which a bias-of-means statistic structurally cannot detect. Second
prescriptive-rule failure (after cube tau-midpoint); consistent with the
instrument's amended scope: applicability and tau-existence, not
prescriptions. Also implies spec serving BENEFITS from sampler diversity,
consistent with best-of-k's positive effect at k(steps)=50.

## FAITHFULNESS PROGRAM (2026-07-30, prep 2306153 read; closed loops
## pre-registered here BEFORE running)

- **k derivation (probe_k, faith-prep task 0): k* = 2.** Bias-to-k50 p50 is
  below the criterion floor (0.098) at EVERY k (0.056 at k=2); dispersion
  grows with k. The sampler converges immediately in the verification
  space. **Confirm run 2306515 fired, bar frozen in its header: k=2 arm
  within 3pp of the k=50 12-seed mean (57.03) at six seeds.** Pass = the
  row runs at 25x fewer diffusion steps per draft; fail = the k rule is
  refuted on this substrate and the row keeps k=50.
- **PushT paired drafter (faith-prep task 1): norm gate PASSED**
  (1.001 lewm / 1.002 dino); beats no-op in BOTH halves at every position
  (m+1 dino 0.099 vs 0.150; m+2 lewm 0.125 vs 1.136). Note the contrast
  with TwoRoom, where the LeWM half sits at chance (~1.37): the drafter is
  faithful wherever the encoder is, measured cross-substrate.
- **PushT DINOv2-verify CONTROL, registered now.** Question: does external-
  space verification also work where the stack's OWN space passes the gap
  probe (answering "why not DINOv2 everywhere" with a measurement)? Arms,
  t=150 protocol (mode long, budget 300, score block, angle 20, H2/RH2,
  n=64, seeds 42-47), BOTH sharing gdm_pusht_s10_paired_e400v.pt so only
  the verifier's half differs:
    C1 paired drafter, verify in the LeWM half, tau=0.20 (the env's
       derived tau);
    C2 paired drafter, verify in the DINO half, tau = criterion floor p50
       from probe_floor --verify-space dino (job 2306521; filled in before
       the battery is submitted, bar does not depend on its value).
  Declared readings: |C1 - C2| <= 3pp -> verification-space choice is free
  when the gap is open; the rule "own space when open (no extra model),
  transplant when not" stands. C2 > C1 + 3pp -> DINOv2 is simply the
  better verifier and the story simplifies. C2 < C1 - 3pp -> transplanting
  costs SR where the native space works; the transplant is a repair, not
  an upgrade. All three outcomes reportable.

## COMPOSITE END-TO-END BATTERY + CROSSOVER SEED SYMMETRY (registered
## 2026-07-30, frozen BEFORE submission; GPUs idle, post-audit)

Motivation (program audit, 2026-07-30): the deployed composite claim
currently CITES its two constituent cells; a reviewer's likeliest
objection is "was the window-rule policy ever RUN as one arm?" This
battery runs it end-to-end. Second, the crossover band's arms are 6-seed
while the composite table's flat comparators are 12-seed; precision-only
extensions close the asymmetry.

COMPOSITE ARM (one policy, driver-internal dispatch: --composite-crossover
45 in surveyor/envs/tworoom/eval.py). At each t the driver routes on the
task's KNOWN goal distance: goal_offset <= 45 (the measured crossover,
midpoint of the confirmed [40,50] band -- the ONE constant) -> flat
branch; else -> certified spec branch (specpaired, tau=0.098 derived,
verify in frozen DINOv2, k(steps)=50). Branch RH frozen by PRIOR-prereg
convention, not tuned here: their-protocol flat = RH5 (LeWM's own serving
config), anchored flat = RH2 (Stage A's strongest flat), spec = RH2 (the
headline arm) everywhere. Protocols = the banked ones verbatim: t=25
their-protocol (mode short, start random, budget 50); t in {40,50,60,75}
anchored (mode long, budget 2t, eval-filter none, episode-min 4000,
goal_proprio). n=64, seeds 42-47 (same seeds as the banked cells, so the
comparison is paired and reads as an integration test).

FROZEN BARS AND PREDICTION:
- **B-COMP-1 (primary, per t): composite mean >= banked mean of its own
  selected branch arm at that t, minus 1pp.** Banked branch references:
  t=25 flat_rh5 84.89 | t=40 flat_rh2 65.36 | t=50 spec_rh2 64.59 |
  t=60 spec_rh2 55.73 | t=75 spec_rh2 (seeds 42-47) 55.47.
- **B-COMP-2 (the paper-facing reading): composite >= max(best flat, best
  spec) - 1pp at every t.** Identical references except t=60, where the
  branch (spec 55.73) and best flat (55.47) are a recorded tie.
- Prediction (falsifiable): the composite reproduces its branch arm
  per-seed up to CEM nondeterminism; a mean deviation beyond 1pp is an
  INTEGRATION FAULT and is reported as such, never adjusted away.
- No banked number changes under any outcome; the composite gets its own
  row. A miss is reported as-is.

SEED-SYMMETRY EXTENSIONS (precision-only; NO registered verdict may
change under any pooled outcome):
- their-protocol pair_bok_rt_rh5 (77.08 at 6 seeds), seeds 48-53 -> 12.
  B25-1 remains FAILED regardless of the pooled mean.
- crossover t=40 and t=50, flat_rh2 + spec_rh2, seeds 48-53 -> 12 each.
  The 6-seed crossover verdicts stand as registered at their own n. The
  12-seed pooled means are reported alongside; if pooling moves a point
  across the tie line the crossover band is RE-STATED as measured
  (widening the reported band is a permitted outcome; tightening the
  claim post hoc is not). t=60 stays 6-seed (declared tie, descriptive).

FILMSTRIPS (cube + reacher, qualitative-only, same convention as
run_poster_strips.sbatch): n=8 single-seed captures at the paper cells;
no number from these runs may be quoted anywhere.

### VERDICTS (2306763/2306764/2306765, all 65 tasks COMPLETED, read
### 2026-07-30 afternoon -- same day as registration)

**COMPOSITE: B-COMP-1 AND B-COMP-2 PASSED AT EVERY t, zero deviation.**
Means (seeds 42-47): t=25 **84.89** | t=40 **65.36** | t=50 **64.59** |
t=60 **55.73** | t=75 **55.47** -- each cell reproduces its banked branch
arm EXACTLY, per-seed (the dispatch is bit-identical to the fixed arm's
code path and the serving stack is deterministic under matched
seed/cem-seed). The [composite] routing line fired correctly in all 30
logs (flat RH5 at t=25, flat RH2 at t=40, spec RH2 at t=50/60/75). The
prediction held in its strongest form: no integration fault, and the
window-rule policy has now been RUN END-TO-END AS ONE ARM at all five
goal distances. The paper may state: one invocation, one constant (the
measured crossover), never loses to the best fixed arm at any t.

**SEED-SYMMETRY EXTENSIONS: verdicts unchanged, band CONFIRMED at 12
seeds.** Pooled 12-seed means (42-53):
- t=40: flat_rh2 **65.10** vs spec_rh2 **61.59** -> flat +3.5 (6-seed
  read was +4.9; same sign, flat still ahead -> lower edge holds).
- t=50: spec_rh2 **63.28** vs flat_rh2 **59.90** -> spec +3.4 (identical
  margin to the 6-seed read -> upper edge holds).
- The crossover stays in [40, 50] at doubled seed count; no re-statement
  of the band is required.
- their-protocol pair_bok_rt_rh5: ext seeds 76.82, pooled 12-seed
  **76.95** (vs 77.08 at 6 seeds). B25-1 remains FAILED as registered;
  the composite table is now 12-seed on BOTH sides at t=25.

**FILMSTRIPS: all 5 captured** (cube spec champion + cube flat-lerp,
reacher spec t=150 + reacher flat-lerp, tworoom flat-lerp t=75); with the
existing pusht pair and tworoom spec strip, all four environments now
have spec-vs-flat side-by-side panels. Qualitative only; no numbers
quotable from the n=8 runs. All 8 npz pulled to the local repo
(strips/).

## PRECISION EXTENSIONS II: FAIRNESS CONTROL + ORACLE CEILING TO 12 SEEDS
## (registered 2026-07-30 evening, frozen BEFORE submission)

Motivation: after the composite/ext12 pass, the two softest inferential
statistics left in the arc are both t=75 anchored controls at 6 seeds --
the compute-matched flat margin (spec +8.6 over 48.44, paired t ~ 2.5,
the weakest test statistic the paper quotes) and the oracle ceiling
(57.81, 6 seeds, against which the 12-seed spec 57.03 is described as
"statistically at the ceiling"). Both extend seeds 48-53 -> 12; log
names pool with the originals (2room_fair2x_flat_seed*.log,
2room_anchA_oracle_rh1_seed*.log).

FROZEN RULES:
- Precision-only intent, but with the honest exposure stated up front:
  * FAIRNESS: the original re-frame trigger (control within 3pp of
    57.03) was adjudicated at its registered n=6 and did not trip. If
    the POOLED 12-seed control comes within 3pp of 57.03, the re-frame
    clause RE-OPENS and is applied (headline re-stated as partially
    compute-driven). We do not get to extend seeds and ignore a tripped
    trigger.
  * ORACLE: the ceiling is re-stated at the pooled value. The "spec
    statistically at the oracle ceiling" sentence is re-checked against
    the pooled oracle (unpaired t, spec 12-seed pool vs oracle 12-seed
    pool); if spec falls statistically below, the sentence downgrades
    to "near the ceiling". Config unchanged from Stage A (oracle s25
    RH1; static-table staleness caveat rides along as always noted).
- No other verdict may change; +15.4pp headline and all banked cells
  untouched under every outcome.

### VERDICT (2306870, all 12 tasks COMPLETED, read 2026-07-30 evening —
### both extensions STRENGTHEN)

- **FAIRNESS: pooled 12-seed control = 47.00** (ext seeds 42.19/45.31/
  45.31/50.00/39.06/51.56 = 45.57; original 48.44 regressed down).
  Margin vs spec 57.03 = **+10.0pp, paired t ~ 4.9, 11/12 seeds
  positive** (only seed 45 negative). The re-frame trigger (within 3pp)
  stays untripped by a wide margin. Rescue over plain flat re-states as
  41.67 -> 47.00 (+5.3pp, still real, still reported). The arc's
  weakest statistic (t~2.5 at 6 seeds) is now t~4.9 at 12.
- **ORACLE CEILING: pooled 12-seed = 56.90** (seeds 42-47 reproduce
  57.81 exactly; ext 59.38/60.94/51.56/60.94/46.88/56.25 = 55.99).
  Spec 57.03 vs pooled ceiling: diff +0.13pp, t ~ 0.06 — statistically
  indistinguishable; **"at the oracle ceiling" STANDS**, now 12-seed on
  both sides. Ceiling re-stated as 56.9 wherever quoted.
