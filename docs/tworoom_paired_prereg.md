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
