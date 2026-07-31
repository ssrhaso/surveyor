# Certification pre-registration (frozen 2026-07-31, BEFORE any run or readout)

Purpose: give the word "certified" measured teeth. Two legs — (M1) a verifier
CALIBRATION readout against ground-truth state over per-replan traces, and
(R1) a drafter CORRUPTION sweep showing what the accept test buys at serving
time. Definitions and predictions frozen here before any number is computed.
Trust order: this doc > memory > paper prose. Companion trace-collection runs
(R2) produce traces only; NO success-rate from them may be quoted anywhere.

## M1 — VERIFIER CALIBRATION (offline, existing + R2 traces)

Data: per-replan trace dumps (`--dump-traces`): for every draft event,
`z_cond` (the achieved latent at that replan) and `block` (the served block).
Decisions (accept/advance vs reject/re-draft) are REPLAYED offline: the
verifier is a deterministic function of the recorded latents and the frozen
tau, so the replay reproduces the deployed decisions exactly. Arms: the
banked headline spec arm per env (cube gc15-gate seeds 42-45; reacher
s10spec/gatev3-spec t100+t150 seeds 42-45; pusht spec t150 + tworoom
spec t75 from R2 below).

Probe: latent -> ground-truth state, trained per env on dataset (latent,
state) pairs from the h5 (states are recorded in every sim dataset; the
substrate's own paper reports state probeable at r ~ 0.97-0.999 on these
latents). PROBE QUALITY GATE, frozen: an env's calibration row is reported
only if the probe reaches R^2 >= 0.90 on held-out dataset frames; below
that the row is declared unavailable, never fudged. Caveat recorded in
advance: probing a DRAFTED latent assumes near-manifold drafts; the LeWM
drafters pass norm/faithfulness gates (ratios ~1.0), and probe outputs on
drafts are sanity-checked against state bounds.

FROZEN READOUTS (computed per env, over all replayed events):
1. Accept-vs-reject separation: distribution of probe-state distance
   (achieved vs pursued waypoint's decoded state) for accepted vs rejected
   events. Declared reading: accepted events sit at/below the env's own
   success-criterion state scale; rejected events sit above.
2. Monotonicity: Spearman rho between the verifier's rel-L2 and the
   probe-state distance across events. Declared reading: rho >= 0.5 =
   the latent test tracks physical truth; 0.2-0.5 = weak, reported;
   < 0.2 = the verifier does not track state and the "certified" language
   must be weakened to "latent-verified". All three outcomes reportable.
3. False-accept rate: fraction of accepts whose probe-state distance
   exceeds the env's success-criterion radius. False-reject rate:
   fraction of rejects below it. No bar (measurement, not hypothesis);
   both quoted as the verifier's operating characteristic at the derived
   tau alongside the tau-robustness sweeps.
These definitions may not be altered after the first number is seen.

## R1 — DRAFTER CORRUPTION SWEEP (closed-loop, PushT t=150 headline cell)

Mechanism question: what does reality-verification BUY when the drafter
degrades? Every drafted waypoint w is displaced by sigma * ||w|| along a
random unit direction (relative-norm corruption, applied at EVERY draft
including re-drafts — the deployed semantics; implementation
`--draft-noise` in specaccept/envs/pusht/eval.py + sources.py, corruption
upstream of verification and of traces).

Arms: verified (spec-accept, tau=0.20 derived, k=3, S=10, RH2 — the
headline config) vs BLIND consumption (identical except tau=999, which
accepts everything = blind commit-N, the banked -18pp negative arm).
sigma in {0, 0.1, 0.2, 0.4}; seeds 42-44; n=64; mode long, t=150, budget
300, block-20 scoring. 24 cells.

FROZEN PREDICTIONS:
- P-CERT-1 (ignition at the derived tau): an off-manifold displacement of
  relative size sigma lower-bounds the achievable rel-L2 of any real
  state to the corrupted waypoint, so accepts require sigma <~ tau.
  Prediction: the verified arm's call_ratio rises from its banked
  baseline toward ~1.0 as sigma crosses the [tau/2, 2*tau] band — i.e.
  THE OFFLINE-DERIVED tau LOCALIZES THE CLOSED-LOOP IGNITION POINT.
  sigma=0.1 mild, sigma=0.2 substantial, sigma=0.4 near-total rejection.
- P-CERT-2 (graceful vs compounding): SR_verified >= SR_blind at every
  sigma > 0, and verified's total degradation (sigma 0 -> 0.4) is
  smaller than blind's. Mechanism: rejection converts corrupted-block
  consumption into per-replan re-anchoring; blind marches N-deep through
  corrupted waypoints.
- P-CERT-3 (anchors reproduce): the sigma=0 cells match the banked spec
  and blind-commit behavior within seed noise.
Failure of any prediction is reported as-is; no sigma grid extension or
tau adjustment after readout.

## R2 — TRACE COLLECTION (no quotable numbers)

pusht spec t=150 (tau=0.20, k=3, S=10, RH2) seeds 42-44 and tworoom spec
t=75 (specpaired, tau=0.098, RH2, anchored protocol) seeds 42-44, each
with --dump-traces, n=64. These cells' SUCCESS RATES ARE NOT QUOTABLE
(banked versions exist); the runs exist solely to produce per-replan
traces for M1. TwoRoom's verification lives in the DINOv2 half; its M1
row uses the paired trace's recorded latents in that half and the state
probe is trained on DINOv2-encoded dataset frames accordingly (gap-dense
encodes exist); if the paired trace lacks the DINO half per event, the
row is declared unavailable rather than approximated.

### R2 AMENDMENT (2026-07-31, same day, BEFORE any readout was computed)

Schema audit of the legacy traces found they record latents only at DRAFT
events, so accepted replans' achieved latents are unrecoverable from them
-- a calibration on legacy traces alone would see only rejects. Fix,
applied before any trace-collection run started: the sources now log a
record-gated CALIBRATION EVENT LIST -- one (env, rel, accepted,
z_achieved, target) tuple per VERIFICATION event, in the verify-half the
accept test actually reads (native 192-d for pusht/reacher/cube; DINOv2
384-d for tworoom paired). The queued trace jobs were cancelled and
resubmitted with the extended recording, and R2 EXPANDS to all four
environments so every M1 row is built from full accept+reject event logs:
  + cube champion arm (specaccept gc, goal-gate, tau=0.20, k=3, RH2,
    certified protocol) seeds 42-44
  + reacher spec t=150 (tau=0.20, k=8, S=10, RH2, max-offset-150
    population) seeds 42-44
Legacy cube/reacher traces are retained for the reject-side only and are
not used for the calibration rates. All cells remain no-quote.

## M2 — ACCEPT-RATE vs REGIME (log mining, descriptive)

Mine call_ratio / advances / rejects from every banked grid cell's log
across envs and horizons. Declared reading (descriptive, no bar): if the
accept rate rises with goal distance beyond the planning window, the
serving loop itself measures the regime the window rule describes; if
not, nothing is claimed.
