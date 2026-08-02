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

### M1 SENSITIVITY AMENDMENT (2026-08-01, registered after the ridge rows
### were read but BEFORE any MLP number was computed)

Probe-class robustness row: rerun the identical frozen readouts with an
MLP probe (256 hidden, sklearn, same 80/20 split and quality gate) on the
same traces. Declared reading (descriptive): the operating characteristic
is probe-independent if the false-accept rates agree within 2pp per env;
larger shifts are reported as-is and the more conservative (higher-FA)
row is the one the paper quotes. No other definition changes.

### M1 REACHER PROBE AMENDMENT (2026-08-01, registered BEFORE computing)

The reacher ridge probe failed its quality gate (R^2 0.41 on raw qpos) --
the textbook circular-regression artifact: joint angles are periodic and
linear regression on raw angles breaks at the wrap. Amendment, justified
a priori by standard circular-regression practice: probe target becomes
[sin(q), cos(q)] per joint (--circular), decoded back via atan2; the
criterion distance becomes the WRAPPED angular difference (max-abs per
joint, radius 0.05 unchanged); the R^2 gate applies to the transformed
targets. BOTH outcomes are reported: "row unavailable under the frozen
raw-angle probe" stands in the record, and the amended row (clearly
labeled) is quoted only if its probe passes the gate. Ridge + MLP both
rerun; the conservative-row rule applies unchanged. If the amended probe
also fails the gate, reacher's row stays unavailable, final.

## M2 — ACCEPT-RATE vs REGIME (log mining, descriptive)

Mine call_ratio / advances / rejects from every banked grid cell's log
across envs and horizons. Declared reading (descriptive, no bar): if the
accept rate rises with goal distance beyond the planning window, the
serving loop itself measures the regime the window rule describes; if
not, nothing is claimed.

# ============================================================
# VERDICTS (read 2026-07-31 evening .. 2026-08-01; every number below was
# produced under the frozen definitions above; nothing re-run, nothing
# re-defined after readout)
# ============================================================

## M1 VERDICT — the calibration table (jobs 2308841/2308848/2308850/2308862)

All rows: replay mismatches = 0 (the offline replay reproduces every
deployed accept/reject decision exactly) and d_accept < d_reject.

| env | probe | R^2 | gate | rho | FA | FR | acc p50 / radius |
|---|---|---|---|---|---|---|---|
| cube | ridge | .992 | pass | .723 | .000 | .590 | .013 / .04 |
| cube | mlp | .190 | **FAIL** | - | - | - | sensitivity row unavailable (probe underfit) |
| pusht | ridge | .977 | pass | .415 | .024 | .798 | 6.5 / 20 |
| pusht | mlp | .983 | pass | .492 | .102 | .577 | - |
| reacher | ridge raw-angle | .412 | **FAIL** | - | - | - | row unavailable under the FROZEN probe (recorded) |
| reacher | ridge circ (amendment) | .999 | pass | **.978** | .347 | .003 | .040 / .05 |
| reacher | mlp circ (amendment) | .997 | pass | .879 | .400 | .045 | .042 / .05 |
| tworoom | ridge | .998 | pass | .363 | .522 | .126 | 17.2 / 16 |
| tworoom | mlp | .997 | pass | .396 | .424 | .127 | - |

RULE APPLICATIONS (all pre-frozen): probes disagree >2pp on pusht
(.024 vs .102), reacher-circ (.347 vs .400) -> the CONSERVATIVE row is
quotable. tworoom conservative row = ridge .522. cube MLP failed its own
gate -> ridge stands with that noted. **QUOTABLE FA: cube .000, pusht
.102, reacher .400 (registered circular amendment; frozen-probe row
recorded unavailable), tworoom .522.**

READING: accepted waypoints are physically within the task's own success
radius 100% (cube) / 90% (pusht) of the time; against the two tightest
criteria (reacher 0.05 rad, tworoom 16 px) accepts sit at ~80-107% of
the radius with 2.5-3x separation from rejects; reacher's verifier ranks
TRUE angular distance at rho = 0.978 (near-perfect ordinal sensor).
Errors are predominantly conservative (high FR / low FA) except reacher,
whose tight criterion makes the accept test liberal in the tail while
its ranking stays near-perfect. Spearman: >= 0.5 on cube/reacher
(tracks); pusht/tworoom in the declared 0.2-0.5 weak band (reported
as such, per the frozen reading; the FA/FR/separation stand as
measurements regardless).

## R1 VERDICT — corruption sweep (2308689, 24/24 COMPLETED)

- **P-CERT-1 CONFIRMED (the headline):** verified arm call_ratio .59 ->
  .66 -> 1.00 -> 1.00 across sigma = 0 / .1 / .2 / .4; advances 518 ->
  420 -> 2 -> 0. The rejection ignition sits between tau/2 and tau —
  **the offline-derived tau=0.20 localizes the closed-loop detection
  point exactly as frozen.** Blind arm flat at its mechanical 1/N floor
  (.34) at every sigma.
- **P-CERT-2 REFUTED as stated:** SR flat for BOTH arms at every sigma
  (verified 61.5/59.9/61.5/60.9; blind 61.5/64.6/64.6/61.5 incl. the
  final sigma=.4 cells). Mechanism: fresh-draw zero-mean isotropic
  jitter is temporally averaged by both serving modes; no compounding.
  Verification's SR value is against AUTOCORRELATED error — already
  banked three ways (blind-commit-3 −18pp on the spine, the stale-oracle
  results, the crossover). Reported as-is; no grid extension.
- **P-CERT-3:** the sigma=0 verified cells (~61.5) match the banked
  unfiltered-population drafted level (~59). Caveat recorded: the banked
  blind-commit −18pp was measured on the filtered spine population; on
  THIS unfiltered population blind ~= verified at sigma=0, so the blind
  anchor is population-mismatched and only the spec anchor is read.

## R2 VERDICT — trace collection (2308713, 12/12 COMPLETED)

All four envs' headline arms produced full accept+reject calibration
event logs under the amended recording (3,552 / 1,923 / 991 / 2,012
events for pusht / tworoom / cube / reacher). Success rates from these
cells remain non-quotable, as registered.

## 08-01 EXTRACTION ADDENDUM (registered BEFORE computing; descriptive, no bars)

Zero-new-evidence extractions from the SAME frozen artifacts, for reviewer-
probe defense; registered here before any number is read:

1. **M1 radius-sensitivity readout.** From the identical calibration events
   and probe outputs (nothing re-run, tau and every accept/reject decision
   FIXED), dump per-event (rel, accepted, state_dist) to CSV and report
   FA(r)/FR(r) as functions of a swept READOUT radius r around the env's
   criterion radius, plus the rel-vs-state-dist operating curve with the
   derived tau marked. This is a descriptive sensitivity axis on the readout
   side only — explicitly NOT a tau re-derivation or re-thresholding
   (graveyard rule 3 untouched: decisions and tau are the deployed ones).
   Declared reading (descriptive): where FA(r) collapses just above the
   criterion radius, the false accepts are near-boundary, consistent with the
   accept-percentile table already in the M1 verdict; wherever it does not,
   that is reported as-is.
2. **M2 tworoom completion.** The M2 verdict noted tworoom rows need a
   dedicated filename regex. Mine the banked anchored-protocol tworoom t=75
   spec cells (Stage B 2306077 + 12-seed pool 2306141 logs) with a correct
   regex and tau label (0.098) and add the rows to the accept-rate table.
   Descriptive, no bar, exactly as M2 was declared.

Companion closed-loop control registered separately in
docs/randreject_prereg.md (matched-rate random-rejection, frozen the same
day BEFORE any run).

### Extraction readout (08-01, same day; jobs 2308864 4/4)

Per-event CSVs pulled local (Results/calibration/events_{env}.csv; 3552 /
2012 / 991 / 1923 events pusht/reacher/cube/tworoom); the rerun pipeline
reproduces every banked FA exactly (.102/.400/.000/.522). FA(r) declared
reading CONFIRMED for the two scary rows — false accepts are near-boundary:
- reacher: FA .400 at the 0.05-rad criterion -> .290 (r=0.06) -> .174
  (0.075) -> **.064 at 2x radius** -> .003 (0.15); accept p90 = 0.090.
- tworoom: FA .522 at 16px -> .380 (20) -> .283 (24) -> **.130 at 2x
  radius** -> .000 (48px).
- pusht: .102 -> .045 (25) -> .020 (30). cube: .000 at criterion (and
  .194 at HALF the radius — the verifier is sharp exactly at scale).
Operating-curve data (rel vs state_dist with derived tau) now plottable
from the same CSVs; figure work belongs to the writing bundle.

## M1-FM — FOUNDATION-SCALE CALIBRATION ROW (registered 2026-08-01 BEFORE
## any number was computed; dataset-pair variant, no closed loop)

Purpose: the certification-frame matrix's one empty cell — the calibration
methodology beyond LeWM. Substrate: V-JEPA 2 ViT-g (vjepa2_ac_vit_giant
encode path, layer-normed tokens, MEAN-POOLED — exactly the space the
transplant's verifier read; tau=0.20 = the frozen transfer value). Data:
the 2,000 cached DROID episodes (data/droid2k_lat, (T,256,1408) fp16
tokens + recorded 7-d proprio states). NO serving loop exists on this
stack (chapter closed by supervisor agreement; this uses its assets
without reviving its claims): this row is a DATASET-PAIR operating
characteristic and must be labeled as such wherever quoted — the LeWM
rows replay deployed serving decisions; this row scores the same accept
statistic over recorded-frame pairs.

Frozen definitions:
- pool = token mean -> (T,1408); rel(i->j) = ||p_i - p_j|| / ||p_i||
  (i = achieved role, j = target role); accept = rel <= 0.20.
- Pairs: 400 episodes sampled (numpy rng seed 0) from the 2,000;
  50 within-episode pairs each, (i, i+Delta), Delta ~ U{1..32} (the
  far-goal horizon) -> 20k pairs; plus 5k cross-episode pairs (different
  episodes, uniform frames).
- Ground truth: RECORDED states (no probe error — both frames are real
  frames); criterion dims = EE position (dims 0-2), distance = L2 in
  meters. A ridge probe pooled->state is additionally fit (4k frames,
  80/20 split) and its R^2 on dims 0-2 reported as the decodability row
  under the standard 0.90 gate label.
- Readouts, within-episode pairs: Spearman rho(rel, EE distance);
  FA(r)/FR(r) at tau=0.20 for r in {0.02, 0.04, 0.08, 0.16} m (DROID has
  no task criterion radius — the CURVE is the deliverable, no
  single-radius headline); accept-vs-reject EE-distance p10/50/90;
  accept rate. Cross-episode pairs: accept rate ONLY (scene
  discrimination check).

FROZEN PREDICTIONS:
- **P-FM-1:** consistent with the July gap verdict on this substrate
  (gap INVERTED: equiv p90 .112 > hop p10 .076; cross p50 .153 < tau),
  the accept test at tau=0.20 is DEGENERATE here: within-pair accept
  rate > 0.9 AND cross-episode accept rate > 0.5. This is the gap
  instrument's scope verdict tested against ground truth at foundation
  scale; confirmation = the two certification legs AGREEING about where
  the method does not apply.
- **P-FM-2 (open, both outcomes reportable):** rho per the M1 bands
  (>=0.5 tracks / 0.2-0.5 weak / <0.2 fails). No bar; quoted as
  measured.
No tau re-derivation on this substrate follows from any readout
(graveyard rule 3). Single run, no pair-population extension.

### M1-FM VERDICT (read 2026-08-01, job 2308926; every number under the
### frozen definitions above)

Results/calibration/cal_vjepa2_droid.json (+ events CSV, 20k within /
5k cross pairs, 400 episodes):

- **P-FM-1 REFUTED, in the favorable direction, reported as-is:** the
  accept test at tau=0.20 is NOT degenerate at foundation scale. Cross-
  episode accept rate = **0.0008** (predicted > 0.5; cross rel p50 =
  0.496, far above tau — near-perfect scene discrimination) and within-
  episode accept rate = 0.850 (predicted > 0.9). The prediction's
  premise (July gap_droid.json: cross p50 0.153 on the 98-episode
  droid_lat pool) is contradicted by the 2,000-episode measurement; a
  descriptive reconciliation diagnostic (identical code path over the
  old 98-episode pool) is in flight and its outcome will be recorded
  here either way. No frozen readout depends on it.
- **P-FM-2: rho = 0.526 — the TRACKS band (>= 0.5),** better than
  pusht's own deployed row (.49) and tworoom's (.36): the pooled
  V-JEPA 2 verifier statistic rank-tracks physical end-effector
  distance over recorded-frame pairs.
- Separation: accepts EE p10/50/90 = 0.020/0.127/0.340 m vs rejects
  0.136/0.283/0.474 (2.2x at p50); FR tiny (.007-.14 across the radius
  grid). FA(r) high against tight radii (.90/.80/.65/.40 at
  2/4/8/16 cm): tau=0.20 is LOOSE in EE-meters here — DROID has no task
  criterion radius, so this is the declared curve readout, no headline.
- Decodability row: ridge probe R^2 (EE dims) = 0.585 -> **below the
  0.90 gate; row recorded unavailable** (the operating characteristic
  is probe-free by construction — recorded states).

READING (within the frozen declarations): the calibration methodology
runs unchanged at foundation scale, and the verifier statistic it
audits is better-behaved there than the July gap verdict assumed —
scene-discriminating and distance-tracking (rho 0.53), with tau=0.20
transferring as a rank-sensible but metrically loose threshold. The
paper may quote this row ONLY with the dataset-pair label and the
tau-looseness sentence attached.

### RECONCILIATION READOUT (same day, job 2308927): key semantics, not
### a data conflict

The identical frozen code path over the July 98-episode droid_lat pool
gives true cross-EPISODE rel p50 = 0.507 (vs 0.496 on the 2k pool) and
within stats agreeing with July (within p50 .128 vs July hop(10) p50
.121; rho there 0.593, cross-episode accepts 0/2000). Root cause found
in code: gap_stat.py's "cross" row measures a random frame against the
SAME episode's final frame (a goal-distance statistic, line 89), not
cross-episode pairs — July's 0.153 sits inside the within-pair span and
is consistent with every new number. So no measurement contradicts any
other; P-FM-1's cross clause was frozen on a misreading of that JSON
key (recorded as such); the scene-discrimination finding (cross-episode
accept rate 0.0008 / 0.0000) was never measured in July, is new, and
stands. The within clause (predicted >.9, measured .85) is a near-miss:
acceptance over the far-goal hop family is loose but not total,
consistent with the July tau-above-gap serving concern and with the
tracking rho. M1-FM row cleared for the paper under its required
labels.

## M2 VERDICT — CONFIRMED (Results/acceptrate_mine.csv, 1,514 rows)

call_ratio falls with goal distance: pusht .88 (t25) -> .68 (t50) ->
.61 (t75) -> .57 (t150); reacher .75 -> .66 -> .64 -> .58 (t25->t150);
cube (single-horizon, arrival-gated) .84. The accept rate rises as the
goal exits the planning window: the serving loop independently measures
the regime the window rule describes. (tworoom rows need a dedicated
filename regex; its banked t=75 call ratio is 0.95 — the weak-encoder
exception, consistent with its scope row.)

### M2 tworoom completion readout (08-01, per the extraction addendum)

All 12 banked anchored-protocol t=75 spec cells mined
(2room_anchB_pair_t098_seed42-47 + 2room_anchExt_pair_t098_seed48-53,
Results/acceptrate_mine_tworoom.csv): per-seed call_ratio 0.938-0.971,
pooled 8243/8608 = **0.958** at tau=0.098. Confirms the noted ~0.95
weak-encoder exception row exactly; the M2 table is now complete for all
four environments.
