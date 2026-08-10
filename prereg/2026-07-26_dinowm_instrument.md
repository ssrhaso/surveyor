# DINO-WM PushT: pre-registered instrument prediction (prospective test #4)

**Status: SKELETON - numbers get filled from `gap_dinowm_pusht_h*.json` and
frozen BEFORE any closed-loop spec-accept run on this stack. The gap-stat
record as of freezing: 3-for-3 prospective (droid readout 8x, v3 drafter
fidelity 20x, v3 planning value 20x).**

## Substrate

DINO-WM (gaoyuezhou/dino_wm @ 0a9492f), released pusht checkpoint: frozen
DINOv2 ViT-S/14 `x_norm_patchtokens` (384-d, pooled per frame for the
verifier metric), their ViT predictor, MPC-CEM planner (300 samples, 30 opt
steps), frameskip 5, goal_H 5, success = pos<20px AND angle<pi/9, n_evals 50
seed 99, max_iter capped at 12 (episode budget; same cap in every arm).
Anchor: flat baseline success rate = 0.86 (job 2299529; paper's number = 0.90).
(Field filled 2026-08-02 from the job log: final `Success rate: 0.86` printed at
the last planning iteration, n=50 seed 99, log line 3607; the run then OOM'd in
the post-eval VQVAE video decode - after scoring, cosmetic only.)

## Instrument reading (measured 2026-07-26, gap_dinowm_*.json - FROZEN)

Gap stat on the VALID split, pooled DINOv2 frame latents, rel L2, equiv=1:

**pusht** (episodes ~200+ frames):
| hop (raw frames) | equiv p90 | hop p10 | gap | tau=0.20 in gap? |
|---|---|---|---|---|
| 5 (1 model step)  | 0.108 | 0.069 | closed | - |
| 25 (goal_H hop)   | 0.108 | 0.135 | **OPEN [0.108, 0.135]** | **NO - tau 0.20 ABOVE gap** |
| 50 (2x goal_H)    | 0.108 | (see json) | - | - |

**point_maze**: h10 equiv p90 0.061 vs hop p10 0.050; h25 0.061 vs 0.056 -
closed, but bulks separated 2.1x (equiv p50 0.036 vs hop p50 0.075) with tail
overlap = the REACHER SHAPE.
**wall_single**: h10 0.078 vs 0.037; h20 0.078 vs 0.052 - closed, bulks
separated ~2.7-3.2x, tail overlap (fat equiv tail p90 = 2x p50) = REACHER
SHAPE again.
**tworoom-dino (encoder-swap probe, job 2299550, measured 7/26)**: equiv
p10/50/90 = 0.047/0.081/0.159; hop(25) = 0.099/0.214/0.338; cross p50 0.222.
**Saturation CURED by the encoder swap** (LeWM ViT-tiny equiv p50 was 0.876 ~
sqrt(2) with cross ~ equiv): bulks separated 2.6x, tails overlap = the REACHER
SHAPE. Verdict: TwoRoom's verification failure was an ENCODER property -
generic frozen features restore metric structure; a fixed-tau verifier still
needs router-class machinery there. (Caveat: first 200 eps, unfiltered
population.) Closed-loop TwoRoom-on-DINO not planned (would need a trained
DINO-WM for tworoom); recorded as the diagnosis-confirmation row.

## Frozen prediction (2026-07-26, BEFORE any closed-loop spec run - prospective #4)

Decision rule (same as the five-substrate figure):
- gap EMPTY/INVERTED at every usable hop -> inapplicable (TwoRoom-on-LeWM case)
- gap OPEN -> applicable with tau INSIDE the gap (necessary condition; after
  the 7/24 cube result we claim tau-in-gap as necessary, NOT midpoint-optimal)
- tau below gap -> over-reject; tau above gap -> degenerate-accept (DROID case)
- bulks separated but tails overlap -> router territory (reacher case)

**PREDICTION (FROZEN):**
- **P1 (pusht, primary):** spec-accept is APPLICABLE at S=25 raw frames
  (= goal_H) with derived tau = 0.121 (gap midpoint). Certified spec-accept at
  tau=0.121 achieves SR >= flat - 2pp at the anchored protocol.
- **P2 (pusht, mechanics):** at transfer tau=0.20 (ABOVE the gap) the verifier
  is degenerate-accept: reject rate ~0, spec degenerates to blind chaining.
  (SR direction under P2 not barred - DROID showed blind chaining can limp.)
- **P3 (maze/wall, predictions only, no closed-loop planned):** plain
  spec-accept at any fixed tau underperforms flat; per-episode routing would
  be required. Recorded as instrument predictions.

## P3 amendment (2026-07-26 night, lens battery, frozen before any closed-loop)

The time-contrastive lens OPENS all three reacher-shaped DINO substrates on
held-out episodes (job 2299566): point_maze -0.006 -> +0.167 (derived tau
0.548), wall -0.012 -> +0.146 (tau 0.426), tworoom-dino -0.055 -> +0.229
(tau 0.540). Calibration-pipeline claim only (probe -> lens -> tau); the
reacher lesson stands (an open lens gap buys verification efficiency, not
guaranteed SR). For the TwoRoom recovery leg, the lens + tau=0.540 is the
designated verifier if closed-loop goes ahead. DROID remains the lens's one
failure (video-scale pooled compression).

## Closed-loop bars (frozen with the prediction)

- B1: certified spec-accept SR >= flat SR - 2pp at the anchored protocol
  (n_evals=50, seeds 99-102, max_iter 12, same CEM budget per replan).
- B2: reject/advance mechanics non-degenerate (not ~0%, not ~100% accept).
- B3: if the instrument predicted a mechanism (gate/router), the PLAIN spec
  arm's failure mode matches the diagnosis.
- Compute: spec arm's diffusion calls logged (call_ratio analog) - efficiency
  is reported, not barred.

## Wall closed-loop extension (frozen 2026-07-27, BEFORE the wall battery ran)

User approved running all DINO-WM envs. point_maze closed-loop is BLOCKED
(their env imports mujoco_py AND d4rl; not worth the dependency swamp) and
stays a prediction-only row. WALL runs the full chain: anchor (their
plan_wall protocol verbatim: n_evals=50 seed=99 goal_source=random_state
goal_H=5, MPC-CEM 300x10, max_iter=12 cap) -> grids -> lens retrained at the
serving hop 10 -> drafter S=10 N=3 -> battery. FROZEN BARS:
- **P3-wall:** spec at fixed tau=0.20 UNDERPERFORMS flat (the tail-overlap
  gap says no fixed tau certifies; this is the applicability prediction
  tested closed-loop on a never-run task).
- **P4-wall:** lens-verified spec (tau = the h10 lens's derived value)
  achieves SR >= flat - 2pp (the instrument's prescribed mechanism works).
- Mechanics non-degenerate in both spec arms (not ~0%, not ~100% accept).
P3 pass + P4 pass = the instrument both predicts failure modes AND
prescribes the working fix on a task it has never seen closed-loop. P4 fail
= the reacher lens lesson generalizes (lens opens gaps offline but does not
buy SR); report as-is.

## PushT long-horizon extension (frozen 2026-07-27, queued behind the H5 battery)

The H5 battery tests spec in flat's best regime (goal one CEM solve away;
early flat iter-1 = 0.78-0.86). The method's claim curve on LeWM grows with
horizon. Extension: goal_H=15 (75 raw frames; the drafter's N=3 x S=25 block
spans the goal distance exactly), arms flat x4 / spec tau=0.121 x4, seeds
99-102, n_evals=50, max_iter=12, otherwise the anchored protocol verbatim.
**P5 (frozen): at goal_H=15, certified spec >= flat - 2pp; the flat-vs-spec
margin IMPROVES for spec relative to the H5 cells (the LeWM horizon
pattern reproduces on the second architecture).** Descriptive expectation,
not barred: flat's own SR degrades from its H5 value.

## TwoRoom recovery, closed loop (frozen 2026-07-27, BEFORE any TwoRoom cell ran)

TwoRoom is the substrate the method has never worked on. On LeWM it failed for
TWO measured reasons, and the DINO-WM leg addresses each one separately:

1. **Encoder saturation.** LeWM's ViT-tiny put equivalent frames at rel L2
   p50 0.876 (approximately sqrt 2, cross approximately equal to equiv): no
   metric structure, tau underivable. The frozen-DINOv2 swap cured it
   (job 2299550, equiv p50 0.081).
2. **Goal unobservability.** TwoRoomEnv never renders the target (measured on
   tworoom.h5: zero green pixels in a frame whose agent sits 95 px from the
   target; the target patch is pure background). Goal-free drafting there is
   ill-posed in principle, which is why the 2026-07-05 and 2026-07-21 attempts
   could not work. **DINO-WM plans toward a goal IMAGE**, and under
   goal_source=random_state that image is the scene rendered with the AGENT AT
   the goal position. The destination is therefore fully specified by obs_g,
   and the task is well posed on this stack. This is a protocol difference, not
   a fix we invented, and it must be stated as such.

**Substrate.** DINO-WM trained from scratch on TwoRoom (job 2299632 plus resume
slices 2299982/2299983): frozen DINOv2 ViT-S/14, their ViT predictor, no
decoder, img 224 (256 patch tokens), frameskip 5, num_hist 3, 3800 train
episodes. Walltime-bounded, so it serves far fewer epochs than the released
DINO-WM checkpoints; every claim stays internal to this stack (spec vs THIS
stack's flat), as everywhere else in the paper.

**Eval env.** Vendored dependency-free re-implementation
(env/tworoom/tworoom_env_wrapper.py), anchored against the recording before any
cell: renderer mean-abs-pixel-diff **0.0000** over 96 frames, dynamics worst
error **0.000000 px** over 1090 transitions including 95 that the wall or
border constrained (validate_tworoom_env.py). Layout is fixed in the data
(door centre (112, 49) in every one of 400 episodes checked), so update_env is
a genuine no-op. Init and goal are drawn in OPPOSITE rooms, matching the data
(opposite-room fraction 1.0000 in both, door-routed path p50 192.6 px recorded
vs 185.7 px sampled).

**Instrument reading at the SERVING configuration (job 2299811, FROZEN).**
Pooled DINOv2 frame latents at img 224, hop 25 raw frames (= goal_H 5 x
frameskip 5), valid split:

| space | equiv p90 | hop p10 | gap | verdict |
|---|---|---|---|---|
| raw pooled | 0.096 | 0.062 | **CLOSED (-0.034)** | no fixed tau certifies |
| time-contrastive lens | 0.421 | 0.574 | **OPEN (+0.153)** | derived tau = **0.497** |

This is the WALL shape (and the reacher shape): bulks separated, tails overlap
in the raw space, lens opens it. The instrument therefore prescribes the
lens-verified verifier here, and predicts that a fixed tau will not work.

**PREDICTIONS (FROZEN, prospective test #5):**
- **P6-tworoom:** plain spec-accept at the transfer tau=0.20 UNDERPERFORMS
  flat. (Raw gap closed; 0.20 is above equiv p90 = 0.096, so the verifier
  should be near degenerate-accept and spec should degrade toward blind
  chaining.)
- **P7-tworoom:** lens-verified spec at the derived tau=0.497 achieves
  SR >= flat - 2pp at the anchored protocol.
- **Mechanics:** non-degenerate in the lens arm (accept rate neither ~0 nor
  ~100 percent). Reported, not barred: call_ratio.
- P6 pass + P7 pass = the instrument diagnoses the failure AND prescribes the
  working fix on the one substrate the method has never worked on. P7 fail =
  the reacher lens lesson generalises (an open lens gap buys verification
  efficiency, not SR); report as-is and TwoRoom stays a negative row.

**ANCHOR GATE (binding, the cube lesson).** No spec cell may be run or quoted
until the flat anchor (run_tworoom_anchor.sbatch, flat only, seeds 99-102) is
in hand. If flat SR is near the floor, the world model is undertrained and the
correct report is "substrate not reached", NOT a method result. Every arm
serves ONE frozen checkpoint (staged once by anchor task 0).

**Protocol** (identical across every TwoRoom arm): goal_source=random_state,
opposite rooms, goal_H=5, frameskip=5, MPC-CEM 300 samples x 10 opt steps,
n_taken_actions=5, max_iter=12 (= 300 env steps, about 6x the p90 door-routed
requirement of 54 steps, so the planner cannot be budget-starved),
n_evals=50, seeds 99-102, success = final agent within 16 px of the goal
(TwoRoomEnv's own terminated criterion). Drafter gdm_dinowm_tworoom_s25.pt
(S=25 raw frames, N=3, goal-conditioned, 257 tokens).

## Log

- 2026-07-25: skeleton written while anchor (2299529) and encode+gap job run.
- 2026-07-26: gap readings measured, P1-P3 frozen; lens battery opens
  maze/wall/tworoom-dino; P3 amendment added.
- 2026-07-27: pusht battery in flight (P1/P2); wall extension frozen above.
- 2026-07-27 (later): TwoRoom eval wrapper built and validated bit-exact
  against the recording; serving-configuration gap measured; P6/P7 frozen
  above before any TwoRoom closed-loop cell existed. Wall lens amendment:
  the h10 lens retrained at the wall SERVING hop did NOT open (width -0.131),
  so P4-wall has no derived tau at the serving hop; see the wall note.

## VERDICTS (appended 2026-07-31; the leg was closed at the pre-declared
## Aug-1 kill-switch. Recorded here so this doc carries its own outcomes,
## per the program trust-order convention.)

| prediction | verdict |
|---|---|
| P1 (pusht spec@tau=0.121 >= flat-2pp) | CONFOUNDED, claimed in neither direction: flat healthy 0.82-0.88, spec ~0 across three batteries with one signature (drafter passes offline gates + beats no-op; served waypoints produce no progress) = unresolved serving-integration fault |
| P2 (tau=0.20 degenerate-accept) | CONFOUNDED, same fault |
| P3 (wall: fixed tau underperforms) | CONFIRMED, cleanest prospective hit: spec 0.685 vs flat 1.00 (-31pp), called before the run |
| P4 (wall recovers under lens) | UNAVAILABLE: lens never opened a gap at the serving hop, no tau derivable; arm not run, reported rather than improvised |
| P5 (goal_H=15 margin improves) | CONFOUNDED, same serving fault (spec 0.0 all seeds, job dwm_bat15) |
| tworoom closed-loop completion | CUT at the kill-switch (WM training cancelled epoch ~32; eval wrapper never built). The OFFLINE encoder-swap cure stands as measured: equiv 0.876 -> 0.081 under DINOv2 on identical frames |

Autopsy of the serving fault is post-poster work only, time-boxed, with a
re-kill switch (see PLAN.md).

## AUTOPSY REGISTRATION (2026-07-31, frozen BEFORE any diagnostic GPU run;
## user authorized pulling it forward from post-poster - GPUs idle, poster
## blocked on a template decision)

TIME-BOX: three days, ends 2026-08-02 EOD. If the fault is not localized to
a code-level or representational cause by then, the leg stays dead, the
neither-direction verdicts above stand verbatim, and no further DINO-WM GPU
is spent. Diagnostic runs adjudicate NOTHING about P1/P2/P5 (n=20, 2 seeds);
they only localize. The original P1/P2 predictions stand UNCHANGED and may
only be adjudicated by fresh full batteries registered after a fix.

LOG-AUDIT FINDINGS (read 2026-07-31 morning, BEFORE new runs; these CORRECT
the provenance of the verdict table above, not its conclusions):
1. Batteries 2299661 (12 tasks) and 2299902 (8 tasks) = 100% TIMEOUT at
   walltime. Every number previously cited from them was a MID-RUN
   truncation (~7 of 12 MPC iters). "Three batteries, one signature" was
   really: one completed small battery + two walltime-killed ones.
2. The gate-test battery 2301459 (6 tasks, n=20, seeds 99/100) COMPLETED
   2026-07-29 and carries the decisive mechanics, read now:
     flat 0.90/0.90 | spec nogate 0.20/0.05 | spec gate 0.20/0.10
     verifier stats: advances 1-8 vs rejects ~230 per run, call_ratio
     ~1.0, gate retired only 1-2/20 envs.
   => The verifier essentially NEVER accepts: drafted waypoints are never
   verifiably reached, and pursuing them actively hurts vs flat. The
   overshoot/arrival-gate hypothesis is REFUTED as the primary cause.
3. The cem.py latent-goal branch is shape-clean by inspection; the spec
   planner instantiates and serves (constructor + stats lines present).

FROZEN DIAGNOSIS HYPOTHESIS: drafted token grids are OFF-MANIFOLD for the
token-level CEM cost while pooled-space conditioning and verification
cannot see it (the DROID compression lesson, drafter-side). Predicted
signature match: CEM chases a token target no real state resembles ->
progress ~0; pooled offline gates pass; verifier never accepts because no
achieved state approaches the drafted grid.

DIAGNOSTIC ARMS (all: n_evals=20, goal_H=5, max_iter=12, tau=0.121, k=8,
seeds 99/100 unless noted; serving protocol identical to 2301459):
- D1 GOAL-SERVE TAUTOLOGY (spec code path, target := the goal grid at every
  iteration - real encoded tokens through the same _latent_goal branch).
  Frozen prediction: ~= flat (0.85-0.90). If instead it collapses, the
  latent-goal hand-off itself is broken -> bounded code-fix hunt.
- D2 DRAFT-AND-SNAP (each drafted grid replaced by its nearest REAL grid
  from the train-split episode cache $DINOWM_DATA/
  pusht_lat/grids/, matched in pooled-visual rel L2 - the TwoRoom snap-bank
  mechanism, no new constant). Frozen prediction: advances >> 8 and SR
  materially above the 0.05-0.20 nogate band. If D1 healthy AND D2 lifts ->
  off-manifold diagnosis CONFIRMED and the fix is identified. If D1 healthy
  but D2 ~= nogate -> drafted content itself does not advance toward the
  goal (deeper than manifold) -> expect re-kill at the box.
- D3 SPEC AS-IS (seed 99 only): fault reproduction under the current code
  state, control for any drift since 2301459.
Instrumentation added for all spec-path arms: per-iteration mean pooled rel
L2 of served target vs current and vs goal, plus mean token-space MSE of
the served target to its nearest bank grid (the off-manifold score).

PROMOTION RULE (frozen): only if D1 ~= flat and D2 confirms does a snap
arm get promoted to a FRESH pre-registered P1 battery at the ORIGINAL bar
(spec@tau=0.121 >= flat - 2pp, their protocol, n_evals=50, seeds 99-102),
with walltime sized from the measured 7.6h/run so no battery can time out
again. Bank convention declared now: snap bank = train-split grids only
(the drafter's own training population; the eval episodes are dset-drawn
and disjoint).

## AUTOPSY VERDICT (2308656, 5/5 COMPLETED, read 2026-07-31 evening -
## the decision tree resolves to the honest-negative branch)

- **D1 GOAL-SERVE TAUTOLOGY: CONFIRMED HEALTHY.** 0.90 / 0.80 vs flat's
  0.90 / 0.90 (gate-test), rel_to_goal = 0.000, achieved state converges
  to rel_to_now ~ 0.12 ~ tau. **The latent-goal hand-off, cem branch,
  grid encoding, and the whole serving integration are exonerated** -
  and D1 doubles as a validation of the serving stack + derived tau on
  the second architecture.
- **D2 DRAFT-AND-SNAP: REFUTED.** 0.05 / 0.10 (the nogate band);
  advances 11 / 7 of ~250 (prediction was >> 8). Real-manifold targets
  do not restore acceptance or SR. The diag localizes why: snapped
  targets sit rel_to_now ~ 0.17 AND rel_to_goal ~ 0.20 - the drafter's
  proposals are neither near reality nor nearer the goal.
- **D3 AS-IS CONTROL:** fault reproduced (0.20); drafted grids' token
  MSE to the nearest real grid 0.47 vs the 0.28-0.32 real-frame bank
  floor - the off-manifold excess is real but NOT the primary cause
  (per D2).

**DIAGNOSIS (replaces "unresolved serving-integration fault"):** the
transplanted drafter's CONTENT fails on frozen DINOv2 features - its
waypoints do not advance toward the goal. The verifier's ~97% rejection
rate across every battery was CORRECT detection of a bad drafter:
certification behaving exactly as designed. Per the frozen tree
(D1 healthy + D2 no-lift): **no promotion battery; the leg re-kills at
the box.** P1/P2 remain unadjudicated for the method served by a
competent drafter; the verdict table above is glossed accordingly.

**DATA-SCALE FOLLOW-UP (2026-08-01, checked against records before any
run):** the "data-starved drafter" hypothesis is REFUTED by the training
log itself - the drafter trained on ALL 18,685 train episodes (~1.6M
pairs, ~100k optimizer steps = 2-3x the budget of every working LeWM
drafter; the grid encode loop is uncapped). The content failure stands
AT FULL DATA SCALE, consistent with the V-JEPA 2 20x-data null:
two foundation-feature stacks, the same drafter-side lesson. This is
the sentence the paper's cross-architecture section now carries.
