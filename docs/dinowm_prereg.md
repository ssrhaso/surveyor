# DINO-WM PushT: pre-registered instrument prediction (prospective test #4)

**Status: SKELETON — numbers get filled from `gap_dinowm_pusht_h*.json` and
frozen BEFORE any closed-loop spec-accept run on this stack. The gap-stat
record as of freezing: 3-for-3 prospective (droid readout 8x, v3 drafter
fidelity 20x, v3 planning value 20x).**

## Substrate

DINO-WM (gaoyuezhou/dino_wm @ 0a9492f), released pusht checkpoint: frozen
DINOv2 ViT-S/14 `x_norm_patchtokens` (384-d, pooled per frame for the
verifier metric), their ViT predictor, MPC-CEM planner (300 samples, 30 opt
steps), frameskip 5, goal_H 5, success = pos<20px AND angle<pi/9, n_evals 50
seed 99, max_iter capped at 12 (episode budget; same cap in every arm).
Anchor: flat baseline success rate = ____ (job 2299529; paper's number = ____).

## Instrument reading (measured 2026-07-26, gap_dinowm_*.json — FROZEN)

Gap stat on the VALID split, pooled DINOv2 frame latents, rel L2, equiv=1:

**pusht** (episodes ~200+ frames):
| hop (raw frames) | equiv p90 | hop p10 | gap | tau=0.20 in gap? |
|---|---|---|---|---|
| 5 (1 model step)  | 0.108 | 0.069 | closed | — |
| 25 (goal_H hop)   | 0.108 | 0.135 | **OPEN [0.108, 0.135]** | **NO — tau 0.20 ABOVE gap** |
| 50 (2x goal_H)    | 0.108 | (see json) | — | — |

**point_maze**: h10 equiv p90 0.061 vs hop p10 0.050; h25 0.061 vs 0.056 —
closed, but bulks separated 2.1x (equiv p50 0.036 vs hop p50 0.075) with tail
overlap = the REACHER SHAPE.
**wall_single**: h10 0.078 vs 0.037; h20 0.078 vs 0.052 — closed, bulks
separated ~2.7-3.2x, tail overlap (fat equiv tail p90 = 2x p50) = REACHER
SHAPE again.
**tworoom-dino (encoder-swap probe, job 2299550, measured 7/26)**: equiv
p10/50/90 = 0.047/0.081/0.159; hop(25) = 0.099/0.214/0.338; cross p50 0.222.
**Saturation CURED by the encoder swap** (LeWM ViT-tiny equiv p50 was 0.876 ≈
√2 with cross ≈ equiv): bulks separated 2.6x, tails overlap = the REACHER
SHAPE. Verdict: TwoRoom's verification failure was an ENCODER property —
generic frozen features restore metric structure; a fixed-tau verifier still
needs router-class machinery there. (Caveat: first 200 eps, unfiltered
population.) Closed-loop TwoRoom-on-DINO not planned (would need a trained
DINO-WM for tworoom); recorded as the diagnosis-confirmation row.

## Frozen prediction (2026-07-26, BEFORE any closed-loop spec run — prospective #4)

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
  (SR direction under P2 not barred — DROID showed blind chaining can limp.)
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
- Compute: spec arm's diffusion calls logged (call_ratio analog) — efficiency
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
