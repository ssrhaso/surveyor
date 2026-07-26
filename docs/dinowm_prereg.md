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

## Closed-loop bars (frozen with the prediction)

- B1: certified spec-accept SR >= flat SR - 2pp at the anchored protocol
  (n_evals=50, seeds 99-102, max_iter 12, same CEM budget per replan).
- B2: reject/advance mechanics non-degenerate (not ~0%, not ~100% accept).
- B3: if the instrument predicted a mechanism (gate/router), the PLAIN spec
  arm's failure mode matches the diagnosis.
- Compute: spec arm's diffusion calls logged (call_ratio analog) — efficiency
  is reported, not barred.

## Log

- 2026-07-25: skeleton written while anchor (2299529) and encode+gap job run.
