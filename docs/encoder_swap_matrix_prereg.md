# Pre-registration: completing the verification-space matrix (Reacher, Cube in DINOv2)

Registered 2026-08-03, BEFORE any result is read. Offline only: no closed loop,
no serving cells, no CEM. Nothing here can change a deployed constant.

## Why

The paper states a rule with a measured reason (App E, point v):

> verify in the stack's own space when its gap is open, transplant only when it is not.

It is currently supported by two decisive cells at both the instrument and the
closed-loop level:

| env | native (LeWM) gap | DINOv2 gap | closed-loop swap run |
|---|---|---|---|
| PushT | OPEN [0.171, 0.406] | measured (`floor_pusht_dino.json`) | yes: SR-indifferent (63.5 vs 62.2) but transplant degenerates, call ratio 1.00, zero free advances |
| Two-Room | INVERTED (equiv p90 1.393 > hop p10 1.018) | measured, saturation cured 0.876 -> 0.081 | yes: +15.4pp at t=75 |
| Reacher | CLOSED by a hair (equiv p90 0.410 > hop p10 0.379) | **not measured** | no |
| Cube | OPEN [0.219, 0.353] | **not measured** | no |

A reviewer can fairly ask why a rule about which space to verify in was tested
on half the environments. The two missing cells are offline gap measurements,
minutes of GPU each on cached frames, so there is no reason not to have them.

## What runs

`specaccept.probes.probe_floor --verify-space dino` on Reacher and Cube, the
same probe and the same flag that produced the existing PushT DINOv2 row. No
`--gdm-ckpt`, so sampler dispersion (part C) is skipped: this measures the
temporal floor, the criterion floor, and the S=10 hop displacement, all in
pooled frozen DINOv2 space.

Gap is read exactly as elsewhere: `[criterion_floor p90, disp_S10 p10]`, OPEN iff
`disp_S10 p10 > criterion_floor p90`.

## Predictions (frozen)

**P-SWAP-1 (Reacher, the informative cell).** Reacher's native gap is closed only
marginally, and DINOv2 cured a far worse saturation on Two-Room. Predict the
DINOv2 gap on Reacher is **OPEN**. If it opens, the rule's "transplant when the
native gap is closed" branch gains a second supporting environment at the
instrument level, and Reacher becomes a named candidate for a future closed-loop
swap. If it stays closed, the honest reading is that DINOv2 is not a universal
repair and Two-Room's cure was specific to a saturated encoder, which we would
state rather than bury.

**P-SWAP-2 (Cube, the control cell).** Cube's native gap is already open, so the
rule says stay native regardless of what DINOv2 does. This cell is registered as
**descriptive**: no outcome changes the deployed threshold or any paper claim. It
exists so the matrix is complete and so P-SWAP-1 has a same-day comparison run
under identical settings.

**Falsifier worth naming.** If DINOv2's gap is closed on Reacher AND open on
Cube, the intuition "DINOv2 is the better verification space" is wrong. That
would not break the deployed rule, which is stated about the native gap and not
about DINOv2 being universally superior, but it would sharpen how we describe
the transplant, and we would say so.

## Declared limits

- Instrument level only. Neither cell licenses a closed-loop claim; a swap arm on
  Reacher or Cube would be new pre-registered work, not an inference from this.
- No tau derived here is deployed anywhere. The serving thresholds stay as
  published (0.20 native, 0.098 Two-Room DINOv2, 0.118 PushT DINOv2).
- Single shot, seed 42, no re-runs on an unwanted result. Whatever comes back is
  what gets reported, including "no change to any paper sentence".

## Job

`batch/isca/run_encoder_swap_matrix.sbatch`, array 0-1 (0 = reacher, 1 = cube),
writing `Results/gap_stat/floor_{reacher,cube}_dino.json`.

---

# VERDICT (read 2026-08-03, job 2309726, 2/2 COMPLETED, ~45s each)

## The completed matrix

Gap read as `[criterion_floor p90, disp_S10 p10]`, OPEN iff hop p10 > criterion p90.

| env | native (LeWM) | margin | DINOv2 | margin |
|---|---|---|---|---|
| PushT | **OPEN** [0.171, 0.406] | +0.235 | **CLOSED** [0.2121, 0.0873] | -0.125 |
| Reacher | CLOSED (0.410 vs 0.379) | -0.031 | **CLOSED** [0.0759, 0.0734] | -0.003 |
| Cube | **OPEN** [0.219, 0.353] | +0.134 | **CLOSED** [0.2668, 0.1859] | -0.081 |
| Two-Room | INVERTED (1.393 vs 1.018) | -0.375 | **OPEN** (saturation cured, 0.876 -> 0.081) | + |

## P-SWAP-1: REFUTED

Predicted the Reacher DINOv2 gap would be OPEN. It is CLOSED, margin -0.0025.
Recorded as a miss. The one thing that survives from the reasoning behind it:
DINOv2 does bring Reacher an order of magnitude closer to opening than its native
space does (-0.003 against -0.031), so the direction was right and the call was
wrong. No paper sentence depended on this prediction.

## P-SWAP-2 (Cube, descriptive): CLOSED, and it is the informative cell

Cube's native gap is OPEN and its DINOv2 gap is CLOSED. Transplanting the verifier
there would **destroy a working accept test**. This is the cell that turns the
deployed rule from a two-environment observation into a four-environment claim
with a mechanism.

## What this establishes

**DINOv2 is not a better verification space, it is a different one.** On every
environment whose native space works, moving the accept test into the general
purpose encoder closes a gap that was open. The Two-Room transplant is therefore
a *targeted repair for a pathologically saturated encoder*, not evidence that a
foundation encoder is a superior place to verify. That is a sharper and more
defensible statement than the paper previously made, and it is the honest reading.

**The instrument retrospectively predicts the closed-loop C1/C2 control.** PushT's
DINOv2 gap is closed (-0.125), and closed loop the DINOv2 verifier rejected
everything it saw (call ratio 1.00, zero free advances, SR unchanged). The offline
statistic and the serving outcome agree without being fitted to each other.

**Corroborates the never-transfer-tau rule.** The deployed tau = 0.20 is off-floor
in DINOv2 space on both new cells (Reacher floor band [0.048, 0.099], Cube
[0.203, 0.347]), which is why a paired arm must re-derive its threshold in the
space it will serve in rather than carry one across.

## Scope

Instrument level only, as declared. No closed-loop swap arm was run on Reacher or
Cube, no deployed constant changed, and the result licenses no new serving claim.
Paper change: one passage in App E point (v), reporting the completed matrix.
