# P-TAU: does the accept test fail, or does the *transferred* threshold fail?

**Frozen 2026-08-11, before any cell below was run.** Submitted after the
P-RATE sweep (`2026-08-11_rate_transfer.md`) had completed and been read.

## Provenance of the hypothesis (stated because it matters)

This is a **post-hoc-motivated** hypothesis, pre-registered before its own run.
It was formed *after* seeing P-RATE refute 1b in both environments and 1c on
Reacher. We say so explicitly rather than presenting it as an independent idea,
and the bars below are inherited from P-RATE (the same 2.0pp band) rather than
chosen to fit the gap we are trying to close.

## Why this exists

P-RATE measured that fixed-rate coins match or beat the adaptive `tau=0.20` arm
at every horizon, worst on Reacher, where `p=0.8/0.9` beat it everywhere and by
up to 9.96pp at `t=25`. Read literally, the accept test does not carry success.

But every SURVEYOR cell in the paper serves `tau=0.20` in **both** environments,
while the criterion floors we derive in `sec:calib-tau` are **not** equal:

| env | derived criterion floor (p50) | tau served | FA at that tau (`tab:calibration`) |
|---|---|---|---|
| PushT | 0.233 | 0.20 | .102 |
| Reacher | **0.106** | 0.20 | **.400** |

On Reacher the served threshold is roughly **twice** its own derived floor, so
the test should systematically **under-reject** — which is exactly what FA=.400
says, and exactly what the winning coins fix by rejecting more often. The
paper's line that 0.20 "sits on both floors" is the weakest sentence in
`sec:calib-tau` for this reason.

So there are two competing readings of P-RATE, and they are distinguishable:

* **(A) The accept rule does not carry success.** Then serving Reacher at its
  own derived floor changes nothing and the P-RATE refutation stands as written.
* **(B) The *transferred constant* does not carry success.** Then serving
  Reacher at its own derived `tau=0.106` closes the gap, and the lesson is
  "derive tau per environment, do not transfer it" — a finding about our own
  protocol, not a rescue of the verifier's selectivity (which stays refuted by
  the August-1 coin control regardless).

## Design

Identical to the P-RATE reference arm in every respect except `--accept-tau`:
same drafters (`gdm_stride10.pt`, `gdm_reacher_s10.pt`), same `k` (3 / 8), same
`S`, same RH, same budget `2t`, same seed-2222 confirmation populations, same
evaluation seeds 42-45, same single 20deg/joint scoring pass. Every comparison
is therefore paired per seed against arms already banked by P-RATE.

* **Reacher, `tau=0.106`** (its own derived criterion floor), `t in {25,50,100,150}`.
* **PushT, `tau=0.233`** (its own derived criterion floor), `t in {25,50,75,100,150}`.

PushT is the **control**, not a second rescue: it is already within 1.18pp of
the best constant, and its floor lies *above* the served 0.20, so if merely
perturbing tau were enough to move SR, PushT would move too.

## Frozen predictions

**P-TAU-1 (the rescue).** On Reacher, the `tau=0.106` arm trails the per-cell
best fixed coin by **no more than 2.0pp at every horizon** (the P-RATE-1c band,
inherited unchanged). *Refuted if it trails by more than 2.0pp at any horizon.*

**P-TAU-2 (the mechanism).** On Reacher, `tau=0.106` re-anchors more often than
`tau=0.20`: the realized call ratio is higher at **at least 3 of 4** horizons.
*Refuted otherwise* — in which case the FA-based explanation above is wrong
whatever P-TAU-1 does.

**P-TAU-3 (the control).** On PushT, moving `tau` from 0.20 to 0.233 changes SR
by **less than 2.0pp at every horizon**. *Refuted otherwise.*

## Consequence rules (frozen before the run)

* **P-TAU-1 and P-TAU-2 both hold.** Reading (B). The paper reports the P-RATE
  refutation *and* this repair together: a transferred tau fails, a
  per-environment derived tau does not, and the protocol lesson is to derive per
  environment. `sec:calib-tau` drops the "sits on both floors" claim, the
  Reacher cells are re-quoted at 0.106 **only if** the whole grid is re-run at
  that value (no mixed-tau tables), and the abstract's rate claim is still
  narrowed, because per-event selectivity remains refuted by the coin control.
* **P-TAU-1 refuted.** Reading (A). The P-RATE refutation stands exactly as
  harvested, the verifier's contribution is restated as **cost control and
  certification, not success**, in the abstract, `sec:results-cert` and
  Limitations, and **no further rescue is attempted**. This document is reported
  as a failed repair, in the main text.
* **P-TAU-2 refuted.** The FA-based mechanism is reported as wrong regardless of
  P-TAU-1's outcome; any P-TAU-1 pass is then reported as unexplained.
* **P-TAU-3 refuted.** The tau-plateau claim in `sec:calib-tau` / `tab:taugrid`
  is narrowed to the swept range and populations where it was measured.

No cell is excluded after the fact. A run that fails to produce an SR line is
re-run once with the identical command and, failing that, reported missing.
Reacher `tau=0.106` is **not** re-derived, re-rounded or tuned: it is the p50
criterion floor already published in `tab:floor`.

## Runner

`batch/isca/run_tau_per_env.sbatch` (array 0-8). Harvest against the banked
P-RATE arms in `docs/rate_transfer.csv`.

## Outcome (appended 2026-08-11, job 2332652, all 9 tasks COMPLETED)

Four evaluation seeds (42--45) per cell, exactly as declared. No cell excluded,
no seed dropped, no re-run needed.

**Reacher, SR %** (`tau=0.106` = its own criterion floor):

| t | tau=0.106 | tau=0.20 (served) | best fixed coin | vs served | vs best coin |
|---|---|---|---|---|---|
| 25 | 62.30 | 55.66 | 65.62 (p=.9) | **+6.64** | **-3.32** |
| 50 | 82.62 | 79.49 | 84.37 (p=.8) | +3.13 | -1.75 |
| 100 | 93.36 | 89.84 | 93.56 (p=.8) | +3.52 | -0.20 |
| 150 | 91.99 | 91.41 | 92.77 (p=.6) | +0.58 | -0.78 |

**PushT control, SR %** (`tau=0.233` = its own floor, against served 0.20):
99.03/96.49/96.88/96.88/98.05 at t=25/50/75/100/150, i.e. +0.10/+0.50/+0.20/
+0.20/-0.19. Every horizon moves by less than 0.5pp.

### Verdicts against the frozen predictions

* **P-TAU-1 (the rescue): REFUTED.** At `t=25` the derived-tau arm trails the
  per-cell best fixed coin by 3.32pp, past the 2.0pp band inherited from
  P-RATE-1c. It is inside the band at the other three horizons (1.75 / 0.20 /
  0.78), but the prediction was quantified over *every* horizon.
* **P-TAU-2 (the mechanism): SUPPORTED, 4/4.** The realized call ratio rises at
  every horizon (~0.62 -> ~0.79-0.83), so the false-accept reading of
  `tab:calibration` is correct: at 0.20 the Reacher test under-rejects.
* **P-TAU-3 (the control): SUPPORTED.** PushT at its own floor moves <=0.5pp
  everywhere, so the Reacher gain is a per-environment threshold effect and not
  a generic sensitivity to perturbing tau.

### Consequence applied

The frozen rule for a P-TAU-1 refutation is reading (A): the P-RATE refutation
stands as harvested, the verifier's contribution is restated as cost control
and certification rather than success, **no further rescue is attempted**, and
this document is reported as a failed repair in the main text. All of that is
now in `main_workshop_final.tex` (abstract, `sec:calib-tau`,
`sec:results-cert`, Limitations, Conclusion).

One reporting note against ourselves: an interim read at two seeds put the
`t=150` gain at +2.34pp; the completed four-seed cell reads +0.58pp. The paper
quotes the full four-horizon range (+6.6 decaying to +0.6), not the favourable
`t<=100` subset that the partial data would have supported.

### What this licenses, and what it does not

Licensed: the statement that a *transferred* tau costs real success (up to
6.6pp on Reacher) and that per-environment derivation is the correct protocol.
Not licensed: re-quoting any Reacher cell at 0.106. Per the design section
above, that requires re-running the whole grid at per-environment tau; no
mixed-tau table is permitted.
