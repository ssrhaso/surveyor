# P-RATE: is the re-anchoring rate one constant, or must it adapt?

**Frozen 2026-08-11, before any cell below was run.** Registers the
fixed-rate transfer sweep that tests the strongest adversarial reading of our
own random-rejection result.

## Why this exists

`2026-08-01_random_rejection.md` banked an honest, uncomfortable result: an
i.i.d. coin rejecting at the verifier's **own measured per-cell rate** ties
SURVEYOR-Base on both `t=150` headline cells (PushT 98.05 vs 98.14; Reacher
90.33 vs 91.11). The paper reports this and concludes that per-event
selectivity is not load-bearing at nominal operation — what verification
supplies is the *rate*.

That conclusion has a hole we have not yet closed by experiment. If one fixed
rejection probability, tuned once, reproduced the method's success at **every**
horizon, then the accept test would be replaceable by a single constant and the
success claim would not need a verifier at all. Our current defence is an
inference from serving telemetry (the realized call ratio falls with goal
distance: PushT 0.88 -> 0.57, Reacher 0.75 -> 0.58 across `t=25 -> 150`), not a
direct measurement. This document registers the direct measurement.

## Design

For each environment and horizon, run the SURVEYOR-Base arm with the accept
decision replaced by an i.i.d. coin at a **fixed** probability `p`, everything
else byte-identical to the tau=0.20 reference arm (same drafter, same `k`, same
`S`, same RH, same budget, same CEM seeds).

* Populations: the frozen seed-2222 confirmation populations
  (`pusht_c2222.episodes{t}.json` over `pusht_c2222.h5`, n=256;
  `reacher_c2222.ep{100,150}.s2222.json`, n=128) — the same populations as the
  same-population `tab:grand` rebuild, so the reference arm and the coins share
  episodes exactly.
* `p in {0.4, 0.5, 0.6, 0.7, 0.8, 0.9}`, chosen to bracket both environments'
  measured call-ratio ranges at both ends. Fixed before the run; no p is added
  afterwards.
* Horizons: PushT `t in {25,50,75,100,150}`, Reacher `t in {25,50,100,150}`.
* Seeds 42-45 (4 evaluation seeds), declared now and pooled without selection.
* Reference arm: SURVEYOR-Base at tau=0.20, same populations, same seeds, run in
  the same batch so the comparison is paired per seed.

Scoring is each environment's operative criterion (PushT block pose at 20 deg,
Reacher all joints within 0.05 rad).

## Frozen predictions

**P-RATE-1a (the optimum moves).** The SR-maximising fixed `p` differs by
`>= 0.15` between the shortest and the longest horizon, in at least one
environment. *Refuted if the same `p` is optimal at every horizon in both.*

**P-RATE-1b (no constant suffices).** No single fixed `p` lies within 2.0pp of
the adaptive tau=0.20 arm at **every** horizon of an environment. *Refuted if
some `p` does.*

**P-RATE-1c (adaptivity is free).** The adaptive arm is within 2.0pp of the
best fixed `p` at every horizon — i.e. adapting the rate costs nothing relative
to an oracle-tuned constant at that cell. *Refuted if the adaptive arm trails
the per-cell best fixed `p` by more than 2.0pp anywhere.*

## Consequence rules (frozen before the run)

* **1a and 1b both hold.** The paper's claim becomes: the verifier supplies a
  *horizon-dependent* re-anchoring rate that no constant matches, and the
  banked coin result is scoped explicitly to coins matched per cell in
  hindsight. This is the intended strengthening.
* **1b refuted.** The paper states plainly that a single constant rejection
  rate reproduces the method's closed-loop success on both dev environments,
  and the accept rule's contribution is restated as **cost control and
  certification, not success**. The headline planning claims stand (they are
  claims about the policy, not about the verifier) but the verifier's role is
  narrowed in the abstract, `sec:results-cert` and the limitations paragraph.
  We report this as a refutation in the main text, not an appendix.
* **1a refuted, 1b held.** Report as: a constant fails everywhere, but not
  because the optimum moves — the coin is simply worse than the test. Weaker
  than the intended result; report as measured.
* **1c refuted.** Report the adaptivity gap as a measured cost of the method
  against an oracle-tuned per-cell constant, in the limitations paragraph.

No cell is excluded after the fact. If a run fails to produce an SR line it is
re-run once with the identical command and, failing that, reported missing.

## Runner

`batch/isca/run_rate_transfer.sbatch` (job submitted 2026-08-11).
Harvest: `batch/isca/build_rate_transfer.py` -> `docs/rate_transfer.csv`.

## Extension A: blind commitment at every depth (frozen 2026-08-11)

The coin is one cheap rival to the accept test; fixed-depth blind commitment is
the other, and it is the sharper one — `d=2` already ties us on PushT at a
*lower* call ratio (.50 against our .53–.63), which we report. We sweep
`d in {1,2,3}` (`--commit fixed --commit-k d`) over the same horizons,
populations and seeds, so coin, commitment and adaptive arms share one table.
`d=1` is every-step drafting (no consumption policy); `d=3` is the full block,
equivalent to `tau=infinity`.

**Frozen prediction P-RATE-2.** No fixed depth is within 2.0pp of the adaptive
arm at every horizon of both environments. *Refuted if some depth is.*

**Consequence.** If P-RATE-2 holds, the method's necessity claim is stated as
measured — every fixed alternative fails somewhere, and none signals which
regime it is in, whereas the adaptive test does not fail anywhere. If refuted,
the paper reports the fixed policy that dominates us and restates the accept
rule's contribution accordingly. Runner:
`batch/isca/run_commit_depth.sbatch`.

## Extension B: drafter training seeds (frozen 2026-08-11)

Every SURVEYOR number in the paper is an *evaluation* seed on one drafter
checkpoint (training seed 42). We report GC-IDM's training-seed spread (8.5pp
at PushT `t=150`) but have never measured our own, and the drafter — unlike the
frozen LeWM encoder — is ours, so "substrate variance is out of scope" does not
cover it. Retrain the PushT drafter at seeds 1 and 2 under the
`gdm_stride10.pt` recipe verbatim, changing only `--seed`, and serve each
through the unchanged headline arm at fixed `tau=0.20`, `k=3` (deliberately not
re-derived per checkpoint).

**Reporting rule, frozen before the runs.** All three training seeds are
reported and the paper quotes the **worst**, exactly as it already does for
GC-IDM. No seed is selected and no cell dropped. If the spread exceeds 3.0pp at
any horizon it is stated in the limitations paragraph and the Table 1 caption.
Runner: `batch/isca/run_drafter_seeds.sbatch`.

## Outcome

*(appended after the runs; see below)*
