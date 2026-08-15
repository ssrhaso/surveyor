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

### Extension A (P-RATE-2), PushT leg: **refuted by `d=2`**

*Appended 2026-08-12, after the PushT array completed. Reacher is reported
missing, not as a result; see "voided run" below for why the first attempt does
not count.*

Adaptive reference = the `tau=0.20` arm run in the same batch
(job `2331806`, tasks 54-58), same episodes and same seeds, so every comparison
below is paired. Mean SR over seeds 42-45, `n=256`/cell, 20 deg criterion;
call ratio in brackets.

| `t` | adaptive | `d=1` | `d=2` | `d=3` |
|-----|----------|-------|-------|-------|
| 25  | 98.93 (.84) | 99.22 (1.00) | 98.93 (.73) | 98.63 (.65) |
| 50  | 95.99 (.65) | 96.19 (1.00) | 96.68 (.56) | 95.90 (.42) |
| 75  | 96.68 (.63) | 95.80 (1.00) | 95.90 (.54) | 95.21 (.38) |
| 100 | 96.68 (.61) | 94.34 (1.00) | 97.46 (.53) | 96.09 (.37) |
| 150 | 98.25 (.61) | 98.34 (1.00) | 98.25 (.52) | 95.60 (.35) |

Deficit against the adaptive arm, worst cell per depth:
`d=1` **-2.34** (`t=100`), `d=2` **-0.78** (`t=75`), `d=3` **-2.64** (`t=150`).

`d=2` is within the frozen 2.0pp band at **every** PushT horizon, at a *lower*
call ratio than the adaptive arm everywhere (.52-.73 against .61-.84). On this
environment the accept test is matched by a constant depth that also drafts
less. `d=1` and `d=3` each fail at some horizon, and neither signals it.

**Consequence applied.** P-RATE-2 is stated over *both* environments, and the
Reacher leg was not run, so the prediction is **not yet decided** and the full
refutation consequence (restating the accept rule's contribution) has **not**
been invoked. What was applied is the narrower correction the PushT data
compels: `sec:method-analysis` previously asserted that blind commitment "fails
without signalling it" and has an environment-dependent safe depth. That
sentence now reads that `d=1` and `d=3` each fall short at some horizon *while
`d=2` matches the adaptive arm at all five for less drafting*, so a safe depth
exists but is a per-stack constant that must be known in advance. The appendix
(`app:negatives`) already said blindness "is not uniformly wrong, it ties where
verification is idle"; the main text now matches it.

Two pre-existing statements were re-read against this table. One survives
untouched; the other does not.

* **`sec:results-executor`, "blind commitment at `tau=infinity` ties the PushT
  cell": stands, and an apparent conflict with this table was a
  misreading on our part.** The banked cell is `subgoal=specgcidm`, i.e. the
  **GC-IDM executor with no CEM in the arm**, on `pusht.episodes150s5.json`;
  its own audit job (`2329658`) ran three arms on that identical cell and reads
  blind `96.09` against the accept rule's `97.27` and every-step's `97.66`, so
  `-1.18`pp at `n=256` is a tie. The sweep above is the **CEM** executor on the
  `c2222` populations: a different executor *and* a different population, so the
  two were never in disagreement. No reconciliation run is needed and none was
  performed.
* **`app:negatives`, fixed depth fails unsignaled at "depth-3 on PushT, any
  depth on Reacher": retracted in part.** The depth-3-on-PushT half is now
  directly confirmed. The "any depth on Reacher" half was never measured, and
  this sweep refutes it: `d=1` does not fail on Reacher, it beats the accept
  rule at every horizon.

### Voided run: job `2331836` (recorded, not used)

The first Extension A array passed
`--subgoal specaccept ... --commit fixed --commit-k D`, but `--commit`/
`--commit-k` are read only by `DSparkSubgoalSource`, built only under
`--subgoal dspark`. `SurveyorSource` never saw them, so the flags were silently
ignored and all fifteen PushT tasks ran the plain `tau=0.20` accept rule.
The tell is the call ratio: .605/.608/.611 at `d=1/2/3`, where `d=1` must be
1.000 and `d=3` about .33. Fifteen tasks were three mislabelled replicas of one
configuration, and their SR differed only by GPU nondeterminism. No number from
that job appears anywhere. The Reacher half of the same array failed loudly
instead, because those flags do not exist in its evaluator at all.

The corrected runner (`run_commit_depth_v2.sbatch`, job `2332870`) uses the raw
open-loop path `--subgoal dspark --no-refine`, which loads no checkpoint, and
was smoke-tested **before** submission (job `2332784`) against a pass criterion
written down in advance: `d=1 -> redraft/advance 1.000`, `d=3 -> about .33`,
"0.61 for both = still inert". Measured 1.000 and .371. The table above comes
entirely from that corrected array, whose per-depth ratios (1.00 / .52-.73 /
.35-.65) confirm the knob engaged in every cell.

### Extension A, Reacher leg: **run 2026-08-12. P-RATE-2 holds by the letter, and its justification clause is false.**

Two blockers had to be cleared first, both found by an engagement gate rather
than by a failed array. (i) Reacher's evaluator had no `dspark` choice and no
`--commit`/`--commit-k`. (ii) `DSparkSubgoalSource` hardcoded
`needs_goal=False` and never forwarded a goal latent: PushT's drafter is
goal-*free* so this never surfaced, but Reacher's is goal-conditioned and the
drafter asserts outright, so the source physically could not draft here. The
goal-free path is pinned bit-identical by regression test, so the banked PushT
depth cells are untouched. Gate before submission (criterion fixed first):
`d=1 -> 1.000`, `d=3 -> 0.371`, gap `0.629`. **PASS**, and `0.371` is the same
value PushT's `t=100` cell produced, as a policy-level property should be.

Adaptive reference = the banked `tau=0.20` arm (job `2331806`, tasks 59-62),
same episodes and seeds, so every comparison is paired. `n=128`, seeds 42-45.

| `t` | adaptive | `d=1` | `d=2` | `d=3` |
|-----|----------|-------|-------|-------|
| 25  | 55.66 (r.64) | **65.04** (r1.00) | 57.42 (r.60) | 50.78 (r.43) |
| 50  | 79.49 (r.61) | **87.69** | 76.17 | 68.16 |
| 100 | 89.84 (r.61) | **96.29** | 91.02 | 84.77 |
| 150 | 91.41 (r.63) | 92.38 | 91.21 | 87.50 |

Deficit against the adaptive arm: `d=1` **+9.37/+8.20/+6.45/+0.97**;
`d=2` +1.76/**-3.32**/+1.17/-0.20; `d=3` -4.88/**-11.33**/-5.08/-3.91.

**Verdict.** No fixed depth is within 2.0pp at every horizon of *both*
environments: `d=2` is on PushT but misses Reacher `t=50` by 3.32pp. By the
letter of the prediction, **P-RATE-2 is not refuted**.

**But the consequence clause cannot be stated as written.** It reads "every
fixed alternative fails somewhere, and none signals which regime it is in,
whereas the adaptive test does not fail anywhere." That is false here. `d=1`,
every-step drafting with no consumption policy at all, **beats** the accept
rule on Reacher at every horizon, by up to `+9.37`pp. It does not fail; the
adaptive arm does. What actually holds is much weaker:

> Per environment, a fixed consumption policy matches the accept rule (PushT,
> `d=2`, at a *lower* call ratio) or beats it (Reacher, `d=1`). No *single*
> depth does both, which is all P-RATE-2 ever claimed. The accept rule's
> defence is therefore that one constant serves two environments, not that it
> is the best policy in either.

This is **corroborated, not an artefact**. `2026-08-11_tau_per_env.md` already
measured Reacher's derived floor at `0.106` against the transferred `0.20`,
worth `+6.6`pp at `t=25`. `d=1` is the `tau -> 0` limit and gains `+9.37`pp at
the same cell: same direction, larger magnitude, monotone in how much
re-anchoring is forced. Three independent probes now agree that `tau=0.20`
under-re-anchors on Reacher, and the cost of that is far larger than the
paper's "SR-neutral" framing admits.

**Applied to the paper.** The "SR-neutral" claim is scoped to PushT, where it
was measured; it is not asserted of Reacher at the served `tau`. The
`app:negatives` claim that fixed depth fails unsignaled at "any depth on
Reacher" is **retracted**: `d=1` does not fail there, it wins. What is true,
and now measured rather than assumed, is that no depth is safe at every
horizon of both environments.

### Extension B (drafter training seeds)

Completed and already reported: two further PushT training seeds move SR by at
most 1.8pp, against 8.5pp for GC-IDM. In the paper's limitations paragraph.

### Pooled update (seed battery 2, 2026-08-15): **P-RATE-2 is REFUTED on the 8-seed pool**

Seeds 46-49 (`2026-08-14_seed_extension_battery2.md`) pooled with 42-45
without selection. The 4-seed "holds by the letter" verdict above rested on
one cell — `d=2` at Reacher `t=50`, `-3.32`pp — which pools to **`-0.98`**
(a seed effect, config-fingerprint-verified). On the pool `d=2` is within
the frozen 2.0pp band at **every horizon of both environments** at a lower
call ratio, and `d=1` qualifies too under the same one-sided scoring
(PushT worst `-1.66`, all-positive Reacher, now `+2.3` to `+9.9`pp).
P-RATE-2's refutation consequence is applied to the live tex: "no single
depth is safe everywhere" is retracted; the surviving asymmetry is
provenance (the safe depth is read off the closed-loop sweep itself,
tau/k are derived offline). The main-sweep and 1c sentences also re-quote
on the pool: every `p` within 2pp in *both* environments now, p=0.9's
t=25 lead `+9.08` (was `+9.96`), and the adaptive arm trails the per-cell
best coin at **all four** Reacher horizons (`-9.08/-6.05/-4.30/-2.15`).
Full pooled tables in the battery-2 document's Outcome.
