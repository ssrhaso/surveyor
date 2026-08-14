# P-PK: does the PushT parity claim survive `k`-matching?

**Frozen 2026-08-13, before any cell below was run.** Post-hoc-motivated and
declared as such: written immediately after P-KPOP
(`2026-08-12_k_population_confound.md`) found the Reacher parity claim rested
on an unmatched `k`.

## Why this exists

P-KPOP established that `tab:spine`(c)'s Reacher comparison ran every-step at
the evaluator's default `k=50` against the accept rule's derived `k=4`, and
that matching `k` reverses the ordering. `sec:results-headline`'s Reacher
parity sentence has been retracted accordingly.

Inspection of the PushT logs shows **the same construction**:

* `tab:spine`(a) "ours, every-step `S=10`" is `pusht_hzs5_s10gdm_*` at
  `ddim/50-steps`.
* `tab:spine`(a) "ours, \methodbase{} `S=10`, `tau=0.20`, `k=3`" is
  `pusht_hzs5_s10speck3_*` at `ddim/3-steps`.

Both arms use the **same** population file (`pusht.episodes150s5as{t}.json`,
`eval_filter=success5`, `n=256`), so unlike the Reacher case the populations
are matched and `k` is the only difference. The surviving PushT parity claim,
"\methodbase{} tracks every-step within `\pm1.1`pp", is therefore a comparison
between `k=50` and `k=3`.

This is the last parity claim standing. It is tested rather than assumed, and
it is tested knowing that the convenient answer is "it holds".

**Why the outcome is not obvious.** On Reacher the `k`-rule "shows no step at
any `k`", so more denoising steps only narrow the distribution and `k=50` is
strictly the weaker setting. On PushT the derivation finds a genuine step at
`k=3` (bias `1.04 -> 0.11`), so `k=50` is *past* convergence rather than short
of it, and the two arms may already be effectively matched. The Reacher result
does not predict this one.

## Design

Everything byte-identical to the banked spine arms except the crossed factors:
`gdm_stride10.pt`, `S=10`, RH$=$2, budget `2t`, `n=256`,
`eval_filter=success5`, single 20deg scoring pass, eval seeds 42--45 (the
banked rows used 2 seeds; 4 here for a tighter estimate).

* **`k`** $\in$ {3, 50}
* **arm** $\in$ {every-step (`--subgoal gdm`), accept rule
  (`--subgoal specaccept --accept-tau 0.20`)}
* **`t`** $\in$ {25, 50, 75, 100, 150}

20 cells $\times$ 4 seeds. Runner: `batch/isca/run_pusht_kmatch.sbatch`.

## Frozen predictions

**P-PK-0 (reproduction gate).** The unmatched pair reproduces `tab:spine`(a)
within `2.0`pp: every-step at `k=50` near `99.6/99.0/98.1/98.1/98.4` and the
accept rule at `k=3` near `99.6/97.9/98.8/97.3/98.1`, at
`t=25/50/75/100/150`. *If this fails, P-PK-1 is not scored and the
non-reproduction is the finding.*

**P-PK-1 (the claim).** At matched `k`, at **both** `k=3` and `k=50`, the
accept rule stays within `2.0`pp of every-step at every horizon.
*Refuted if every-step leads by more than `2.0`pp at any horizon at matched
`k`.*

**P-PK-2 (direction).** If P-PK-1 is refuted, the shortfall is larger at
`k=50` than at `k=3`, i.e. the accept rule is closest to parity at its own
derived operating point.

## Consequence rules (frozen before the run)

* **P-PK-1 holds.** PushT parity is genuine and `k`-matched. The paper says so
  explicitly, the `\pm1.1`pp sentence gains "at matched `k`", and the Reacher
  retraction is reported as environment-specific rather than general.
* **P-PK-1 refuted.** The parity claim is withdrawn on PushT as well, and with
  it the "SR-neutral" framing of verified consumption in
  `sec:results-headline` and `eq:cost`'s second factor. What would remain of
  the accept rule is drafter cost and the certificate, on every environment
  tested, and the paper must say that in the abstract rather than only in
  Limitations. No further rescue is attempted.
* **P-PK-0 fails.** The banked spine row is flagged as non-reproducing and that
  becomes the finding.

No cell is excluded after the fact. A run producing no SR line is re-run once
with the identical command and otherwise reported missing.

## Amendment, before any cell was scored (2026-08-13)

The first submission (job `2335548`) was **cancelled after ~1 minute and its
partial logs deleted**, before any number was read. Recorded here because the
reason changes what this document can test.

`tab:spine`(a)'s population files (`pusht.episodes150s5*.json`) index into
`pusht.h5`, **which no longer exists on the cluster**. Only `pusht_c1111.h5`
and `pusht_c2222.h5` remain, and they are different datasets (1212 and 1211
episodes). The cancelled runner had a fallback that silently substituted
`pusht_c2222.h5`, which would have paired the spine's episode indices with a
different dataset: the indices would resolve, the runs would complete, and
every number would be quietly wrong. The fallback has been removed and the
runner now fails loudly if its dataset is absent.

**Consequences for this pre-registration:**

* **P-PK-0 is withdrawn as unrunnable, not failed.** The banked spine cells
  cannot be reproduced because their dataset is gone. This is a reproducibility
  gap in the banked row, independent of anything measured here, and is recorded
  as such.
* **P-PK-1 and P-PK-2 are retested on the `c2222` population**
  (`pusht_c2222.episodes{t}.json` with `pusht_c2222.h5`, a self-consistent
  pair, `n=256`), which is also `tab:grand`'s population. The bars are
  unchanged.
* **Scope change, stated plainly:** the result will answer "does PushT parity
  survive `k`-matching on a currently available population?" It will **not**
  reproduce `tab:spine`(a)'s numbers, and no claim about that table's specific
  values may be drawn from it. If parity fails here, `tab:spine`(a) is not
  thereby refuted; what follows is that the parity claim is unsupported on the
  population we can still run, which is enough to require the paper to stop
  asserting it without qualification.

## Outcome (job 2335553, 20/20 cells, harvested 2026-08-13)

Means over eval seeds 42--45, `pusht_c2222` population, `n=256`/cell, 20deg
criterion. No cell excluded, no seed dropped, no re-run needed.

| `t` | every `k=3` | spec `k=3` | delta | every `k=50` | spec `k=50` | delta |
|---|---|---|---|---|---|---|
| 25 | 99.12 | 99.12 | 0.00 | 98.93 | 98.93 | 0.00 |
| 50 | 95.90 | 96.48 | +0.59 | 97.76 | 97.66 | -0.10 |
| 75 | 95.60 | 97.07 | +1.47 | 96.58 | 97.95 | +1.37 |
| 100 | 94.53 | 97.07 | +2.54 | 96.58 | 96.88 | +0.29 |
| 150 | 98.24 | 97.95 | -0.29 | 98.83 | 99.12 | +0.29 |

### Verdicts against the frozen predictions

* **P-PK-0: withdrawn as unrunnable** per the amendment above (dataset gone);
  the `c2222` numbers are consistent with `tab:grand`'s cells on the same
  population, as expected for a different population than `tab:spine`(a).
* **P-PK-1 (the claim): HOLDS, decisively.** At matched `k`, at both `k=3`
  and `k=50`, the accept rule never trails every-step by more than **0.29pp**
  at any horizon, against the frozen 2.0pp bar, and *leads* in six of ten
  cells (by up to +2.54 at `k=3`, `t=100`).
* **P-PK-2: moot** (only scored on a P-PK-1 refutation).

One mechanism observation, reported not claimed: every-step improves from
`k=3` to `k=50` (e.g. 94.53 -> 96.58 at `t=100`) while the accept rule is
nearly `k`-invariant (97.07 vs 96.88), consistent with the verifier absorbing
noisier `k=3` drafts by re-anchoring more often, which is the same telemetry
seen when the derived `k=3` was first served.

### Consequence applied (the P-PK-1-holds rule)

PushT parity is genuine and `k`-matched. In `main_workshop_final.tex`:
`sec:results-headline`'s parity sentence gains the matched-`k` clause ("at
worst 0.3pp behind, ahead in six of ten cells"), the `tab:spine` caption's
disclosure notes that a `k`-matched re-test passes the frozen bar, and the
Reacher retraction stands as environment-specific rather than general.
