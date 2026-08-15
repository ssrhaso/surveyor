# P-KPOP: does the accept rule's Reacher advantage depend on `k`, on the population, or on neither?

**Frozen 2026-08-12, before any cell below was run.** Written after the
P-RATE Extension A Reacher sweep (`2026-08-11_rate_transfer.md`) had completed
and been read, and stated as such: this is a **post-hoc-motivated** hypothesis,
pre-registered before its own run.

## Why this exists

The paper now contains two Reacher results that a careful reader will place
side by side, and they point opposite ways.

* `tab:spine`(c), banked: on the max-offset-150 population at `k=4`, the accept
  rule **leads** every-step drafting by `+1.0/+2.3/+0.6`pp at `t=100/125/150`.
  This is the source of the "+0.6 to +2.3pp on long-range Reacher" sentence in
  `sec:results-headline`.
* P-RATE Extension A, measured 2026-08-12: on the seed-2222 confirmation
  population at `k=8`, every-step drafting (`d=1`) **beats** the accept rule at
  every horizon, by up to `+9.37`pp.

Both are in the paper. Neither is wrong on its own terms, but the paper does
not currently say why they differ, and "our own results disagree about whether
the method helps on Reacher" is the single most obvious thing for a reviewer to
find.

Two candidate explanations were verified to be genuinely confounded, by
inspection of the runs rather than assumption:

* **`k` differs**: `4` in the banked row, `8` in the sweep.
* **Population differs**: the banked cells use the sampling-seed-42 population
  (`reacher_horizon150.ep.t12.json`, and `reacher_speck4.ep150.t*.json`, which
  are **byte-identical** to it, md5 `06a1f9b...`, so `tab:spine`(c) is itself a
  properly controlled comparison). The sweep uses
  `reacher_c2222.ep150.s2222.json`, sampling seed 2222. The two episode sets
  share **zero** (episode, start) pairs.

## Design

Fully crossed, everything else byte-identical to the banked arms
(`gdm_reacher_s10.pt`, `S=10`, `RH=2`, budget `2t`, `n=128`, eval seeds 42--45,
joint criterion, single scoring pass):

* **population** $\in$ {spine (`reacher_horizon150.ep.t12.json`), c2222
  (`reacher_c2222.ep150.s2222.json`)}
* **`k`** $\in$ {4, 8}
* **arm** $\in$ {every-step (`--subgoal gdm`, the banked row's own
  configuration), accept rule (`--subgoal specaccept --accept-tau 0.20`)}
* **`t`** $\in$ {100, 150}

16 cells $\times$ 4 seeds. Runner: `batch/isca/run_k_population.sbatch`.

## Frozen predictions

**P-KPOP-0 (reproduction gate).** At `k=4` on the spine population, the accept
arm reproduces `tab:spine`(c) within `2.0`pp ($95.5$ at `t=100`, $95.7$ at
`t=150`) and every-step likewise ($94.5$, $95.1$). *If this fails the banked
cells do not reproduce, the rest of the grid is not interpretable, and P-KPOP-1
is not scored.*

**P-KPOP-1 (the claim: `k` explains it).** The sign of (accept $-$ every-step)
is set by `k` and not by population: **positive at `k=4` on both populations,
negative at `k=8` on both**. *Refuted if the sign tracks population at fixed
`k`, or if it tracks neither.*

**P-KPOP-2 (magnitude).** Wherever the accept rule trails, it trails by less at
`k=4` than at `k=8` on the same population and horizon.

## Consequence rules (frozen before the run)

* **P-KPOP-1 holds (`k` explains it).** The accept rule's Reacher advantage is
  a **`k`-dependent scope condition**, currently unstated. `sec:results-headline`'s
  "+0.6 to +2.3pp on long-range Reacher" gains "at `k=4`", and the paper says
  plainly that at the `k` the sweep used the ordering reverses. This is a
  narrowing, not a rescue: it does not restore a success claim, it explains
  where the existing one applies.
* **P-KPOP-1 refuted, sign tracks population.** The banked `tab:spine`(c)
  advantage is population-specific. It is reported as such, and the
  Extension A retraction stands unchanged.
* **P-KPOP-1 refuted, sign tracks neither.** The two results are reported as
  jointly unexplained, with both cited and no mechanism claimed. We do not
  search for a third factor after the fact.
* **P-KPOP-0 fails.** The banked row is flagged as non-reproducing and that,
  not the confound, becomes the finding.

No cell is excluded after the fact. A run producing no SR line is re-run once
with the identical command and otherwise reported missing.

## Outcome (job 2334817, 64/64 cells, 2026-08-13)

**P-KPOP-0 FAILS, and the reason is the finding.** Per the frozen rule,
P-KPOP-1 is therefore not scored.

| population | `k` | `t` | every-step | accept | accept $-$ every |
|---|---|---|---|---|---|
| spine | 4 | 100 | 98.44 | 96.09 | **-2.35** |
| spine | 4 | 150 | 96.68 | 95.50 | **-1.18** |
| spine | 8 | 100 | 97.66 | 94.72 | **-2.93** |
| spine | 8 | 150 | 95.12 | 93.56 | **-1.56** |
| c2222 | 4 | 100 | 95.12 | 94.73 | -0.39 |
| c2222 | 4 | 150 | 94.14 | 92.97 | **-1.17** |
| c2222 | 8 | 100 | 93.94 | 91.02 | **-2.93** |
| c2222 | 8 | 150 | 92.38 | 92.58 | +0.20 |

The gate failed on one cell of four: every-step at `t=100` on the spine
population read `98.44` against the banked `94.5`, `+3.94`pp outside the
`2.0`pp band. The **accept** arm reproduced on both cells (`96.09` vs `95.5`,
`95.50` vs `95.7`), so this is not a general failure to reproduce.

**Diagnosis, from the run logs rather than inference.** `tab:spine`(c) is
**not `k`-matched**. Its every-step row (`reacher_hz150_s10gdm_*`) ran at
`ddim/50-steps`, the evaluator's default, never overridden; the accept row it
is compared against (`reacher_hz150k4_s10spec_*`) ran at `k=4`. The table
labels the accept row "`k{=}4`" and the every-step row only "`S{=}10`", so
nothing is misstated in the table itself, but the two arms do not share a
sampler budget and the prose in `sec:results-headline` reads them as a
like-for-like comparison.

The direction is against us. This paper's own "spread is load-bearing" result
means more denoising steps narrow the drafted distribution, and Reacher is the
environment where the `k`-rule "shows no step at any `k`". So `k=50` is the
**weaker** setting, and the banked comparison gave every-step the weak `k` and
the accept rule the strong one. Matching `k` on that same population reverses
the sign: `+1.0/+0.6`pp in favour of the accept rule becomes `-2.35/-1.18`
against it.

**What the grid says once `k` is matched.** The accept rule trails every-step
drafting in **7 of 8 cells**, on both populations and at both `k`, by up to
`2.93`pp; the single exception is `c2222`/`k=8`/`t=150` at `+0.20`pp. Neither
`k` nor population rescues it, which is why P-KPOP-1's "the sign is set by `k`"
is refuted in substance as well as unscored by rule: the sign is essentially
constant and negative.

P-KPOP-2 holds in 3 of 4 comparisons (the accept rule trails less at `k=4` than
at `k=8`), consistent with `k=4` being closer to its derived operating point,
but it does not change the sign anywhere.

**The cost side, which the SR table alone understates.** Reading drafter cost
next to success makes the matched-`k` result a trade rather than a loss:

| population | `k` | `t` | every-step SR/NFE | accept SR/NFE | $\Delta$SR | cost |
|---|---|---|---|---|---|---|
| spine | 4 | 100 | 98.44 / 4.00 | 96.09 / 2.54 | -2.35 | $1.57\times$ |
| spine | 4 | 150 | 96.68 / 4.00 | 95.50 / 2.58 | -1.18 | $1.55\times$ |
| spine | 8 | 100 | 97.66 / 8.00 | 94.72 / 5.10 | -2.93 | $1.57\times$ |
| spine | 8 | 150 | 95.12 / 8.00 | 93.56 / 5.02 | -1.56 | $1.59\times$ |
| c2222 | 4 | 100 | 95.12 / 4.00 | 94.73 / 2.63 | -0.39 | $1.52\times$ |
| c2222 | 4 | 150 | 94.14 / 4.00 | 92.97 / 2.57 | -1.17 | $1.56\times$ |
| c2222 | 8 | 100 | 93.94 / 8.00 | 91.02 / 5.13 | -2.93 | $1.56\times$ |
| c2222 | 8 | 150 | 92.38 / 8.00 | 92.58 / 5.04 | +0.20 | $1.59\times$ |

At matched `k` the accept rule buys `1.52`--`1.59`$\times$ fewer drafter
evaluations for `0.4`--`2.9`pp of success. The saving is exactly what
`eq:cost` predicts from a call ratio near `0.63` ($1/0.63 = 1.59$), so the
mechanism is behaving as derived; what is new is that on Reacher it is **not
free**.

This also locates where the headline `26\times` comes from. The banked
comparison set every-step at `k=50` against the accept rule at `k=4`, so most
of that factor is the `k` gap (12$\times$) rather than verified consumption
(1.6$\times$). `tab:headline2` is not affected: it carries an explicit
NFE/replan column ($50$ vs $1.9$), so its asymmetry is disclosed and is the
point of the table rather than a hidden confound. `sec:results-headline`
already separates the two factors correctly ("removing ${\sim}40\%$ of calls at
matched `k`"); the error was calling that second factor SR-neutral on Reacher.

**Consequences applied.**

1. `sec:results-headline`'s "leads by `+0.6` to `+2.3`pp on long-range Reacher"
   is **retracted**. Under matched `k` on the same population the accept rule
   trails. The Reacher parity claim against every-step drafting does not hold.
2. `tab:spine`(c) gains an explicit note that its every-step row is at `k=50`
   and is therefore not `k`-matched to the accept row.
3. This is the fourth independent line pointing the same way (rate-matched
   coin, fixed rate, fixed depth, and now matched-`k` every-step). The paper's
   position is unchanged in kind and firmer in evidence: at the `\tau` we serve,
   the accept rule buys drafter cost and an auditable certificate, not
   closed-loop success. PushT parity (`\pm1.1`pp) is unaffected and was
   separately measured.

No cell was excluded. The banked cells themselves remain valid at their own
`k=50`; what fails is the comparison drawn across them.

### Pooled update (seed battery 2, 2026-08-15): 7/8 becomes **8/8**

Seeds 46-49 pooled with 42-45 (`2026-08-14_seed_extension_battery2.md`).
The lone exception cell (`c2222`/`k=8`/`t=150`, +0.20 at 4 seeds) flips to
**-2.05** on the pool: the accept rule trails every-step drafting in all
8 matched-`k` cells, for **0.4-3.8pp** (was 0.4-2.9) at an unchanged
1.5-1.6x call saving. P-KPOP-2 is now 4/4. The P-KPOP-0 gate cell re-reads
98.44 +/- 0.33 SE at n=8, confirming the banked 94.5 non-reproduction
harder; the k-mismatch diagnosis is unchanged. Paper sentences re-quoted
accordingly (three sites).
