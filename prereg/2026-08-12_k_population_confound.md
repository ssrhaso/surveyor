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

## Outcome

*(appended after the runs)*
