# Seed extension battery: seeds 46-49 for every 2026-08-13 result

**Declared 2026-08-13, before any extension cell ran.** Purpose: every result
folded into the paper this week rests on evaluation seeds 42-45. This battery
adds seeds 46-49 to the same cells, byte-identical otherwise, halving the
standard errors under the paper's newest sentences. No new claims; no new
arms; no new thresholds.

## Rule, fixed now

Extension seeds are **pooled with 42-45 without selection**. Every frozen bar
is re-evaluated on the 8-seed pool and reported whichever way it moves:

* **P-PK-1** (matched-`k` parity, 2.0pp band, both `k`, every horizon).
* **P-ORC-1** (oracle within 2.0pp of the strongest in-batch flat at `t=25`).
* **P-DRIFT** deltas (test vs coin at every `rho`, PushT scoping and the
  Reacher re-match sentence as currently written).
* **P-REMATCH-0** (realised-rate gate, 0.015): re-evaluated on the 8-seed
  pooled realised ratio. The `rho=0.9` cell's validity may flip in either
  direction and is reported as it lands.

If any pooled bar flips, the corresponding paper sentence is amended per the
original document's consequence rules; if none flips, captions gain the
larger `n` and nothing else changes.

## Cells

1. **P-PK** (`2026-08-13_pusht_k_matched_parity.md`): all 20 cells
   (`k in {3,50}` x {every-step, accept} x `t in {25,50,75,100,150}`),
   `pusht_c2222`, `n=256`. Runner: the banked `run_pusht_kmatch.sbatch` with
   only the seed list and log prefix changed (edited on the cluster from the
   original so the invocation cannot drift).
2. **P-ORC** (`2026-08-13_reacher_oracle_shortrange.md`): all 6 cells (oracle
   `S=10` goal-terminated, flat RH2, flat RH5 x `t in {25,50}`), post-patch
   builder, `K=4` gate re-checked at `t=25`.
3. **P-DRIFT stage 1+2** (`2026-08-11_autocorrelated_divergence.md`):
   `sigma=0.1` only, {accept, coin, blind} x {PushT, Reacher} x
   `rho in {0,0.9,0.99}`, coin at the stage-2 specified rates (convention
   preserved).
4. **P-REMATCH** (`2026-08-13_drift_coin_rematch.md`): the 3 calibrated-`q`
   Reacher coin cells (`q = 0.5558/0.5464/0.5607`).

47 array tasks, seeds 46-49 in-task, submitted together overnight.

## Amendment: the P-REMATCH extension cells are void (2026-08-14, recorded before the corrected run finished)

All 47 tasks completed overnight (jobs 2335888 pkext / 2335889 orcext /
2335890 rematchext / 2335891 driftext, zero failures, zero re-runs). Config
fingerprinting against the originals (checkpoint, stride, `q`, budget,
population, criterion) verified the pkext / orcext / driftext arrays
byte-identical to their banked counterparts, and the orcext oracle `t=25`
cells all print the required `K (subgoals/ep) = 4`.

The rematchext array is **not** byte-identical: `run_rematch_ext.sbatch`
dropped the `--horizon 2 --receding-horizon 2` line from the eval invocation,
so all 12 cells ran at the eval's default `S=25` instead of the registered
`S=10` (confirmed in every log's summary line; SR collapsed 20--25pp exactly
as a stride mismatch on this checkpoint predicts). The 12 cells are **void as
extensions** — they measure a different configuration, not seed noise. Their
logs are kept as `drift_rematchext_*`. The corrected array
(`run_rematch_fix.sbatch`, job 2336226, logs `drift_rematchfix_*`) restores
the missing line and is otherwise byte-identical; the P-REMATCH pooled
re-scoring below is **pending** until it lands. Incidental observation from
the voided cells, reported not claimed: the realised call ratio at `S=25`
(0.647--0.659) matches the `S=10` originals, consistent with the ratio being
a property of the coin + exhaustion process rather than of the stride.

## Outcome (pooled 8-seed re-scoring; P-REMATCH cells pending job 2336226)

No seed dropped, no cell excluded beyond the voided rematchext array above.
All pooled values are means over seeds 42--49 (equal `n` per seed, so the
pool equals the mean of per-seed SRs).

### P-PK (pusht_c2222, n=2048/cell)

| `t` | every `k=3` | spec `k=3` | delta | every `k=50` | spec `k=50` | delta |
|---|---|---|---|---|---|---|
| 25 | 99.17 | 99.02 | -0.15 | 99.02 | 99.12 | +0.10 |
| 50 | 96.39 | 97.12 | +0.73 | 97.46 | 97.51 | +0.05 |
| 75 | 95.31 | 97.17 | +1.86 | 96.73 | 97.56 | +0.83 |
| 100 | 95.12 | 96.88 | +1.76 | 96.83 | 97.02 | +0.20 |
| 150 | 97.51 | 97.95 | +0.44 | 98.54 | 98.78 | +0.24 |

**P-PK-1 pooled: HOLDS, tightened.** Worst accept-rule deficit is 0.15pp
(`k=3`, `t=25`) against the 2.0pp bar, and the rule now leads in 9 of 10
cells (4-seed: worst 0.29pp, ahead 6/10). The matched-`k` parity sentence
survives the larger `n` unchanged.

### P-ORC (reacher_c2222, n=1024/cell)

| arm | `t=25` | `t=50` |
|---|---|---|
| oracle `S=10` (goal-terminated) | 98.14 (SE 0.42) | 99.32 (SE 0.26) |
| flat RH2 | 95.70 (SE 0.63) | 86.82 (SE 1.06) |
| flat RH5 | 82.32 (SE 1.19) | 93.26 (SE 0.78) |

**P-ORC-0 pooled: PASSES** (+40.0pp over the banked adaptive arm).
**P-ORC-1 pooled: HOLDS, Reading A.** Oracle leads the strongest flat arm by
+2.44pp at `t=25` (z ~ 3.2) and +6.06pp at `t=50`. The draft-noise account
of the short-range tax stands at halved SEs; the hindsight-goal caveat on
the oracle's edge is unchanged.

### P-DRIFT (sigma=0.1, t=100; PushT n=2048, Reacher n=1024)

| env | arm | `rho=0` | `rho=0.9` | `rho=0.99` |
|---|---|---|---|---|
| PushT | test | 95.95 | 96.29 | 95.21 |
| PushT | coin | 96.83 | 96.58 | 96.78 |
| PushT | blind | 96.44 | 95.70 | 95.75 |
| Reacher | test | 92.09 | 91.80 | 91.70 |
| Reacher | coin | 93.46 | 93.46 | 92.87 |
| Reacher | blind | 82.42 | 83.50 | 82.81 |

Test-minus-coin: PushT -0.88 / -0.29 / -1.56; Reacher -1.37 / -1.66 / -1.17.

**One pooled bar flips, and it flips against the scoping rather than any
claim.** The Reacher `rho=0` stage-2 control, refused at 4 seeds (2.54pp,
outside the 2.0pp band), pools to **1.37pp — inside the band**. Per the
original P-DRIFT rule the Reacher leg is therefore scorable on the pool, and
it scores as refuting: the test trails the (over-drafting, coin-favoured)
coin at every `rho` there, matching the P-REMATCH matched-rate result. The
paper's scoping sentence ("Reacher leg failed its `rho=0` control and is
unscored") must be replaced by the pooled result: **P-DRIFT-1 is refuted in
both environments on the 8-seed pool**, with the rate caveat still disclosed.
P-DRIFT-2 holds in both environments (0.88 / 1.37pp). P-DRIFT-3 stays
refuted, but the 4-seed mechanism garnish dissolves: the PushT margin is no
longer monotone in `rho` (-0.88/-0.29/-1.56) and the coin no longer
"improves with correlation" (96.83/96.58/96.78, flat) — neither observation
survives the larger `n`, and neither may be cited.

### P-REMATCH (pooled with the corrected cells, job 2336226, S=10 gate passed in all 12 logs)

| `rho` | coin SR | coin realised | target | gate (0.015) | test SR | test - coin |
|---|---|---|---|---|---|---|
| 0.0 | 91.80 | 0.6476 | 0.6608 | pass (0.0132) | 92.09 | +0.29 |
| 0.9 | 92.68 | 0.6386 | 0.6625 | **FAIL (0.0239)** | 91.80 | (-0.88, not scored) |
| 0.99 | 91.21 | 0.6480 | 0.6570 | pass (0.0090) | 91.70 | +0.49 |

**P-REMATCH-0 pooled: 2 of 3 valid, same cells as at 4 seeds.** The
flagged `rho=0.9` cell flips *away* from validity: its miss widens from
0.0043 beyond the gate to 0.0089 (the corrected extension seeds realised
0.629--0.640 against the 0.6625 target). **P-REMATCH-1 pooled: HOLDS,
tightened to a dead tie** — test +0.29pp at `rho=0` (4-seed: coin +1.76).
**P-REMATCH-2 pooled: REFUTED as before** — test +0.49pp at the valid
`rho=0.99`, far short of the 2.0pp bar. The paper's "ties at every valid
`rho`" sentence survives the pool verbatim at doubled `n`.

### Consequence status

The live manuscript is `paper/main_workshop_final.tex` and it carries the
whole 08-12/08-13 applied pass (matched-`k` clause, Reacher-parity
retraction, regime-map Reacher numbers, scoped drift control, re-match
sentence — all verified by grep, 2026-08-14). A **stale pre-08-13 duplicate
at the repo root** (`main_workshop_final.tex`) briefly read as a missing
consequence pass during this scoring; it is vestigial and should be deleted
or refreshed so it cannot mislead again. Amendments applied to the live tex
(2026-08-14, after job 2336226 landed):

* parity clause: "at worst 0.3pp behind, ahead in six of ten cells" ->
  pooled "at worst 0.2pp behind, ahead in nine of ten";
* P-ORC numbers gain the pooled values/`n` (98.1/99.3 vs 95.7/93.3);
* drift sentence: pooled PushT trails are 0.3--1.6pp and *not* ordered in
  `rho`, the "coin improves with correlation" clause dissolves
  (96.83/96.58/96.78), and the Reacher `rho=0` control *passes* on the pool
  (1.37pp), so the Reacher leg scores directly as refuting rather than only
  via the re-calibrated coin;
* P-REMATCH clause: unchanged — "ties at every valid `rho`" holds verbatim
  on the pool (the amended drift sentence now carries the pooled ranges).
