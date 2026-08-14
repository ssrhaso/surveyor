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

## Outcome

*(appended after the pooled re-scoring)*
