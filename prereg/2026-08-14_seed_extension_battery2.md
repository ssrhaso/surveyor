# Seed extension battery 2: seeds 46-49 for the 2026-08-11/12 control families

**Declared 2026-08-14, before any extension cell ran.** Battery 1
(`2026-08-13_seed_extension_battery.md`) extended the 08-13 result families
to 8 seeds; after it, every control sentence from those families stands at
halved SEs while the equally load-bearing 08-11/12 families still rest on
seeds 42-45. This battery closes that asymmetry. Same rules as battery 1: no
new claims, no new arms, no new thresholds; extension seeds are **pooled
with 42-45 without selection** and every quoted sentence is re-evaluated on
the pool and reported whichever way it moves.

## Families and the paper sentences they re-score

1. **P-RATE fixed-rate sweep + in-batch adaptive arms**
   (`run_rate_transfer.sbatch`, job 2331806; 63 cells: PushT
   `p in {0.4..0.9}` x `t in {25..150}` + Reacher x 4 horizons + adaptive
   comparators). Sentences: "one constant tracks the adaptive arm in both
   environments (PushT: every `p` within 2pp; Reacher: `p=0.9` ahead by
   9.96pp at `t=25`)".
2. **Commitment-depth arms** (`run_commit_depth_v2.sbatch` 15 PushT cells,
   `run_reacher_commit_depth.sbatch` 12 Reacher cells). Sentences: "`d=2`
   matches the adaptive arm at every horizon on PushT at lower call ratio";
   "`d=1` beats the accept rule on Reacher by up to +9.37pp"; the P-RATE-2
   letter/justification split as currently written.
3. **Per-environment tau floors** (`run_tau_per_env.sbatch`, 9 cells).
   Sentences: "Reacher at its derived 0.106 gains +6.6pp at `t=25` yet
   trails the best constant by 3.3pp, past the frozen 2pp band; PushT at its
   floor moves <= 0.5pp".
4. **k-matched Reacher grid** (`run_k_population.sbatch`, 16 cells, both
   populations). Sentences: "once `k`-matched the accept rule trails in 7 of
   8 cells, buying ~1.6x fewer drafter calls for 0.4-2.9pp"; the spine-pop
   ordering reversal that retracted the Reacher parity claim.

## Construction rule (fixed now, after battery 1's S=25 artifact)

Extension runners are generated from the banked originals by `sed` changing
**only** the seed list (`42 43 44 45` -> `46 47 48 49`), the job name
(suffix `ext`), and the log prefixes; the full diff of each ext runner
against its original is inspected before submission and must show nothing
else. A cell whose config fingerprint (checkpoint, stride, horizon flags,
population, budget, criterion) differs from its original is void, exactly as
battery 1's rematchext cells were.

If any pooled sentence flips, the paper is amended per the original
documents' consequence rules; if none flips, captions gain the larger `n`
and nothing else changes.

115 array tasks (63 + 15 + 12 + 9 + 16), seeds 46-49 in-task. Submitted
2026-08-14 as jobs 2336784 (ratetxext) / 2336785 (commitd2ext) / 2336786
(rcommitdext) / 2336787 (tauenvext) / 2336800 (kpopext); the sed diffs were
inspected before submission and showed only the three allowed change
classes in all five runners.

## Outcome

*(appended after the pooled re-scoring)*
