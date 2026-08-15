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

## Outcome (appended 2026-08-15, after all 115 tasks COMPLETED)

All five arrays finished overnight: 115/115 tasks COMPLETED, exit 0, no
re-runs needed, no cell missing an SR line. Fingerprint spot-check per the
construction rule: `rcommitdext_reacher_d2_t50_seed46` against
`rcommitd_reacher_d2_t50_seed42` shows identical episodes file, identical
`episodes_idx[:5]`/`start_steps[:5]`, identical
`commit=fixed mean_commit_depth=2.00`, budget and offset — only the seed
differs. No cell voided. Every number below is the 8-seed pool (42-49),
means of per-seed SR; n=256/cell PushT, 128/cell Reacher.

### Family 1 — P-RATE sweep + adaptive: holds, and hardens against the rule

* PushT adaptive 98.73/96.34/96.63/96.19/97.66 at t=25/50/75/100/150.
  Worst coin deficit per `p`: -1.42/-1.22/-0.10/-0.15/-0.98/-1.37
  (p=0.4..0.9) — **every `p` within 2pp at every horizon**, as at 4 seeds.
* Reacher adaptive 56.93/78.71/90.04/90.33 at t=25/50/100/150. Now **every
  `p` is within 2pp there too** (worst -1.08, p=0.4 at t=50); p>=0.6 beats
  the adaptive arm at most horizons. The quoted sentence's number moves:
  **p=0.9 ahead by +9.08pp at t=25** (was +9.96). Sentence stands, updated.
* P-RATE-1c on the pool: the adaptive arm now trails the per-cell best coin
  **at all four Reacher horizons** (-9.08/-6.05/-4.30/-2.15), not only at
  the short end; PushT worst -0.93, inside the band. The adaptivity gap
  sentence strengthens against us and is re-quoted accordingly.

### Family 2 — commitment depth: **the one flip. P-RATE-2 is REFUTED on the pool.**

Deficits vs the paired adaptive arm (positive = fixed depth ahead):

| env | t | d=1 | d=2 | d=3 |
|---|---|---|---|---|
| PushT | 25 | +0.49 | +0.15 | +0.00 |
| PushT | 50 | -0.44 | +0.34 | -0.05 |
| PushT | 75 | -1.22 | -0.59 | -0.93 |
| PushT | 100 | -1.66 | +1.42 | +0.05 |
| PushT | 150 | +0.39 | +0.83 | -1.71 |
| Reacher | 25 | +7.42 | +0.88 | -6.44 |
| Reacher | 50 | +9.86 | -0.98 | -9.67 |
| Reacher | 100 | +4.98 | +1.08 | -6.83 |
| Reacher | 150 | +2.34 | +0.59 | -3.22 |

* The 4-seed verdict "P-RATE-2 holds by the letter" rested on a single cell:
  d=2 at Reacher t=50, -3.32pp. On the pool that cell reads **-0.98**
  (per-seed: 75.0/75.8/78.1/75.8 then 81.2/77.3/82.8/75.8 for d=2, against
  an adaptive arm that came DOWN on the ext seeds — a plain seed effect,
  verified against the config banners above). **d=2 is now within the frozen
  2.0pp band at every horizon of both environments, at a lower call ratio.**
  d=1 also qualifies under the one-sided scoring the 4-seed verdict used
  (PushT worst -1.66, all-positive on Reacher). P-RATE-2's refutation
  condition — some fixed depth within 2pp everywhere in both environments —
  is met, twice over.
* Collateral: d=3's PushT t=150 miss (-2.64 at 4 seeds) pools to -1.71,
  inside the band; d=3 now fails only on Reacher (-3.2 to -9.7).
* The quoted d=1 sentence updates upward: beats the accept rule on Reacher
  by **+2.3 to +9.9pp** (was "up to +9.37", with a +0.97 floor at 4 seeds).
* **Consequence applied per the frozen P-RATE-2 rule** ("the paper reports
  the fixed policy that dominates us and restates the accept rule's
  contribution accordingly"): `sec:why-verifier`'s "no single depth is safe
  everywhere" and `app:negatives`' "one constant cannot serve both" are
  **retracted on the pool** in the live tex. The surviving asymmetry is
  provenance, stated as such: the safe depth is read off the closed-loop
  sweep itself, while tau/k are derived offline before any rollout.

### Family 3 — per-environment tau floors: verdicts unchanged, numbers move

Reacher tau=0.106 vs served 0.20 (paired): **+4.88/+4.59/+3.23/+1.76** at
t=25/50/100/150 (was +6.64/+3.13/+3.52/+0.58 — still positive everywhere,
still shrinking with horizon, but the t=25 gain was partly seed luck). Vs
the pooled per-cell best coin: -4.20/-1.47/-1.07/-0.39 — the t=25 cell is
**still past the frozen 2pp band, so P-TAU-1 stays REFUTED** (slightly
harder than at 4 seeds: -4.20 vs -3.32). P-TAU-2 mechanism: call ratio
0.80-0.83 vs the served arm's 0.61-0.65, higher at 4/4 horizons —
supported. P-TAU-3 control: PushT at its own floor moves at most **+0.73pp**
(was <=0.5) — supported; the paper's "<=0.5pp" becomes "<=0.8pp".

### Family 4 — k-matched Reacher grid: 7/8 becomes **8/8**

| pop | k | t | every-step | accept | diff | cost |
|---|---|---|---|---|---|---|
| spine | 4 | 100 | 98.44 | 96.68 | -1.76 | 1.57x |
| spine | 4 | 150 | 96.48 | 96.09 | -0.39 | 1.54x |
| spine | 8 | 100 | 97.56 | 94.73 | -2.83 | 1.57x |
| spine | 8 | 150 | 95.41 | 93.36 | -2.05 | 1.58x |
| c2222 | 4 | 100 | 95.60 | 95.22 | -0.39 | 1.54x |
| c2222 | 4 | 150 | 94.34 | 93.16 | -1.17 | 1.56x |
| c2222 | 8 | 100 | 94.43 | 90.62 | -3.81 | 1.57x |
| c2222 | 8 | 150 | 93.07 | 91.02 | -2.05 | 1.59x |

The lone 4-seed exception (c2222/k=8/t=150, +0.20) flips to -2.05 on the
pool: the accept rule now trails every-step drafting in **all 8 cells**, for
**0.4-3.8pp** (was 0.4-2.9) at an unchanged 1.5-1.6x call saving. P-KPOP-2
(trails less at k=4 than k=8) is now 4/4 (was 3/4). The P-KPOP-0 gate cell
re-reads 98.44 +/- 0.33 SE at n=8 — the banked 94.5 non-reproduction is
confirmed harder; the k-mismatch diagnosis is unchanged.

### Paper edits applied (live `paper/main_workshop_final.tex`)

1. `sec:why-verifier`: depth-sweep passage rewritten — d=2 serves both
   environments on the pool; four-seed reading retracted; provenance
   asymmetry stated as the surviving defence.
2. `app:negatives`: same retraction in full, d=3 numbers re-scoped to
   Reacher, +9.4 -> +9.9, worst-cell numbers added.
3. `sec:results-cert`: coin sentence re-quoted (every `p` within 2pp in
   both environments; +9.96 -> +9.1; adaptive behind best coin at all four
   Reacher horizons); tau-repair numbers +6.6/-3.3/<=0.5 ->
   +4.9/-4.2/<=0.8.
4. `sec:calib-tau`: +6.6pp -> +4.9pp.
5. `sec:results-headline` + `fig:pareto` caption + `app:eff` prose:
   "7 of 8" -> all 8; "0.4--2.9pp" -> "0.4--3.8pp" (three sites).

Net reading: pooling doubled the evidence and every family moved the same
way — **against** per-event selectivity and **for** the cost/certificate
framing the paper already adopted. The only sentence that flipped
(P-RATE-2) flipped in the direction the paper's own August-12 blockquote
("per environment, a fixed policy matches or beats the accept rule") had
already conceded halfway.
