# Matched-rate random-rejection control (frozen 2026-08-01, BEFORE any run)

Purpose: the decision-content control the 08-01 meat audit identified as the
one experiment a reviewer will punish the paper for lacking. The banked
evidence triangle (blind-commit -18pp = re-anchor rate zero; 2x-CEM fairness =
compute; crossover / stale-oracle = regime) varies rate, compute, and regime —
none holds the RE-ANCHORING RATE fixed while destroying the DECISION CONTENT.
Our own R1 verdict (P-CERT-2: verified SR ~= blind SR under white-noise
corruption) hands a reviewer the question directly: does the accept test buy
closed-loop SR beyond scheduling re-anchors?

Mechanism under test: replace the verifier's per-event decision with an
i.i.d. Bernoulli coin — reject with probability p, ignoring the latent test —
at the SAME per-event rejection probability the deployed verifier exhibited on
the banked cells. Everything else (drafter, populations, seeds, budgets, CEM,
tau bookkeeping, block mechanics incl. exhaustion) is byte-identical to the
banked fixed-spec arm. Implementation: `--random-reject p` in the pusht and
reacher drivers -> `SpecAcceptSubgoalSource(random_reject=p)`; the coin draws
from a DEDICATED generator (seed = cem_seed + 777001) so the draft-sampling
RNG stream is untouched relative to a normal spec run.

## Matched rates: measured, not tuned (derivation shown, verifiable from logs)

Every verification event is exactly one of {advance (verified, ptr<N),
reject-redraft (not verified), verified-but-exhausted redraft}. First-ever
draft per env produces no event, and goal_gate=False in both banked arms, so:

    verified_exhausted = redrafts - rejects - n_envs*n_seeds
    n_events           = advances + rejects + verified_exhausted
    p                  = rejects / n_events

From the banked cells' own printed `[specaccept]` counters:

- **pusht t150** (logs pusht_v4_r_spec_t150_seed{42,43} +
  pusht_ext_spec_t150_seed{44,45}): redrafts 9583, advances 6441, rejects
  7406, first-draws 4x256=1024 -> exhausted 1153, events 15000,
  **p = 7406/15000 = 0.4937**.
- **reacher t150** (logs gatev4c_spec_t150_seed{42..45} +
  rea_ext_spec_t150_seed{46..49}): redrafts 6873, advances 4364, rejects
  5007, first-draws 8x128=1024 -> exhausted 842, events 10213,
  **p = 5007/10213 = 0.4902**.

Same populations, same seeds as the banked cells; nothing tuned against any
closed-loop outcome. (Cross-check, different population: the R2 calibration
event logs give per-event not-verified rates 0.4707 pusht / 0.5239 reacher.)

## Arms (the banked spec cells are NOT re-run; program rule stands)

- **pusht-coin**: pusht.eval --subgoal specaccept --gdm-ckpt gdm_stride10.pt
  --accept-tau 0.20 --gdm-steps 3 --random-reject 0.4937 --mode long
  --episodes-file pusht.episodes150s5.json --goal-offset 150 --eval-budget 300
  --num-eval 256 --score block --angles 20 5 --horizon 2 --receding-horizon 2,
  seeds 42-45 (seed = cem-seed). Quoted stat: block-20, as everywhere.
- **reacher-coin**: reacher.eval --subgoal specaccept --gdm-ckpt
  gdm_reacher_s10.pt --gdm-steps 8 --accept-tau 0.20 --random-reject 0.4902
  --episodes-file reacher_gatev4c.ep150.s999.json --goal-offset 150
  --eval-budget 300 --num-eval 128 --horizon 2 --receding-horizon 2,
  seeds 42-49.

## Banked comparison columns (recorded NOW, per-seed, from the ISCA logs)

| env | seed | banked spec SR |
|---|---|---|
| pusht | 42 | 98.44 (252/256) |
| pusht | 43 | 98.05 (251/256) |
| pusht | 44 | 98.44 (252/256) |
| pusht | 45 | 97.66 (250/256) |
| pusht | pooled | **98.14 (1005/1024)** |
| reacher | 42 | 90.62 (116/128) |
| reacher | 43 | 89.84 (115/128) |
| reacher | 44 | 93.75 (120/128) |
| reacher | 45 | 89.06 (114/128) |
| reacher | 46 | 92.19 (118/128) |
| reacher | 47 | 90.62 (116/128) |
| reacher | 48 | 92.19 (118/128) |
| reacher | 49 | 90.62 (116/128) |
| reacher | pooled | **91.11 (933/1024)** |

(pusht seeds 42/43 come from the v4 confirm's r_spec cells, which at t=150
are the full 256-episode population — the router sends 256/256 to drafting,
recorded in the paper; seeds 44/45 from the seed extension's full-pop spec.)

## Validity gate (mechanics matched, frozen)

The coin arm's pooled call_ratio must land within +/-0.05 of the banked arm's
pooled call_ratio (pusht 0.598 = 9583/16024; reacher 0.612 = 6873/11237).
A miss means the rate-matching failed mechanically; the cell is then reported
as an invalid control AS-IS. No p adjustment, no resubmission — single shot.

## Frozen bar and declared readings (all outcomes reportable)

Margin = banked spec pooled SR - coin pooled SR, per env.

- **B-COIN-1 (per env): margin >= +3pp AND per-seed paired t >= 2**
  (pusht df=3, reacher df=7; paired on the shared seeds/populations)
  -> the verifier's decision content — WHICH replans get rejected — carries
  SR beyond rate-matched random re-anchoring; "certified" keeps its
  closed-loop teeth.
- Coin within +/-3pp of spec -> at these cells the accept test's closed-loop
  SR contribution is indistinguishable from rate-matched random re-anchoring;
  the verification-value claim reverts to the offline calibration table, the
  tau-ignition result, and the banked autocorrelated-error results, and the
  paper must say so.
- Coin ABOVE spec by >3pp -> reported as a refutation.

Expected and explicitly NOT claimed against: the coin arm will likely still
beat flat by a wide margin (drafting + re-anchoring are most of the lift);
the control isolates the verifier's MARGINAL contribution only.

No sigma/p grid, no seed extension, no re-definition after readout.

# ============================================================
# B1 VERDICT (read 2026-08-01, job 2308863, 12/12 COMPLETED)
# ============================================================

Validity gates PASSED both envs: pusht realized per-event p = .4944 (target
.4937), call_ratio .590 (banked .598); reacher p = .4895 (target .4902),
call_ratio .595 (banked .612). The control is valid.

- **PushT: coin 98.05** (1004/1024; per-seed 249/254/254/247) vs banked spec
  98.14 -> margin **+0.10pp, paired t 0.16. TIE.**
- **Reacher: coin 90.33** (925/1024; per-seed 116/115/117/113/117/115/116/116)
  vs banked 91.11 -> margin **+0.78pp, paired t 2.65** — positive on 5/8 seeds
  and never negative, but far below the bar. TIE per the declared +/-3pp band.
- **B-COIN-1 FAILS both envs.** The declared middle reading applies: at
  nominal operation (healthy drafter, in-regime cells) the accept test's
  PER-EVENT SELECTIVITY is not load-bearing for closed-loop SR — rate-matched
  random re-anchoring reproduces it. Verification's demonstrated closed-loop
  value is therefore: (a) SETTING the re-anchoring rate (the coin's p was
  parasitic on the verifier's measured rate; without verifying there is no
  signal for what p should be), (b) ADAPTING it under divergence (R1
  tau-ignition: cr .59 -> 1.00 exactly at the derived tau — a fixed-p coin
  cannot ignite), and (c) the offline calibration table (M1 truth-tracking).
  Consistent with the paper's own sec:why-verifier framing ("verification is
  idle where drafts track reality, and correctly so"; blind commit-2 ties
  spec on pusht anchors) and with the banked reacher blind loss (-4.1pp on
  its own population family: blind3 89.84 vs on-pop spec 93.95 pooled 42-45).
  The paper must state the refined decomposition; nothing here contradicts a
  banked claim.

# ============================================================
# RATE-SWEEP AMENDMENT (registered 2026-08-01 AFTER the B1 verdict above
# was read, BEFORE any sweep cell runs — stated openly)
# ============================================================

B1's tie forces the rate-controller reading; this sweep is the dose-response
that makes it a measured curve instead of an inference. Same coin mechanism,
same populations/seeds as B1, sub-matched rejection probabilities:

- **reacher t150** (s999 pop, n=128, seeds 42-45): p in {0, 0.125, 0.25}.
  p=0 == blind consumption on THIS population (never banked here; the banked
  blind3 cells live on the ep150.t4/t5 population family, not re-run).
- **pusht t150** (spine s5 pop, n=256, seeds 42-45): p in {0.125, 0.25}.
  p=0 on this population IS banked (blind commit-3, the -18pp negative,
  Isambard-era logs, cited as the endpoint; graveyard 10 forbids re-running
  it). The matched-rate endpoint (p=.4937 -> 98.05) is B1's cell.

FROZEN PREDICTIONS:
- **P-RATE-1 (reacher):** SR increases with p toward the matched rate:
  SR(p=0) <= coin(.4902)'s 90.33 - 3pp. Declared alternative reading if it
  ties instead: the re-anchoring rate is ALSO not load-bearing on the s999
  population, the reacher verification-value claim localizes to the
  ep150.t4/t5 population family, and sec:why-verifier gets scoped
  accordingly. Reported as-is either way.
- **P-RATE-2 (pusht):** SR(p) interpolates the banked bracket (commit-3
  ~-18pp at p=0; tie at p=.4937): both mid-p cells sit BELOW spec-3pp or the
  curve is reported flat as-is (which would localize the -18pp to its
  population and must be said).
- **P-RATE-3 (mechanics, descriptive):** call_ratio decreases toward the
  mechanical 1/N floor (~0.36 reacher, ~0.34 pusht) as p -> 0.

Single battery (20 jobs), no extension, no p added after readout.

# ============================================================
# RATE-SWEEP VERDICT — REACHER LEG (read 2026-08-01, job 2308892,
# reacher 12/12 COMPLETED; pusht 8 cells still running, readout pending)
# ============================================================

| p | SR pooled (seeds 42-45) | per-seed | call_ratio |
|---|---|---|---|
| 0 (blind) | **84.57** (433/512) | 107/112/109/105 | .353-.361 |
| 0.125 | 85.74 (439/512) | 113/107/111/108 | .402-.419 |
| 0.25 | 90.43 (463/512) | 116/116/120/111 | .458-.485 |
| .4902 (B1 matched) | 90.33 | 116/115/117/113 | .591-.598 |
| verifier (banked spec) | 90.82 (42-45) / 91.11 (42-49) | — | .612 |

- **P-RATE-1 CONFIRMED:** SR(p=0) = 84.57 <= 87.33 (coin - 3pp); paired
  per-seed diffs vs the matched coin +9/+3/+8/+8 of 128. The re-anchoring
  RATE is load-bearing on the s999 population: blind costs -5.8pp vs the
  matched-rate coin. SR(p) is monotone non-decreasing across the whole
  measured curve (84.6 -> 85.7 -> 90.4 -> 90.3 -> 91.1).
- **P-RATE-3 CONFIRMED:** call_ratio at p=0 sits at the mechanical 1/N
  floor (~0.357 vs predicted ~0.36) and rises with p.
- Descriptive (declared axis): the SR curve saturates by p ~ 0.25 — the
  verifier's operating rate (0.49) sits ON the plateau, ~2x the knee. Read
  with R1: the adaptive headroom is what ignites to 1.0 under corruption;
  a fixed p=0.25 coin has no such response. Rate-controller decomposition
  now fully measured on reacher: rate -5.8pp, content +0.8pp (B1).
- Pusht leg (P-RATE-2) pending; endpoints banked (-18pp commit-3 at p=0,
  B1 tie at p=.4937).

# ============================================================
# RATE-SWEEP VERDICT — PUSHT LEG (read 2026-08-01, job 2308892,
# 20/20 COMPLETED; battery closed)
# ============================================================

| p | SR pooled (seeds 42-45) | per-seed | call_ratio |
|---|---|---|---|
| 0 (banked commit-3) | ~80 (-18pp, banked; not re-run) | — | — |
| 0.125 | **96.97** (993/1024) | 251/248/248/246 | .396-.406 |
| 0.25 | 96.39 (987/1024) | 246/248/249/244 | .455-.460 |
| .4937 (B1 matched) | 98.05 | — | .587-.594 |
| verifier (banked spec) | 98.14 | — | .598 |

- **P-RATE-2 primary prediction FAILS; the declared alternative reading
  applies and is reported as-is:** both mid-p cells sit ABOVE spec - 3pp
  (96.97 / 96.39 vs bar 95.14). The pusht curve is flat from p = 0.125
  upward (within 1.2-1.7pp of the verifier); the banked -18pp blind loss
  is confined to strict p = 0 (full commit-N consumption). On pusht, a
  tiny amount of random re-anchoring recovers nearly everything.
- Combined rate-controller decomposition, now measured on both envs:
  reacher = graded rate dependence (84.6 -> 90.4 over p 0 -> .25, knee
  ~.25); pusht = step at p=0 then flat. In BOTH, the verifier sits on the
  SR plateau, and per-event selectivity adds 0.1-0.8pp (B1).
- HONEST NUANCE, recorded before a reviewer finds it: the verifier's
  operating rate (.59-.61) is ABOVE the SR-knee on both envs — a fixed
  coin at p~.25 matches SR with FEWER diffusion calls (call_ratio .46 vs
  .59). What the coin cannot do: (a) its p was DISCOVERED from the
  verifier's own telemetry — there is no a-priori signal for the knee;
  (b) it cannot ignite (R1: verifier cr .59 -> 1.00 exactly at the
  derived tau under corruption; a fixed coin stays fixed); (c) it carries
  no per-episode regime diagnostic (the call-ratio signal the paper
  already uses). The claim the paper may keep: verification = a
  CALIBRATED, ADAPTIVE rate controller whose fixed-rate shadow is only
  available in hindsight. The claim it must drop: per-event selectivity
  drives closed-loop SR at nominal operation.

Battery closed. Nothing further runs under this prereg.
