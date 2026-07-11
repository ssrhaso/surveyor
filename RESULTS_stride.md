# Stride-Spacing Round — Results

Direction doc: `DIRECTION_stride_spacing.md`. Question: does a finer subgoal stride
(re-anchored at the finer cadence, control loop scaled with it) extend the validated
"re-anchor more = better" trend, or is a fine-stride target degenerate (too close to be
informative)?

Fixed anchors from the DSpark rounds (all at native stride-25): gdm commit-1 SR
91.9%/83.1% (20°/5°, n=62, seeds 42–45, goal_offset 150) > commit-3 73.8% > open-loop-6
72.6% — SR monotone in re-anchor cadence.

---

## Stage 0 — CPU pre-gate: fine-stride latent displacement vs encoder noise floor

**Verdict: PROCEED (decisive).** The stride-5 target is well-posed: latent displacement
at S=5 sits ~4.7× above the encoder noise floor and the distributions barely overlap.

Setup: `ffjepa/probe_stride_displacement.py`, full `pusht_expert_train.h5` (local laptop
run, CPU, frozen local encoder), 128 successful@20° episodes × 4 uniform-random anchors
= 512 anchors, seed 42. `disp(S) = ‖E(t+S)−E(t)‖/‖E(t+S)‖`; floor = disp(1) (nothing
task-relevant happens in one env step). Raw stats: `Results/stage0_stride_displacement.json`.

| S | disp p10 | p50 | p90 | ×floor (p50) | cos_long | cos_succ |
|---|---|---|---|---|---|---|
| 1 (floor) | 0.027 | 0.074 | 0.148 | 1.00 | 0.21 | 0.71 |
| 5 | 0.151 | 0.346 | 0.597 | **4.69** | 0.39 | 0.58 |
| 10 | 0.365 | 0.656 | 1.014 | 8.88 | 0.60 | 0.22 |
| 15 | 0.565 | 0.926 | 1.227 | 12.54 | 0.78 | −0.06 |
| 25 | 0.875 | 1.254 | 1.439 | 16.99 | 1.00 (def) | −0.34 |

(`cos_long(S)` = cos(Δ_S, Δ_25) at the same anchor; `cos_succ(S)` = cos of successive
S-segments Δ_[t,t+S] vs Δ_[t+S,t+2S].)

Read-offs:

1. **Not degenerate.** disp(5) p10 (0.151) > disp(1) p90 (0.148): even the quietest
   decile of 5-step moves exceeds the noisiest decile of the floor. The kill condition
   (disp(5) ≈ disp(1)) is cleanly absent.
2. **Directionally structured, not jitter.** cos_long(5)=0.39 (short moves point where
   the 25-step move goes); cos_succ(5)=0.58 — successive 5-step segments are MORE
   directionally consistent than successive 25-step segments (cos_succ(25)=−0.34,
   long segments anti-correlate as PushT trajectories curl through arcs). Fine strides
   see a locally-smooth trajectory; coarse strides see direction reversals.
3. **Calibration.** disp(25) p50 = 1.254 reproduces the known stride-25 noop scale:
   gdm_faithful's m+1 rel_err 0.159 vs this noop ⇒ ~7.9× beat, matching the recorded
   ~7.2× from the suffix-decay probe — the probe measures the same quantity the drafter
   is scored on.
4. Implied Stage 2 context: a stride-5 drafter faces noop_err ≈ 0.35. The degeneracy
   gate at Stage 2 is `rel_err ≈ 0.35`, i.e. any useful drafter must land well under it.

## Stage 1 — faithful-recipe retrains at S ∈ {5,10,15,25} (in flight, 2026-07-07)

**Dense rebuild done (Isambard job 5540587, 4m50s on GH200 vs "overnight" on ofs-v01):**
full h5 re-downloaded from HF byte-identical (46,300,921,856 B), `subgoals_dense_full.pt`
rebuilt = 1,456,688 latents / 12,042 kept@20° episodes (9,xxx flagged 5°), ‖z‖ mean 13.91
std 0.82 (≈√192) — healthy. Train array = job 5542760 (S∈{5,10,15,25}, faithful recipe,
in-job Stage 2 diag at matching `--subgoal-step S --stride S`); gate job 5542761
(`batch/run_stride_gate.sbatch`) parses the diags after the array and auto-submits the
Stage 3 A/B (`batch/run_stride_ab.sbatch`, 8×1-GPU: S∈{5,25}×seeds 42–45) iff
degeneracy/collapse/S=25-sanity gates all pass; on any fail it writes
`logs/STRIDE_GATE_FAILED.txt` and submits nothing.

Note (caught pre-run): `diag_gdm` Probe B takes its own `--stride` for the eval-
distribution target offset — the in-job diag passes `--stride S` alongside
`--subgoal-step S`, otherwise every arm would have been diagnosed against 25-step
targets. First array submission (5540588) predated this fix and was cancelled.

## Stage 2 — gate outcome and resolution protocol (2026-07-07)

Trains completed (all 4 arms, ~2h each). In-job diag, B-success block (rel_err /
noop_err / collapse): S=5 0.230/0.454/0.994 · S=10 0.318/0.742/0.994 ·
S=15 0.330/0.940/0.998 · S=25 0.400/1.228/0.994. Probe A (S=25): 0.432/1.253,
cos_move 0.90. Every arm beats no-op 2–3×, zero collapse, B-success ≫ B-failed
(structure as expected).

**The mechanical gate (job 5542761) FAILED the array and blocked Stage 3** on
(a) S=25 sanity: B-success rel_err 0.400 outside [0.10,0.25], and (b) S=5
degeneracy: 0.230 ≥ 0.5×noop (by 0.007). Assessment: both thresholds were
implementation choices of the auto-gate, not the brief's pre-registered
criteria, and (a)'s band was calibrated against the WRONG probe — the 0.159
anchor comes from probe_suffix_decay's end-anchored runway windows on the
success5 subset, while diag Probes A and B sample uniform / arbitrary windows
over whole episodes (incl. PushT's fast early-push regime and end-clamped
windows). Recorded evidence that window placement alone is a big lever on the
SAME weights: gdm_faithful m+1 = 0.16 (runway) vs 0.25–0.52 (start-mode) from
the DSpark round. The brief's actual Stage-2 kill was `rel_err ≈ noop_err`;
0.23 vs 0.45 is not that.

Data-level forensics already in hand: the rebuilt dense file yields **9,094
mask-5 episodes and 1,100,325 pairs ⇒ 171,940 iters — EXACTLY the recorded
gdm_faithful values** (episode count and iteration count both match to the
digit), so the rebuilt training population is byte-equivalent in size.

**Pre-registered resolution (job 5550512, submitted before results known),
Stage 3 submits ONLY if BOTH legs pass:**
1. `gdm_faithful.pt` through the IDENTICAL diag the S=25 arm got. Pass:
   |faithful − arm| ≤ 0.05 rel_err on BOTH Probe A and B-success. If faithful
   lands near 0.16–0.20 while the arm sits at 0.40–0.43, the retrain genuinely
   diverged → debug, no Stage 3.
2. `gdm_stride25.pt` on the EXACT anchor protocol (probe_suffix_decay,
   subset_longeval + episodes150, seed 42). Pass: within ±0.03 per position of
   the recorded 0.1593/0.2288/0.3179.
Marginal cases: report, human call — no silent pass.

**RESOLUTION (job 5550512, 2026-07-07 evening): BOTH LEGS PASS — pipeline faithful,
gate bands were miscalibrated, Stage 3 submitted (job 5551767).**
- Leg 1, identical-diag comparison, faithful vs S=25 arm: Probe A 0.4207 vs 0.4322
  (Δ=0.012), B-success 0.4023 vs 0.3996 (Δ=0.003), B 0.5857 vs 0.5801 — all ≤ 0.05.
  `gdm_faithful` ITSELF scores ~0.40 on the diag's uniform-window distribution while
  scoring 0.159 on the anchor's runway windows: the "sanity fail" was 100% probe
  distribution, 0% model.
- Leg 2, anchor protocol on gdm_stride25: **0.1593 / 0.2362 / 0.3246** vs recorded
  0.1593/0.2288/0.3179 — m+1 exact to 4 decimals, all within ±0.03.
Stage 3 (s5/s25/cad × seeds 42–45, n=62) runs with `--dump-traces` (per-episode
success flags + per-replan (z_cond, drafted subgoal) pairs) for failure anatomy.

## STAGE 3 RESULT (job 5551767, 2026-07-07 night): SPACING IS A LIVE LEVER — precision headline

Means over seeds 42–45 (n=62, block criterion, goal_offset 150, budget 300):

| arm | 20° | 5° (strict) | per-seed 5° |
|---|---|---|---|
| **s5** (stride-5 drafter, horizon=receding=1) | **92.34%** | **91.94%** | 91.9 / 96.8 / 85.5 / 93.5 |
| s25 (baseline package, rebuilt pipeline) | 91.13% | 86.29% | 85.5 / 90.3 / 82.3 / 87.1 |
| cad (stride-25 targets, re-anchor every 5) | 78.63% | 77.82% | 79.0 / 74.2 / 69.4 / 88.7 |

1. **Decision rule: PROCEED.** 20° is a tie (+1.2pp, noise — saturated criterion);
   at the strict 5° criterion stride-5 wins by **+5.7pp, positive in every seed
   individually** (+6.5/+6.5/+3.2/+6.5). Fine-grained control buys PRECISION:
   s5's strict-angle SR ≈ its loose SR (91.9 vs 92.3), while s25 drops ~5pp
   between criteria.
2. **s25 re-derivation matches the historical anchor** (91.1/86.3 vs 91.9/83.1,
   within noise) — the rebuilt pipeline is faithful in closed loop too.
3. **The cadence-isolation control FAILED (−12.5pp vs s25)** — re-drafting a
   25-step-away subgoal every 5 steps makes CEM chase a churning target (each
   re-draft resamples from a wide horizon-25 posterior). So the round's claim is
   sharper than "re-anchor more = better": **match the control loop to the
   subgoal scale; finer scales win.** cad is the attribution control: s5's win
   is the granularity package, not cadence alone.
4. Cost: s5 pays 60 diffusion calls/episode vs s25's 12 — the 5× premium the
   DSpark Round 2 machinery (k=8 drafts × τ-verified acceptance ≈ 12×) is built
   to recover. Fine stride at below-baseline diffusion cost is the combined story.
5. Stage 4 sweep (S ∈ {10,15}, same protocol, job 5553714) submitted to map the
   knee. Failure traces captured for all 12 Stage 3 arms (`runs/stride/traces_*`).

## STAGE 4 SWEEP RESULT (job 5553714, same night): CLEAR INTERIOR OPTIMUM AT S=10

Full SR-vs-stride curve, means over seeds 42–45 (n=62, block, 20°/5°):

| S | 20° | 5° | per-seed 5° |
|---|---|---|---|
| 5 | 92.34 | 91.94 | 91.9/96.8/85.5/93.5 |
| **10** | **97.18** | **96.37** | 95.2/98.4/96.8/95.2 |
| 15 | 96.77 | 94.36 | 93.5/91.9/96.8/95.2 |
| 25 | 91.13 | 86.29 | 85.5/90.3/82.3/87.1 |

**S=10 beats the baseline package by +6.1pp (20°) and +10.1pp (5°), positive in
every seed at both criteria** (5° paired: +9.7/+8.1/+14.5/+8.1), and beats S=5 as
well (+4.8/+4.4) — the curve has a genuine interior optimum, exactly the "knee"
outcome. Mechanistic read: S=5 sits at the weakest drafter-SNR point (rel/noop
0.51) with the shallowest CEM lookahead; S=25 is imprecise; S=10 balances target
informativeness against control granularity. One seed hit 62/62 at 20°.
Diffusion premium at S=10: 30 calls/ep = 2.5× baseline — cheaper to recover than
S=5's 5×. Overnight portfolio (2026-07-07→08): spec-accept smoke 5563729 →
arms 5563730 (5 configs × 4 seeds incl. k-isolation control), seeds 46–49 across
the curve (5563731), random-init n=256 for s10/s5 (5563732), blind commit-2 @s10
(5563733, the verifier-value contrast), Phase-A gates @s10 (5563735).

## OVERNIGHT RESULTS (2026-07-08 morning; 44/44 tasks completed)

**1. Random-init n=256 — the paper's weakest cell — S=10 is PERFECT.**
s10 seed42: **100.00/100.00 (256/256 at BOTH criteria)**; s10 seed43:
**100.00/100.00**; s5 seed42: 100.00/94.53. Reference points on the SAME fixed
per-episode-goal protocol and code path: our stride-25 repro = 96.48/78.12
(seed 42, n=256); paper = 82.42%. Fine-grained control takes the strict-angle
random-init number from **78% → 100% across 512 episodes with zero failures.**
(Not the old 100% artifact: that bug was fixed and the s25 control scores 78.12
on this exact code — the delta is the treatment. Confirmation seeds 44/45
queued, job 5578762.)

**2. Speculative acceptance (Phase B closed loop): SR holds (or rises) at
~0.4–0.6× the diffusion calls.** Means over seeds 42–45 (20°/5° / mean
call_ratio; every-step references: s10 = 97.18/96.37, s5 = 92.34/91.94):

| arm | 20° | 5° | call_ratio |
|---|---|---|---|
| s10 τ=0.15 k=8 | 97.98 | 95.16 | 0.67 |
| s10 τ=0.20 k=8 | 97.98 | 94.76 | 0.56 |
| s10 τ=0.30 k=8 | 96.77 | 94.76 | 0.45 |
| s10 τ=0.20 k=50 | 97.98 | 95.56 | 0.55 |
| **s5 τ=0.20 k=8** | **97.98** | 94.76 | **0.42** |

- vs its own every-step arm, spec-accept is SR-neutral (Δ within noise) at
  ~0.45–0.67× the calls; with k=8 drafts the total denoising compute is
  **~11× below every-step s10** and **~4.4× below the stride-25 BASELINE
  (135 vs 600 NFE/episode) while scoring +6.9/+8.5pp above it.**
- **At S=5, spec-accept BEATS its own every-step arm by +5.6pp (20°)** —
  skipping verified re-drafts *stabilizes the target* (the anti-churn effect
  the cad arm exposed). The verifier isn't just cheaper; at fine strides it's
  better control.
- k-isolation: k=8 vs k=50 at τ=0.20 differs by ≤0.8pp (noise) — few-step
  drafting is confirmed free in closed loop.

**3. Blind commit-2 @ S=10: 98.79/96.37 — blind consumption does NOT hurt at
fine stride** (contrast: commit-3 at S=25 lost 18pp). Target coherence, not
verification, is the main SR effect at S=10 on this population; the verifier's
value is adaptivity + the divergence guarantee at equal-or-lower cost
(call_ratio 0.45–0.56 adaptive vs 0.50 fixed), which is the safe default under
distribution shift.

**4. 8-seed stride curve (5° criterion, seeds 42–49):** s5 92.94 · **s10
96.37** · s15 94.36 · s25 87.70. Interior optimum unchanged with doubled
seeds; s10−s25 = **+8.7pp at SE ≈ 0.9pp**. s10's per-seed spread is remarkably
tight (95.2–98.4).

**5. A2 simulator validated as a design tool:** it predicted call_ratio 0.617
at s10/τ=0.20; the closed loop measured 0.54–0.59 (slightly MORE acceptance —
CEM actively drives toward the target, so verification passes more often than
expert replay). Offline gate → closed-loop transfer confirmed.

**6. Failure anatomy (all 8 S=10 seeds, `ffjepa/probe_failure_anatomy.py`,
traces + goal latents): ALL 18/496 residual strict-criterion failures are
NEAR-MISSES** — final latent distance-to-goal within the success
distribution's p95 (one at d=7.3, closer than the median success: an
angle-only miss). Zero unreachable-subgoal failures, zero diverged, zero
stuck. Success episodes' commanded/achieved step ratio p50 = **1.06** — the
S=10 drafter commands subgoals the agent reaches almost exactly, i.e.
**FF-JEPA's own stated #1 failure mode (unreachable subgoals) is eliminated
at fine stride.** No further SR mechanism is warranted on PushT; the residue
is tolerance-boundary quantization. Lever 4 closed.

## REGRESSOR CLOSED-LOOP ARM (job 5578787, 2026-07-08): diffusion EARNS ITS KEEP — spread thesis vindicated

The deferred diffusion-vs-regression question, closed. Deterministic z_cond→m+1
regressor (the offline-DOMINANT JointMLP from the probe battery) as subgoal source,
re-anchoring every replan, same protocol as the stride arms (n=62, goal_offset 150,
budget 300, seeds 42–45). Means (20°/5°), references in parentheses:

| arm | 20° | 5° | per-seed 5° |
|---|---|---|---|
| regressor S=10 | 91.94 (gdm 97.18) | 91.94 (gdm 96.37) | 90.3/90.3/91.9/95.2 |
| regressor S=25 | 86.29 (gdm 91.13) | 74.60 (gdm 86.29) | 85.5/71.0/67.7/74.2 |

1. **The registered prediction ("loses at S=25, coin-flip at S=10") was half right —
   the regressor loses at BOTH strides**: −4.8/−11.7pp at S=25, −5.2/−4.4pp at S=10
   (5°: behind in 3/4 seeds, ties the 4th). Diffusion is NOT replaceable by a 2ms MLP
   even at the winner config.
2. **Offline rel_err is the wrong selection metric for closed loop.** The regressor
   DOMINATED the drafter on offline rel_err (probe battery, RESULTS_dspark.md) yet
   loses in closed loop at every cell — regression-to-the-mean targets carry zero
   conditional spread, and CEM needs target diversity across replans. This is the
   cleanest evidence yet for the spread thesis, and it means the paper's cost story
   keeps the drafter: spec-accept's ~135 NFE/ep is the honest floor, not 1 MLP pass.
3. Cost framing for the writeup: the right cost comparison stays
   "spec-accept vs every-step drafting" (4.4× under baseline at +7/+9pp SR), with the
   regressor as the ablation that shows why the cheap deterministic alternative
   sacrifices SR. Traces at `runs/regressor/traces_s{10,25}_seed*.pt`.

## Drafter battery (job 5550807, 2026-07-07): decay is horizon-intrinsic, not positional

All four stride drafters, anchor protocol (n=256, runway windows, K=8 seeds), rel_err
(cond_spread) by position:

| S | m+1 | m+2 | m+3 |
|---|---|---|---|
| 5 | 0.093 (0.065) | 0.112 (0.077) | 0.140 (0.098) |
| 10 | 0.105 (0.080) | 0.147 (0.103) | 0.178 (0.126) |
| 15 | 0.132 (0.093) | 0.184 (0.130) | 0.219 (0.159) |
| 25 | 0.159 (0.112) | 0.236 (0.171) | 0.325 (0.236) |

(S=25 row = the anchor triplet again, third independent reproduction.)

**Headline: prediction error collapses onto ABSOLUTE horizon (env-steps ahead),
nearly independent of which drafter/position produces it** — +10 steps: 0.105 (S10
first hop) ≈ 0.112 (S5 deep position); +15: 0.132 ≈ 0.140; +30: 0.178 ≈ 0.184.
Consequences:
1. "Suffix decay" is horizon-intrinsic uncertainty, NOT an architectural deficiency
   of the block — deep positions are already at horizon-optimal fidelity. This
   retires the remaining motivation for intra-block architecture work (semi-AR
   sampling head etc.) independently of probe-2's earlier NULL.
2. The stride-5 block's deepest position (+15 steps, 0.140) outpredicts the
   stride-25 drafter's FIRST hop (+25, 0.159) — fine-stride blocks give speculative
   acceptance real runway at better-than-baseline fidelity.
3. The cost of coasting (skipping a re-draft) is exactly a walk up this horizon
   curve (e.g. consuming +15 drafted at t vs re-drafting at t+10: 0.140 vs 0.093);
   the A2 acceptance simulator prices that walk with verification filtering.
4. Spread is healthy at every stride (grows with position, no collapse anywhere;
   relative to target displacement, finer strides are ~2× richer).
All strides clear the degeneracy gate by 3.6–6×.

## Stage 3 design amendment (pre-results): cadence-isolation arm

`horizon = receding_horizon = S/5` means the s5 arm changes stride, re-anchor
cadence, and CEM search depth together — deliberate (the doc's "control
granularity" package) but a loss would be unattributable between "target too
close" and "shallow search". Added arm **cad**: stride-25 drafter, `horizon=5,
receding_horizon=1` → re-anchors every 5 env-steps (like s5) with 25-step
targets and full-depth rollouts (like s25). Well-posed (rollout ≈ target
distance); note the converse control (s5 targets with horizon-5 rollouts) is
NOT run because a 25-step terminal cost toward a 5-step target is the
degenerate objective mismatch the direction doc rules out. Read-off: cad vs
s25 isolates cadence at fixed stride/depth; s5 vs cad isolates the finer
target beyond cadence. 12 tasks total (3 arms × seeds 42–45), same n=62.

Statistical resolution note: n=62 × 4 seeds ⇒ SE ≈ 1.7pp per arm at 90% SR;
differences under ~3pp are noise — pre-commit to not over-reading them.

## Side probe — CEM cost-signal SNR at fine stride (2026-07-07, local CPU)

**The degeneracy mechanism is absent in-model — in fact inverted.**
`ffjepa/probe_cost_snr.py`: 64 end-anchored runway windows, per window the
frozen predictor rolls the TRUE demo continuation block + 128 real decoy
blocks (from other success episodes; same decoy windows serve both strides)
+ a zeros hold block, cost = terminal L2² to the TRUE E(t+S) — byte-matched
to SubgoalCostModel's math and Stage 3's loop geometry (horizon = S/5, H=1).
Raw: `Results/cost_snr.json`.

| metric (mean±sd over windows) | S=5 (horizon 1) | S=25 (horizon 5) |
|---|---|---|
| true-block percentile among decoys | **0.929 ± 0.062** | 0.823 ± 0.266 |
| decoy-cost cv (dynamic range) | **0.271** | 0.085 |
| best_gap (median−min)/median | 0.564 | 0.236 |
| act_sens (terminal-latent spread) | 0.063 | 0.169 |
| hold percentile | 0.546 | 0.507 |

Read-offs: (1) the 5-step cost function ranks the true plan HIGHER and far
more consistently than the 25-step one (whose per-window ranking is blurred by
long-horizon rollout error in a subset of windows — ±0.27 sd); (2) relative
cost dynamic range is ~3× larger at S=5; (3) actions displace the predicted
terminal latent less at 5 steps (0.063 vs 0.169) but the signal CEM consumes
(ranking) is sharper. Mechanistic support for fine-grained control and for WHY
frequent re-anchoring wins (short rollouts sit in the predictor's trustworthy
regime). Caveat: in-model ranking of realistic candidates, not closed-loop SR
— the arbiter remains Stage 3; what this kills is the "CEM picks noise at
stride 5" explanation for any outcome.

- Faithful recipe pinned from `batch/run_recipe.sh` (produced `gdm_faithful.pt`):
  dense file + `--mask 5 --subgoal-step 25 --batch-size 128 --lr 5e-5 --weight-decay 1e-3
  --grad-clip 1.0 --lr-schedule warmup_cosine --epochs 20 --ema-decay 0 --amp`.
  Stride-S arm = identical invocation with `--subgoal-step S` (dense file_stride=1 ⇒
  horizon = S frames). Pair counts are identical across arms (sliding conditions over
  every frame), so training compute is matched by construction.
- `subgoals_dense_full.pt` lives only on unreachable ofs-v01 ⇒ rebuilt on Isambard:
  full h5 re-downloaded from HF (`quentinll/lewm-pusht`), dense build
  `batch/run_build_dense_isambard.sbatch` (`build_subgoals --stride 1`, frozen local
  encoder), then `batch/run_train_stride_array.sbatch` (array over S, dependency-chained,
  Stage 2 diag in-job).
- S=25 arm doubles as the re-derivation sanity check: its diag must reproduce
  gdm_faithful.pt's offline curve (Probe A/B rel_err ≈ 0.16) before any cross-stride
  read.

## PAPER-PROTOCOL WINNER CELLS (job 5579718, 2026-07-09 overnight): beats the paper at its own protocol

Winner configs at the paper's own two protocols (n=256, seeds 42/43, means at 20°/5°;
short = success5 population, budget 50; long75 = episodes75 runway, budget 150):

| cell | S=10 every-step | S=10 + spec-accept | paper (FF-JEPA DM) | our repro |
|---|---|---|---|---|
| short t=25 | 98.44 / 94.14 | 98.44 / 95.31 | 96.09 / — | 95.31 |
| long t=75 | 97.07 / 93.75 | 97.86 / 91.61 | 91.80 / — | 88.67 (multiseed 90.0) |

1. **Both winner configs beat the paper's published numbers on the paper's own
   protocols** — +2.4pp short, +6.1pp long at 20° — and spec-accept is SR-neutral
   vs every-step at both.
2. **call_ratio is horizon-dependent, as the mechanism predicts:** ~0.57 at t=75
   (matching the controlled-protocol 0.56) but 0.81–0.87 at t=25 — a 50-step episode
   has ≤5 replan boundaries and the first draft is unavoidable, so there is little
   room to coast. The cost win is a long-horizon phenomenon; frame it that way.

## FIXED-POPULATION HORIZON SWEEP (job 5579798, 2026-07-09 overnight): baseline degrades with horizon, ours does NOT — gap widens monotonically

The BeyondNext-style question, answered on OUR protocol with population controlled:
same 256 episodes (episodes150 reindexed per offset), budget 2×offset, seeds 42/43,
means at 20°/5°:

| t | s25 baseline | s10 every-step | s10 + spec-accept | 5° gap (s10−s25) |
|---|---|---|---|---|
| 25 | 98.05 / 93.75 | 99.80 / 97.07 | 99.61 / 95.12 | +3.3 |
| 50 | 93.56 / 84.96 | 97.46 / 93.75 | 97.07 / 93.94 | +8.8 |
| 75 | 92.58 / 83.20 | 98.24 / 93.75 | 97.86 / 92.97 | +10.5 |
| 100 | 93.75 / 81.84 | 97.86 / 93.75 | 98.63 / 92.58 | +11.9 |

1. **The stride-25 baseline decays with horizon at the strict criterion**
   (93.75 → 81.84, −11.9pp from t=25 to t=100) while **S=10 is FLAT at 93.75
   from t=50 onward** (and 97–99 at 20° everywhere). The advantage is not a
   constant offset — **the gap widens monotonically with horizon** (+3.3 → +11.9pp),
   which is the long-horizon claim in its strongest form, free of the population
   confound (same 256 episodes at every offset).
2. **Spec-accept tracks every-step at every horizon** (max gap 1.2pp, sign varies)
   — the cost cut stays free across the whole curve.
3. Protocol note for the writeup: this is the FF-JEPA final-window protocol with
   budget 2×offset, NOT BeyondNext's fixed-budget-50 LeWM protocol — compare curve
   SHAPES to their Fig. 2, not absolute numbers.
4. All 24 tasks completed (5.5–31 min each), no timeouts; the t=100/budget-200
   concern was unfounded.

## VLWM-REGIME ARM (job 5593795, 2026-07-09): survives budget starvation far above the flat baselines

Our three arms under BeyondNext/VLWM's fixed-budget-50 regime (same fixed 256
episodes, budget 50 at every offset; t=25 cells = the existing budget-50 t=25
results). Means @20°/5°, seeds 42/43:

| t | s25 | s10 every-step | s10 + spec-accept | VLWM paper (their protocol) | LeWM paper |
|---|---|---|---|---|---|
| 50 | 83.6 / 72.5 | 88.7 / 76.8 | **90.0 / 81.3** | 60 | 46 |
| 75 | 55.7 / 35.0 | 59.8 / 42.2 | **60.9 / 43.6** | 20 | 12 |
| 100 | 23.4 / 9.8 | 24.4 / 13.7 | **27.3 / 16.4** | 12 | 8 |

1. **Budget starvation reproduces the collapse shape** — with 50 steps to cover
   2–4× that distance, everyone falls off a cliff. This confirms the VLWM paper's
   PushT collapse is largely budget, not only compounding error, and is why our
   main horizon table (budget 2t) shows flat curves.
2. **Within their regime we sit far above their published bars** (t75 ~61 vs
   their 20/12) — cite as "same budget regime", NOT same protocol (start/goal
   sampling still differs); shapes + regime-matched magnitudes only.
3. **Spec-accept beats every-step at every offset under time pressure**
   (t50 +1.3/+4.5, t100 +2.9/+2.7) — when steps are scarce, target stability
   beats re-drafting; the S=5-like sign-reversal appears at S=10 once budget
   is the binding constraint.
4. The s10−s25 gap compresses under starvation (+3–5pp @20° vs +8–12 at 2t
   budget) — scale matching buys precision and reachability, not raw speed.
