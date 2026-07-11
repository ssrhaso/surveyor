# Direction: DSpark → GDM, Round 2 (draft–verify on the cost frontier)

Written 2026-07-07 with the actual DSpark paper (DeepSeek-AI) in hand, against our own
closed-loop evidence (`RESULTS_dspark.md`, `RESULTS_stride.md`). Goal of this doc: one
concrete, gated plan — no more component-level debate.

## 0. What "highest results on this paper" can honestly mean

FF-JEPA's PushT numbers are **saturated**: short 96.09% (our faithful repro 95.31%,
same-set oracle 93.36%, n=256 ⇒ SE ~1.2pp), long 91.80% (ours 91.9%). There is no
beatable SR headroom on the paper's own benchmarks — any "improvement" claim there is
noise-chasing. The beatable axes are the ones the paper itself concedes or cannot reach:

1. **Cost.** GDM = 242.6 ms per 25-step planning cycle vs 2.1 ms deterministic (paper
   Fig. 5) — a ~115× diffusion premium for +4–14pp SR. At stride-5 (our round in
   flight) the premium multiplies ×5 (60 vs 12 diffusion calls/episode). DSpark's own
   paper is a *Pareto-frontier* paper (throughput vs interactivity); the transplant is
   the planning edition: **SR vs diffusion-cost frontier for hierarchical latent
   planning.**
2. **Capability regimes the paper doesn't enter**: finer control granularity (stride
   round, Stage 3 pending), beyond-paper horizons, and the V-JEPA 2 transplant
   (headline plan) where drafter cost actually bites.

"Heavily improve GDM" therefore means: **same SR, much cheaper; a block that is
*usable* deeper than m+1; and mechanisms that transfer** — not +0.5pp on PushT.

## 1. Component-by-component verdict: DSpark paper → GDM

Now with the paper's actual mechanisms, against our closed-loop data:

| DSpark component (paper §) | Port to GDM | Verdict |
|---|---|---|
| Semi-AR **sequential head** as post-hoc block refiner (§3.1) | DSparkHead refiner | **DEAD** — measured: refined m+1 at fixed cadence −11pp (81.1 vs 91.9); mechanism = spread collapse 0.37–0.42. |
| **Commit** >1 subgoal blind (fixed-k / verify-free) | fixed-3, open-loop-6 | **DEAD** — SR monotone in re-anchor cadence (91.9 → 73.8 → 72.6). |
| **Learned confidence head** scheduling depth (§3.2.1) | ConfidenceHead θ-commit | **DEAD** — AUC 0.67; adaptive (64.5%) < fixed-3 (73.8%). |
| **Draft–verify–accept skeleton** (§2.1) with a *ground-truth verifier* | env/reality as verifier | **ALIVE — the core of this round.** Draft the block once; after S steps compare achieved latent to predicted m+1; accept → advance to m+2 with no re-draft; reject → re-draft from reality. Never sacrifices re-anchoring when needed; SR-neutral by construction; pure diffusion savings. |
| **Cheap draft, expensive verify** (γ-block in one pass) | few-step DDIM draft of the full model (self-speculative) | **ALIVE, gated** — `sample_next(n_steps=k)` is native; measure the acceptance×cost curve for k ∈ {2,4,8,16} vs 50. |
| "Position-1 capacity" analysis (§4.3.1: parallel drafters win at position 1) | the deferred **regressor-vs-diffusion closed-loop arm** | **OPEN — the one untested SR question from the DSpark round.** JointMLP dominated offline; the spread thesis predicts it LOSES in the loop (like the refiner). Either outcome is a finding; one cheap arm closes it. |
| **K-wide** posterior as CEM target (set-valued cost) | draft K samples of m+1, min-over-set cost | **GATED, expectations low** — requires a multimodal m+1 posterior; our JointMLP≈oracle-floor result already implies near-unimodal futures on success-filtered PushT. Cheap probe closes it either way. |
| Position-weighted loss w_k = exp(−(k−1)/γ) (§3.3) | one-flag retrain of the DiT | **OPTIONAL quirk**, only if spec-accept shows suffix quality is the binding constraint (and remembering rel_err↔SR inversion). |
| Semi-AR **at sampling time** (Markov-head-style conditioner *inside* the sampler — biases the sampling distribution, preserves stochasticity; ≠ our MSE refiner) | extra conditioning token in the DiT on realized previous position | **OPTIONAL, gated** — the one architectural transplant our data does NOT falsify (our refiner was a deterministic post-processor; this keeps the draft stochastic). Only worth a retrain if A2 shows acceptance depth is limited by m+2/m+3 quality. Note probe 2's causal≈non-causal NULL caps expectations. |
| Hardware-aware scheduler (§3.2.2) | tolerance τ as the budget knob | Concept survives as: acceptance tolerance τ trades diffusion calls vs staleness; the τ-sweep IS the scheduler, sim-sized. |

The user-level point this table settles: "DSpark-on-GDM has no surviving configuration"
was wrong, and "refine/commit is the way" was also wrong. The surviving configuration
is precisely DSpark's *skeleton* (draft cheap → verify → accept/reject) with the
verifier upgraded from a learned head to reality — which our data says is the only
verifier strong enough.

## 2. The plan (pre-registered gates, cheapest-first)

### Phase A — offline probes (no CEM, GPU-minutes; can run alongside stride Stage 3)

- **A1. Few-step-draft acceptance curve.** For k ∈ {2,4,8,16,50}: draft m+1 with
  k-step DDIM, compare against the 50-step reference sample AND the true latent on the
  anchor protocol; report rel_err/spread per k and the fraction within tolerance of
  the full sampler's draw. Gate: k ≤ 8 delivering ≈50-step quality → self-speculative
  drafting is free savings (≥6× per call) and goes in the closed-loop arm.
- **A2. Env-verifier speculative-acceptance simulator** (the decisive probe). On held
  -out demos: draft the N-block at t; advance along the demo S steps; verify achieved
  E(t+S) against predicted m+1 (tolerance τ); on accept, advance to m+2 without
  re-draft; sweep τ. Report: accepted-depth distribution, diffusion-call ratio vs
  every-step, and the *counterfactual subgoal error* while coasting (accepted m+2 vs a
  fresh re-draft from reality). Gate: mean accepted depth ≥ ~1.5 at a τ where coasting
  error ≈ re-draft error → build the closed-loop arm. (Probe-3's acceptance AUC
  0.83→0.72 by position suggests depth ~2 is realistic.)
- **A3. Multimodality probe** (closes the K-wide question). K=32 full-sampler draws
  per z_cond on val; cluster; report effective mode count; if >1 mode is
  control-relevant (inter-mode distance ≫ intra-mode spread), cross-reference against
  baseline failure states. Expected outcome per existing evidence: unimodal → K-wide
  scoring is dead on PushT and we say so in one paragraph.

### Phase B — closed loop (one sbatch array, after stride Stage 3 resolves)

At the winning stride from Stage 3, same n=62/seeds 42–45 protocol:
- **B1. Regressor arm**: JointMLP-class z_cond→m+1 source, re-anchor every step —
  closes the deferred SR question and tests the spread thesis causally.
- **B2. Speculative-acceptance arm**: env-verified advancement with τ from A2 (+
  few-step drafting from A1 if it passed). Success criterion: SR within noise of the
  every-step baseline at ≥2× fewer diffusion calls (stretch: ~accept-depth × few-step
  ≈ 10×+ combined).

### Phase C — the paper story (contingent on Stage 3)

- Stride-5 ≥ stride-25: "finer control granularity buys SR at a 5× diffusion premium;
  speculative acceptance with a reality verifier recovers the premium" — a capability
  + efficiency Pareto shift, DSpark's framing ported to hierarchical latent planning.
- Stride-5 < stride-25: the frequency–SR curve is complete on both sides (DSpark round
  = coarse, stride round = fine), and the efficiency result stands alone at stride-25:
  same SR, fraction of the diffusion cost. Scoop-resistance in both branches comes
  from the planning-specific verifier (reality/reachability), which no speculative-
  diffusion work has, plus our inversion/spread findings motivating WHY learned-
  fidelity verifiers are the wrong tool.

## 3. Interaction with the in-flight stride round

- Battery job 5550807 already measures per-position decay + conditional spread for the
  stride-5 block — that's the raw material for A2 at fine stride (N=3 × stride 5 = 15
  -step runway → up to 3× fewer re-drafts before any retrain; a larger-N fine-stride
  retrain extends the runway and is a one-flag change).
- Stage 3's `cad` arm (re-anchor every 5 with stride-25 targets) separates cadence
  from target distance — which also tells us how much acceptance-induced *staleness*
  the loop can tolerate, informing τ.

## 4. The SR axis (added 2026-07-07: cost is not the only target)

Constraint from round 1 that any SR lever must respect: do NOT improve drafts by
reducing latent error toward the conditional mean (inversion result, −11pp) and do NOT
trade away re-anchoring. Within that, four levers, ordered by evidence:

1. **Stride (in flight).** Stage 3 IS the primary SR bet: finer control granularity.
   The cost-SNR probe already showed the 5-step cost landscape is *sharper* than the
   25-step one — mechanistic room for a real SR gain.
2. **Random-init is the non-saturated cell of the paper's own table.** Short/long are
   at ceiling; random-init (paper: 82.42%) is not. Any SR-improvement claim on the
   paper's own benchmarks should be evaluated THERE, not on the saturated cells.
3. **Reachability-selected drafting (best-of-K, selection not refinement).** The
   drafter's spread becomes an asset: sample K drafts of m+1, score each by in-model
   reachability from the CURRENT state (predictor-rollout cost, the cost-SNR probe's
   machinery), hand CEM the most reachable sample. Every sample stays on-manifold (no
   averaging), cadence stays every-step, and it directly targets FF-JEPA's own stated
   #1 failure mode ("subgoal looks right, agent can't reach it"). **Gated by A3**: if
   the m+1 posterior is effectively unimodal, K samples are near-identical and
   selection is dead — same gate as the K-wide cost, read the same number.
4. **Failure anatomy before any further SR mechanism.** ~8% of long-horizon episodes
   fail; we have never classified them. Instrument the Stage 3 baseline arms (dump
   per-episode success vector + subgoal traces) and bucket failures: unreachable
   subgoal vs drift vs budget-exhaustion vs bad draft. Whatever bucket dominates is
   the next mechanism's target — build for the measured failure mode, not the
   hypothesized one.

Explicitly out of scope for SR: touching the frozen encoder ("more representative
latents" at the encoder level is a different paper — the whole reproduction and the
goal-free premise rest on E staying fixed), and any refine-toward-mean operator.

## 5. PHASE A RESULTS (2026-07-07 evening; local CPU + Isambard job 5550997 agree)

**A1 — PASS, decisively (both strides).** k=2 is broken (rel_err ~1.3); k=4 is already
at 50-step quality; **k=8 is indistinguishable from 50 steps** (s25: 0.1485 vs 0.1500;
s5: 0.0949 vs 0.0933; 92–98% of matched-noise draws within 0.05 of the full draw;
spread preserved). Few-step drafting = **6.25× fewer denoising steps, free.**

**A2 — PASS (both strides), with a surprise at stride 25.** Fresh every-step baseline
error varies strongly by episode phase (mean 0.433 at s25 / 0.211 at s5 across hops —
the 0.159 anchor is the best-case window). Acceptance simulator:
- s25 @ τ=0.40: call_ratio 0.52 (**1.9× fewer diffusion calls**), depth 1.92,
  stale penalty **−0.027** — verification-filtered consumption is *more* accurate
  than fresh every-step drafting (the verifier only lets deep positions be consumed
  on trajectories that are proving predictable). Negative penalty at every τ ≤ 0.60.
- s5 @ τ=0.15–0.20: call_ratio 0.53–0.61 (**~1.7–1.9× fewer calls**), depth 1.65–1.90,
  penalty +0.004–0.008 (≈0). At τ=0.60 depth 2.71 approaches the N=3 cap → if Phase B
  wants more savings, the binding constraint is BLOCK LENGTH, not acceptance: an N=6
  stride-5 retrain (one flag) extends the runway.
- Combined with A1: **~12× reduction in diffusion NFEs** at ≈zero offline fidelity
  cost. Phase B τ picks: s25 → 0.40, s5 → 0.20.

**A3 — K-wide dead at native stride; marginal at fine stride.** s25: **0/256**
candidate-multimodal (median 2-means separation 1.64, mode-gap 0.12× move) — the m+1
posterior is unimodal, K-wide CEM targets and best-of-K selection are DEAD on PushT
at stride 25, exactly as the JointMLP evidence predicted. s5: 24/256 (9.4%) candidates
with mode-gap ~0.25× move — small but real structure at fine stride. Not enough to
build a mechanism that idles 90% of the time; the one follow-up worth doing is free:
when Stage 3 failure traces land, check whether failures cluster at the 9.4%
multimodal states. If yes, selection earns a targeted arm; if no, closed for good.

**Net Phase B spec:** implement `SpecAcceptSubgoalSource` (verify-against-reality,
τ from above, k=8 DDIM drafts) + the JointMLP regressor arm; run both at the Stage 3
winning stride, same n=62/seeds protocol; report SR + measured diffusion calls.

## Out of scope (still)

MSE refinement in any form; blind commit; learned confidence heads as schedulers;
K-wide scoring unless A3 surprises us. These are measured-dead, not assumed-dead.
