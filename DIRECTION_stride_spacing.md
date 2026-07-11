# Direction: Subgoal Spacing as a Control Lever (and DSpark's surviving role)

## 0. Where we are — what the DSpark rounds established

Two rounds of probes on FF-JEPA / PushT (frozen LeWM encoder + `gdm_faithful.pt`
diffusion subgoal drafter, DiT N=3, stride 25), all on Isambard GH200. Full detail in
`RESULTS_dspark.md`; the load-bearing conclusions:

1. **Offline latent fidelity is a liar.** `rel_err` anti-correlates with closed-loop SR. A
   draft-free MSE regressor minimises `rel_err` and *looks* best offline, but that is
   near-tautological and inverts in the loop.
2. **The validated offline signal is conditional sample spread** (across-seed, same input).
   On the refiner we *know* hurt SR (81% vs 92%), spread collapsed to 0.37–0.42 while
   `rel_err`, batch-collapse and ‖z‖ all looked fine. Minimising `rel_err` regresses drafts
   to the conditional mean → collapses spread → a mean, less-reachable subgoal.
3. **Closed-loop SR is monotone in re-anchor cadence** (env-steps between diffusion
   re-drafts), measured at fixed stride-25 targets by varying commit depth:

   | re-anchor every … | SR (20°) |
   |---|---|
   | 25 env-steps (gdm, commit-1) | **91.9%** |
   | 75 (commit-3) | 73.8% |
   | 150 (raw chain, commit-6, open-loop) | 72.6% |

   Frequent re-anchoring is the dominant factor. Every "refine / commit / lengthen" scheme
   trades it away and loses — **including a perfectly healthy raw chain** (the open-loop-6
   arm used the raw on-manifold chain, no refiner, and still fell ~19pp). This closes the
   refine/commit/lengthen space.

**What was never varied: the stride itself.** Every number above is at native stride-25;
only commit depth moved. Spacing is the untested axis, and it is the one axis that *pushes
the winning lever further* (re-anchor more often) rather than trading it away.

---

## 1. Central hypothesis and the honest counter-risk

**Hypothesis.** A finer-stride drafter, re-anchored at the finer cadence, extends the
"re-anchor more = better" trend: SR(stride-5) ≥ SR(stride-25).

**Why it is genuinely open, not a safe extrapolation.** The monotone curve above was
measured by varying *re-draft cadence at a fixed 25-step target*. Changing the stride moves
**two** coupled things at once:
- **re-anchor cadence** (finer → more re-anchoring → predicted to help), and
- **target horizon** (finer → the m+1 subgoal sits only S steps out).

The second is the real research question. At stride-5 the target is 5 steps out, so:
- (a) it may be **too close to be informative** — `rel_err → noop_err`, and CEM's cost
  signal shrinks toward rollout noise (a degenerate target), and
- (b) the drafter faces a **low-SNR** task (5-step latent displacement is small; the
  predict-no-change null is strong).

So finer stride either keeps paying (control granularity wins) or hits negative returns
(target too close). **Both outcomes are publishable**; the goal of this round is to
find out which, cleanly, before any DSpark work.

---

## 2. Correction to the proposal: the control loop must scale with stride

The subgoal is the CEM cost target: cost = terminal L2 of the CEM rollout to the subgoal.
For that to be well-posed, the **CEM rollout length must ≈ the subgoal distance**. So a
finer stride is *not* a config flag on the drafter alone — it is a **control-granularity**
change, and three quantities move together, by construction:

| quantity | stride-25 (baseline) | stride-S |
|---|---|---|
| subgoal stride S (env-steps to m+1) | 25 | S |
| CEM rollout `horizon × action_block` | 25 | **S** |
| replan / re-anchor cadence `receding_horizon × action_block` | 25 | **S** |

With `action_block = 5` held fixed: set `horizon = receding_horizon = S / action_block`
(S=5→1, 10→2, 15→3, 25→5). Hold `action_block`, `num_samples`, `n_steps`, `topk` fixed.

> **This overrides the proposal's "do not touch receding_horizon."** That instruction
> contradicts its own "stride-5 replans ~60× vs stride-25 ~12×": 60 replans over 300 steps
> *requires* a 5-step replan cadence, i.e. `receding_horizon × action_block = 5`. Leaving
> `receding_horizon = 5` would make CEM plan 25 steps toward a 5-step target — a degenerate
> mismatch, not the intended experiment. Scaling the loop with the stride **is** the single
> lever (control granularity); it is a definitional coupling, not a second confound.

Acknowledge (do not hide) the intrinsic consequence: a shorter CEM horizon is an easier
search at fixed `num_samples`. That easier-per-solve-but-more-solves tradeoff is *part of*
finer control, and it is exactly the diffusion-cost premium that motivates the DSpark layer
in Round 2. State the per-arm replan and diffusion-call counts so the SR/cost tradeoff is
explicit.

---

## Stage 0 — CPU pre-gate: is a fine-stride target even well-posed? (no retrain, minutes)

Front-load the degeneracy risk before spending a single GPU-hour on a retrain. From
`pusht_expert_train.h5` + the frozen encoder (CPU, no model training, no CEM): for a sample
of successful episodes and strides S ∈ {5, 10, 15, 25}, measure the realised **latent
displacement** distribution ‖E(t+S) − E(t)‖ / ‖E(t+S)‖ and compare it to the encoder's
noise floor (e.g. displacement at S=1, or augmentation jitter of the same frame).

- **Kill:** if the stride-5 displacement collapses toward the encoder noise floor — 5-step
  targets are latent-indistinguishable from "stay put" — the target is degenerate
  *regardless of any drafter*, and the whole direction dies here for the price of a CPU job.
- **Proceed:** if fine-stride displacements are well above noise and directionally
  structured, the target is well-posed → retrain.

This is the cheapest possible gate and needs no Isambard time.

## Stage 1 — Retrain finer-stride GDM drafters (one-lever; GPU)

Hard dependency: `gdm_faithful.pt` was trained on stride-25-aligned transitions and is OOD
at other phases (`diag_gdm.py` shows arbitrary-phase conditioning degrades it), so a finer
cadence *requires* a finer-stride drafter — this is not a runtime flag.

- Rebuild transition targets at stride S: `build_subgoals.py --stride S` (confirmed to
  support any stride; z_{sg,m}=E(t), z_{sg,m+1}=E(t+S), … from the successful expert set).
- Retrain with the **exact faithful recipe** — *only the stride changes*. **Prerequisite:
  pin the faithful recipe** (the trainer's argparse defaults are explicitly NOT faithful;
  the real recipe is LeWM's config — batch 128, lr 5e-5, and the rest — locate the run
  script / handoff that produced `gdm_faithful.pt` and match every hyperparameter). A null
  must be attributable to spacing, not to an arch/training delta.
- Retrain S ∈ {5, 10, 15, 25}. **S=25 is a re-derivation sanity check**: it must reproduce
  `gdm_faithful.pt`'s offline curve, proving the rebuild+retrain pipeline is faithful.

## Stage 2 — Offline drafter-quality gate (GPU-cheap, no CEM; diagnostic only)

Per finer drafter, m+1 on the val split: `rel_err`, `cos_move`, **`noop_err`**
(‖z_cond − z_true‖/‖z_true‖), and conditional spread (health).

- **Degeneracy gate:** if `rel_err ≈ noop_err`, the subgoal is barely distinguishable from
  "stay put" → CEM gets no signal → dead, skip the closed-loop run.
- Otherwise proceed. **These metrics gate *out* an obviously-degenerate drafter; they do
  NOT greenlight.** `rel_err` does not decide SR (validated inversion). A good `rel_err`
  means proceed-to-closed-loop, not proceed-with-confidence.

## Stage 3 — Closed-loop A/B: stride-5 vs stride-25 (the actual test)

Both at every-step commit-1, matched control loop (§2), everything else identical: same
LeWM, same `num_samples`/`n_steps`/`topk`/`action_block`, same n=62 held-out episodes,
seeds 42–45, block criterion, goal_offset 150, total eval-budget 300 env-steps. Only the
drafter checkpoint + stride (and the definitional loop scaling) differ.

Report: SR 20° (primary) / 5° (supplementary) vs the 91.9 / 83.1 anchor, **plus per-arm
replan counts and diffusion-call counts** so the SR/cost tradeoff is explicit (stride-5
re-drafts ~60× vs ~12× over 300 steps — a ~5× diffusion premium; that premium is the
tradeoff being characterised and the motivation for Round 2, not a confound to remove).

**Decision rule.**
- stride-5 ≥ stride-25 (ideally >, since it re-anchors more) → **spacing is a live lever →
  Stage 4**.
- stride-5 < stride-25 → denser targets don't help (too close / low SNR) → **report and
  stop**; the spacing direction is dead and we rethink.

## Stage 4 — Frontier (only if Stage 3 clears)

Stride sweep {5, 10, 15, 25} at commit-1 → map SR-vs-stride, find the knee. Plus the single
**commit-2** point at the best stride — the one commit-depth point genuinely unmeasured at
fine spacing (does one denser block buy a skipped re-draft?).

---

## Round 2 — DSpark's surviving role: speculative acceptance (gated on Stage 3/4 headroom)

The refine / commit / lengthen readings of DSpark are dead. The reading that **survives** is
DSpark's *actual* LLM role — **speculative decoding for speed, not quality** — ported with
the **environment as the verifier**:

- Draft a block of K future fine-stride subgoals in one diffusion call.
- Execute toward subgoal 1. After S steps, **verify against reality**: does the achieved
  latent match the predicted subgoal (within tolerance)? **Accept** → advance to subgoal 2
  with no re-draft. **Reject** (trajectory diverged) → re-draft from the achieved state.
- Diffusion cost ≈ (steps / S) × reject-rate; an accurate drafter pays little.

**Why this survives when refine/commit/lengthen died:** it *never* sacrifices re-anchoring
when it is needed — it skips a re-draft only while reality confirms the prediction, and
re-anchors the instant reality diverges. It is **SR-neutral by construction** (the verifier
is the ground-truth achieved state, not a weak learned confidence head) and is a **pure
speedup**. It earns its place *only if* Stage 3/4 show fine stride buys SR at a diffusion-cost
premium — precisely the setup where speculative acceptance recovers the cost. That is the
next brief, not this one.

---

## Out of scope this round
No `DSparkHead` refiner, no RNN/MGU sequential cascade, no commit > 2, no learned confidence
head. Those are the efficiency/extension layer and earn a place only if Stage 3/4 show SR
headroom. DSpark's surviving role is speculative-accept speedup on a mechanism that already
works — never refine-and-commit.

## Compute / logistics
- Isambard GH200, container `pytorch_2412.sif` + `venv3`, account `brics.u6ko`, partition
  `workq`. Submit small (1-GPU) jobs / arrays — they backfill in seconds; a 4-GPU ask sat
  ~2h in queue. Retrains are the only non-trivial cost; the four strides are embarrassingly
  parallel (one array task each).
- CPU for fast tests (Stage 0, offline metric readouts, health checks) — no local GPU.
- Reuse the fixed held-out episode set (`dspark_val.episodes.json`, n=62) and seeds 42–45
  throughout so every SR number is on a byte-identical population.
