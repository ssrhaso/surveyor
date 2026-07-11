# DSpark → GDM Offline Probe Battery — Results

**Compute:** Isambard GH200 (`u6ko`, `brics.u6ko`, `workq`), container `pytorch_2412.sif` + `venv3`.
**Criterion:** per-position `rel_err` in the native LeWM latent space (‖z‖≈13.9, D=192), vs the frozen-drafter
anchor **0.159 / 0.229 / 0.318** (m+1 / m+2 / m+3). `cos_move`/`collapse` tracked; a head that lowers
`rel_err` with `collapse ≪ 1` predicts the mean and is flagged (none did — all `collapse` ≈ 0.95–1.0).

Every head number is on a **byte-identical, episode-split held-out val set** (`episode_id % 5 == 0`; no
window from a train episode leaks into val). The raw-diffusion baseline is recomputed on that same val set
(= raw draft `z_draft` vs `z_true`), so each head delta is measured against raw diffusion on the identical
population. Heads train on real frozen-drafter chains only (4 DDIM seeds, stored separately, never averaged;
never on ground-truth `z_{i-1}` except the Probe-0 ceiling head; never on injected Gaussian noise).

---

## Step 0 — anchor reproduction (hard gate) — **PASS**

`probe_suffix_decay.py` · `gdm_faithful.pt` (DiT, N=3, WG=1, T=1000, DDIM η=0, param `eps`, schedule
`linear`) · n=256 · stride 25 · seed 42 · `subset_longeval.episodes150.json`:

| pos | rel_err | anchor | cos_move | collapse |
|---|---|---|---|---|
| m+1 | **0.1593** | 0.159 | 0.9868 | 0.997 |
| m+2 | **0.2288** | 0.229 | 0.9806 | 0.998 |
| m+3 | **0.3179** | 0.318 | 0.9648 | 0.996 |

Exact reproduction → every downstream delta is measured against a valid curve. CPU isotropic-proxy head
table (`compare_heads*.py`) also reproduced (JointMLP 0.294/0.340, SeqMLP-oracle 0.216/0.282, SeqGRU
0.226/0.303, Ensemble-oracle 0.203/0.194).

## Step 1 — real-drafter chains — **DONE**

`build_real_chains.py`: frozen `gdm_faithful.pt` real DDIM drafts (seeds 42–45) paired with
`z_true[k]=E(frame@start+(k+1)·stride)`, on `subset_longeval.h5` (504 episodes, already 100% success5 =
paper-faithful population), end-anchored "runway" windows matching `eval_ffjepa.sample_long` (the anchor's
convention).

| file | N | stride | drafter | M | split (win) | raw-draft rel_err |
|---|---|---|---|---|---|---|
| `real_chains_n3.pt` | 3 | 25 | native | 624 | 500 / 124 | 0.158 / 0.233 / 0.302 (≈ anchor ✓) |
| `real_chains_ar6.pt` | 6 | 25 | AR-chained ×2 hops | 437 | 350 / 87 | 0.164/0.219/0.303 / **0.556/0.697/0.742** |

The AR-chained N=6 block exposes the compounding-error seam at the re-conditioning boundary
(m+3→m+4: 0.30→0.56). Literal N=12/stride=10 is infeasible with this native-N=3/stride-25 drafter (max
episode = 246 steps < 12·25 = 300; off-stride is off-distribution), so N=6 AR is the honest real-draft
portability test — and a stronger one, since it crosses a real re-conditioning seam.

---

## Probe 0 — Oracle-conditioned horizon floor · **KILL GATE → PASS (not killed)**

High-capacity head conditioning on **ground-truth** `z_{i-1}` (+ `z_cond` + full raw block), held-out val:

| pos | OracleFloor | raw diffusion | anchor |
|---|---|---|---|
| m+1 | **0.0663** | 0.1576 | 0.159 |
| m+2 | **0.0911** | 0.2333 | 0.229 |
| m+3 | **0.1203** | 0.3022 | 0.318 |

Perfect previous-position info more than halves m+3 error (0.302 → 0.120) — conditioning has **large
headroom**, it is *not* horizon-intrinsic. Gate: **PASS**, proceed. (Contra the brief's prior expectation
that ~75% of m+3 error might be horizon-intrinsic — it is largely recoverable.)

## Probe 1 — Real-drafter head bake-off · FORK resolves **AGAINST draft-refinement**

Held-out val, per-position `rel_err` (all `collapse` 0.98–1.0, no mean-collapse):

| head | uses draft? | m+1 | m+2 | m+3 |
|---|---|---|---|---|
| raw diffusion (no head) | — | 0.1576 | 0.2333 | 0.3022 |
| **JointMLP** (regress block from `z_cond`) | **no** | **0.0623** | **0.1023** | **0.1507** |
| Ensemble(K=5) DSpark-causal | yes | 0.0737 | 0.1159 | 0.1784 |
| SeqMLP (per-pos residual) | yes | 0.0772 | 0.1192 | 0.1810 |
| DSparkHead causal (residual) | yes | 0.0804 | 0.1255 | 0.1899 |
| — OracleFloor (cheats, ref.) | gt-prev | 0.0663 | 0.0911 | 0.1203 |

Two facts:
1. **Every head beats raw diffusion** — the diffusion drafter is far from optimal on this population; the
   refiners genuinely cut suffix decay (DSpark m+3 0.302 → 0.190).
2. **But the head-family fork resolves to direct regression, not refinement.** `JointMLP` — which *ignores
   the draft entirely* and regresses `[z1,z2,z3]` straight from `z_cond` — dominates every draft-refiner at
   every position, and nearly reaches the oracle-floor ceiling. Conditioning on the stochastic draft is a
   **liability**, not an asset: the draft is a lossy, noisy re-encoding of information already in `z_cond`.

**Verified not an overfitting artifact** (`verify_jointmlp.py`, train-vs-val):
- JointMLP wd=0: gap +0.015–0.022; **wd=1e-3: val 0.066/0.105/0.157, gap ≤ 0.010**; tiny hid=64 (train≈val,
  no overfit): val 0.113/0.156/0.224 — *still beats raw diffusion at every position.*
- DSparkHead overfits **more** than JointMLP (gap +0.018/+0.031/+0.039) — the noisy draft input is memorized.

Gate: **PASS** (something beats raw diffusion), but the architectural verdict is that **the DSpark
draft-refinement head is dominated by a draft-free regressor.**

## Probe 2 — Causal vs non-causal conditioning · **NULL → use causal**

| variant | m+1 | m+2 | m+3 |
|---|---|---|---|
| DSparkHead causal | 0.0806 | 0.1251 | 0.1900 |
| DSparkHead non-causal (full raw block) | 0.0843 | 0.1305 | 0.1964 |

Non-causal (exposing the full raw draft block to every position) does **not** help — it is marginally worse.
The "free full-block info" claim is empty here, consistent with Probe 1: the draft carries little useful
signal. **Verdict: null; use causal (simpler).**

## Probe 3 — Confidence / disagreement calibration · **PARTIAL**

Fidelity label = refined `rel_err < τ_k` (per-position train-median τ for balanced classes). Held-out val:

| pos | AUC (ensemble-var) | AUC (learned) | ECE pre | ECE post (STS) |
|---|---|---|---|---|
| m+1 | 0.819 | 0.834 | 0.117 | 0.117 (T=1.0) |
| m+2 | 0.753 | 0.760 | 0.143 | 0.143 (T=1.0) |
| m+3 | 0.715 | 0.726 | 0.064 | 0.056 (T=1.4) |

A usable acceptance signal exists (AUC ≳ 0.8 at m+1) but **weakens with depth** (0.72–0.73 at m+3); ECE is
modest and only slightly improved by sequential temperature scaling. **Verdict: partial** — an adaptive
commit-depth `k* = max{k : Πc_i > θ}` is viable at shallow depth; the fixed-block-reuse fallback is the safe
default for deeper commits.

## Probe N-gen — Extended-horizon (N=6 AR) bake-off · advantage **holds and grows**

Held-out val `rel_err`, N=6 AR-chained real drafts:

| head | m+1 | m+2 | m+3 | m+4 | m+5 | m+6 |
|---|---|---|---|---|---|---|
| raw diffusion (AR-chained) | 0.164 | 0.219 | 0.303 | 0.556 | 0.697 | 0.742 |
| **JointMLP** (regress from `z_cond`) | 0.066 | 0.110 | 0.165 | **0.218** | **0.273** | **0.337** |
| Ensemble DSpark-causal | 0.097 | 0.140 | 0.200 | 0.273 | 0.331 | 0.398 |
| SeqMLP | 0.087 | 0.133 | 0.214 | 0.299 | 0.385 | 0.452 |
| DSparkHead causal | 0.104 | 0.149 | 0.213 | 0.295 | 0.363 | 0.439 |

The AR-chained drafter degrades catastrophically past the seam (0.74 at m+6). **JointMLP eliminates the
compounding entirely** — it predicts each position directly from `z_cond`, no chaining, staying at 0.34 at
m+6 (2.2× better than raw). The shared-weight refiner generalizes to N=6 too (advantage over raw grows with
depth), but is again out-performed by direct regression. **Soft gate: PASS** (advantage holds/grows), same
verdict as Probe 1.

---

## Readout

**Forks resolved.**
- **Head family (Probe 1):** the draft-refinement head — DSpark's core scoped mechanism — **is dominated by
  a draft-free direct regressor** (`z_cond → block`). Refinement works (it beats raw diffusion and cuts
  suffix decay), but ignoring the draft works better and is verified robust to overfitting. On this
  success-filtered PushT population the future subgoal latents are ~a deterministic function of `z_cond`
  (JointMLP ≈ oracle-floor), so the diffusion drafter's stochastic error is liability, not multimodality.
- **Scope (Probe 2):** null — causal = non-causal; use causal.
- **Acceptance oracle (Probe 3):** partially alive — AUC 0.83 (m+1) → 0.72 (m+3). Adaptive commit-depth is
  viable shallow; **fixed-block-reuse is the safe fallback** (still a valid "fewer replans at equal fidelity"
  systems result, and both regressor and refiner de-compound the AR horizon).

**Recommended head to carry into the (still-gated) closed-loop build.** Not the DSpark semi-AR
draft-refiner. Carry a **lightweight deterministic block-regressor conditioned on `z_cond`** (JointMLP-class,
shared-weight + sinusoidal pos-emb for arbitrary N; the current `DSparkHead` collapses to this if the draft
input is dropped). It gives ~2× better offline fidelity than the diffusion GDM, matches the oracle-floor
ceiling, and eliminates AR compounding at extended horizon.

**Caveats before over-committing (why this stays offline for now).**
1. Small population (624 / 437 windows); absolute head numbers may be mildly optimistic, though the *ordering*
   (regressor > refiner > raw) is verified stable under regularization and capacity reduction, and replicates
   independently on the N=3 and N=6 chain sets.
2. **Offline latent `rel_err` is not closed-loop success.** The next gate (the deferred 3-arm closed-loop
   eval) must now include a **direct-regressor arm** vs the diffusion-GDM arm — the live question is no longer
   "refine the drafted suffix?" but "should the suffix be drafted (diffusion) at all, vs regressed?"
3. Multimodality: PushT-success is near-unimodal, which is *why* a regressor wins. On non-success-filtered or
   genuinely multimodal-future populations the diffusion drafter may recover its edge — do not generalize the
   "drop diffusion" claim beyond this population without re-running the bake-off there.

**Artifacts.** Chains `real_chains_n3.pt`, `real_chains_ar6.pt`; metrics `runs/dspark/probes_n3.json`,
`probes_ar6.json`; code `ffjepa/{dspark_head,build_real_chains,dspark_probe_runner}.py`; sbatch
`batch/run_{suffix_decay_probe,build_real_chains,build_ar6_chains,dspark_probes}.sbatch`.

---

# Closed-loop DSpark port — CEM success rate (the real test)

The DSpark mechanism was fully ported into the FF-JEPA planner and evaluated closed-loop, resolving the
offline caveat above (`rel_err ≠ SR`). Mechanism: `DSparkSubgoalSource` (subgoal_planner.py) drafts the
N-block with the frozen GDM, refines it (`DSparkHead`), scores each position (`ConfidenceHead`), and commits
a speculative prefix k* = max{k : Π c_i > θ} to a per-env queue consumed one-per-replan — so the diffusion
re-draft runs only every k* replans (decoupling subgoal cadence from CEM's 25-step action replan). Heads
trained on the held-out TRAIN split of `real_chains_n3.pt`; eval on the disjoint VAL episodes
(`dspark_val.episodes.json`, n=62), long horizon (goal_offset 150 = 6 subgoal-steps), block criterion.
Job 5525827, 4 arms × seeds 42-45 + θ sweep.

**Success rate (mean over seeds 42-45; per-seed spread in parens for 20°):**

| arm | difference from baseline | 20° SR | 5° SR | re-draft ratio |
|---|---|---|---|---|
| **gdm** (baseline) | raw m+1, re-draft every replan | **91.9%** (88.7–95.2) | **83.1%** | 1.00 |
| dspark fixed-1 | *refined* m+1, re-draft every replan | 81.1% (74.2–83.9) | 72.2% | 1.00 |
| dspark fixed-3 | commit whole refined block | 73.8% (69.4–79.0) | 50.0% | 0.35 (2.9× fewer) |
| dspark adaptive θ=0.5 | confidence commit-depth (mean k*≈2.34) | 64.5% (61.3–67.7) | 52.0% | 0.47 (2.1× fewer) |

θ sweep (seed 42, 20°): θ=0.3 (k*≈2.73) 59.7% · θ=0.5 (k*≈2.33) 66.1% · θ=0.7 (k*≈1.95) 61.3%.

**Verdict: the DSpark mechanism does NOT help closed-loop on FF-JEPA/PushT — NEGATIVE result.**

1. **Baseline wins outright (91.9%/83.1%)** — re-drafting the raw diffusion m+1 every replan, no refinement,
   no block commit. (Matches the paper's long-horizon ≈92%, so the harness/reference is sound.)
2. **Refinement itself HURTS: fixed-1 (81.1%) < gdm (91.9%)** at identical cadence/commit-depth — the *only*
   difference is refined vs raw m+1. Offline the refiner halved latent error (m+1 0.16→0.08); in closed loop
   that lower-`rel_err` subgoal is a **worse CEM target** (−11pp at both 20° and 5°). **Offline latent
   fidelity was actively misleading** — the caveat above, confirmed and pointing the wrong way. Likely the
   refiner's regression-toward-demo pulls the subgoal slightly off the predictor's reachable manifold, so the
   terminal-L2 cost plans toward it less successfully than toward the raw (noisier but on-manifold) sample.
3. **Deeper commit hurts more** (fixed-3 73.8%, adaptive 64.5%) — relying on committed z2'/z3' instead of
   re-anchoring every replan degrades control (stale subgoals, the `OracleSubgoalSource` staleness pattern).
   The weak single-model confidence (AUC≈0.67) makes adaptive *worse* than fixed-3 at 20°, and θ is
   non-monotone — the acceptance signal is too weak to schedule commit-depth usefully.
4. The 2–3× diffusion-call savings are real but bought at 18–27pp SR — not a favorable trade.

**Conclusion across offline + closed-loop.** Both directions the draft-refinement / speculative-commit idea
could win — offline latent fidelity and closed-loop SR — go against it: offline, a draft-free `z_cond`
regressor dominates the refiner; closed-loop, the *unmodified* diffusion baseline dominates refinement and
every commit-depth > 1. The suffix-decay premise is real, but neither refining the drafted suffix nor
committing it deeper improves the deployed planner. Recommended pivot: abandon draft-refinement; if pursuing
the "fewer replans" systems angle, the only lever left is a better acceptance oracle (the AUC≈0.67 single-model
confidence is the bottleneck), and even a perfect one is capped by the fixed-3 ceiling (73.8%) < baseline.
The stronger remaining thread is the offline regressor question (does a `z_cond`→block regressor beat the
diffusion GDM in *closed loop*?) — untested here and worth one more arm before closing the book.

**Closed-loop artifacts.** `dspark_head.pt`, `dspark_val.episodes.json`, logs `logs/dspark_ov_*.log`
(job 5525827); code `ffjepa/{dspark_head,train_dspark_head,build_val_episodes}.py`,
`subgoal_planner.py::DSparkSubgoalSource`, `eval_ffjepa.py --subgoal dspark`; sbatch
`batch/run_dspark_{sanity,overnight}.sbatch`.

---

# Follow-up round: the "longer subgoal chain" thesis (capability, not efficiency)

A separate claim from the refinement mechanism: *generate MORE subgoals (a longer chain) and use the extra
lookahead for long-horizon planning.* Probed cheaply, reframed around the one offline signal that survives.

**Spread is the validated offline gate (rel_err is not).** On the N=3 refiner we KNOW hurt closed-loop
(81% vs 92%), the offline signal that separated it from the baseline was **conditional sample spread**
(across-seed, same input): ratio refined/raw = **0.37–0.42** (refinement shrank the drafter's per-input
diversity ~60%), while ‖z‖ stayed ~13.9 and rel_err/batch-collapse both looked fine. This *explains* the
rel_err↔SR inversion: minimizing rel_err drives every draft to the conditional mean → collapses spread →
mean, less-reachable subgoal. The MSE regressor "winning" offline is just maximal spread-collapse.

**Chain-health to N=21 (job 5532490).** AR-chaining the native-3 drafter out to 21: the **raw** chain stays
on-manifold the whole way (‖z‖ ~13.0–13.2, spread grows honestly 0.11→0.69) — generating a long chain is
feasible. The **refiner** breaks it two ways: collapses spread where trained (ratio 0.45 at pos 1–3) and
drifts off-manifold where not (‖z‖ 13.9→10.7 at pos ≥4, untrained extrapolation with no supervision to fix).

**GATE — raw open-loop N=6 (job 5532843, seeds 42–45), goal_offset 150, block criterion, vs gdm 91.9%:**

| re-anchor cadence | 20° SR | 5° SR | re-draft ratio |
|---|---|---|---|
| every step — gdm (raw m+1) | **91.9%** | 83.1% | 1.00 |
| every step — refined m+1 (fixed-1) | 81.1% | 72.2% | 1.00 |
| every 3 — refined block (fixed-3) | 73.8% | 50.0% | 0.35 |
| **every 6 — raw chain open-loop** | **72.6%** | 63.7% | 0.185 (5.4× fewer) |

**Verdict: the longer-chain thesis is FALSIFIED at the root.** The open-loop arm used the *raw, healthy,
on-manifold* chain (no refiner confound), and still fell ~19pp below gdm — so the loss is **intrinsic to
committing to a longer chain, not a refinement artifact.** Across every closed-loop point, SR is **monotone
in re-planning frequency** (every-1 92% > every-3 74% > every-6 73%); committing further trades away the
re-anchoring that actually drives success. Mechanism: the drafter's uncertainty grows with horizon (spread
0.11→0.69), so a committed far subgoal bets on an increasingly uncertain future with no correction. A
spread-preserving / distributional refiner (the natural "step 3") is therefore *not worth building* — even a
perfect raw chain loses when committed. Both levers of the direction (refine; lengthen+commit) fail
independently. The only closed-loop thread still untested is orthogonal to this thesis: a per-step
`z_cond`→block **regressor** that re-anchors every step (keeps the winning cadence) vs the diffusion GDM.

**Follow-up artifacts.** `ffjepa/probe_chain_health.py`, `DSparkSubgoalSource(chain_n=, refine=)`,
`eval_ffjepa.py --no-refine --chain-n`; jobs 5532490 (health), 5532843 (raw open-loop); logs
`logs/dspark_{health,rawopen}_*.log`.
