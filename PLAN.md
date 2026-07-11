# PLAN

Status date: 2026-07-11. Paper source: `writeup/main.tex` (+ compiled `main.pdf`), in-repo.
Method naming: our contributions are the subgoal-scale law and **spec-accept**
(reality-verified speculative consumption). "DSpark" refers only to the ported
negative-result baseline (`specaccept/dspark/`).

---

## 1. Done (verified results, in the paper with real numbers)

All on PushT, controlled or paper protocol as marked. Section numbers refer to `writeup/main.tex`.

| Sec | Result | Key numbers |
|-----|--------|-------------|
| 2 | Faithful reproduction, audit closed | short 95.31 (paper 96.09), t75 multiseed 90.0 (91.80), rand-init 96.5/78.1; training set rebuilt to the episode (9,094) and iteration (171,940); offline anchor 0.1593 to 4 decimals |
| 2 | 20deg/5deg criterion finding | 20deg is the coded criterion; 5deg is prose-only; both reported throughout |
| 3 | Subgoal-scale law (Table 1) | interior optimum S=10: 97.2/96.4 vs 91.1/86.3 at S=25; +8.7pp at 5deg over 8 seeds, positive in every seed |
| 3 | Mechanism controls | cadence-isolation collapses to 78.6 (scale matching, not re-plan frequency); prediction error is horizon-intrinsic (retires intra-block architecture work) |
| 4 | DSpark port negative results (Table 2) | refinement -11pp, blind commit-3 -18pp, learned confidence worse than blind |
| 5 | Random init (Table 3) | S=10: 1024/1024, zero failures, 4 seeds, both criteria (paper's hardest setting, published 82.42) |
| 6 | Spec-accept (Table 4) | ~11x NFE cut (1500 -> 135/ep) at neutral 20deg SR; S=5 sign reversal (spec-accept beats every-step) |
| 7 | Regressor negative result (Table 5) | offline-matched deterministic regressor loses everywhere closed-loop; -11.7pp mirrors refinement's -11pp (spread thesis closed causally) |
| 8 | Failure anatomy | 18/496 failures, all tolerance-boundary near-misses; FF-JEPA's reported failure mode eliminated at fine scale |
| 9 | Fixed-population horizon sweep (Table 6) | baseline decays -11.9pp, S=10 flat; gap widens +3.3 -> +11.9pp |
| 9 | VLWM fixed-budget-50 regime (Table 7) | reproduces the published collapse; ~3x above VLWM bars within the identical budget |
| 10 | Headline table (Table 8) | complete for PushT |

Infrastructure done (2026-07-11): repo restructured into `specaccept/` (envs split,
DSpark quarantined); full battery committed twice over (Isambard SLURM chain + no-SLURM
ISCA MIG toolchain); 20GB RAM-cap trainer fix committed; smoke passed on ISCA v03
(baseline 84.4 vs paper 86); dense Reacher latents built (2.01M frames, on v03 disk).

## 2. To do

### A. Reacher battery (sec 11.1) -- the active experiment, blocked on compute only

Dependency chain (each step gates the next):

1. **Compute up** (see blockers). Two interchangeable routes:
   - Isambard back: `bash batch/deploy_reacher_isambard.sh reacher` (one command; ships
     the specaccept tree, removes the stale ffjepa dir on the box, submits the chain).
   - ISCA v03/v05: `git pull` (or re-paste `batch/box_bootstrap.txt`) FIRST -- boxes
     still have the old ffjepa layout -- then `batch/run_reacher_local.sh train`.
2. **train**: 4 goal-conditioned drafters, S in {5,10,15,25}, faithful recipe, in
   parallel across MIG slices. Expect 30-60 min of CPU pair-build before any GPU
   process appears (do not misdiagnose). Hours-scale.
3. **arms** (v03): 36 jobs = 9 arms (baseline + gdm x4 strides + spec-accept x4 strides)
   x seeds 42-45, n=128. Fills the scale-curve half of the sec 11.1 table.
4. **horizon** (v05): 24 jobs = {s25gdm, s10gdm, s10spec} x offsets {25,50,75,100} x
   seeds {42,43}, both budget regimes (2t and fixed-50) per cell.
5. **aggregate**: `python batch/aggregate_reacher_results.py` -> fill the sec 11.1
   tbdblock, update Table 8 and the conclusion.

Success criteria (pre-registered): (i) scale law transfers (interior optimum beats
S=25); (ii) spec-accept SR-neutral at call_ratio < 1; (iii) gdm >= baseline.
Judge on the **horizon** regime -- the single-hop reach is near-solved (baseline ~85%)
and has no headroom; a null there is uninformative, not a failed transfer.

### B. Paper scope decisions (user's call, before results land)

- **sec 11.2 TwoRoom**: parked by decision (goal-free is structurally ill-posed there),
  but the paper still promises the table. Decide: un-park (chain exists, Isambard-only)
  or cut the subsection and say why in Limitations.
- **sec 11.3 OGBench-Cube**: promised in the paper; **zero code exists**. This is the
  only place where the paper writes a check the repo cannot cash. Decide: scope it
  (biggest remaining engineering item, new env port) or cut it.

### C. Writing work -- no compute needed, can start now

1. **Bibliography**: all 6 entries in `writeup/references.bib` are TBD stubs. Real
   metadata is known for LeWM (arXiv 2603.19312) and FF-JEPA (arXiv 2606.09311); CEM,
   VLWM, OGBench, DSpark need real citations.
2. **Related work** tbdblock (main.tex ~L115): position vs hierarchical latent planners,
   diffusion planners, speculative decoding.
3. **Algorithm box** tbdblock (~L159): spec-accept pseudocode + CEM cost definition.
4. **Conclusion** tbdblock (~L574): placeholder text, rewrite once sec 11 resolves.
5. Terminology sweep: "spec-accept" everywhere for the method; "DSpark" only for the
   negative-result port.

## 3. Immediate blockers

1. **Compute** (blocks A only):
   - Isambard: down. When back, deploy is one command.
   - ISCA v03/v05: viable again in principle -- the 20GB RAM-cap fix is committed but
     has **never completed a full training run** (verified output-identical logic +
     paste parses; the actual sub-20GB peak is unproven). First train launch is also
     the fix's validation run. Session-culler hazard stands: keep a browser tab open.
2. **Push to GitHub** (blocks the git-pull resume path): all 7 commits
   (57f770b..7db3bed) are local-only. Boxes bootstrap from the GitHub clone, so
   pushing main is the first move before any box work. (User pushes, per workflow.)
3. **Nothing blocks C** -- the writing items can proceed immediately, in parallel.

Priority order if compute stays down: C1 (bibliography) -> C3 (algorithm box) ->
C2 (related work) -> B decisions, so the paper is submission-shaped the moment the
Reacher numbers arrive.
