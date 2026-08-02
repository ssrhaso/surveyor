# Agent brief: adversarial meat audit — is anything left to RUN for main-track ICLR?

Copy-paste everything below the line as the agent's prompt. (Written 2026-08-01,
after the certification batteries closed; supersedes docs/next_agent_audit_prompt.md,
whose Part-B items are all resolved.)

---

You are a hostile ICLR area chair plus an experiment planner. A research program
believes its experimental ledger is COMPLETE and that all remaining value is
writing. Your ONE job: try to REFUTE that position — find any experiment whose
absence a main-track reviewer would punish and whose addition materially raises
acceptance probability. An honest empty list is an acceptable answer, but it must
be argued, not asserted. You run nothing; you read, simulate reviewers, and rank.
Target: two pages of markdown, no more.

## The program (context you'd otherwise lack)

METHOD ("certified spec-accept"): on a frozen latent world-model planner, draft a
block of N subgoal latents once; at each replan verify the ACHIEVED latent against
the pursued waypoint (rel-L2 <= tau -> serve the next waypoint free; else re-draft
from reality). Constants derived, never tuned: tau = the success criterion's own
latent floor, k = sampler convergence, S by rule. Novelty stack: (1) reality as
the acceptance test (speculative-decoding transplant; no learned confidence, no
new parameters), (2) derived constants, (3) a pre-flight instrument (the
verification gap) predicting applicability from cached latents, (4) a regime law
(decomposition pays iff goal beyond the planning window x goal inferable x budget
headroom). Competitor context: SAGE is concurrent subgoal work on the same
substrate (cite as premise validation; differentiate on certification/instrument;
protocols not cross-comparable).

WHAT IS BANKED (every claim pre-registered; verdicts live in the docs below):
- 11-cell grid: certified spec-accept >= strongest flat in every env x horizon
  cell of the substrate's ENTIRE published benchmark suite (PushT 5/5, Reacher
  4/4, Cube +14.0); 384-1536 eps/cell; one declared -1.0pp self-miss.
- TwoRoom arc: anchor attribution (their 87 reproduced; gap = protocol), +15.4pp
  at t=75 AT the oracle ceiling (12 seeds, every seed), crossover [40,50]
  confirmed at 12 seeds both edges, composite window-rule policy RUN END-TO-END
  as one arm (zero deviation), fairness (+10.0pp over 2x-CEM flat, t~4.9),
  timing (+2% wall-clock), verify-space C1/C2.
- CERTIFICATION (new, 07-31..08-01): verifier CALIBRATED against ground-truth
  state — quotable false-accept 0.000 (cube) / 0.102 (pusht) / 0.400 (reacher,
  tight 0.05-rad criterion, Spearman rho 0.978 = near-perfect ordinal tracking) /
  0.522 (tworoom, the known weak-encoder scope row); conservative-probe rule and
  R^2>=0.90 gates pre-frozen; decision replay exact. CORRUPTION sweep: the
  derived tau=0.20 localized the closed-loop rejection-ignition point EXACTLY
  (call_ratio .59 -> 1.00 as sigma crosses tau) — prospective validation of the
  derivation; the SR-robustness prediction honestly refuted (white jitter
  averages out; value-vs-staleness is banked elsewhere). M2: accept rate rises
  with horizon (call_ratio pusht .88->.57, reacher .75->.58) — the serving loop
  itself measures the regime law.
- SECOND ARCHITECTURE (DINO-WM), resolved 07-31: serving stack + derived tau
  VALIDATED (goal-serve tautology through the full spec path = flat parity
  0.85-0.90); the transplanted drafter's CONTENT fails — at FULL data scale
  (18,685 episodes, ~1.6M pairs, 2-3x the optimizer budget of every working
  drafter; the data-starvation hypothesis is refuted by the training log) and
  snapping drafts to real-manifold grids does not rescue (advances 7-11/250).
  The verifier's ~97% rejection was CORRECT detection of a bad drafter. Leg
  killed at a pre-declared time-box. Consistent with the V-JEPA 2 20x-data null
  (tau/mechanism transferred; drafter did not; chapter dropped by supervisor
  agreement).
- Instrument prospective record 3-for-3 incl. a 31pp cross-architecture hit
  called before the run; refutation ledger with ~10 retained negative results.

TRUST ORDER: PLAN.md top STATUS block (incl. the 08-01 addendum) >
docs/certification_prereg.md + docs/dinowm_prereg.md + docs/tworoom_paired_prereg.md
(frozen bars AND verdicts) > paper/main_v2.tex (+ paper/sections/) > raw logs on
ISCA (/lustre/home/ha676/le-wm/logs/). Results/RESULTS.md is STALE — never cite
it. NOTE: the paper does NOT yet contain the certification-day results (that
surgery is queued writing) — do not count their absence from the PDF as a
missing experiment.

## THE GRAVEYARD — already run, refuted, or rule-forbidden

Re-proposing any item below requires a NEW argument that defeats the recorded
reason, not a restatement of the idea:
1. Third architecture / DINO-WM closed-loop revival — drafter-content failure
   measured at full data scale; time-boxed kill executed twice; V-JEPA dropped
   by supervisor agreement.
2. Retrain/scale the DINO-WM drafter — data-starvation hypothesis refuted by its
   own training log (best-fed drafter in the program).
3. tau re-derivation or tuning anywhere — forbidden: tau is never tuned against
   closed-loop numbers; calibration FA values are closed-loop-derived.
4. More seeds / bigger n anywhere — every quotable cell >= 12 seeds or >= 384
   episodes; headline cells 12-seed with paired tests.
5. Cube best-of-k; Reacher -1.0pp self-miss chase — weak priors, declared
   misses read as integrity; rejected three times.
6. Drift/staleness corruption sweep — redundant: autocorrelated-error value of
   verification already banked three ways (blind-commit -18pp, stale oracle,
   crossover).
7. A 5th environment — the four envs ARE the substrate's entire published
   suite; a 5th requires training a new substrate and dilutes that sentence.
8. Block-size N ablation — N is the frozen stack's own architecture; the
   method's premise is "stack as given".
9. Observation-noise robustness axis — a claim the paper doesn't make; scope
   creep.
10. Re-running banked cells for any reason — forbidden by program rule.

RULES for anything you DO propose: it must have a pre-registerable bar frozen
before running; claims stay internal to a stack (no cross-stack or cross-paper
absolute comparisons); poster deadline Aug 11 (experiments must not block it),
ICLR deadline late September (hard cutoff for anything new); ISCA A100s are
idle (GPU cost is real but not the binding constraint — reviewer impact per
GPU-hour is).

## DELIVERABLE (three parts, ranked, terse)

PART A — REVIEWER SIMULATION. The three most probable rejection vectors for
this paper at ICLR main track (think: toy-scale envs, single-stack closed-loop
positive, "just a threshold" novelty compression, missing baselines). For each:
what banked evidence currently answers it, and your verdict — SUFFICIENT (a
writing/framing job) or INSUFFICIENT (name the missing experiment).

PART B — NEW EXPERIMENTS, ranked by acceptance-probability-per-GPU-hour. For
each: the claim it would add, a sketch of the pre-registerable bar, GPU cost,
deadline fit, and the strongest argument AGAINST running it (steelman the
graveyard). If your honest list is empty, say so and defend it against the
strongest single candidate you rejected.

PART C — FREE MEAT. Anything extractable from EXISTING artifacts with zero GPU
(banked logs, 10 calibration JSONs in Results/calibration/, trace .pt files on
ISCA runs/, 1,514-row Results/acceptrate_mine.csv, 16 strip npz) that would
strengthen a claim a reviewer will probe.

Do not audit writing quality, figures, bibliography, or packaging — a separate
writing bundle covers those. Your scope is solely: does the EVIDENCE need
anything more, and if so, exactly what.
