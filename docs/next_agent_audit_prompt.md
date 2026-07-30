# Agent brief: audit the spec-accept program and map extensions

Copy-paste everything below the line as the agent's prompt.

---

You are auditing a finished research program so its author can decide what, if
anything, to add before submission. Produce ONE concise deliverable (target: two
pages of markdown, no more): PART A, what the program has, as claim -> evidence
-> where the number lives; PART B, a ranked list of potential extensions with
cost, benefit, and risk. You run NO experiments and edit NO files; you read,
verify, and map. Where a claim and a source disagree, say so loudly; finding rot
is more valuable than summarizing.

CONTEXT. The method is CERTIFIED SPEC-ACCEPT on latent world-model planners:
draft a block of N subgoal latents, serve them to the stack's own frozen
planner, verify the achieved latent against the pursued waypoint at every
replan (rel L2 <= tau -> advance free; else re-draft from reality). Constants
are derived, not tuned (tau = criterion floor, k = sampler convergence, both
offline). A companion instrument (the verification gap) predicts applicability
from cached latents. Everything is pre-registered; refuted predictions are
retained deliberately as evidence of discipline.

AUTHORITATIVE SOURCES, in trust order. (1) PLAN.md's top STATUS block
(2026-07-31) is the program map. (2) docs/tworoom_paired_prereg.md and
docs/dinowm_prereg.md hold frozen bars AND verdict tables; every TwoRoom and
DINO-WM number must trace to one of these. (3) paper/main_v2.tex (+
paper/sections/) is the write-up under audit. (4) Raw logs live on the ISCA
cluster under /lustre/home/ha676/le-wm/logs/ (ssh isca, PowerShell only,
script-files not one-liners; only spot-check if a number smells wrong).
Results/RESULTS.md is STALE (July 3) — never cite it. Memory files summarize
but can lag; the prereg docs win every conflict.

PART A (what we have). For each of: the 11-cell LeWM grid (PushT/Reacher/
Cube), the TwoRoom arc (anchor attribution, +15.4pp at t=75, their-protocol
t=25, crossover, composite, fairness, timing, C1/C2), the instrument's
prospective record, the refutation ledger (k=2, bok-feas, nofilt, B25-1, cube
tau-midpoint, V-JEPA certificate), and the DINO-WM kill-switch outcome: state
the claim in one line, the evidence in one line, and the file/section where a
reviewer would check it. Flag any paper sentence whose number you could not
trace to a prereg verdict or log.

PART B (potential extensions). Rank by expected paper impact per unit cost;
for each give one line of benefit, one of cost (GPU-hours or writing-hours),
one of risk. Seed the list with these candidates and add any the audit
surfaces, but prune ruthlessly — a short honest list beats coverage:
  * Cube best-of-k battery (bok is TwoRoom-only; one more env upgrades it to
    a method component; pre-register >= +2pp, 6 seeds).
  * Best-of-k on Reacher t=150 (could close the one -1.0pp self-miss).
  * DINO-WM serving-fault autopsy (unblocks P1/P2/P5, currently claimed in
    neither direction; unbounded debugging risk — the kill-switch cut it once).
  * 12-seed extension for any cell still at 6 seeds that a reviewer would
    call thin (check the their-protocol spec arms and the crossover sweep).
  * Formal composite evaluation (run the window-rule policy end-to-end as ONE
    arm rather than citing its two constituent cells).
  * Qualitative additions: cube/reacher filmstrips (drivers lack --dump-strip
    plumbing that pusht/tworoom have).
  * Writing-only: full-PDF verbosity pass, official ICLR style kit swap,
    real author names in references.bib for SAGE/TRM, appendix table of every
    refuted prediction (the honesty ledger as a single artifact).
  * SAGE positioning check (concurrent subgoal work, arXiv 2607.17973): cite
    as premise validation, differentiate on certification; protocols are NOT
    cross-comparable — verify the related-work section says exactly this.

RULES. Do not propose anything that violates the program's discipline: no
experiment may be suggested without a pre-registerable bar; claims stay
internal to a stack (never compare absolute numbers across stacks or papers);
tau is never tuned against a closed-loop number. The known honest weak spots,
so you do not "discover" them: bok's 12-seed margin is within noise (recorded);
flat+2xCEM recovers +6.8 on TwoRoom (recorded, margin survives); B25-1 missed
(recorded, oracle explains); the t=60 crossover point is a tie within noise.
Deadline context: poster due Aug 11 — weight extensions accordingly.
