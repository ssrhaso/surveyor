# PLAN (tick sheet)

Paper: `paper/main_v2.tex` (ACTIVE — method-paper restructure, 2026-07-13;
`paper (1).tex` = v1 forensic archive feeding the appendices; writeup/main.tex = older interim doc).
Rule: update the paper ONCE per completed block, not per job.
Method = spec-accept; "DSpark" = the ported negative-result baseline only.

---

## ================================================================
## STATUS 2026-07-31 (READ THIS FIRST — supersedes everything below)
## THE ENTIRE EXPERIMENTAL PROGRAM IS CLOSED. Writing/poster only.
## ================================================================
##
## WHAT EXISTS (each claim pre-registered; verdict tables live in
## docs/tworoom_paired_prereg.md and docs/dinowm_prereg.md; grid claims in
## paper tab:grand; NEVER re-run any of this):
##  * Core: certified spec-accept >= strongest flat in all 11 pre-reg
##    env x horizon cells (PushT 5/5, Reacher 4/4, Cube +14.0). Constants
##    derived (tau = criterion floor, k = sampler convergence), fairness
##    (2x-CEM flat) + timing + tau-robustness closed on all envs.
##  * TWOROOM (the 4th env, closed 07-29..31 in three days): LeWM's own
##    weakest benchmark (their 87 vs 97-100 for baselines; they blame the
##    representation). Arc: 21pp "reproduction gap" = PROTOCOL (V5
##    verbatim replication 85.33; success filter ~14.6pp selects a harder
##    population). Their-protocol t=25: flat 83.72 (12 seeds, ties 87),
##    oracle -21pp (goal inside one planning window -> decomposition
##    structurally counterproductive), spec 77.08 = honest B25-1 miss.
##    ANCHORED t=75: spec (verify in frozen DINOv2, tau=0.098 derived)
##    57.03 vs flat 41.67 = +15.4pp, ALL 12 seeds, paired t~11, AT the
##    oracle ceiling (57.81). Crossover MAPPED t in [40,50], frozen
##    prediction confirmed. DEPLOYED CLAIM = composite: flat up to the
##    measured crossover (ONE measured constant), certified spec beyond;
##    never loses to flat. Controls: fairness flat+2xCEM 48.44 (real +6.8
##    rescue, reported; spec +8.6 above it, margin NOT compute-driven);
##    timing +2% wall-clock (140ms on 6.9s eps; CEM 94%); C1/C2 on pusht:
##    verify-space SR-indifferent where own gap open BUT external space =
##    degenerate-reject (cr 1.00 vs 0.66) -> rule "own space when gap
##    open, transplant when not" stands with a measured reason.
##  * REFUTATIONS ON RECORD (the discipline is a paper feature): k-from-
##    bias k*=2 refuted closed-loop (31.25; sampler diversity collapse,
##    invisible to bias-of-means — spec serving NEEDS diversity); bok-feas
##    -23pp (stay-put pathology, predicted in advance); nofilt 3.4x-data
##    drafter null; bok-goal passed 6-seed bar (+2.9) but +1.8/t~1.1 at 12
##    seeds -> optional add-on, out of headline; B25-1 non-inferiority
##    miss at t=25 (oracle explains).
##  * DINO-WM leg KILLED per the pre-declared Aug-1 switch: P3 = confirmed
##    prospective hit (31pp, called in advance); P4 unavailable; P1/P2/P5
##    confounded by an UNRESOLVED serving fault (flat healthy, spec ~0,
##    3 batteries, same signature; drafter passes offline gates) — claimed
##    in NEITHER direction; tworoom-dinowm WM training cancelled, offline
##    encoder-swap cure (0.876 -> 0.081) stands as the measured claim.
##  * V-JEPA 2: mechanism + tau transfer positive; certificate dead (3x);
##    15x compute claim retired. Chapter dropped by supervisor agreement.
##  * Assets: filmstrip pipeline (--dump-strip + specaccept/render_strips
##    .py; pusht + tworoom rendered), crossover figure
##    (paper/figures/fig_tworoom_crossover.png), proximity router
##    (--route-goal-hop), verify-half switch, timing in tworoom driver.
##
## PAPER STATE: main_v2.tex compiles (22pp). TwoRoom is in abstract /
## contributions (verifier-modularity bullet) / grid paragraph /
## held-out section (full arc + controls) / conclusion; both
## instrument_generality \inflights resolved with measured verdicts;
## composite switch stated as ONE MEASURED CONSTANT. Style shim is still
## the PLACEHOLDER ICLR kit — swap before submission.
##
## OUTSTANDING (no GPUs involved):
##  1. COMMITS: done 07-30 evening (7 commits, FEAT:/CHORE:, no
##     co-author; user pushes). Rule stands: commit ONLY when asked.
##  2. POSTER (hard deadline Aug 11, supervisor promise): awaiting the
##     user's format/template decision; filmstrip panels now exist for
##     ALL FOUR envs (strips/, spec-vs-flat pairs).
##  3. Paper polish: full-PDF read-through, remaining [CHECK authors]
##     bib entries (6), official ICLR kit swap (kits committed as zips),
##     reproducibility statement + refuted-ledger appendix;
##     Results/RESULTS.md is STALE (7/3) — do not trust it, prereg docs
##     + logs are the truth.
##  4. Extension candidates: see docs/next_agent_audit_prompt.md +
##     docs/audit_map.md. NOTE 07-30 post-audit batteries CLOSED the top
##     run candidates (all prereg-frozen first, verdicts same-day):
##     composite window-rule run END-TO-END as one arm (--composite-
##     crossover 45, reproduces its branch cells exactly, all 5 t);
##     crossover band 12-seed both edges (t40 flat +3.5 / t50 spec +3.4,
##     band [40,50] unchanged); fairness control 12-seed (pooled 47.00,
##     spec +10.0pp, paired t~4.9); oracle ceiling 12-seed (56.90, spec
##     statistically AT it); statistical-conventions disclosure para in
##     sec:setup. Experimental ledger at optimum — writing/poster only.
## ================================================================

---

## ================================================================
## STATUS 2026-07-29 (READ THIS FIRST — supersedes everything below)
## TwoRoom re-opened, three retractions, one live positive.
## ================================================================
##
## THE BLOCKING FACT IS RESOLVED (2304691, read 2026-07-29 midday).
## LeWM's paper (arXiv 2603.19312, Fig. 6) reports **87 on Two-Room**; our
## flat read 64-66. The anchor ablation peeled our protocol differences off
## cumulatively, three seeds each:
##   V0 ours-as-run 65.10 | V1 -cross-room 65.10 (vacuous: success is a
##   subset of cross-room in this dataset) | V2 -success-filter 79.69 |
##   V3 -holdout 80.21 | V4 start-random 82.29 | V5 +goal_proprio,n=50
##   **85.33** (seeds 86/82/88; bar was >=82: PASSED, their 87 inside our
##   seed range).
## The 21pp gap is PROTOCOL, not the checkpoint: success filter ~14.6pp
## (it selects a strictly HARDER population, opposite of its intent),
## start source ~2.1pp, goal source + n ~3.0pp, cross-room 0, holdout
## ~0.5pp. Every TwoRoom margin we measured was under a uniformly harder
## protocol, now attributed pp-by-pp. Also noted: with goal_state on
## unfiltered episodes the goal IMAGE (agent at demo end) mismatches the
## scored target; LeWM's goal_proprio is the coherent goal source on the
## unfiltered population, not just their convention.
##
## IN FLIGHT (submitted 2026-07-29 midday, docs/tworoom_paired_prereg.md
## 07-29 amendment frozen FIRST — anchored protocol, Stage A/B bars, gate):
##  * 2304722 Stage A: flat RH {1,2,5} x 6 seeds + oracle s25 RH=1 x 6
##    seeds at the ANCHORED long protocol (t=75, eval-filter none,
##    episode-min 4000, goal_proprio, n=64). Sets the Stage B bar.
##    Stage B gate: oracle >= best flat + 3pp, else no-headroom scope row.
##  * 2304723 best-of-k offline probe (probe_bok.py; k* = 80% of k=16
##    gain; kill if no gain or LeWM-half fidelity degrades). Mechanism
##    implemented in paired.py (--best-of-k, --bok-score goal|feas).
##  * 2304733 item-3 chain: unfiltered pair pool (--no-success-filter in
##    build_subgoals.py, ~2x pairs, matches anchored serving population)
##    -> 576-d gc drafter, proven recipe -> offline norm gate + bok probe.
##    NOT closed-loop until gated + pre-registered.
##
## CONTEXT FROM THE PAPER (newly read, changes the framing):
##  * LeWM is the WEAKEST method on Two-Room: 87 vs DINO-WM 100, PLDM 97,
##    GCBC/GCIVL/GCIQL 100. It is the only env where LeWM loses.
##  * Their stated reason: SIGReg in a low-intrinsic-dimensionality env
##    gives "a less structured latent representation".
##  * We measured exactly that independently: TwoRoom equiv p50 0.876 vs
##    cross 1.456, by far the worst of the four envs (PushT 0.085,
##    Cube 0.091, Reacher 0.230).
##  => The valuable claim is NOT "4th env win". It is: LeWM's TwoRoom
##     deficit is representational, we can measure it, and certifying
##     subgoals in an external structured space recovers part of it
##     WITHOUT retraining the world model.
##  * Other LeWM numbers (we reproduce these): Reacher 86, PushT 96,
##    Cube 74.
##
## THREE RETRACTIONS (all mine, all caught by discipline or by the user):
##  1. "+18.8pp drafting margin at t=75" -> flat had never been swept over
##     replan rate. RH=1 lifts flat from 27.34 to 44.53. Real margin +1.6pp.
##  2. "+10.9pp at t=25" -> two seeds. At six seeds flat rose 60.94 -> 66.41
##     and the margin became -2.1pp.
##  3. Stride-saturation as a scope rule -> REFUTED by its own frozen
##     prediction. Distance saturation does not imply the planner has no
##     signal. The "explains the interior optimum in S" claim goes with it.
##
## THE BUG THAT INVALIDATED JULY. Both TwoRoom drafters emitted latents
## ~100x too large (109.7 and 132.0 drafted/true norm) against a PushT
## control at 0.988 through the identical code path. Cause: 11,807 training
## pairs x 20 epochs = 940 gradient steps for a 53M DiT. Fixed by
## v-parameterisation + cosine + min-SNR 5 at 400 epochs (ratio 1.014/1.001).
## => The July "TwoRoom parked FINAL / intervention failed" conclusion rests
## on a broken drafter and MUST be retracted. Its replacement is not a win,
## it is a properly measured null-to-small-positive.
## An audit confirmed every paper drafter is clean (pusht 0.991/0.992,
## reacher 1.001, cube 0.987), so the 11-cell grid is unaffected.
##
## THE ONE LIVE POSITIVE (t=75, RH=2, 12 seeds, filtered protocol):
##   flat 43.36 | pair no-verify 49.22 | pair+verify tau=.098 48.83 (cr .962)
##   => +5.47pp, t ~ 2.0, p ~ 0.06. Survived doubling the seeds, which the
##      two retracted margins did not. Against strongest flat anywhere at
##      t=75 (RH=1, 44.01) it is +4.8pp.
##   CEILING: oracle 52.60 at t=75 RH=1 vs flat 44.01 (+8.6) => headroom is
##      real at range. At t=25 oracle is 46.61 vs flat 64.84 (-18.2) =>
##      decomposition CANNOT win at short horizon, so TwoRoom must be a
##      horizon-extension result, as PushT and Reacher already are.
##   voracle 15.62 / 1.56 is not a ceiling: verification in LeWM space never
##      fires so it sticks on waypoint 1. Consistent with the dead verifier.
##
## WHAT WORKS AND WHAT DOES NOT, on TwoRoom:
##  WORKS   certify in frozen DINOv2 instead of LeWM space: +5.2pp at t=25
##          over the same drafter unverified (specaccept/paired.py)
##  WORKS   jointly-trained 576-d drafter: +3.1pp over the 192-d one
##  DEAD    verification in LeWM's own latent space: call_ratio 0.998-1.000,
##          numerically identical to no verification in every cell
##  NULL    retrieval snapping to real training frames
##  NULL    progress-constrained snapping
##  NULL    arrival gate (also null on DINO-WM: gated and ungated traces
##          byte-identical, so that frozen prediction is refuted too)
##
## OTHER LEGS:
##  * WALL P3 PASSED: flat 1.00, spec tau=0.20 0.685. The instrument
##    predicted a fixed tau underperforms on a never-seen task; it does, by
##    31pp. Cleanest prospective hit we have. P4 unavailable (its lens never
##    opened at the serving hop, so no derived tau; reported as such).
##  * DINO-WM pusht P1/P2 FAILED: flat 0.82-0.88, spec 0.00-0.06. Drafter is
##    weak but NOT norm-broken (1.08-1.29x no-op, loses to lerp twice). The
##    arrival gate did not fix it. Unresolved; do not write up as a clean
##    negative until the serving path is understood.
##  * pusht H15 (2299902) running. TwoRoom DINO-WM WM training continues.
##
## EVENING UPDATE (2026-07-29): STAGE A+B BOTH LANDED SAME DAY. ALL
## PRE-REGISTERED BARS READ. TwoRoom = a WIN at the anchored protocol.
##  * Stage A (2304722): flat RH1/2/5 = 41.67/42.71/38.54, oracle 57.81.
##    Gate PASSED (+15.1). P-A1 refuted+recorded (anchored flat t=75 is NOT
##    higher than filtered; the success-filter cost is horizon-dependent).
##  * bok probe (2304723): k*=8 both rules; 'goal' primary ('feas' =
##    stay-put risk, recorded in advance).
##  * nofilt drafter (2304733): 39,637 pairs (3.36x), norm gate 1.003/1.002.
##  * Stage B (2306077), six-seed means vs bars (prereg has full table):
##      pair+verify t098 55.47 vs bar 46.71  -> B1 PASSED (+12.8 over flat,
##        wins ALL SIX seeds, paired t~5.9; > 2x the filtered margin)
##      + bok8-goal 58.33 vs bar 57.47       -> B2 PASSED (>= pair every
##        seed; sits AT the oracle ceiling 57.81)
##      no-verify 56.25, verify -0.78 at cr .94-.97 -> B3 PASSED
##      nofilt drafter 53.65                 -> B4 FAILED (null, recorded:
##        3.4x matched data does not beat the filtered pool)
##      bok8-feas 32.29                      -> stay-put pathology CONFIRMED
##        (-23pp; offline fidelity misleads again, predicted in advance)
##      tau=0.20 52.08                       -> robustness row, derived tau
##        better but not fragile
##  * 12-seed extension 2306141 IN FLIGHT (flat RH2 + pair t098, seeds
##    48-53, fired unconditionally BEFORE Stage B was read; pooled bar =
##    margin >= +3pp AND t >= 2).
##  * Poster strips: DONE + rendered (pusht spec/flat t150, tworoom spec
##    t75; strips/*.npz + render script specaccept/render_strips.py).
##  * Paper: July TwoRoom conclusion retracted, anchor + representational
##    story in, \inflight marker awaiting the 12-seed pool. Compiles, 22pp.
##
## MORNING 2026-07-30: HEADLINE BANKED + FAITHFULNESS PROGRAM RUNNING.
##  * 12-seed pool (2306141): flat 41.67 vs pair+verify 57.03 =
##    **+15.4pp, positive on ALL 12 seeds, paired t~10.9** vs bar
##    (>=+3pp, t>=2) -> PASSED. Ext seeds STRONGER for spec (58.6): no
##    regression-to-mean. 57.03 = statistically AT oracle (57.81).
##    Paper \inflight swapped for measured text; compiles, 22pp.
##  * k derivation: k* = 2 (bias below floor at every k; sampler converges
##    immediately in the verification space). Confirm 2306515 in flight,
##    bar: within 3pp of 57.03 at 6 seeds -> pass = 25x fewer diffusion
##    steps per draft in the faithful row.
##  * PushT paired drafter: norm gate PASSED (1.001/1.002), beats no-op in
##    BOTH halves (LeWM half faithful on pusht, 0.12-0.17, vs chance ~1.37
##    on tworoom = the encoder story from the drafter's side).
##  * PushT verify-space CONTROL pre-registered (prereg doc): C1 verify
##    LeWM-half tau=0.20 vs C2 verify DINO-half tau=derived, SAME drafter,
##    t=150, 6 seeds; tau from probe_floor --verify-space dino (2306521,
##    in flight). Answers "why not DINOv2 everywhere" with a measurement.
##
## NIGHT 2026-07-30 (user asleep; all read + fired same night):
##  * k=2 confirm REFUTED (31.25 vs 57.03): k-from-bias rule dead closed-
##    loop; cause visible offline = diversity collapse at low steps
##    (disp 0.049 vs 0.127); k stays 50. Second prescriptive-rule failure.
##  * THEIR-PROTOCOL battery (2306523, LeWM Fig.6 setup verbatim + n=64
##    x6 seeds): flat RH5 84.89 (their 87 in noise) | oracle-s10 63.54
##    (-21pp: goal sits INSIDE one planning window; decomposition
##    structurally counterproductive) | best spec arm (bok+proximity-router
##    RH5) 77.08 -> B25-1 non-inferiority FAILED (-7.8pp), P25-1 confirmed.
##    NEW router mechanism built (--route-goal-hop 0.163, derived).
##  * => FINAL TWOROOM CLAIM = WINDOW-RULE COMPOSITE (zero new constants,
##    goal_offset is a known task parameter): within one window -> flat
##    (84.9 = ties LeWM's own game); beyond -> certified spec (57.0 vs
##    41.7, +15.4pp). Both sides measured at THEIR-anchored protocols.
##    Reacher shape; never loses to flat. Prereg has the full table.
##  * PushT C1/C2 verify-space control chained + queued (floor probe
##    2306565 -> battery 2306566, tau_dino read from JSON at runtime;
##    specpaired arm + --verify-half ported to the pusht driver).
##
## MORNING 2026-07-31 (all overnight batteries read + banked; prereg has
## the full verdict tables):
##  * CROSSOVER PREDICTION CONFIRMED: flat +4.9 at t=40, spec +3.4 at
##    t=50, tie t=60 (noise), spec +15.4 at t=75 -> crossover in [40,50],
##    where the goal exits the ~25-step window + one stride. The window
##    rule is now a measured curve (spine panel material).
##  * bok 12-seed pool 58.85 vs 57.03 = +1.8 (t~1.1): 6-seed bar passed
##    but margin within noise at 12 -> bok stays optional, OUT of headline.
##  * their-protocol flat 12-seed: 83.72 (composite table symmetric).
##  * C1/C2: SR indifferent (63.54 vs 62.24) BUT C2's derived-tau DINO
##    verifier is degenerate-reject (cr 1.00, 0 advances) vs C1 cr 0.66:
##    external space loses ALL call-savings where the own gap is open ->
##    the verify-space rule stands WITH a measured reason.
##  * Paper updated (their-protocol + composite + crossover + k=2/bok
##    honesty in one block); compiles 22pp. TWOROOM IS EXPERIMENTALLY
##    CLOSED end-to-end; zero TwoRoom compute outstanding.
##
## 2026-07-31 LATER: PROGRAM FULLY CLOSED. DINO-WM leg killed per the
## Aug-1 switch (2299983 cancelled; P3 = confirmed prospective hit, P4
## unavailable, P1/P2/P5 = confounded serving fault, claimed in neither
## direction; tworoom closed-loop cut, offline cure stands). TwoRoom
## fairness control (2306686): flat+2xCEM = 48.44 -- rescues +6.8 over
## flat (prediction too strong, reported as-is) but spec stays +8.6pp
## above the control (paired t~2.5) with less compute -> margin NOT
## compute-driven; last control in the program. Paper solidity pass DONE:
## TwoRoom in abstract/contributions (verifier-modularity bullet)/grid
## para/conclusion; both instrument_generality \inflights resolved with
## measured verdicts; composite switch re-stated as ONE MEASURED CONSTANT
## (crossover), fixing the zero-constants overclaim. Crossover figure
## rendered (paper/figures/fig_tworoom_crossover.png). Compiles, 22pp.
## ZERO experiments outstanding anywhere in the program.
##
## TIMING BANKED (2306731): tworoom overhead = ~140ms on ~6.9s episodes
## (+2%; CEM 94% of wall-clock; lerp tautology reproduced flat 42.7 =
## instrumentation validated). fig:cost column complete for all 4 envs.
## Paper sentence added; compiles 22pp. TRULY ZERO experiments left.
##
## NEXT, IN ORDER:
##  1. Poster assembly (Aug 11; awaiting format/template decision;
##     filmstrips + crossover figure ready as panels).
##  2. Commit the 07-30/31 work when asked.
##  3. Post-poster: cube best-of-k battery (pre-register >= +2pp bar).
## ================================================================

---

## ================================================================
## STATUS 2026-07-27 (superseded by the block above)
## New-agent handoff: the full picture in one block.
## ================================================================
##
## WHAT THIS PROJECT IS. Method = CERTIFIED SPEC-ACCEPT: draft a block of
## N subgoal latents, serve them to the stack's own frozen planner, and at
## every replan boundary VERIFY the achieved latent against the waypoint
## pursued (rel L2 <= tau -> advance at zero drafter cost; else re-draft
## from reality). Constants derived, not tuned (tau = criterion floor /
## gap; k = sampler convergence). Companion instrument = the VERIFICATION
## GAP [equiv p90, hop p10] computed from cached latents in minutes:
## predicts applicability, tau, and which auxiliary mechanism (none /
## arrival gate / c* router / lens) BEFORE any closed-loop run.
##
## PROVEN & FROZEN (never re-run):
##  * LeWM grid: certified spec >= strongest flat in ALL 11 pre-registered
##    env x horizon cells (PushT 5/5, Reacher 4/4, Cube +14.0 vs +5 bar);
##    fairness/timing/tau-robustness closed. paper tab:grand.
##  * Gap instrument: explains every closed-loop outcome on 9 substrates;
##    3/3 prospective predictions confirmed (droid lens 8x, v3 drafter
##    fidelity + planning value at 20x). Cube tau-midpoint PRESCRIPTION
##    failed its bar -> claim is applicability/tau-existence only.
##  * TwoRoom-on-LeWM: provably out of scope (gap inverted, encoder
##    saturated ~sqrt2); ENCODER SWAP to frozen DINOv2 CURES the metric
##    (equiv 0.876 -> 0.081) = scope condition is an encoder property.
##  * V-JEPA 2 (1B): mechanism + tau transfer; certificate dead (3x);
##    15x compute claim RETIRED (failed powered pre-reg replication; no
##    offline instrument can referee planning there). Honest scale bound.
##
## IN FLIGHT ON ISCA (bars frozen in docs/dinowm_prereg.md BEFORE runs):
##  * pusht-DINO battery 2299661 (P1 spec@tau=0.121 >= flat-2pp; P2
##    tau=0.20 degenerate-accept). Flat anchor banked: 86% (paper 90%).
##  * wall anchor 2299790 + prep 2299789 -> battery 2299791 (P3 fixed-tau
##    underperforms; P4 lens-verified recovers). Lens taus derived.
##  * pusht goal_H=15 battery 2299902 (P5: margin improves at range).
##  * TwoRoom recovery: WM training 2299632 + drafter prep 2299811; the
##    LAST unbuilt piece = tworoom eval wrapper for dino_wm (vendor
##    stable_worldmodel/envs/two_room/env.py, 710 lines, pymunk; then
##    pre-register bars incl. long-horizon cells where LeWM flat
##    collapsed 68->31; lens verifier tau=0.540 designated).
##  * point_maze = prediction-only row FOREVER (env needs mujoco-py+d4rl).
##
## INFRA MAP (ISCA, ssh isca, user ha676; PowerShell ssh only, Git Bash
## cannot auth; beware PowerShell quoting — use script files):
##  * le-wm stack: /lustre/home/ha676/le-wm (.venv py3.11 torch2.5).
##  * dino_wm: /lustre/home/ha676/dino_wm (.venv py3.9 torch2.3); compat
##    patches REQUIRED and already applied in-place: cem.py latent-goal
##    branch, evaluator decode gate (DWM_DECODE opt-in), train.py
##    save_ckpt guard, env/__init__ mujoco guard, dinov2 hub py3.9 patch
##    (rerun patch_dinov2_py39.py if TORCH_HOME cache purged), max_iter=12
##    cap in runners (their null = infinite loop). Copies of all patches +
##    runners live in batch/isca/dinowm/ (committed).
##  * Serving stack: batch/isca/dinowm/specaccept_dinowm.py (SpecMPCPlanner;
##    verify on POOLED VISUAL tokens only = the gap space; spec serves
##    drafted VISUAL target + goal's own proprio — drafted proprio token is
##    untrained, serving it flies away). Runner env vars: ARM/TAU/PLAN_CFG/
##    MODEL_NAME/GOAL_H/READOUT_CKPT/MAX_ITER.
##  * Data: /lustre/home/ha676/data/dinowm/* (pusht grids = (T,197,384)
##    fp16, 196 visual + 1 proprio token); QUOTA IS NEARLY FULL — check
##    lfs quota before big encodes.
##  * Drafter training reused verbatim: vjepa2/specaccept_vjepa2/
##    train_drafter.py on grid npy files (dims inferred from data).
##
## DISCIPLINE (non-negotiable, the paper's credibility rests on it):
##  * Anchor flat FIRST on any new substrate (cube lesson).
##  * Gap probe + freeze predictions in docs/dinowm_prereg.md BEFORE any
##    closed-loop spec run. Never change arms mid-battery.
##  * Claims are internal to a stack (spec vs THAT stack's flat); never
##    compare absolute numbers across stacks or papers (SAGE's protocol
##    differs wildly from ours).
##  * Commit only when the user asks; no co-author lines; no em dashes in
##    paper prose or commit messages; user prefers concise headline-first.
##
## PAPER STATE: main_v2.tex compiles clean (21pp; shim
## iclr2026_conference.sty is a PLACEHOLDER — swap official kit before
## submission; bst = plainnat copy). New: fig:method + fig:gap (TikZ),
## sections/instrument_generality.tex with \inflight markers awaiting
## battery verdicts. Abstract/contributions/conclusion tightened, em
## dashes purged (table cell placeholders remain). Supervisor brief:
## claude.ai artifact dde8b3df (concise, same URL on republish).
##
## RELATED WORK (positioning, memory: related-work-sage-trm): SAGE
## (2607.17973) = CONCURRENT subgoal drafting on LeWM, NO verification /
## instrument / generality -> cite as premise validation, differentiate
## on certification. TRM (2605.22164) = metric-repair, lens-adjacent.
## PLDM port rejected (optional cheap gap-probe only).
##
## NEXT, IN ORDER: (1) read off pusht/wall/H15 battery verdicts, swap
## \inflight boxes for measured sentences; (2) build tworoom eval wrapper
## -> pre-register -> final battery; (3) deep verbosity pass on main_v2
## body + cube spine panel plot; (4) related-work re-sweep + real
## SAGE/TRM author names in references.bib; (5) official ICLR style kit.
## Kill-switch: anything DINO-WM-side unfinished by 2026-08-01 is cut and
## reported as frozen predictions.
## ================================================================

---

## ================================================================
## STATUS 2026-07-15 (READ THIS FIRST — supersedes cube/EXP-3 blocks)
## ================================================================
## The paper's method is now UNIFIED SPEC-ACCEPT (spec-accept v2): draft
## subgoals only while the planner certifies the goal is out of reach
## (c* > tau); episode-start c* commit = the router; per-replan c*-retire
## replaces the latent goal-gate. One rule, one tau=0.20 (criterion-floor
## derived), S=10 universal (user decision 2026-07-15: NO per-env stride),
## k derived per env (pusht 3 / reacher 8 / cube 3). Implemented as
## `CstarRetireSource` (sources.py) + `--subgoal unified` in all 3 drivers;
## episode-level routing via probes/route_horizon.py + routed-subset eval
## (`--episodes-file`). Old goal-gate + plain-router results = ablations.
##
## BANKED 2026-07-14/15 (all at certified protocols, ISCA):
##  * CUBE FLIPPED TO A WIN then CONFIRMED: gc10+goal-gate k=3 vs flat on
##    8 FRESH seeds (50-57, job 2276035, pre-registered bar >= flat+5pp):
##    76.46 vs 65.92 = +10.5pp PASS (exploratory margin +10.3 reproduced;
##    oracle 82.4 for context). Mechanism: overshoot tax (same as reacher
##    short) +32pp via gate + goal-conditioning +49pp; both needed on cube.
##  * REACHER gate v4 ternary router (exploratory, pop 888): composite
##    98.05/97.07/93.75/93.55 vs LeWM-best 97.07/96.88/83.79/75.20 ->
##    beats BOTH flat arms at all four t; the one leak = -2.15 vs our own
##    spec at t150. r_flat2 subsets 98-100% pure at every t (c* selection
##    is the mechanism proof). Gate v3 = 2/4 pre-reg (t50 miss diagnosed
##    -> fixed by v4's RH5 flat branch).
##  * PUSHT already won by PURE spec k=3 everywhere (s5 spine, 20deg:
##    99.6/97.9/98.8/97.3/98.0 vs flat 98.1/60.4/42.6/12.7/2.7) — the one
##    env whose expert data contains arrival (episodes end at goal), which
##    is WHY it needed no gate: independent confirmation of the overshoot
##    mechanism. Routing pass: c*-optimism REFUTED on pusht (t100/t150
##    route 256/256 to spec; flat fires only at t25/t50).
##  * Goal-gate scope finding: rescues short horizon (+35pp reacher t25)
##    but HURTS long horizon on reacher (-10/-15pp vs ungated; latent
##    distance saturates at range) — hence c*-retire in the unified method.
##
## IN FLIGHT (all pre-registered, frozen bars in sbatch headers):
##  * reacher v4c confirm, fresh pop 999: 2276033(done)->2276034 (~1h left)
##  * pusht v4 eval (flat2 refs + routed subsets): 2276094 (overnight)
##  * UNIFIED batteries: reacher spec-leg 2276371 (16c), pusht spec-leg
##    2276360 (10c), cube full arm seeds 50-57 2276361 (8c)
##  KNOWN RISK (declared): cube c*-retire may never fire (criterion floor
##  0.76-0.79 >> tau) -> behaves like ungated gc spec ~76-77; principled
##  fix if so = retire threshold := criterion floor (derived, new pre-reg).
##
## NEXT after the grid lands: (1) assemble 3-env x 11-cell table (unified
## vs flat refs vs plain spec; two-sided bar: >=LeWM everywhere, >=spec
## within noise) + efficiency column (NFE + CEM solves + pusht wall-clock);
## (2) write into Results/RESULTS.md + main_v2 tables; (3) ICLR push =
## V-JEPA 2 TRANSPLANT (the generality result; ~8 weeks to deadline);
## more seeds on headline cells; failure-anatomy figure (trace format
## reconciliation pending in probe_failure_anatomy).
## ================================================================

---

## GATE 0 - unblock
- [ ] push main to GitHub (user; everything is committed locally, boxes clone from GitHub)
- [x] compute up — RESOLVED 2026-07-11: full SLURM ISCA (`login.isca.ex.ac.uk`, ha676, `ssh isca`),
      6 live A100 nodes (gpu11-16). Battery deployed via `batch/isca/` (commit bdb74b3):
      venv + dataset (99GB h5, schema-verified) + encoder staged; full chain QUEUED
      (smoke 2270234, dense 2270235, train 2270236, arms 2270237, horizon 2270238,
      gpu-probe 2270232), waiting on the last 6 ECHELON `isca_scratch` tasks to free nodes.
      Redeploy: `bash batch/isca/deploy_reacher_isca.sh`. Unused alternates:
  - [ ] Isambard back -> `bash batch/deploy_reacher_isambard.sh reacher`
  - [ ] ISCA v03/v05 JupyterHub boxes -> `git pull` (or re-paste `batch/box_bootstrap.txt`) then
        `batch/run_reacher_local.sh` (v03 had smoke-pass + dense done + 4 trainers in flight
        2026-07-10 22:19, fate unknown — cull hazard; check browser tab, may yield a free replication)

## WRITE 0 - paper work with no experiment dependency (do anytime)
- [x] references.bib CREATED 2026-07-13 (verified IDs only; DSpark = arXiv 2607.05147
      confirmed via live sweep; author-list CHECK notes remain; VLWM still unresolved)
- [ ] algorithm box tbdblock: spec-accept pseudocode + CEM cost
- [x] related work WRITTEN 2026-07-13 (main_v2, with live concurrent-work sweep:
      closest = BID 2408.17355 / RTC 2506.07339 / DCDP 2603.01953; SPO 2603.19418 =
      concurrent but speculates over network transport, not drafter compute; re-sweep at submission)
- [ ] terminology sweep: spec-accept vs DSpark usage
- [x] NFE-currency defense paragraph + abstract honest-ordering skeleton (2026-07-13, per review)

## DECIDE - scope (anytime before WRITE 2)
- [x] sec 11.2 TwoRoom: DECIDED 2026-07-12 — boundary-case paragraph written into
      the paper (unobservable goal = sharpest form of the data/observability
      condition); no full battery, not pursued further
- [ ] sec 11.3 OGBench-Cube: build (dataset exists on HF; port = reacher pattern;
      needs reacher.h5 shrink for disk) or CUT — decide after WRITE 1

---

## EXP 1 - Reacher drafters                          [gated by: GATE 0]   == DONE 2026-07-12 ==
- [x] smoke 2270234: oracle 90.62 / baseline 93.75 (n=32) - stack validated, 5 min
- [x] dense 2270235: 2.01M frames @ 790 f/s, subgoals_reacher_dense.pt
- [x] train 2270236[0-3]: all 4 ckpts, ~4h6m each, final losses 0.006-0.009 (finite, falling)
- [x] RAM-cap fix validated (no OOM at --mem=80G)

gated ->

## EXP 2 - Reacher battery: LeWM + GDM + spec-accept  [gated by: EXP 1]   == DONE 2026-07-12 ==
- [x] arms 2270237[0-35]: baseline 83.99; gdm 52.5-60.6 (s10 top); spec 46.1-60.9 at cr 0.59-0.88
- [x] horizon 2270238[0-23] + baseline arm 2270556[0-7] (was missing from the PushT-derived design!)
      + seed extension 2270564[0-11] (t75/t100 2t cells to n=4)
- [x] aggregate: logs merged to combined naming on box; aggregator output = the §11.1 tables
- [x] criteria: (i) interior optimum s10 > S=25 YES (but whole curve < baseline natively);
      (ii) spec-accept SR-neutral at cr<1 YES, every cell every regime (t100: 91.0 at cr=0.61);
      (iii) gdm >= baseline: NO natively (-24pp) and NO starved-budget (-15..-25pp);
      YES at long horizon + adequate budget: t100 2t = 90.8 vs baseline 83.2 (n=4, paired pop).
      STORY: goal-image planning degrades once the goal is remote (91.8@t50 -> 83.2@t100);
      drafted subgoals restore ~8pp exactly there; spec-accept delivers it at 61% of calls.
      Scoping: subgoal detours COST under starved budgets; drafting needs headroom to pay off.

gated ->

## ICLR ROADMAP (added 2026-07-12; target: main track. Paper identity:
## "spec-accept = SR-neutral at 10-20x less drafter NFE, across envs/drafter
## types/regimes + first careful characterization of WHEN subgoal drafting
## helps (goal beyond planner sight + budget headroom + data quality)")

### EXP 2b - comparison hygiene: strong baseline + extensions   [IN FLIGHT on ISCA]
- [x] goal-gate v1 (--goal-gate, greedy latent progress): REFUTED 2026-07-12 —
      collapses the t100/t150 crossover to ~baseline (91->83, 94->80); t25 "win"
      (91.6) explained by the RH confound. Code stays flagged-off in sources.py.
- [x] RH=2 flat-baseline control: t25 = 95.7 (!), t100 = 67.6 (collapses).
      -> NO flat config is good everywhere; replan-rate trades short vs long;
      crossover SURVIVES the strongest per-cell flat baseline. Every paper table
      must anchor to strongest-flat-per-cell, not protocol RH=5.
- [x] RH grids COMPLETE both envs: Reacher RH{1,2,5} x t{25,100,150} (95.7/83.2/76.0
      best-per-cell); PushT RH{1,2} short=81.8/93.4 (RH5 already strongest), horizon
      collapse at every RH (RH1 88->18, RH2 96->47) — replan-rate is a brittle
      trade-off on both envs; crossover survives strongest-flat everywhere
- [x] frontier n=4: tau.1/k4=95.9, tau.2/k4=95.5 at 2.6 NFE/replan
- [x] goal-free ablation: 8.6-11.3% ~= RANDOM FLOOR (vs goal-cond 60.6-95.1) — maximal form
- [x] AUDIT ROUND (2026-07-12 evening, all n=512 unless noted):
      * oracle FULL POWER: 84.2 +-1.0 = flat RH5 (84.0) — parity bound certified
      * straight-line (lerp) oracle t25: 21.9/70.7/95.5 for frac .25/.5/1.0 —
        DECOMPOSITION ITSELF is the tax (mechanism separated from data meander);
        frac=1.0 reproduces flat RH2 within 0.2pp (harness self-check)
      * k=8 every-step mirror t100: 94.9 — BEATS k=50 ref (90.8); HAIRCUT: spec-accept
        marginal factor ~3x (95.5@2.6 vs 94.9@8), NOT 20x; cheap-drafts effect separate
      * population identity verified (byte-identical md5 across batteries); budget
        parity verified by code path
- [x] gate v2: NOT pursued — v1 refuted, scoping guideline ships instead

### EXP 2c - PushT restage on ISCA (removes dead-box dependency; enables PushT
###          frontier, strong-baseline sweep, goal-cond ablation, timing figure)
- [x] recon: datasets/quentinll/lewm-pusht EXISTS on HF (13GB h5.zst); encoder =
      --source pretrained (swm hub auto-download, drivers default to it); box2d-py
      builds with --no-build-isolation after swig (plain pip fails)
- [x] STAGE A LAUNCHED 2026-07-12 (batch/isca/pusht/): deps -> download ->
      waiter auto-submits sanity -> smoke(baseline n=32) + dense -> train x3
      (gdm_stride10 goal-free = headline replication; gdm_stride25 goal-free =
      faithful gate, diag must reproduce rel_err ~0.16; gdm_stride10_gc
      goal-cond final-rule = the last 2x2 ablation cell). Drafters by morning.
- [x] STAGE B CERTIFICATION PASSED (2026-07-12 ~16:30): replication anchors
      baseline 94.6 / oracle 92.6 / s25 94.8 / s10 99.4 vs box-era 93.4/93.4/95.3/98.4
      — every PushT number in the paper now reproduces from public assets on ISCA.
      (s25 offline diag 0.43 vs 0.16 anchor = cross-rebuild probe incomparability;
      closed loop is the certification. gc-drafter diag crash was cosmetic — diag_gdm
      can't feed goals; ckpt fine.)
- [x] AUDIT ROUND 2 LANDED (2026-07-12 evening):
      * lerp oracle at t100: 33/44/55 (frac .25/.5/.75) vs drafter 94.5 —
        THE LEARNED DRAFTER IS NECESSARY (geometric interpolation fails at horizon)
      * every-step k=4 at t100 (Reacher): 96.1 — k-curve monotone 50->8->4;
        spec-accept honest marginal = ~1.5x NFE at SR-parity (2.6 vs 4.0)
- [x] PUSHT SPINE LANDED (unfiltered population): baseline 95.5->13.7 vs drafters
      ~87->59 over t25->150 — crossover +45pp at t150, in-population. CAVEAT
      DISCOVERED: regenerated population lacked the success filter (includes
      failed demos), so absolutes sit below box-era table; success5-filtered
      respin queued (2271135) for box-era comparability; unfiltered spine kept
      as harder-population robustness row. NEVER merge the two populations.
- [x] OVERNIGHT GATES LANDED 2026-07-13 (all COMPLETED; 763 logs pulled, parsed,
      archived Results/isca_gates_logs/ + README job map; main_v2 tables updated):
      gamma x40 final (S25 49.0->57.9->4.7; S10 gentler) | k-grids + cliff (es
      55.9/55.0/63.3/64.1 at k1/2/4/8 t100; spec k<=2 cr=1.00) | tau-on-val flat
      (63.5-64.5, tau=.20 re-selected) + off-axis separable | timing (drafter
      1.0%@k50/0.4% frozen; S10 12% faster) | reacher S-long 91-96 flat |
      s5/s15 t150 (interior optimum persists: 55.3/59.6/58.3/55.9) |
      success5 spine x40 (reproduces box-era: flat 98.1->2.7, s10 ~98-99) |
      gc 2x2 closed (98.9 vs 99.4 short; 60.1 vs 59.6 t150)

### EXP 2e - VERDICT BATTERY: blind-vs-verified + derivability (2026-07-13) == DONE ==
### (triggered by external review: "tau may be the encoder's noise floor; blind
###  commit-2 at matched k is the single highest-information cell in the paper")
- [x] blind commit-2 at MATCHED k (--subgoal dspark --no-refine --commit fixed --commit-k 2):
      PushT in-distribution TIES spec at cr .50 — t100: 65.3 vs 65.0 (k4), 64.3 vs 63.3 (k8);
      t150: 58.1 vs 58.9 (k4). Verifier buys nothing in-distribution on PushT.
- [x] k-cliff INVERSION: blind2@k2 = 63.5 BEATS spec@k2 = 55.6 (+8pp; ~k4 quality at 1 NFE/replan).
      The cliff was CHURN (re-drafting under sampler noise), not "drafts unusable".
      ** RETRACTION: the "verifier self-test" claim (call_ratio->1.00 at k<=2 as a
      safety feature) is FALSE REJECTION. Never present it as a feature. **
- [x] Reacher blind-3 (spec tau=999, zero code changes): LOSES 5-6pp — 87.9/89.8 vs spec
      94.5/93.9 (t100/t150). Verifier LOAD-BEARING exactly where rejections fire (cr .62).
      Blind depth = hidden hyperparam with its own cliff (box-era commit-3 = -18pp).
      VERDICT: verification is REGIME-DEPENDENT and predictable, not universal.
- [x] FLOOR PROBE both envs (specaccept/probes/probe_floor.py -> Results/floor_*.json):
      * criterion floor (success-equivalent latent distance): PushT p50 .233 (20deg) /
        .173 (5deg), temporal .080; Reacher p50 .106 / p90 .153, temporal .223
      * tau* = 0.20 sits on both floors; cross-env SHAPE prediction PASSES (lower floor
        -> plateau ends earlier: Reacher degrades at tau=.3 while PushT still flat) —
        tau derivable to its plateau (band ~ floor p50..1.3xp90). Caveats: retrospective,
        n=2, quantile differs per env (reviewer surface).
      * sampler convergence: PushT SHARP unconvergence cliff k3->k2 (bias .11->1.04,
        dispersion .08->.65); Reacher NO cliff (wide conditional = spread, not error;
        k=1 works closed-loop, pre-existing)
- [x] k=3 PROSPECTIVE CONFIRMATION (2271747): the rule named k=3, never swept (grid
      bracketed [2,4]); measured es 63.3±0.5 (= k4 plateau, NOT the 55 cliff), spec
      64.5±1.1 at cr .63 = 1.9 NFE/replan (~18-25% under frozen k=4). DERIVED, NOT SWEPT.
- [x] Reacher spec k=4 respin (k-freeze consistency fix; review caught headline cells at
      gdm_steps=8): t100/125/150 = 95.5/95.5/95.7 — >= old k8 cells; use for headline.
- [x] 2271772 LANDED 2026-07-13 eve: ALL 24 PushT headline/spine/repl spec cells
      re-stamped at the DERIVED config (tau .20, k=3), logs *_s10speck3_*; every cell
      within noise of its k=8 predecessor (repl short 98.7±1.0 incl. a 256/256 seed;
      s5-spine 99.6/97.9/98.8/97.3/98.1; unfilt 86.5/79.1/70.5/64.1/58.4). Call ratio
      drifts up at k=3 on some cells (up to .81 vs .54) — verifier absorbing noisier
      drafts, SR unchanged; NFE/replan still ~1.9-2.4 vs ~4.3-4.5 at k=8.
      == DEV-ENV EXPERIMENTAL PROGRAM CLOSED (PushT + Reacher: nothing left to run) ==
- MECHANISM for paper §7 ("churn thesis", unifies 5 results): control pays for TARGET
      CONSISTENCY; re-draft churn ∝ draft noise. Evidence: cadence collapse, gamma curve,
      S=5 anti-churn (+5pp), k2 blind>spec (+8pp), Reacher blind<spec (-6pp).
- DERIVATION STATUS: validated retrospectively 2/2 envs + ONE prospective hit (k=3).
      NOT yet derive-FIRST. Graduates via pre-registered Cube (or TwoRoom / DINO-WM
      encoder swap). Confidence ~70-80%. Do NOT write unconditional "derived" in the
      paper until a derive-first battery lands.

### CONFIG FREEZE POLICY (adopted 2026-07-12; SUPERSEDED IN PART 2026-07-13)
- 2026-07-13 UPDATE: tau and k are now DERIVED, not frozen-by-sweep (EXP 2e):
  tau = the encoder's criterion floor (per encoder, offline, minutes);
  k = the sampler's convergence point (per drafter, offline, minutes).
  PushT derived = (0.20, k=3); Reacher derived = (0.20 in-band; k=4 recipe cell,
  k=1 cheapest Pareto point). The sweeps below are demoted to VALIDATION evidence.
  S remains the one tuned knob (the finding). NEW ENVIRONMENTS: derive-first,
  pre-registered, NO grids.
- tau = 0.20 FROZEN globally. Provenance disclosed in paper sec 6: selected on
  PushT, within 0.4pp of Reacher optimum, re-selection on disjoint val split in
  flight (robustness appendix, not tuning).
- k: FREEZE TONIGHT on joint two-env evidence at MATCHED anchors (t100 + t150 both
  envs). Reacher's early vote = k=4. If envs split: freeze ONE, report the other
  env's small loss honestly — never per-env k (undercuts the generality claim).
- S = THE one tuned knob (the finding). Interior-optimum evidence: PushT native
  4-point + t150 4-point (tonight); Reacher native 4-point + long-horizon 3-scale
  (tonight). Populations: t<=100 grid vs max-offset-150 grid NEVER spliced.
- gamma = mechanism experiment, not a knob (pre-registered gates in the sbatch).
- [x] PushT strong-baseline RH sweep DONE (short: RH5 already strongest at 93.4;
      horizon: collapse at every RH). tau/k grids + timing queued overnight.
- [x] SPACE: resolved without the shrink — pusht h5.zst was only 13GB (46GB
      unpacked), fits quota at ~440/500GB. Reacher-h5 shrink deferred to BEFORE
      the Cube download (46GB compressed, ~150GB+ unpacked — will need it).

### ===================================================================
### CUBE STATUS 2026-07-14 (READ THIS FIRST -- supersedes the block below)
### ===================================================================
### HARD LESSON: we NEVER ANCHORED THE CUBE BASELINE before running arms.
### On PushT we refused to trust any drafter number until flat anchored at 94.6;
### on Reacher until 84. For Cube we skipped that step -- and every conclusion we
### drew on 2026-07-13/14 was consequently WRONG. Four retracted claims:
###   (x) "drafting loses on cube"        -> from the INVALID v1 (vacuous cells)
###   (x) "cube-single is structurally short-horizon / untestable" -> premature
###   (x) "multi-segment random targets -> goal-free ill-posed"    -> FALSE, measured:
###        exactly ONE target per episode; the data IS goal-directed
###   (x) "the LeWM world model is too weak for manipulation"      -> FALSE:
###        LeWM's paper reports 74% on OGBench-Cube (user supplied the figure)
###
### MEASURED FACTS about cube_single_expert.h5 (trust these):
###   * ONE fixed target per episode (target changes 0 times in 201 steps)
###   * cube starts ~0.275 m from target, expert delivers it in ~90 steps, then
###     IDLES ~110 steps (cube is AT target for 109/201 frames)
###   * => our protocol was broken two ways:
###       (a) start = ep_len-1-t puts the start AFTER the cube already arrived for
###           small t  -> the cell is VACUOUS (doing nothing scores 100%). This is
###           why t=25/t=100 read 100% for every arm.
###       (b) budget = 2t gave 50 steps for a task that physically needs ~90
###           -> the planner was STARVED, and we misread the failure as a finding.
###   * with a non-vacuous fixed population (v2), flat baseline = ~21% vs LeWM's 74%
###     -> OUR PROTOCOL IS WRONG, not the substrate.
###
### *** ANCHOR ACHIEVED 2026-07-14 11:33 (2272343) ***
###   flat baseline = 71.1% @ goal_offset=150, budget>=300, RH=5  (LeWM: 74%)
###   -> REPRODUCED within noise. THIS IS THE CERTIFIED CUBE PROTOCOL.
###   (RH=2 gives only 65.6-66.4 -> RH=5 is the strongest flat config, as at range
###    on the other envs. offset-200 cells still running, may be closer still.)
###   Why v2 gave 21%: its non-vacuity filter demanded cube displacement > 0.08 m at
###   EVERY offset incl. t=25, which silently selected only the steepest part of the
###   trajectory -- a far harder subset than the real task. Over-corrected one bug
###   into another. DO NOT reuse cube_horizon.ep.json.
### NEXT: re-run the drafting arms (goal-free gdm_cube_s10.pt, + spec at tau=0.20/k=4)
###   at the CERTIFIED protocol and compare against flat=71.1. Only now do the numbers
###   mean anything. Regime-map prediction to score: cube is ONE pick-and-place with the
###   goal inside the planner's reach (flat already 71%) => expect LITTLE/NO drafting
###   headroom (the Reacher-native regime), NOT a win.
###   ALSO: gdm_cube_s10_gc.pt goal-cond drafter training 2272341 (~16:00) -- kept as
###   a control, though its justification (the multi-segment story) was wrong; the
###   goal-FREE drafter is probably fine since the data IS goal-directed and the
###   target marker is rendered in the observation.
### NOTHING about cube may be concluded (or written into the paper) until flat ~74%.
### Retire/ignore: cube v1 battery (2272069), cube v2 battery (2272298) -- both used
###   the unanchored protocol. gdm_cube_s10.pt (goal-free) is fine and reusable.
### OGBench 5 predefined tasks exist (task1_horizontal ... task5_diagonal2, with
###   init_xyzs/goal_xyzs in cube_env.py) -- if the dataset-replay anchor cannot hit
###   74%, switch to the env's TASK mode (reset(task_id), env's own rendered goal).
### CUBE-DOUBLE: parked. Premature until single-cube anchors.
###
### EXP 2d - third env: OGBench-Cube  [THE DERIVATION GRADUATION TEST]
## DERIVE-FIRST PRE-REGISTRATION (2026-07-13, before any closed-loop battery cell):
##  - S = 10 (pre-registered primary arm, zero-shot transfer of the dev-env
##    optimum; gated by floor-probe disp(10) >> criterion floor).
##  - drafter = GOAL-FREE (cube is expert/goal-directed data -> PushT regime;
##    2x2 prediction: goal-free works here, unlike Reacher).
##  - tau = the cube encoder's criterion floor (from Results/floor_cube.json,
##    floor-AB job 2272066); k = the sampler convergence point (from
##    Results/floor_cube_full.json, floor-C after training). BOTH recorded here
##    BEFORE the battery is submitted. Battery = baseline(strongest RH) +
##    every-step + spec-accept at derived (tau,k), horizon sweep.
##  REGIME-MAP PREDICTIONS to score: (i) goal-free works (expert data);
##  (ii) drafting beats flat at long horizon / where goal outruns lookahead;
##  (iii) spec-accept SR-neutral at derived (tau,k). If tau lands on the floor
##  DERIVE-FIRST (never seen this encoder), the derivation rule GRADUATES from
##  retrospective pattern to rule.
- [x] port VALIDATED 2026-07-13: swm/OGBCube-v0 + quentinll/lewm-cube encoder
      both exist; ogbench-1.2.1 installed; smoke baseline 100% (16/16) t=25.
      success = cube within 0.04m of target (env criterion); callables
      set_state(qpos,qvel)+set_target_pos(0, goal block pos, hindsight).
      Files: specaccept/envs/cube/{eval,build_subgoals}.py, probe_floor --env cube.
- [>] FULL OVERNIGHT CHAIN QUEUED 2026-07-13 ~21:30 (autonomous, dependency-linked):
      build 2272065 -> train+floorC 2272067 -> extract 2272068 -> BATTERY 2272069;
      floor-AB(tau) 2272066 + baseline-RH 2272070 parallel. Extract applies the
      PRE-REGISTERED rule (tau=criterion-floor p50; k=sampler convergence step,
      validated PushT->3 / Reacher->4) to Results/floor_cube_full.json -> writes
      cube_derived.env -> battery sources it. NO closed-loop number picks tau/k.
      Battery = {baseline RH5, s10 goal-free every-step k50, s10 spec derived} x
      t{25,50,75,100,150} x 2 seeds, n=128. ETA full spine ~morning.
- [!] CONFOUND FOUND 2026-07-13 (floor-AB 2272066): cube criterion_floor p50=0.967
      -- ~10x PushT/Reacher, BUT temporal_floor disp1 p50=0.087 (normal, ~PushT
      0.080). ENCODER FINE. Cause: cube success criterion covers ONLY the cube
      (0.04m) while the LeWM latent sees the WHOLE ARM; my criterion-equivalence
      matched cube-pos-alone -> paired frames with cube fixed but arm in different
      poses -> huge latent gap. The criterion-floor tau-rule DOES NOT PORT to a
      partial-observation criterion. Genuine scope finding, not a bug.
      => spec-accept overnight tau is PROVISIONAL (extract clamps 0.967->0.40).
      MORNING FIX (needs design decision, not a hack): options -- (a) criterion-
      equivalence = full-scene match (cube 0.04m AND proprio_effector_pos/joints
      within tol) so equivalent frames actually look alike; (b) tau basis = the
      TEMPORAL floor (disp1, confound-free, env-agnostic) -- cube 0.087; check it
      retro-fits PushT(0.080)/Reacher(0.223) vs tau*=0.20. Then re-run floor + spec.
- [!] EARLY SIGNAL: baseline-RH cells 100% so far -> cube-single flat planning may
      have NO drafting headroom (Reacher-native regime) at all horizons; if so the
      honest cube result is "no crossover / bounded", not a drafting win. Confirm
      from the every-step arm vs baseline overnight. (Multi-cube double/triple would
      be the real long-horizon case but we only have single-cube data.)
- [ ] MORNING: pull cube_*_t*_seed* logs -> (a) goal-free every-step works?
      (b) any crossover vs baseline? (c) derived k (floor_cube_full.json part C,
      valid). FIX tau-derivation per above, re-run spec. Then decide cube framing
      (win / bounded-negative) + add to tab:spine, archive logs.
- [x] k-derivation (floor-C part C) UNAFFECTED by the confound -- drafter dispersion,
      not criterion pairs. Valid overnight.
- [x] TAU-FIX ATTEMPTED 2026-07-13 ~22:00 (full-scene: match cube AND effector, both
      0.04m; probe_floor cube branch updated + redeployed; floor-AB 2272078 rerun):
      floor only 0.97->0.79, STILL ~9x temporal (0.087). Full-scene match does NOT
      rescue it -> criterion-floor rule genuinely OUT OF SCOPE on cube (encoder keeps
      task-irrelevant scene variation). SHARPER finding for the paper. Kept both
      floor_cube.json (cube-only) + floor_cube_fullscene.json as evidence.
      => extract now has a SCOPE GUARD: if criterion_floor > 3x temporal, fall back to
      transferred default tau=0.20 (auto, logged); k still derived. floor-C bumped to
      pair-pool 120000. Battery spec now runs at a DEFENSIBLE tau, not a clamp.
- [+] EXTRA overnight (fill idle nodes, finish before cube battery): S x tau grid
      2272079 (S{5,10,25} x tau{.1-.4} k8 t100, closes sec:calib-scale separability
      TBD + documents the check(iv) band-widens-with-S effect). Spine FIGURE generated
      locally (figs/spine.{pdf,png}, make_spine.py) -> wired into paper (fig:spine),
      closes the centerpiece figure TBD.
- [x] recon: datasets/quentinll/lewm-cube EXISTS on HF (2026-07-12)
- [ ] scope after PushT restage: port = reacher pattern (envs/cube/eval.py
      driver + build_subgoals + batch chain); DECIDE sec 11.3 accordingly
- [ ] STAGING ORDER (2026-07-13): shrink reacher.h5 first (disk: 46GB compressed,
      ~150GB+ unpacked vs ~60GB free) -> download -> port -> PRE-REGISTER derive-first:
      run probe_floor.py on the Cube encoder BEFORE any closed-loop cell, commit to
      the derived (tau, k) + declared S rule, one battery at that config only.
      This battery doubles as the derivation rule's graduation test.
      (Optional cheaper alternative/parallel: DINO-WM encoder swap on PushT —
      re-encode + retrain one S=10 drafter overnight, derive tau, one confirmation cell.)

### WRITE RULES (from the 2026-07-12 review; bind WRITE 1/2)
- (2026-07-13) NEVER present call_ratio->1.00 at low k as a "self-test" feature —
  RETRACTED, it is false rejection (blind2@k2 63.5 vs spec@k2 55.6). Verifier claims
  are REGIME-DEPENDENT, evidenced by the Reacher blind loss; blind rows are SHOWN in
  the tables, never hidden — the PushT tie is what the floor theory predicts.
- (2026-07-13) "derived" appears in the paper ONLY with its validation tier attached
  (retrospective 2/2 + prospective k=3) until a derive-first battery lands.
- The Reacher t25 row (spec 58.6 vs flat 95.7) is the paper's most attackable
  number: the oracle-parity bound AND the lerp collapse must sit IN THE SAME
  PARAGRAPH as that table, never in an appendix.
- Efficiency claims state TWO factors separately: cheap-drafts (k effect,
  everyone gets it) x consumption policy (~3x marginal, spec-accept's own).
  Never quote the conflated 11x/20x vs k=50 without the decomposition.
- One k everywhere. One tau everywhere. S varies and says why.

## WRITE 1 - Reacher section, one pass                [gated by: EXP 2 + 2b]
- [x] sec 11.1 DRAFTED 2026-07-12 (paper (1).tex): setup, native table, strong-
      baseline control, horizon table, frontier, transfers/does-not-transfer;
      TwoRoom boundary note; sec 9 replan-rate para; sec 6 denominators +
      selection defence; limitations/conclusion/abstract updated
- [x] FINAL-NUMBER PASS round 1 DONE 2026-07-13 in main_v2 (tau-grid table filled,
      k/S exact numbers, dual-population spine, gamma full curve, gc 2x2, timing,
      Reacher fail-safe note, goal-gate negative, pareto fig regenerated)
- [ ] FINAL-NUMBER PASS round 2 (gated by 2271772): swap headline/spine/repl spec
      cells to the *_s10speck3_* derived-config numbers; NFE figures to 1.9/replan;
      Reacher headline cells to the k=4 respin (95.5/95.5/95.7)
- [ ] §5/§7.1/§8 REWRITE to the churn + derivation story (floor tables in §5;
      regime-dependent verification + blind rows in §7/§8; self-test retraction)
- [ ] update headline table (Table 8) if a Reacher row belongs there
- [ ] compile check (needs iclr2026_conference.sty dropped into paper/ — not on
      this machine; structural checks pass)

gated ->

## EXP 3 - optional second env                        [gated by: DECIDE + WRITE 1]
- [ ] TwoRoom chain (only if un-parked): train array + eval arms -> sec 11.2
- [ ] OGBench-Cube (only if built): port env driver + build_subgoals -> sec 11.3

gated ->

## WRITE 2 - final pass                               [gated by: WRITE 1 (+ EXP 3 if run)]
- [ ] conclusion tbdblock: rewrite from placeholder
- [ ] limitations: update (ii) cross-env generality to match what actually ran
- [ ] cut or fill remaining tbdblocks (related work / algorithm box if not done in WRITE 0)
- [ ] full recompile: zero tbdblocks, zero (?) citations
