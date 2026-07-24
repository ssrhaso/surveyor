# PLAN (tick sheet)

Paper: `paper/main_v2.tex` (ACTIVE — method-paper restructure, 2026-07-13;
`paper (1).tex` = v1 forensic archive feeding the appendices; writeup/main.tex = older interim doc).
Rule: update the paper ONCE per completed block, not per job.
Method = spec-accept; "DSpark" = the ported negative-result baseline only.

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
