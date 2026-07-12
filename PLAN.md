# PLAN (tick sheet)

Paper: `paper/paper (1).tex` (ICLR draft; writeup/main.tex = older interim doc).
Rule: update the paper ONCE per completed block, not per job.
Method = spec-accept; "DSpark" = the ported negative-result baseline only.

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
- [ ] references.bib: replace all 6 TBD stubs (LeWM = arXiv 2603.19312, FF-JEPA = arXiv 2606.09311; CEM, VLWM, OGBench, DSpark)
- [ ] algorithm box tbdblock (~L159): spec-accept pseudocode + CEM cost
- [ ] related work tbdblock (~L115)
- [ ] terminology sweep: spec-accept vs DSpark usage

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
- [ ] OVERNIGHT GATES (queued, ETA ~04:00): gamma dose-response x40 (running;
      gamma=0 confound gate + mechanism curve) | k-grids both anchors + k-cliff
      k{1,2} x32 (k FREEZE decision) | tau-on-val + off-axis separability |
      timing x4 | reacher S-long x24 | s15/s5 train+eval (PushT interior
      optimum at t150) | success5 spine x40

### CONFIG FREEZE POLICY (adopted 2026-07-12)
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

### EXP 2d - third env: OGBench-Cube                  [biggest accept lever]
- [x] recon: datasets/quentinll/lewm-cube EXISTS on HF (2026-07-12)
- [ ] scope after PushT restage: port = reacher pattern (envs/cube/eval.py
      driver + build_subgoals + batch chain); DECIDE sec 11.3 accordingly

### WRITE RULES (from the 2026-07-12 review; bind WRITE 1/2)
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
- [ ] FINAL-NUMBER PASS after overnight gates (needs: k frozen, goal-free row
      swapped in for its TBD, k-mirror haircut applied per WRITE RULES, lerp +
      oracle numbers into the t25 paragraph, gamma verdict sentence)
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
