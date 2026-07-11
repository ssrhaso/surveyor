# PLAN (tick sheet)

Paper: `writeup/main.tex`. Rule: update the paper ONCE per completed block, not per job.
Method = spec-accept; "DSpark" = the ported negative-result baseline only.

---

## GATE 0 - unblock
- [ ] push main to GitHub (user; everything is committed locally, boxes clone from GitHub)
- [ ] compute up, either:
  - [ ] Isambard back -> `bash batch/deploy_reacher_isambard.sh reacher`
  - [ ] ISCA v03/v05 -> `git pull` (or re-paste `batch/box_bootstrap.txt`) then use `batch/run_reacher_local.sh`

## WRITE 0 - paper work with no experiment dependency (do anytime)
- [ ] references.bib: replace all 6 TBD stubs (LeWM = arXiv 2603.19312, FF-JEPA = arXiv 2606.09311; CEM, VLWM, OGBench, DSpark)
- [ ] algorithm box tbdblock (~L159): spec-accept pseudocode + CEM cost
- [ ] related work tbdblock (~L115)
- [ ] terminology sweep: spec-accept vs DSpark usage

## DECIDE - scope (anytime before WRITE 2)
- [ ] sec 11.2 TwoRoom: un-park (chain exists, Isambard-only) or CUT + one line in Limitations
- [ ] sec 11.3 OGBench-Cube: build (zero code exists today) or CUT

---

## EXP 1 - Reacher drafters                          [gated by: GATE 0]
- [ ] train 4 goal-cond drafters S in {5,10,15,25} (`run_reacher_local.sh train` or train array sbatch)
- [ ] collect: train loss curves; ckpts `gdm_reacher_s{5,10,15,25}.pt`
- [ ] sanity: loss finite + falling; note 30-60 min CPU pair-build before GPU shows
- [ ] (ISCA only) this run doubles as validation of the 20GB RAM-cap fix - watch peak RSS

gated ->

## EXP 2 - Reacher battery: LeWM + GDM + spec-accept  [gated by: EXP 1]
- [ ] arms (v03): 36 jobs = {baseline, gdm x4 strides, spec-accept x4 strides} x seeds 42-45, n=128
      collect: SR 20deg/5deg per arm, call_ratio + NFE/ep for spec-accept arms
- [ ] horizon (v05): 24 jobs = {s25gdm, s10gdm, s10spec} x t {25,50,75,100} x seeds {42,43}, budgets 2t AND fixed-50
      collect: SR vs offset per budget regime
- [ ] aggregate: `python batch/aggregate_reacher_results.py` -> one results table
- [ ] check success criteria: (i) interior optimum beats S=25, (ii) spec-accept SR-neutral at call_ratio < 1, (iii) gdm >= baseline
      (judge on horizon; the single-hop reach is near-solved, no headroom)

gated ->

## WRITE 1 - Reacher section, one pass                [gated by: EXP 2]
- [ ] fill sec 11.1 tbdblock: SR vs scale table + spec-accept NFE column
- [ ] add horizon table/figure if the sweep is clean
- [ ] 1-2 paragraphs of prose: does the scale law transfer, does spec-accept stay neutral
- [ ] update Table 8 (headline) if a Reacher row belongs there
- [ ] recompile, confirm sec 11.1 tbdblock gone

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
