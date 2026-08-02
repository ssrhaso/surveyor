# Verification-audit brief (write 2026-08-02; paste body to a cold agent)

You are an independent verification auditor with full repo access. A research
program believes its experimental ledger is complete, its verdicts banked, and
its ICLR 2027 submission build structurally sound. Your ONE job: try to catch
it out. Verify that every quoted number traces to a banked verdict, that no
retracted claim survives anywhere in the manuscript, and that the submission
build has no desk-reject hazards. You run NO experiments and change NO files:
you read, cross-check, and report. Target: a ranked defect list, max 2 pages.

## Trust order (highest wins on any conflict)

1. docs/certification_prereg.md, docs/randreject_prereg.md,
   docs/repro_smoke_prereg.md, docs/tworoom_paired_prereg.md,
   docs/dinowm_prereg.md — frozen definitions AND verdicts
2. PLAN.md top STATUS blocks (07-31 + 08-01 addenda)
3. Results/ artifacts (calibration JSONs/CSVs, acceptrate CSVs)
4. paper/main_iclr2027.tex (live build) + paper/sections/ + paper/figs/make_cert.py
5. Raw logs on ISCA (/lustre/home/ha676/le-wm/logs/) — only if local sources conflict
NEVER trust Results/RESULTS.md (stale since 7/3). paper/main_v2.tex is a
frozen archive; audit main_iclr2027.tex.

## PART A — number tracing (every check lists its ground truth)

1. tab:grand + tab:unified numbers vs the prereg verdict tables and PLAN.md
   (pusht spine 99.5/98.0/98.8/98.1/98.1; reacher policy 98.7/96.3/92.3/90.1;
   cube 79.9; margins row).
2. tab:calibration vs Results/calibration/*.json under the conservative-row
   rule: cube ridge FA .000; pusht MLP FA .102 (NOT ridge .024); reacher
   circ-MLP FA .400 rho .978 (frozen raw-angle row recorded unavailable);
   tworoom ridge .522. Replay mismatches 0 everywhere.
3. sec:results-cert decision paragraph vs docs/randreject_prereg.md verdicts:
   coin p derivations (.4937 = 7406/15000, .4902 = 5007/10213), coin SRs
   98.05/90.33, banked spec 98.14/91.11, sweep points (reacher 84.57/85.74/
   90.43; pusht 96.97/96.39), knee ~0.25, blind −18pp confined to p=0.
4. Ignition numbers vs certification_prereg R1 verdict: cr .59/.66/1.00/1.00,
   advances 518/420/2/0, blind floor .34, band [tau/2, 2tau].
5. M1-FM paragraph vs the M1-FM verdict + reconciliation: within accept .85,
   cross accept .0008 (99.9% rejection), rho 0.526, REQUIRED labels present
   (dataset-pair label + tau-metrically-loose sentence). FA(r) quotes
   (.400->.064 at 2x, .003 at 3x; .522->.130) vs Results/calibration/events_*.csv
   — RECOMPUTE these four from the CSVs yourself.
6. M2 quotes (.88->.57, .75->.58, cube .84, tworoom .958) vs
   Results/acceptrate_mine.csv + acceptrate_mine_tworoom.csv.
7. Held-out bridge paragraph vs tworoom_paired_prereg verdicts (+15.4, 12/12
   seeds, t~11, oracle 57.81/56.90, crossover [40,50], fairness +6.8/+8.6,
   +2% wall-clock) and efficiency bridge vs the moved Appendix B content.
8. fig:cert-decision + fig:cert-cal (paper/figs/make_cert.py): verify the
   hard-coded arrays against the prereg verdicts, and the FA(r)/scatter
   panels against the events CSVs.

## PART B — retraction & no-quote blacklists (grep the WHOLE build incl.
## sections/ and appendix; ANY hit is CLAIM-INTEGRITY severity)

Retracted/refuted claims that must NOT appear as live claims:
- "beats the oracle" at long horizon (oracle staleness artifact, 7/2)
- performance "improves with horizon" / t150 > t75 (population artifact, 7/3)
- +18.8pp TwoRoom drafting margin; +10.9 at t25 (both retracted 7/29)
- stride-saturation as a scope rule (refuted by its own frozen prediction)
- k*=2 from bias-of-means (refuted closed-loop; diversity collapse)
- 15x planning-compute claim on V-JEPA 2 (buried at n=36, 7/25)
- per-event selectivity drives closed-loop SR (refuted by B1 coin tie)
- DINO-WM "unresolved serving fault" as current status (superseded by the
  08-01 autopsy: serving exonerated, drafter content at fault)
- any 5-degree PushT criterion as a headline (20-degree is the criterion)
- P-CERT-2 "graceful vs compounding" as if confirmed (refuted; white jitter)
No-quote numbers that must NOT appear anywhere: R2 trace-cell SRs; repro-smoke
SRs (71.88/90.62/51.56/71.88 at seed 60); gap-reconcile diagnostics as claims.

## PART C — internal consistency

1. Abstract + contribution bullets: every quantitative claim appears and is
   supported in the body (rho 0.98, 31pp prospective call, +94pp, +15.4pp,
   1.9-2.4 evals, 11/11).
2. Post-restructure integrity: the three moved blocks (held-out, efficiency,
   negatives) exist exactly once; bridges' numbers match their appendix
   twins; no orphaned \Cref (compile and grep the log for "undefined" and
   "multiply defined"); tab:spine/fig:cost references still resolve.
3. The certified-definition question: does the paper anywhere imply formal
   guarantees? "Certified" must mean measured/calibrated/prospectively
   validated (the formal definition paragraph is a KNOWN missing item — flag
   any sentence that overreaches ahead of it).
4. sections/instrument_generality.tex: verify the DINO-WM verdict paragraph
   carries the autopsy (exonerated serving / drafter content / 97% correct
   detection) and P1/P2/P5 as unadjudicated-for-method.

## PART D — ICLR 2027 shape & desk-reject hazards (build: main_iclr2027.tex)

1. Style: iclr2027_conference.sty/.bst actually used; compiles 2-pass +
   bibtex with zero undefined citations.
2. Anonymity sweep over the tex, sections/, figs scripts, and everything
   destined for supplementary (incl. batch/ and docs/ if shipped): grep for
   author names, "ha676", "lustre", "isca", "isambard", personal emails,
   non-anonymous repo URLs, acknowledgements. Any hit = DESK-REJECT class.
3. Statements: AI-use (REQUIRED by ICLR 2027) and reproducibility statement —
   both KNOWN missing; confirm nothing else required is missing (ethics not
   needed: sim only).
4. Page trajectory: appendix start page from the aux label `appendixstart`;
   main text must reach <= 9pp — currently ~17, compression in progress; flag
   only if any content was LOST rather than moved.

## What NOT to flag (known work-in-progress, already scheduled)

Compression 17->9; certified-definition paragraph; AI/repro statements;
anonymization pass; six [CHECK authors] bib entries; Fig-1 schematic
check; poster. DO flag if this open-items list is itself incomplete.

## DELIVERABLE

PART 1: pass/fail per category A-D with one-line evidence each.
PART 2: ranked defects — file:line, severity (DESK-REJECT / CLAIM-INTEGRITY /
MINOR), one-line fix. An honest empty list must be argued, not asserted.
PART 3: anything missing from the open-items list above.
