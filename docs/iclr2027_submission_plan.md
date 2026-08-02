# ICLR 2027 submission plan (frozen scope, 2026-08-02)

The packaging pass: turn paper/main_v2.tex (25pp, single blob) into an
ICLR-2027-shaped submission. LeWM's own paper (2603.19312) is the structural
template: tight 9-page main with contribution bullets and an early overview
figure, everything operational pushed to lettered appendices.

## 1. Hard constraints (from the author guidelines; the ones that bind us)

| Constraint | Consequence for us |
|---|---|
| Abstract deadline **Sep 18 AOE**; full paper **Sep 25 AOE**; no exceptions | abstract must be final-quality by Sep 18 (used for reviewer bidding; placeholders deleted); author list FROZEN at abstract deadline |
| **Main text <= 9 pages at submission** (10 at rebuttal/camera), strictly enforced, desk-reject | current 25pp -> 9pp main + appendices; references DO NOT count; appendices unlimited, after references |
| Official ICLR 2027 style files | downloaded to paper/iclr2027_kit.zip (iclr2027_conference.sty/.bst + math_commands + fancyhdr + natbib); swap from placeholder kit |
| Double-blind; identity leak = desk reject | anonymization pass: no authors/acks; own prior work in 3rd person; NO ISCA/user paths or repo links in text; code via anonymized zip or anonymous repo |
| **AI use statement REQUIRED** (no page cost) | must be written (honest: AI assistance in experiment orchestration, analysis tooling, and manuscript drafting; per ICLR AI Policy for Authors) |
| Reproducibility statement recommended, end of main before refs, no page cost | we have unusually strong material: prereg docs, clean-room rebuild (repro_smoke_prereg.md), derived-constant scripts |
| Ethics statement optional | not needed (sim benchmarks, no human subjects, no sensitive data) — omit |
| Supplementary: single PDF (paper+appendix) encouraged; code zip encouraged | one PDF; code as anonymized zip (specaccept/ + vjepa2 calibrate + batch scripts + prereg docs) |
| Reciprocal reviewing: >= 1 author registered + qualified (accepted pub at listed venues); authors on >= 3 papers must review 6 | ADMIN (user + supervisor): OpenReview profiles up to date; decide who registers as reviewer |

## 2. Target structure (LeWM-as-template), 9-page main

LeWM's shape we mirror: Abstract -> Intro (contributions as bullets, overview
figure on p1-2) -> Related Work -> Method -> Results -> Analysis -> Conclusion
+ Limitations -> refs -> appendices A..N each owning one operational topic.

Main text budget (~9.0pp):

1. **Introduction** (~1.25pp) — certification frame; contribution bullets
   (already reframed); Fig. 1 = the method schematic (draft-verify-serve loop
   + the certification layer around it; TO MAKE, tikz or matplotlib).
2. **Related work** (~0.5pp) — compressed; SAGE/TRM/DINO-WM/V-JEPA/LeWM.
3. **Method: certified spec-accept** (~1.5pp) — spec-accept rule; derived
   constants (tau = criterion floor, k = sampler convergence, S rule); c*
   arbiter; the FORMAL certified-definition paragraph (to write: certified =
   measured, calibrated, prospectively validated; explicitly not
   formal-proof). Background (LeWM planner) folded to 2-3 sentences + cite.
4. **The verification gap instrument** (~0.75pp) — gap definition, decision
   rule, the five-substrate figure, prospective record summary.
5. **Results: the frozen recipe** (~2pp) — tab:grand + spine figure;
   one-policy story; TwoRoom held-out arc compressed to one paragraph +
   crossover figure reference; efficiency 2-3 sentences (fig:cost to
   appendix).
6. **Certification: the verifier held to account** (~1.5pp) — tab:calibration
   + fig:cert-cal + fig:cert-decision; ignition; rate-vs-content decomposition;
   M2; M1-FM foundation row.
7. **Generality and scope** (~0.75pp) — DINO-WM autopsy (correct-detection),
   V-JEPA transfer/boundary, TwoRoom encoder-property, all compressed;
   pointer to appendix for full verdicts.
8. **Discussion, limitations, conclusion** (~0.75pp) — honest scope (one
   substrate end-to-end, toy scale), churn thesis summary, future work
   (closed-loop at scale needs hardware).
+ Reproducibility statement + AI use statement (after conclusion, before
  refs; zero page cost).

Appendices (after references, unlimited; each lettered, LeWM-style):
- A. Full protocol tables + populations (tab:protocols, tab:spine detail)
- B. Complete grids and per-cell results (current Controls & Full Grids)
- C. Calibration details: probe classes, amendments, per-env JSONs, FA(r)
  tables, operating scatters for all four envs (3 more panels TO MAKE)
- D. The corruption sweep + rate sweep full tables
- E. Negative results and the refutation ledger (paper feature)
- F. Faithful reproduction & audit (current sec)
- G. DINO-WM + V-JEPA full pre-registered verdicts
- H. TwoRoom arc detail (anchor attribution, fairness, timing)
- I. Pre-registration methodology (how bars were frozen; pointer to the
  prereg docs shipped in supplementary)
- J. Efficiency/timing detail (fig:cost + tables)

## 3. Execution order

1. [x] Fetch official kit (paper/iclr2027_kit.zip)
2. Create paper/iclr2027/ build: new main file from iclr2027_conference.tex
   skeleton; \input the reorganized sections; keep main_v2.tex untouched as
   the working archive until the new build supersedes it
3. Restructure: split main_v2 content into sections/*.tex (main) and
   appendix/*.tex per the map above — move first, compress second
4. Compression pass to <= 9.0pp (order: results prose -> intro -> method)
5. Write: certified-definition para, AI statement, repro statement, Fig. 1
6. Anonymization pass (grep for names/paths/links; third-person self-cites;
   acks removed)
7. Supplementary: single PDF (appendix in same file, after refs); code zip
   (anonymized: strip user paths/hostnames from batch scripts) + prereg docs
8. Bib: fix 6 [CHECK authors]; switch to iclr2027_conference.bst natbib
9. Full-PDF read-through; pdfdiff sanity vs main_v2 claims
10. ADMIN (user): OpenReview profiles for all authors; reviewer registration
    decision; author list final by Sep 18 AOE; abstract text final by Sep 18

Timeline: poster (Aug 11) first, then this pass through late August;
freeze week of Sep 8; abstract in by Sep 15 (buffer); full by Sep 22 (buffer).

## 4. Anonymization checklist (desk-reject class; check ALL)

- [ ] no author names/affiliations/acknowledgements in submission build
- [ ] no /lustre/home/ha676, ISCA, Isambard, or personal paths in text OR
      supplementary code
- [ ] no links to non-anonymous repos; code = anonymized zip
- [ ] own related preprints (if any go up before Sep) cited third-person
- [ ] figure/PDF metadata scrubbed (pdflatex defaults fine; check \author)
- [ ] prereg docs in supplement scrubbed of usernames/hostnames/job IDs kept
      only where meaningless

## 5. Statement drafts (to finalize in step 5)

- **AI use**: AI assistance (large language model) was used for experiment
  orchestration tooling, analysis scripting, and manuscript drafting/editing
  under author direction; all experimental designs, pre-registrations, and
  scientific claims were specified and verified by the authors. (Align with
  final ICLR AI Policy wording before submission.)
- **Reproducibility**: constants derived by the released scripts in minutes
  per environment (App. C); every claim pre-registered with frozen bars
  (App. I + supplementary prereg docs); complete per-cell grids (App. B);
  dependency spec validated by a clean-room rebuild that caught and fixed two
  undocumented packages before release; code + configs in supplementary.
