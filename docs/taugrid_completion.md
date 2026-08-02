# tau-grid table-completion cells (registered 2026-08-02, BEFORE submission)

Purpose: tab:taugrid (the closed-loop validation of the derived threshold) has
seven never-run cells shown as dashes. User-authorized completion (GPU idle;
same category as docs/spine_t125_completion.md): NEW cells, no banked cell
re-run, no verdict can change. DESCRIPTIVE, NO BAR.

Cells (18 jobs, batch/isca/run_taugrid_completion.sbatch), each identical to
its row's original battery except tau:
- A: pusht t100 k4 (episodes150as100 pop, budget 200, RH2, gdm_stride10,
  n=256 x seeds 42-43): tau in {0.15, 0.25, 0.40}.
- B: pusht val-split t150 k8 (episodes150.val pop, budget 300, RH2,
  n=256 x seeds 42-43): tau in {0.10, 0.40}.
- C: reacher t100 k4 (max-offset-100 pop, builder seed 42, episode-min 8000,
  budget 200, RH2, gdm_reacher_s10, n=128 x seeds 42-45): tau in {0.15, 0.25}.

Declared expectations (descriptive, falsifiable, from the banked neighbors):
- A: tau .15 and .25 land on the plateau (neighbors 64.1 / 65.0 / 65.0);
  tau .40 is the band-edge probe — the floor theory says PushT's wider floor
  keeps it flatter than Reacher's .40 (which lost 5.3pp). If PushT tau .40
  drops MORE than Reacher's relative drop, the floor-ordering sentence in
  sec 5.1 is re-scoped as-is.
- B: tau .10 ~ 63.5 with more calls; tau .40 ~ 63.7 (flat band).
- C: tau .15 in [95.5, 95.9]; tau .25 in [94.1, 95.5].
- Call ratios monotone decreasing in tau everywhere.
Any excursion is reported as-is; no re-run, no tau added after readout.

## READOUT (pending)
