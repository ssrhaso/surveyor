# Pre-registration documents

Every quantitative claim in the paper traces to one of these files. Each was
frozen before the runs it governs: it fixes the definitions, the populations,
the thresholds, the pass bars and the predictions, and is then amended only by
appending the banked outcome, including the outcomes that missed. Files are
named by freeze date.

| file | registers |
|---|---|
| `2026-07-26_dinowm_instrument.md` | the verification gap as a prospective applicability test on DINO-WM |
| `2026-07-27_tworoom_paired.md` | the paired external-encoder verifier on Two-Room |
| `2026-07-31_certification.md` | verifier calibration, corruption ignition, and the foundation-scale row |
| `2026-08-01_random_rejection.md` | the matched-rate random-rejection control |
| `2026-08-01_repro_smoke.md` | the clean-room rebuild of the dependency spec |
| `2026-08-02_spine_t125.md` | the interior-offset table-completion cells |
| `2026-08-02_tau_grid.md` | the closed-loop validation grid for the derived threshold |
| `2026-08-03_encoder_swap_matrix.md` | the verification-space matrix across encoders |
| `2026-08-07_baselines.md` | the random floor, GC-IDM, and the cited comparators |
| `2026-08-07_composite_and_executor.md` | the routed composite and the executor swap |
| `2026-08-07_gcidm_branches.md` | which GC-IDM branch each cell serves |
| `2026-08-07_hmax_tradeoff.md` | the horizon-input sweep of the amortized baseline |
| `2026-08-08_overnight_hardening.md` | seed replicates and the blind executor control |
| `2026-08-09_cube_executor_and_tworoom_gcidm.md` | the held-out Cube executor cell and Two-Room GC-IDM |
| `2026-08-10_composite_v2.md` | the single route reassignment, P-COMP-2 |
| `2026-08-11_route_v3_and_executor_seeds.md` | the route repair and executor training-seed robustness |

Three conventions to read them by. Module paths, arm names and the method's name
were updated to the released spellings when the package was renamed, so the
commands quoted here run as written; `surveyor/arms.py` maps the development
names that appear in older logs. Scheduler job identifiers are kept as opaque
provenance labels for the runs behind each verdict. Citations of
`batch/*.sbatch` name the job script that ran a cell: those scripts are
site-specific wrappers around the commands in the root README and are not part
of the released tree.
