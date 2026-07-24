# Verdict battery + derived-config re-stamp: ISCA run logs (2026-07-13)

Third and final dev-env log archive (after `../isca_reacher_logs/` 2026-07-11/12 and
`../isca_gates_logs/` 2026-07-12/13-morning). 146 log files + the 2 floor-probe JSONs
= everything run on ISCA after the gates pull. With this directory, **every log of the
entire PushT + Reacher experimental program is stored in-repo**; the datasets on ISCA
are safe to delete (reacher.h5 approved for deletion 2026-07-13 to make room for Cube;
re-downloadable via HF / `batch/download_inspect_reacher.sh`).

## Job map

| Job ID | Log family | What it is |
|---|---|---|
| 2271628/2271652 | `floor_probe_*` (crashed) | first floor-probe attempts (h5py duplicate-index bug, fixed in probe) |
| 2271674 | `floor_pusht.log`, `floor_reacher.log`, `floor_*.json` | **THE FLOOR PROBE** (dense k-grid): PushT criterion floor p50 .233/.173 (20deg/5deg), temporal .080; Reacher .106/p90 .153, temporal .223. Sampler convergence: PushT step k2->k3 (bias 1.04->.11); Reacher no step |
| 2271629 | `pusht_blind2k{2,4,8}_t{100,150}_*` | blind commit-2 at MATCHED k: ties spec in-distribution (65.3/64.3 t100; 58.1/58.2 t150); **k=2 cliff inversion: blind 63.5 vs spec 55.6 -> self-test framing RETRACTED (false rejection)** |
| 2271630 | `reacher_hz150k4_s10spec_*` | Reacher spec k=4 respin (k-freeze fix): t100/125/150 = **95.5/95.5/95.7**; the paper's headline Reacher cells |
| 2271694 | `reacher_blind3_*` | Reacher blind commit-3 (spec tau=999): **87.9/89.8 vs spec 94.5/93.9; verifier load-bearing on Reacher** |
| 2271747 | `pusht_{gdmk3,speck3}_t100_*` | **k=3 prospective confirmation**: es 63.3+-0.5 (plateau, not cliff), spec 64.5+-1.1 at cr .63 = 1.9 NFE/replan; derived, never swept |
| 2271772 | `pusht_{hzs5,hz}_s10speck3_t*`, `pusht_repl_s10speck3_*` | derived-config (tau .20, k=3) re-stamp, 24 cells: repl short 98.7+-1.0 (one 256/256 seed); s5-spine 99.6/97.9/98.8/97.3/98.1; unfiltered 86.5/79.1/70.5/64.1/58.4; all within noise of k=8 originals; cr drifts up on some cells (to .81), SR unchanged, NFE/replan 1.9-2.4 |

## Populations (NEVER splice)
Same conventions as `../isca_gates_logs/README.md`: `hzs5` = success5 population
(box-era comparable), `hz`/anchors = unfiltered populations, repl = short paper
protocol (success5 filter).

## Where these numbers go
- Reacher headline cells -> `reacher_hz150k4` (95.5/95.5/95.7)
- PushT headline/spine spec cells -> `*_s10speck3_*`
- Sec 5 Calibration by Measurement -> floor JSONs + k=3 confirmation
- Sec 7/8 (churn thesis, regime-dependent verification, retraction) -> blind cells
- k=8 originals in `../isca_gates_logs/` are retained as the in-band-alternative row.

Parse with `../isca_gates_logs/parse_logs.py` (same summary-block format).

`reacher_traces.tgz` (175MB) = the complete `runs/reacher/` eval traces (per-replan
achieved/drafted latents from every Reacher battery), pulled 2026-07-13 before the
reacher.h5 deletion, since without the dataset they are no longer cheaply
re-derivable. Raw material for accept-time distance distributions (sec 7 churn
analysis) without any re-run. Kept out of git (over GitHub's 100MB limit; see
.gitignore); lives only in this working copy and on the ISCA pull.

## Addendum: gap-fill batch + Cube staging (2026-07-13 late)

| Job ID | Log family | What it is |
|---|---|---|
| 2272000 | `pusht_randinit_{gdm,speck3}_*` | rand-init column on the certified pipeline: **every-step k=50 = 100.00 x4 AND spec derived (tau .20, k=3) = 100.00 x4 (1024/1024 each, zero failures)**; retires the box-era footnote, fills the spec dash |
| 2272008 (0-7) | `pusht_t75_{gdm,speck3}_*` | t=75 paper protocol certified: every-step 97.9+-0.8, spec 97.2+-0.7 (published: 91.8); last box-era number in the main text replaced |
| 2272008 (8-15) | `pusht_s25spec_tau0{2,4}_*` | **prospective S x tau band test: CONFIRMED**; at S=25, tau=.4 ties tau=.2 (64.2+-1.4 vs 63.2+-0.4) at 27% fewer calls (cr .64 vs .88); the tau band's upper edge scales with the divergence scale as the floor theory predicts |
| 2271997 | `cube_download_*.log` | Cube staged: 95G h5, 10000 eps x 201 frames (same shape as Reacher), pixels 224x224x3, action(5), privileged block pos/quat/yaw + target block pos/yaw + target task, proprio effector; everything needed for the port, the criterion pairs, and the floor probe |
