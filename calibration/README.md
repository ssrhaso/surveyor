# Calibration

Outputs of the offline tolerance probes (no planner, no closed loop). An environment's tolerance is the median
(`p50`) of the latent distance between pairs of observations that its success criterion counts as the same
state; `p50_ci95` is a bootstrap interval of that median. Dataset paths inside the files are reduced to their
basenames. Tolerances reach the runs as `--accept-tau` (accept rule) and `--retire-tau` (arbiter) on the manifest
command lines, and `reproduce/build_robustness_tables.py` reads these files for the tolerance tables.

- `floor_pusht_U.json`: PushT, tau = 0.233 (`criterion_floor_20deg`; the strict 5-degree reading, `criterion_floor_5deg`, gives 0.173).
  `python -m surveyor.probes.probe_floor --env pusht --h5 $DATA/pusht_expert_train.h5 --source pretrained --encoder-id quentinll/lewm-pusht --device cuda --stride 10 --tau 0.20 --seed 42 --n-anchors 512 --json-out calibration/floor_pusht_U.json`
- `floor_reacher_U.json`: Reacher, tau = 0.106 (`criterion_floor_reacher`: both joints within 0.05 rad).
  `python -m surveyor.probes.probe_floor --env reacher --h5 $DATA/reacher.h5 --source local --local-dir encoder_reacher --device cuda --stride 10 --qpos-tol 0.05 --tau 0.20 --seed 42 --n-anchors 512 --episode-min 8000 --json-out calibration/floor_reacher_U.json`
- `floor_cube_U.json`: Cube, tau = 0.794 (`criterion_floor_cube`: block and effector within 0.04 m).
  `python -m surveyor.probes.probe_floor --env cube --h5 $DATA/cube_single_expert.h5 --source pretrained --encoder-id quentinll/lewm-cube --device cuda --stride 10 --cube-tol 0.04 --tau 0.20 --seed 42 --n-anchors 512 --pair-pool 120000 --json-out calibration/floor_cube_U.json`
- `gap_tworoom_paired_Z_fullscene.json`: Two-Room accept rule (DINOv2 space), tau = 0.127 (pairs from different episodes, agent and target within 16 px).
  `python -m surveyor.envs.tworoom.probe_dino_gap --h5 $DATA/tworoom.h5 --device cuda --hops 5 10 25 --episodes 600 --episode-max 4000 --pairing different --match-target --out calibration/gap_tworoom_paired_Z_fullscene.json`
- `gap_tworoom_paired_Z_same.json`: Two-Room, same-episode pairs: 0.098 (the transferred Two-Room tolerance).
  `python -m surveyor.envs.tworoom.probe_dino_gap --h5 $DATA/tworoom.h5 --device cuda --hops 5 10 25 --episodes 250 --episode-max 4000 --pairing same --out calibration/gap_tworoom_paired_Z_same.json`
- `gap_tworoom_paired_Z_different.json`: Two-Room, agent only, different episodes: 0.142 (battery Z reading).
  `python -m surveyor.envs.tworoom.probe_dino_gap --h5 $DATA/tworoom.h5 --device cuda --hops 5 10 25 --episodes 250 --episode-max 4000 --pairing different --out calibration/gap_tworoom_paired_Z_different.json`
- `gap_tworoom_lewm_TR_fullscene.json`: Two-Room arbiter (LeWM space), retire tau = 1.323 (the pairs of the accept-rule probe, encoded by LeWM; battery TR).
  `python -m surveyor.envs.tworoom.probe_lewm_gap --h5 $DATA/tworoom.h5 --device cuda --source pretrained --encoder-id quentinll/lewm-tworooms --hops 5 10 25 --episodes 600 --episode-max 4000 --pairing different --match-target --out calibration/gap_tworoom_lewm_TR_fullscene.json`
- `gap_tworoom_lewm_TR2_same.json`: Two-Room arbiter (LeWM space), same-episode reading: 1.160 (battery TR2).
  `python -m surveyor.envs.tworoom.probe_lewm_gap --h5 $DATA/tworoom.h5 --device cuda --source pretrained --encoder-id quentinll/lewm-tworooms --hops 5 10 25 --episodes 250 --episode-max 4000 --pairing same --out calibration/gap_tworoom_lewm_TR2_same.json`
- `gap_tworoom_lewm_TR2_agentonly.json`: Two-Room arbiter (LeWM space), agent-only reading: 1.364 (battery TR2).
  `python -m surveyor.envs.tworoom.probe_lewm_gap --h5 $DATA/tworoom.h5 --device cuda --source pretrained --encoder-id quentinll/lewm-tworooms --hops 5 10 25 --episodes 250 --episode-max 4000 --pairing different --out calibration/gap_tworoom_lewm_TR2_agentonly.json`

Cube's block-only reading (only the scored block within 0.04 m, the effector unconstrained) gives 0.967. Its
probe output was not kept, so the value is a literal: `CUBE_BLOCK_ONLY` and the `table_tolsens` spec in
`reproduce/build_robustness_tables.py`, and `--accept-tau 0.967` in the battery Z rows
(`reproduce/battery_Z.tsv`, arms `*_tau0.967`).
