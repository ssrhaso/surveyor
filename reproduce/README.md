# Reproducing the tables and figures

Everything here runs from this directory's own files and `../calibration/`; nothing outside the repository is read.

## Tables and figures from the harvested results

```
python reproduce/build_table1_latex.py out/tables          # Tables 1 and 2; per-environment, ablation, range, cost, LeFlow, GC-IDM tables
python reproduce/build_arbiter_table.py out/tables         # arbiter activity (also rewrites reproduce/arbiter_activity.csv)
python reproduce/build_robustness_tables.py out/tables     # tolerances, transfer, tolerance sensitivity, drafter seeds, budget k, demonstrations
python reproduce/build_competence_table.py out/tables      # executor search budget
python reproduce/build_beyond_table.py out/tables          # beyond t = 150
python reproduce/build_costbenefit_table.py out/tables     # cost per solved episode at the longest distance
python reproduce/build_costbenefit_table.py out/tables short   # the same at the shortest distance
python reproduce/build_fig_main.py out/figures             # Figure 1 and the cost figure
python reproduce/build_fig_sr_distance.py out/figures      # success against distance, per executor
```

Inputs, all in this directory:

- `rb<X>_cells.csv`: one line per (environment, arm, goal distance t) of battery X: the success rate per evaluation seed,
  its mean and standard error over the eight seeds, and the drafter call ratio.
- `battery_<X>.tsv`: the manifest of battery X, one run per row (idx, env, arm, t, seed, log, cmd).
- `rb*_sacct.txt`: scheduler wall-clock per array task (`JobID|State|Elapsed`); `build_table1.JOBS` maps each array job
  to its manifest, whose `--num-eval` turns seconds per task into seconds per episode.
- `*_banners_*.txt`: the arbiter's final banner line per router log (`<log name>\t<line>`); `arbiter_activity.csv` is
  what `build_arbiter_table.py` derives from them.

`build_table1.py` switches: `SURVEYOR_TAU=served` (uniform tolerance 0.20 instead of each environment's own),
`SURVEYOR_TR=off` (Two-Room without battery TR), `SURVEYOR_TIMING=gh200` (the cluster B timing set). Cluster A has
NVIDIA A100 GPUs, cluster B NVIDIA GH200 GPUs; every seconds column in the paper comes from cluster A.

## Re-running experiments

Each manifest row is one run: one evaluation seed of one configuration at one goal distance. Commands run from the
repository root and need `DATA` (the directory holding `pusht_expert_train.h5`, `reacher.h5`, `cube_single_expert.h5`
and `tworoom.h5`, see `external/README.md`), the evaluation populations in `episodes/`, `pusht_c2222.h5` (data
preparation below) and the checkpoints each command names (`--gdm-ckpt`, `--gcidm-ckpt`, `--leflow-ckpt`), which are
not part of this repository.

```
bash reproduce/run_manifest.sh reproduce/battery_A.tsv 17        # one row
sbatch --array=0-711 --export=ALL,MANIFEST=reproduce/battery_A.tsv,DATA=$DATA reproduce/run_manifest.sbatch
python reproduce/harvest_rbA.py rbA          # logs/rbA_*.log -> reproduce/rbA_cells.csv, reproduce/rbA_grid.md
python reproduce/extract_banners.py logs     # router logs -> reproduce/arbiter_banners_pulled.txt, gcrouter_banners_pulled.txt
python reproduce/harvest_union.py --prefix rbL --src logs_A logs_B   # a battery whose logs sit in two directories
```

## Data preparation

The outputs of these steps that the manifests read are already in `episodes/`.

```
python reproduce/data_prep/tworoom_add_state.py --h5 $DATA/tworoom.h5
python -m surveyor.envs.pusht.extract_subset --h5 $DATA/pusht_expert_train.h5 --out pusht_c2222.h5 \
    --seed 2222 --eval-filter success5 --goal-offsets 25 50 75 100 150 --num-eval 256 --compression lzf
python reproduce/data_prep/make_pusht_long_populations.py --h5 pusht_c2222.h5 --goal-offsets 175 200
for T in 100 150 175 195; do
  python -m surveyor.envs.reacher.build_populations --h5 $DATA/reacher.h5 \
      --out reacher_c2222.ep$T.s2222.json --n 128 --max-offset $T --episode-min 8000 --seed 2222
done
python reproduce/data_prep/gen_cube_pairs.py --h5 $DATA/cube_single_expert.h5 --out-dir . --ts 25 50 75 100 150 175 195
python reproduce/data_prep/make_pusht_holdout_mask.py --subgoals subgoals_dense_full.pt \
    --out pusht_train_mask.npy --exclude 'episodes/pusht_c2222.source_episodes*.json'
python reproduce/data_prep/make_split.py --h5 $DATA/pusht_expert_train.h5 --name pusht_expert_train \
    --out pusht_split.json --exclude-json episodes/pusht_c2222.source_episodes*.json
```

`extract_subset` writes `pusht_c2222.h5`, `pusht_c2222.episodes<t>.json` (indices into the subset) and
`pusht_c2222.source_episodes<t>.json` (the same episodes as indices into the full dataset). The Reacher populations use
the held-out episodes (index >= 8000); a file built for the largest distance serves every shorter one. The holdout mask
keeps the evaluation episodes out of the PushT drafter's training set; `make_split.py` does the same for LeFlow.
