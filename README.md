# SURVEYOR

Code for *SURVEYOR: Training-Free Decomposition for Long-Range Latent World-Model Planning*.

SURVEYOR is a training-free layer between an executor (a planner or policy) and a frozen LeWM world model. A drafter
proposes latent subgoals a fixed number of steps ahead. The accept rule reuses the remaining draft while the achieved
latent stays within a tolerance of the subgoal just pursued, and re-drafts otherwise. The arbiter plans flat toward the
goal and skips or retires drafting once that plan is predicted to reach it. The tolerances are measured offline.

## Contents

| Path | What it holds |
|---|---|
| `surveyor/sources.py` | the accept rule (`SurveyorSource`) and the arbiter (`CstarRetireSource`) |
| `surveyor/drafter.py` | the diffusion drafter and its sampler |
| `surveyor/solvers.py` | selects the CEM, iCEM, MPPI or AdamW gradient executor of stable-worldmodel |
| `surveyor/gcidm_executor.py`, `surveyor/gcidm_official.py`, `surveyor/leflow_executor.py` | the GC-IDM and LeFlow executors |
| `surveyor/envs/<env>/eval.py` | one evaluation driver per environment (PushT, Reacher, Cube, Two-Room); every run is one call |
| `surveyor/probes/probe_floor.py`, `surveyor/envs/tworoom/probe_*_gap.py` | the offline tolerance measurements |
| `episodes/` | the fixed evaluation episodes of PushT, Reacher and Cube (Two-Room draws its episodes from the seed) |
| `calibration/` | the tolerance measurements and the command behind each |
| `reproduce/` | the command of every run (`battery_<X>.tsv`), the per-run results, and the scripts that rebuild every table and figure |
| `external/README.md` | the external code (official GC-IDM, LeFlow), package versions, datasets and encoders |

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Run everything from the repository root. Datasets and LeWM encoders are public (see `external/README.md`); set
`DATA` to the directory holding the `.h5` files. Trained checkpoints (drafters, GC-IDM, LeFlow) are not included;
each command names the checkpoint it loads.

## Rebuild the tables and figures

The per-run results are in `reproduce/`, so this step needs no GPU:

```bash
python reproduce/build_table1_latex.py out/tables     # Tables 1 and 2 and the per-environment tables
python reproduce/build_fig_main.py out/figures        # Figure 1 and the cost figure
```

`reproduce/README.md` lists the builder of every other table and describes each input file.

## Re-run an experiment

Each row of a manifest is one run: one seed of one configuration at one goal distance.

```bash
bash reproduce/run_manifest.sh reproduce/battery_A.tsv 0     # one row, locally
python reproduce/harvest_rbA.py rbA                          # logs -> reproduce/rbA_cells.csv
```

`reproduce/run_manifest.sbatch` runs a manifest as a SLURM array. `calibration/README.md` gives the command behind each
tolerance, and `reproduce/README.md` the data preparation that produced `episodes/`.

## Attribution

This repository extends [LeWM](https://github.com/lucas-maes/le-wm), whose world model, training script, evaluation
harness and configs (`config/`, `jepa.py`, `module.py`, `train.py`, `eval.py`, `utils.py`) are carried here as released,
under the MIT licence in `LICENSE`. The drafter follows FF-JEPA's subgoal model; GC-IDM and LeFlow are used through their
authors' public code.
