# SURVEYOR

Code for *SURVEYOR: Certified Speculative Plan Consumption for Latent
World-Model Planning*.

A drafter proposes a block of `N` subgoal latents once. At every replan
boundary the achieved latent is verified against the waypoint just pursued:
within a relative distance `tau` the next pre-drafted waypoint is served at no
drafter cost, otherwise the block is redrafted from the current state. There is
nothing to train and nothing to tune. `tau` is the encoder's criterion floor and
`k` the sampler's convergence point, both read off offline measurements in
minutes.

The substrate is a frozen LeWM JEPA encoder with a CEM controller. The same rule
is served, unchanged, through a GC-IDM executor, a DINO-WM encoder and V-JEPA 2.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The dependency list was validated by a clean-room rebuild. Run everything from
the repository root, which is what puts `surveyor` on the import path.

## Data and checkpoints

Datasets and LeWM encoder checkpoints come from the
[LeWM collection](https://huggingface.co/collections/quentinll/lewm); place the
extracted `.h5` files under `$STABLEWM_HOME` (default `~/.stable-wm/`). The eval
drivers take the dataset path directly with `--h5`, and resolve the encoder
either from the Hub (`--source pretrained --encoder-id quentinll/lewm-cube`) or
from a local directory (`--source local --local-dir encoder_reacher`).

## Quickstart

Four steps, PushT as the example. Steps 1 to 3 are per environment and run once.

```bash
H5=$STABLEWM_HOME/pusht_expert_train.h5

# 1. dense subgoal latents: every frame of every episode through the encoder
python -m surveyor.envs.pusht.build_subgoals --source pretrained --h5 "$H5" \
    --out subgoals_dense_full.pt --stride 1 --device cuda --batch-size 256

# 2. derive the two constants offline: tau from the criterion floor, k from
#    where the sampler stops moving. No closed loop, no search.
python -m surveyor.probes.probe_floor --env pusht --h5 "$H5" \
    --source pretrained --device cuda --stride 10 --tau 0.20 --seed 42 \
    --n-anchors 512 --pair-pool 120000 --ks 1 2 3 4 5 6 8 12 16 50 \
    --json-out floor_pusht.json

# 3. train the drafter at the matched subgoal scale
python -m surveyor.train_drafter --subgoals subgoals_dense_full.pt \
    --out gdm_stride10.pt --device cuda --mask 5 --subgoal-step 10 \
    --batch-size 128 --lr 5e-5 --weight-decay 1e-3 --grad-clip 1.0 \
    --lr-schedule warmup_cosine --epochs 20 --ema-decay 0 --amp

# 4. fixed evaluation population, then one certified cell at t=150
python -m surveyor.envs.pusht.build_populations "$H5"
python -m surveyor.envs.pusht.eval --subgoal surveyor \
    --gdm-ckpt gdm_stride10.pt --accept-tau 0.20 --gdm-steps 8 \
    --horizon 2 --receding-horizon 2 \
    --source pretrained --h5 "$H5" --device cuda \
    --mode long --episodes-file pusht.episodes150.json \
    --goal-offset 150 --eval-budget 300 \
    --num-eval 256 --score block --angles 20 5 --seed 42 --cem-seed 42
```

Each run prints one `[RESULT]` line with the cell's success rate as a pooled
numerator over denominator. Cells combining seeds pool those counts rather than
averaging percentages.

## Arms

`--subgoal` selects the policy, and the values are the rows of the paper's
summary table. `surveyor/arms.py` lists them all, including the controls and the
negative results, and maps the development names that appear in older logs.

| arm | policy |
|---|---|
| `flat` | LeWM CEM against the goal image |
| `ffjepa` | a freshly drafted subgoal at every replan |
| `surveyor` | draft a block, verify against reality, serve or redraft |
| `gcidm` | GC-IDM, the amortized goal-reaching baseline |
| `gcidm+surveyor` | the same rule with GC-IDM as the executor |
| `router+surveyor` | `c*` routes the episode, then retires the drafter on arrival |
| `paired+surveyor` | the accept test verified in a second, frozen encoder |

Swapping the arm is the only difference between the compared runs. The
configurations behind the paper's cells:

```bash
# PushT, flat and FF-JEPA references for the cell above
--subgoal flat   --horizon 5 --receding-horizon 5
--subgoal ffjepa --gdm-ckpt gdm_stride10.pt --horizon 2 --receding-horizon 2

# Reacher, routed. Its population file comes from
#   python -m surveyor.envs.reacher.build_populations --h5 "$H5" \
#       --out reacher_horizon150.json --n 128 --max-offset 150 --episode-min 8000
python -m surveyor.envs.reacher.eval --subgoal router+surveyor \
    --gdm-ckpt gdm_reacher_s10.pt --gdm-steps 8 --accept-tau 0.20 \
    --source local --local-dir encoder_reacher --h5 "$H5" --device cuda \
    --num-eval 128 --seed 42 --episodes-file reacher_horizon150.json \
    --goal-offset 150 --eval-budget 300 --horizon 2 --receding-horizon 2

# Cube, GC-IDM as the executor, held out
python -m surveyor.envs.cube.eval --subgoal gcidm+surveyor \
    --gcidm-ckpt gcidm_cube_h300.pt --gdm-ckpt gdm_cube_s10_gc.pt \
    --gdm-steps 3 --accept-tau 0.20 --sg-steps 10 --goal-gate \
    --source pretrained --encoder-id quentinll/lewm-cube --h5 "$H5" \
    --device cuda --start final --goal-offset 150 --eval-budget 300 \
    --num-eval 128 --seed 42 --cem-seed 42 --episode-min 8000

# Two-Room, verifier transplanted into a frozen external encoder
python -m surveyor.envs.tworoom.eval --subgoal paired+surveyor \
    --gdm-ckpt gdm_tworoom_s10_gc_paired.pt --gdm-steps 50 --accept-tau 0.098 \
    --h5 "$H5" --device cuda --mode long --goal-offset 75 --eval-budget 150 \
    --num-eval 64 --episode-min 4000 --goal-from-proprio --eval-filter none
```

## Offline instruments

None of these run a closed loop, and all of them are read before the cells they
govern.

```bash
# is this substrate a candidate at all? the verification gap
python -m surveyor.probes.gap_stat --name pusht --subgoals-pt subgoals_dense_full.pt \
    --equiv-stride 1 --hop-stride 10 --tau 0.20 --json-out gap_pusht.json

# does the accept test track ground truth? calibration at the task's own radius
python -m surveyor.probes.calibrate_verifier --env pusht --h5 "$H5" \
    --traces 'runs/pusht/traces_*.pt' --tau 0.20 --out cal_pusht.json --device cuda

# how far into a block can a draft be trusted?
python -m surveyor.probes.probe_suffix_decay --gdm-ckpt gdm_stride10.pt \
    --h5 "$H5" --episodes-file pusht.episodes150.json
```

`surveyor/probes/__init__.py` lists the rest, including the routing post-mortem
and the retired horizon gates, which are kept because the paper reports them.

## Layout

```
surveyor/            the method: encoder IO, drafter, accept rule, sources,
                     policies, training and the offline probes
surveyor/envs/       per-environment eval drivers and dataset builders
surveyor/dinowm/     the DINO-WM transplant (second architecture)
surveyor/vjepa2/     the V-JEPA 2 transplant (foundation scale)
surveyor/dspark/     the ported DSpark head, kept as a negative result
config/, *.py        the LeWM training and evaluation entry points, upstream
```

`python -c "import surveyor; help(surveyor)"` prints the module map.

## Pre-registration

Every quantitative claim traces to a pre-registration document that fixed the
definitions, populations, thresholds and pass bars before the runs, and was
amended only by appending the outcome, including the outcomes that missed.
Those documents are held with the manuscript rather than in this repository;
the identifiers cited in the source (`P-EXEC-1`, `M1`, and the rest) name
them.

## Attribution

This repository extends [LeWM](https://github.com/lucas-maes/le-wm) (Maes, Le
Lidec, Scieur, LeCun and Balestriero), whose JEPA world model, training script,
evaluation harness and configs are carried here as released, under the original
MIT licence in `LICENSE`. The substrate we build on is FF-JEPA's subgoal
drafting on that model. The transplants target
[DINO-WM](https://github.com/gaoyuezhou/dino_wm) and
[V-JEPA 2](https://github.com/facebookresearch/vjepa2), and the amortized
baseline is GC-IDM. Cluster job scripts are not included: they wrap the commands
above with site-specific accounts and paths.
